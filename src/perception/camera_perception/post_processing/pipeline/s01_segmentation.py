"""1단계. Segmentation — best.pt 로 클래스 맵을 얻는다.

**`lane_detection.py` 를 import 하지 않는다.** 이 폴더가 그것을 대체하는 것이
목적이라, 대체 대상을 붙들고 있으면 영원히 떼어낼 수 없다.

---------------------------------------------------------------------------
규격을 상수로 박지 않고 체크포인트에서 읽는다
---------------------------------------------------------------------------
학습 쪽(`seg_model` / `seg_dataset`)을 import 하던 이유는 입력 크기·클래스 수가
조용히 어긋나는 것을 막기 위해서였다. 그 목적은 **체크포인트를 읽는 것으로 더
확실하게** 달성된다.

    input_size     -> 전처리 크기            (박아 두지 않는다)
    class_names    -> 클래스 이름/개수        (박아 두지 않는다)
    num_classes    -> 출력 채널 수
    args.backbone  -> 어떤 ResNet 인지

복사본이 되는 것은 **모델 정의 하나**뿐이다. 그리고 구조가 바뀌면
`load_state_dict` 가 키 불일치로 **즉시 터진다** - 조용히 틀리는 게 아니라
시끄럽게 실패하므로, 이 복사는 안전하다. (`strict=True` 가 기본값이고,
여기서 끄지 않는 것이 핵심이다.)

---------------------------------------------------------------------------
보닛을 여기서 지우지 않는다
---------------------------------------------------------------------------
돌려주는 마스크는 **모델이 뱉은 그대로**다. 보닛 제거는 2단계의 일이고, 여기가
미리 해 버리면 1단계 사진에서 모델이 실제로 무엇을 내는지 볼 수 없다. 그리고
볼 것이 많다 - 실측(last_test 39장) 클래스 픽셀의 68~92% 가 보닛 위 노이즈다.
"""

import os
import sys

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet34, resnet50

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from _common import (BONNET_DILATE_PX, BONNET_POLY, CLASS_BG,  # noqa: E402
                     CLASS_NAMES, CROP_TOP, DEFAULT_SENSOR_ID, IMAGENET_MEAN,
                     IMAGENET_STD, default_cam_set, default_checkpoint,
                     load_camera)

# 모델 정의의 기본 클래스 수. 실제로는 체크포인트 값을 명시적으로 넘긴다.
NUM_CLASSES = len(CLASS_NAMES)

# ==========================================================================
# 모델 정의 - training/seg_model.py 에서 소스째 가져왔다.
#
# 손실(SegLoss)과 혼동행렬은 학습 전용이라 가져오지 않는다. 추론에 필요한 것은
# 그래프뿐이고, 그래프가 어긋나면 load_state_dict 가 즉시 터진다.
#
# ResNet 인코더 + U-Net 디코더다. 차선은 폭 3~7px 로 얇아서, 1/32 특징만 확대
# 하면 선이 뭉개진다. 스킵 연결로 1/4, 1/8 의 고해상 특징을 되살려야 위치가
# 픽셀 단위로 남는다.
# ==========================================================================
_BACKBONES = {"resnet18": resnet18, "resnet34": resnet34,
              "resnet50": resnet50}
# layer1..layer4 의 출력 채널 수
_CHANNELS = {'resnet18': (64, 128, 256, 512), 'resnet34': (64, 128, 256, 512), 'resnet50': (256, 512, 1024, 2048)}


class _Up(nn.Module):
    """2배 확대 후 스킵과 이어붙이고 conv 두 번."""

    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if skip is not None:
            # 입력 크기가 32 의 배수가 아니면 1px 어긋날 수 있다
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
            x = torch.cat([x, skip], dim=1)
        return self.block(x)


class LaneSegNet(nn.Module):
    def __init__(self, backbone="resnet34", pretrained=True, num_classes=NUM_CLASSES):
        super().__init__()
        if backbone not in _BACKBONES:
            raise ValueError(f"모르는 백본: {backbone} (가능: {sorted(_BACKBONES)})")
        net = _BACKBONES[backbone](weights="DEFAULT" if pretrained else None)
        c1, c2, c3, c4 = _CHANNELS[backbone]

        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu)   # 1/2, 64ch
        self.pool = net.maxpool                                    # 1/4
        self.layer1, self.layer2 = net.layer1, net.layer2          # 1/4, 1/8
        self.layer3, self.layer4 = net.layer3, net.layer4          # 1/16, 1/32

        self.up4 = _Up(c4, c3, 256)     # 1/16
        self.up3 = _Up(256, c2, 128)    # 1/8
        self.up2 = _Up(128, c1, 64)     # 1/4
        self.up1 = _Up(64, 64, 32)      # 1/2  (stem 과 이어붙임)
        self.up0 = _Up(32, 0, 16)       # 1/1
        self.head = nn.Conv2d(16, num_classes, 1)

    def forward(self, x):
        s = self.stem(x)                # 1/2
        f1 = self.layer1(self.pool(s))  # 1/4
        f2 = self.layer2(f1)            # 1/8
        f3 = self.layer3(f2)            # 1/16
        f4 = self.layer4(f3)            # 1/32
        d = self.up4(f4, f3)
        d = self.up3(d, f2)
        d = self.up2(d, f1)
        d = self.up1(d, s)
        d = self.up0(d)
        return self.head(d)             # [B, C, H, W] 로짓



class Segmenter:
    """best.pt 를 한 번 올려 두고 프레임마다 클래스 맵을 낸다."""

    def __init__(self, checkpoint=None, cam_set=None, device=None,
                 sensor_id=DEFAULT_SENSOR_ID, crop_top=CROP_TOP):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu"))

        path = checkpoint or default_checkpoint()
        if not os.path.isfile(path):
            raise SystemExit(f"체크포인트를 못 찾았습니다: {path}")
        ck = torch.load(path, map_location=self.device, weights_only=False)

        # --- 규격은 전부 체크포인트에서 ---
        self.input_w, self.input_h = (int(v) for v in ck["input_size"])
        self.class_names = list(ck.get("class_names", CLASS_NAMES))
        n_cls = int(ck.get("num_classes", len(self.class_names)))
        backbone = ck.get("args", {}).get("backbone", "resnet34")

        self.model = LaneSegNet(backbone, pretrained=False, num_classes=n_cls)
        # strict=True (기본값) - 구조가 어긋나면 여기서 터져야 한다
        self.model.load_state_dict(ck["model"])
        self.model.to(self.device).eval()

        self.info = {"epoch": ck.get("epoch"), "backbone": backbone,
                     "num_classes": n_cls, "scheme": ck.get("scheme"),
                     "input_size": (self.input_w, self.input_h),
                     "iou": ck.get("iou"), "path": path}

        # --- 카메라 (4단계 Calibration 이 쓴다) ---
        cs = cam_set or default_cam_set()
        if not cs or not os.path.isfile(cs):
            raise SystemExit("cam_set.json 을 못 찾았습니다")
        self.crop_top = int(crop_top)
        self.cam = load_camera(cs, sensor_id, "horizontal").cropped(self.crop_top)
        self.src_w, self.src_h = self.cam.width, self.cam.height

        # --- 보닛 (2단계 Morphology 가 쓴다) ---
        self.bonnet = self._build_bonnet()

    def _build_bonnet(self):
        m = np.zeros((self.src_h, self.src_w), np.uint8)
        cv2.fillPoly(m, [np.array(BONNET_POLY, np.int32)], 1)
        if BONNET_DILATE_PX:
            k = np.ones((2 * BONNET_DILATE_PX + 1,) * 2, np.uint8)
            m = cv2.dilate(m, k)
        return m > 0

    @torch.no_grad()
    def apply(self, frame_bgr):
        """프레임 -> (mask, crop). mask 는 보닛 제거 **전** 원본 출력이다."""
        img = (frame_bgr[self.crop_top:]
               if frame_bgr.shape[0] > self.src_h else frame_bgr)
        crop = img
        x = cv2.resize(img, (self.input_w, self.input_h),
                       interpolation=cv2.INTER_LINEAR)
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        x = (x - IMAGENET_MEAN) / IMAGENET_STD
        x = torch.from_numpy(x.transpose(2, 0, 1).copy()).unsqueeze(0).to(self.device)

        small = self.model(x).argmax(1)[0].to(torch.uint8).cpu().numpy()
        # **NEAREST 여야 한다.** 클래스 번호를 보간하면 1 과 3 사이에 없던 2 가
        # 생긴다 (white_solid 와 yellow 사이에 white_dashed 가 끼는 식).
        mask = cv2.resize(small, (self.src_w, self.src_h),
                          interpolation=cv2.INTER_NEAREST)
        return mask, crop

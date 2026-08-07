import argparse
import importlib.util
import json
import os
import sys

import onnx
import torch


def parse_model_kwargs(kwargs_list):
    kwargs = {}
    for item in kwargs_list:
        if "=" not in item:
            raise ValueError(f"Invalid model kwarg: {item}. Use key=value format.")
        key, value = item.split("=", 1)
        value = value.strip()
        if value.lower() in {"true", "false"}:
            value = value.lower() == "true"
        else:
            try:
                if "." in value:
                    value = float(value)
                else:
                    value = int(value)
            except ValueError:
                pass
        kwargs[key] = value
    return kwargs


def load_model_class(script_path: str, class_name: str):
    script_path = os.path.abspath(script_path)
    if not os.path.isfile(script_path):
        raise FileNotFoundError(f"Model script not found: {script_path}")

    module_name = os.path.splitext(os.path.basename(script_path))[0]
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to import module from {script_path}")

    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, os.path.dirname(script_path))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)

    if not hasattr(module, class_name):
        raise AttributeError(f"Class '{class_name}' not found in {script_path}")
    return getattr(module, class_name)


def compatible_state_dict(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            new_state_dict[key[7:]] = value
        else:
            new_state_dict[key] = value
    return new_state_dict


def convert_to_onnx(model_script: str, class_name: str, checkpoint_path: str, output_path: str,
                    input_size: tuple, model_kwargs: dict, opset: int):
    model_cls = load_model_class(model_script, class_name)
    model = model_cls(**model_kwargs)

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = compatible_state_dict(checkpoint["state_dict"])
    elif isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = compatible_state_dict(checkpoint["model"])
    else:
        state_dict = compatible_state_dict(checkpoint)

    model.load_state_dict(state_dict, strict=False)
    model.eval()

    dummy = torch.randn(1, *input_size, dtype=torch.float32)

    torch.onnx.export(
        model,
        dummy,
        output_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=opset,
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        do_constant_folding=True,
        verbose=False,
    )

    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)

    print(f"ONNX export completed: {output_path}")
    print(f"Input size: {input_size}, opset: {opset}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert ENet-SAD PyTorch checkpoint to ONNX")
    parser.add_argument("--model-script", required=True, help="Path to the Python script defining the ENet-SAD model class")
    parser.add_argument("--class-name", default="ENet_SAD", help="Model class name to instantiate")
    parser.add_argument("--checkpoint", required=True, help="Path to the PyTorch checkpoint file (.pth or .pt)")
    parser.add_argument("--output", required=True, help="Output ONNX file path")
    parser.add_argument("--input-size", nargs=3, type=int, default=[3, 360, 640],
                        help="Input tensor shape as C H W (default: 3 360 640)")
    parser.add_argument("--opset", type=int, default=11, help="ONNX opset version")
    parser.add_argument("--model-kwargs", nargs="*", default=[],
                        help="Additional model constructor kwargs in key=value format")
    args = parser.parse_args()

    kwargs = parse_model_kwargs(args.model_kwargs)
    convert_to_onnx(
        args.model_script,
        args.class_name,
        args.checkpoint,
        args.output,
        tuple(args.input_size),
        kwargs,
        args.opset,
    )

#!/usr/bin/env bash
# Shared installation/launch contract. Include whole Python packages so new
# helpers cannot be omitted while their importing node is updated.
curvature_runtime_files() (
  set -euo pipefail
  cd -- "$1"
  {
    printf '%s\n' \
      src/bringup/morai_bringup/launch/final_ws_curvature_signal.launch \
      src/bringup/morai_bringup/launch/morai_udp_ekf_purepursuit.launch \
      src/control/purepursuit_mgeo/launch/frenet_avoidance_debug.launch \
      src/detection/camera_perception/launch/camera_perception.launch \
      src/detection/camera_perception/models/yolov8s.pt \
      src/detection/camera_perception/models/best0917.pt \
      src/detection/camera_perception/lane/mgeo/R_KR_PR_K-city_2025/link_set.json \
      src/detection/camera_perception/lane/mgeo/R_KR_PR_K-city_2025/node_set.json \
      src/detection/camera_perception/lane/mgeo/R_KR_PR_K-city_2025/intersection_controller_set.json \
      src/detection/camera_perception/lane/mgeo/R_KR_PR_K-city_2025/traffic_light_set.json \
      data/mgeo/R_KR_PR_K-city_2025/link_set.json \
      src/control/purepursuit_mgeo/src/purepursuit_mgeo/longitudinal_controller.py \
      data/routes/2026_molit_comp_global_path.txt \
      data/routes/roundabout_circulating_path.txt \
      config/roundabout_v61_main.yaml \
      src/control/purepursuit_mgeo/config/highway_overtake.yaml \
      src/control/purepursuit_mgeo/config/highway_overtake_escape4.yaml \
      docker/final_ws/curvature_runtime_files.sh \
      docker/final_ws/install_curvature_signal.sh \
      docker/final_ws/run_test.sh \
      docker/final_ws/highway-test.env.example \
      docker/final_ws/diagnose_curvature_signal.py \
      docker/final_ws/check_highway_launch.py \
      docker/final_ws/smoke_models.py
    local package
    for package in \
      src/experimental/curvature_speed_purepursuit \
      src/control/purepursuit_mgeo src/detection/lidar_perception \
      src/control/turn_signal_controller src/control/stopline_control \
      src/detection/camera_perception \
      src/localization/ekf_local_enu src/localization/gps_mgeo_converter \
      src/localization/morai_udp_bridge src/localization/morai_udp_drive_bridge; do
      printf '%s\n' "$package/CMakeLists.txt" "$package/package.xml" "$package/setup.py"
      find "$package/src" "$package/scripts" -type f -name '*.py'
      if [[ -d "$package/test" ]]; then
        find "$package/test" -type f -name '*.py'
      fi
    done
    find src/localization/ekf_local_enu/launch src/localization/gps_mgeo_converter/launch \
      src/localization/morai_udp_bridge/launch src/localization/morai_udp_drive_bridge/launch \
      src/detection/lidar_perception/launch \
      -type f -name '*.launch'
    find src/detection/camera_perception/lane src/detection/camera_perception/post_processing \
      -maxdepth 1 -type f -name '*.py'
    find src/detection/camera_perception/web -type f
  } | LC_ALL=C sort -u
)

curvature_runtime_manifest() (
  set -euo pipefail
  cd -- "$1"
  local file files
  files="$(curvature_runtime_files "$1")"
  while IFS= read -r file; do
    sha256sum -- "$file" || exit
  done <<< "$files"
)

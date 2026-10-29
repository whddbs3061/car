#!/usr/bin/env bash
# final_ws driving stack with Competition-only status; no implicit Docker use.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WS="$ROOT/morai_ws"
MODE="${1:-monitor}"
[[ $# -eq 0 ]] || shift
case "$MODE" in
  monitor) CONTROL=false ;;
  drive) CONTROL=true ;;
  *) printf 'Usage: bash RUN_NEW_FINAL.sh [monitor|drive] [name:=value ...]\n' >&2; exit 2 ;;
esac
if [[ ! -f /opt/ros/noetic/setup.bash || ! -f "$WS/devel/setup.bash" ]]; then
  printf 'Build the new_final_ws catkin workspace first: %s\n' "$WS" >&2
  exit 2
fi
set +u
source /opt/ros/noetic/setup.bash
source "$WS/devel/setup.bash"
set -u
ACTUAL="$(rospack find morai_bringup)"
if [[ "$ACTUAL" != "$WS/src/bringup/morai_bringup" ]]; then
  printf 'Wrong ROS workspace: %s; expected %s/src/bringup/morai_bringup\n' "$ACTUAL" "$WS" >&2
  exit 2
fi
exec roslaunch morai_bringup final_ws_curvature_signal.launch \
  workspace_path:="$WS" enable_control:="$CONTROL" \
  morai_host_ip:=192.168.10.14 ego_status_port:=9099 "$@"

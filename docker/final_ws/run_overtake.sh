#!/usr/bin/env bash
# Separate MORAI test setup for the highway overtaking planner.
#
# Runs in its own container (CONTAINER_NAME in THIS folder's highway.env,
# morai-overtake) made from a snapshot of the competition container.  The
# competition container, its image and its checkout are only read; this script
# refuses to run against them.
#
#   bash run_overtake.sh init [COMPETITION_FINAL_WS]  once: this folder's settings
#   bash run_overtake.sh setup         once: snapshot -> own image/container, install, test
#   bash run_overtake.sh update        after a code change: install and test again
#   bash run_overtake.sh monitor|drive run_test.sh drive 5 with the overtaking planner
#   bash run_overtake.sh highway CRUISE_KMH [MAX_KMH]   highway zone speeds
#   bash run_overtake.sh speed KMH     ceiling outside the zone (MAX_SPEED_KPH here)
#   bash run_overtake.sh start|stop|shell|status|show|test
#   bash run_overtake.sh remove --yes  delete the overtake container and image
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WS="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
ACTION="${1:-help}"
[[ $# -eq 0 ]] || shift
COMPETITION_CONTAINER=morai-highway
COMPETITION_IMAGE=morai-final:highway
DEST=/opt/AutoVehicle/morai_ws
LAUNCH=final_ws_highway_overtake_bringup.launch
FILES=(
  src/control/purepursuit_mgeo/src/purepursuit_mgeo/highway_overtake.py
  src/control/purepursuit_mgeo/scripts/highway_overtake_node.py
  src/control/purepursuit_mgeo/config/highway_overtake.yaml
  src/control/purepursuit_mgeo/tools/highway_overtake_sim.py
  src/control/purepursuit_mgeo/test/test_highway_overtake.py
  src/control/purepursuit_mgeo/test/test_highway_overtake_node.py
  src/control/purepursuit_mgeo/test/test_overtake_launch.py
  docker/final_ws/make_overtake_launch.py
)

die() { echo "ERROR: $*" >&2; exit 2; }
number() { [[ "$1" =~ ^[0-9]+([.][0-9]+)?$ && "$1" =~ [1-9] ]]; }
set_value() {   # FILE NAME VALUE: replace NAME=... or append it
  if grep -q "^$2=" "$1"; then sed -i "s|^$2=.*|$2=$3|" "$1"; else printf '%s=%s\n' "$2" "$3" >> "$1"; fi
}

if [[ "$ACTION" == help || "$ACTION" == -h || "$ACTION" == --help ]]; then
  sed -n '2,16p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  exit 0
fi

if [[ "$ACTION" == init ]]; then
  SOURCE_DIR="${1:-$HOME/AutoVehicle/morai_ws/docker/final_ws}"
  [[ "$(cd -- "$SOURCE_DIR" 2>/dev/null && pwd || true)" != "$SCRIPT_DIR" ]] ||
    die "give the COMPETITION final_ws folder, not this one"
  for name in highway.env highway-test.env overtake.env; do
    [[ ! -e "$SCRIPT_DIR/$name" ]] || die "$SCRIPT_DIR/$name already exists; edit it, or delete it to start over"
  done
  BASE=$COMPETITION_CONTAINER BASE_IMAGE=$COMPETITION_IMAGE
  if [[ -f "$SOURCE_DIR/highway.env" ]]; then
    # Same IPs and GPU flavour as the competition setup; its container is the snapshot source.
    BASE="$(source "$SOURCE_DIR/highway.env"; echo "${CONTAINER_NAME:-$COMPETITION_CONTAINER}")"
    BASE_IMAGE="$(source "$SOURCE_DIR/highway.env"; echo "${IMAGE_NAME:-$COMPETITION_IMAGE}")"
    cp "$SOURCE_DIR/highway.env" "$SCRIPT_DIR/highway.env"
    echo "Copied $SOURCE_DIR/highway.env (read only)"
  else
    cp "$SCRIPT_DIR/highway.env.example" "$SCRIPT_DIR/highway.env"
    echo "No competition highway.env in $SOURCE_DIR; using highway.env.example"
  fi
  set_value "$SCRIPT_DIR/highway.env" CONTAINER_NAME morai-overtake
  set_value "$SCRIPT_DIR/highway.env" IMAGE_NAME morai-final:overtake
  if [[ -f "$SOURCE_DIR/highway-test.env" ]]; then
    cp "$SOURCE_DIR/highway-test.env" "$SCRIPT_DIR/highway-test.env"
    echo "Copied $SOURCE_DIR/highway-test.env: same speed/steering/sensor settings"
  else
    cp "$SCRIPT_DIR/highway-test.env.example" "$SCRIPT_DIR/highway-test.env"
    echo "No competition highway-test.env; using highway-test.env.example (MAX_SPEED_KPH=5)"
  fi
  cp "$SCRIPT_DIR/overtake.env.example" "$SCRIPT_DIR/overtake.env"
  set_value "$SCRIPT_DIR/overtake.env" BASE_CONTAINER "$BASE"
  set_value "$SCRIPT_DIR/overtake.env" BASE_IMAGE "$BASE_IMAGE"
  echo "Settings in $SCRIPT_DIR: container morai-overtake, snapshot source $BASE."
  echo 'Next: stop the competition container, then bash run_overtake.sh setup'
  exit 0
fi

for name in highway.env highway-test.env overtake.env; do
  [[ -f "$SCRIPT_DIR/$name" ]] || die "missing $SCRIPT_DIR/$name; run: bash run_overtake.sh init"
done
source "$SCRIPT_DIR/highway.env"
source "$SCRIPT_DIR/highway-test.env.example"
source "$SCRIPT_DIR/highway-test.env"
source "$SCRIPT_DIR/overtake.env"
: "${CONTAINER_NAME:?}" "${IMAGE_NAME:?}" "${BASE_CONTAINER:?}"
UBUNTU_IP="${UBUNTU_IP:-192.168.0.185}"
MORAI_IP="${MORAI_IP:-192.168.0.147}"
TORCH_FLAVOR="${TORCH_FLAVOR:-cpu}"
for name in "$BASE_CONTAINER" "$COMPETITION_CONTAINER"; do
  [[ "$CONTAINER_NAME" != "$name" ]] ||
    die "CONTAINER_NAME=$CONTAINER_NAME is the competition container; set CONTAINER_NAME=morai-overtake in $SCRIPT_DIR/highway.env"
done
for name in "${BASE_IMAGE:-}" "$COMPETITION_IMAGE"; do
  [[ "$IMAGE_NAME" != "$name" ]] ||
    die "IMAGE_NAME=$IMAGE_NAME is the competition image; set IMAGE_NAME=morai-final:overtake in $SCRIPT_DIR/highway.env"
done

if [[ "$ACTION" == highway ]]; then
  number "${1:-}" || die 'Usage: bash run_overtake.sh highway CRUISE_KMH [MAX_KMH], e.g. highway 54 79.2'
  MAX="${2:-$HIGHWAY_MAX_KPH}"
  number "$MAX" || die "MAX_KMH must be a positive number"
  awk -v c="$1" -v m="$MAX" 'BEGIN {exit !(m >= c)}' || die "MAX_KMH $MAX is below CRUISE_KMH $1"
  set_value "$SCRIPT_DIR/overtake.env" HIGHWAY_CRUISE_KPH "$1"
  set_value "$SCRIPT_DIR/overtake.env" HIGHWAY_MAX_KPH "$MAX"
  echo "Highway zone: cruise $1 km/h, overtaking up to $MAX km/h. Restart the drive to apply."
  exit 0
fi
if [[ "$ACTION" == speed ]]; then
  # This folder's highway-test.env only; the competition folder is not touched.
  exec bash "$SCRIPT_DIR/run_test.sh" speed "$@"
fi

profile_args() {
  # Exactly the arguments of `bash run_test.sh drive 5` (profile full).
  mapfile -t ARGS < <(bash "$SCRIPT_DIR/run_test.sh" show 5 | sed -n '/^Launch=/,$p' | tail -n +2)
  [[ ${#ARGS[@]} -gt 10 ]] || die "cannot read the run_test.sh profile 5 arguments"
  ARGS+=("highway_cruise_kph:=$HIGHWAY_CRUISE_KPH" "highway_max_speed_kph:=$HIGHWAY_MAX_KPH")
}

if [[ "$ACTION" == show ]]; then
  profile_args
  echo "Container=$CONTAINER_NAME image=$IMAGE_NAME snapshot_source=$BASE_CONTAINER"
  echo "Launch=$LAUNCH outside MAX_SPEED_KPH=$MAX_SPEED_KPH highway ${HIGHWAY_CRUISE_KPH}/${HIGHWAY_MAX_KPH} km/h"
  printf '%s\n' "morai_host_ip:=$MORAI_IP" "${ARGS[@]}"
  exit 0
fi

DOCKER=(docker --context default)
if ! "${DOCKER[@]}" info >/dev/null 2>&1; then DOCKER=(sudo docker --context default); fi
exists() { "${DOCKER[@]}" container inspect "$1" >/dev/null 2>&1; }
running() { [[ "$("${DOCKER[@]}" inspect --format '{{.State.Running}}' "$1" 2>/dev/null || true)" == true ]]; }
in_container() { "${DOCKER[@]}" exec -i "$CONTAINER_NAME" /usr/local/bin/morai-entrypoint "$@"; }
need_running() { running "$CONTAINER_NAME" || die "$CONTAINER_NAME is not running; bash run_overtake.sh start"; }

check_container() {
  need_running
  in_container bash -c "cd $DEST/src/control/purepursuit_mgeo && python -m unittest discover -s test -p 'test_*overtake*.py'"
  in_container python -c "import roslib.packages as p; m = p.find_node('purepursuit_mgeo', 'highway_overtake_node.py'); assert len(m) == 1, m; print('NODE_OK', m[0])"
  NODES="$(in_container roslaunch --nodes morai_bringup "$LAUNCH" "workspace_path:=$DEST" enable_control:=false)"
  grep -qx /highway_lane_strategy <<<"$NODES" || die "launch check failed: $NODES"
  grep -qx /adaptive_curvature_purepursuit <<<"$NODES" || die "launch check failed: $NODES"
  echo "OVERTAKE_CHECK_OK container=$CONTAINER_NAME"
}

install_files() {
  need_running
  for file in "${FILES[@]}"; do [[ -f "$WS/$file" ]] || die "missing $WS/$file"; done
  "${DOCKER[@]}" exec "$CONTAINER_NAME" mkdir -p "$DEST/src/control/purepursuit_mgeo/tools"
  for file in "${FILES[@]}"; do
    "${DOCKER[@]}" cp "$WS/$file" "$CONTAINER_NAME:$DEST/$file"
  done
  "${DOCKER[@]}" exec "$CONTAINER_NAME" chmod 755 \
    "$DEST/src/control/purepursuit_mgeo/scripts/highway_overtake_node.py" \
    "$DEST/docker/final_ws/make_overtake_launch.py"
  # Launch and regions are derived from the container's own files, so
  # everything outside the highway planner stays identical to them.
  in_container python "$DEST/docker/final_ws/make_overtake_launch.py"
  check_container
}

case "$ACTION" in
  setup)
    exists "$BASE_CONTAINER" || die "competition container $BASE_CONTAINER not found (see: docker ps -a)"
    ! running "$BASE_CONTAINER" ||
      die "stop the competition container first: in its final_ws folder run bash run_highway.sh stop"
    ! exists "$CONTAINER_NAME" || die "$CONTAINER_NAME already exists; use update, or remove --yes to start over"
    echo "Snapshot $BASE_CONTAINER -> image $IMAGE_NAME (the competition container is not changed)"
    "${DOCKER[@]}" commit "$BASE_CONTAINER" "$IMAGE_NAME"
    GPU_ARGS=()
    if [[ "$TORCH_FLAVOR" == cu121 ]]; then GPU_ARGS=(--gpus all); fi
    "${DOCKER[@]}" run -d --init --name "$CONTAINER_NAME" --network host --shm-size 1g \
      "${GPU_ARGS[@]}" -e "ROS_IP=$UBUNTU_IP" -e "ROS_MASTER_URI=http://$UBUNTU_IP:11311" \
      -e "MORAI_IP=$MORAI_IP" -v morai-overtake-logs:/root/.ros \
      "$IMAGE_NAME" sleep infinity
    install_files
    ;;
  update) install_files ;;
  test) check_container ;;
  start)
    exists "$CONTAINER_NAME" || die "$CONTAINER_NAME does not exist; bash run_overtake.sh setup"
    "${DOCKER[@]}" start "$CONTAINER_NAME"
    ;;
  stop) "${DOCKER[@]}" stop --time 10 "$CONTAINER_NAME" ;;
  shell)
    need_running
    exec "${DOCKER[@]}" exec -it "$CONTAINER_NAME" /usr/local/bin/morai-entrypoint bash --noprofile --norc
    ;;
  status)
    for name in "$BASE_CONTAINER" "$CONTAINER_NAME"; do
      state="$("${DOCKER[@]}" inspect --format '{{.State.Status}}' "$name" 2>/dev/null || echo missing)"
      echo "$name: $state"
    done
    ;;
  monitor|drive)
    need_running
    ! running "$BASE_CONTAINER" ||
      die "competition container $BASE_CONTAINER is running; stop it first (same ROS master and MORAI ports)"
    profile_args
    # A second launch could keep sending commands; refuse an occupied command topic.
    in_container python - <<'PY'
import os
import socket
import sys
from xmlrpc.client import ServerProxy
socket.setdefaulttimeout(3.0)
try:
    master = ServerProxy(os.environ.get('ROS_MASTER_URI', 'http://127.0.0.1:11311'))
    code, message, state = master.getSystemState('/overtake_launch_check')
except (OSError, socket.timeout):
    sys.exit(0)      # roslaunch starts a master
if code != 1:
    sys.exit('Cannot inspect ROS master: ' + str(message))
occupied = {topic: nodes for topic, nodes in state[0]
            if topic in ('/ctrl_cmd', '/control/ctrl_cmd') and nodes}
if occupied:
    sys.exit('Existing command publishers: %s. Stop the other launch first.' % occupied)
PY
    CONTROL=false
    [[ "$ACTION" != drive ]] || CONTROL=true
    echo "Launch=$LAUNCH control=$CONTROL outside MAX_SPEED_KPH=$MAX_SPEED_KPH highway ${HIGHWAY_CRUISE_KPH}/${HIGHWAY_MAX_KPH} km/h"
    exec "${DOCKER[@]}" exec -it "$CONTAINER_NAME" /usr/local/bin/morai-entrypoint \
      xvfb-run -a roslaunch morai_bringup "$LAUNCH" "morai_host_ip:=$MORAI_IP" \
      "enable_control:=$CONTROL" "${ARGS[@]}"
    ;;
  remove)
    [[ "${1:-}" == --yes ]] || die "this deletes $CONTAINER_NAME and $IMAGE_NAME; run: bash run_overtake.sh remove --yes"
    if exists "$CONTAINER_NAME"; then "${DOCKER[@]}" rm -f "$CONTAINER_NAME"; fi
    "${DOCKER[@]}" image rm "$IMAGE_NAME" || true
    ;;
  *) die "unknown action $ACTION; bash run_overtake.sh help" ;;
esac

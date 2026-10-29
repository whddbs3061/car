#!/usr/bin/env bash
# The overtaking highway planner in the team's v61 obstacle integration.
#
# Runs in its own container (morai-overtake-v61) made from a snapshot of the
# team's container; the team's container is only read, and this script refuses
# to run against it.  In the copy only new files are added: the team's 513
# files must match docker/final_ws/team_v61_manifest.txt, or nothing runs.
#
#   bash run_overtake_v61.sh init [TEAM_CONTAINER]   once: settings in overtake_v61.env
#   bash run_overtake_v61.sh setup     once: snapshot -> own image/container, install, check
#   bash run_overtake_v61.sh update    after a code change: install and check again
#   bash run_overtake_v61.sh monitor|drive   integrated_obstacle_v61_overtake.launch
#   bash run_overtake_v61.sh monitor-escape|drive-escape   ..._overtake_escape.launch
#                                   (lead-escape planner; the stable node is kept)
#   bash run_overtake_v61.sh monitor-escape2|drive-escape2   ..._overtake_escape2.launch
#                                   (escape + entry double line + map lane check)
#   bash run_overtake_v61.sh monitor-escape3|drive-escape3   ..._overtake_escape3.launch
#                                   (escape2 + camera line crossings, gentler changes)
#   bash run_overtake_v61.sh team-monitor|team-drive   the team's launch, same container
#   bash run_overtake_v61.sh speed KMH                 ceiling outside the zone (max_speed_kph)
#   bash run_overtake_v61.sh highway CRUISE_KMH [MAX_KMH]   highway zone speeds
#   bash run_overtake_v61.sh start|stop|shell|status|check
#   bash run_overtake_v61.sh remove --yes   delete the copy container and image
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WS="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
ENV_FILE="$SCRIPT_DIR/overtake_v61.env"
ACTION="${1:-help}"
[[ $# -eq 0 ]] || shift
V61=/root/morai_integration_test/AutoVehicle/morai_ws
LAUNCH=integrated_obstacle_v61_overtake.launch
ESCAPE_LAUNCH=integrated_obstacle_v61_overtake_escape.launch
ESCAPE2_LAUNCH=integrated_obstacle_v61_overtake_escape2.launch
ESCAPE3_LAUNCH=integrated_obstacle_v61_overtake_escape3.launch
TEAM_LAUNCH=integrated_obstacle_v61.launch
MANIFEST=docker/final_ws/team_v61_manifest.txt
FILES=(
  src/control/purepursuit_mgeo/src/purepursuit_mgeo/highway_overtake.py
  src/control/purepursuit_mgeo/src/purepursuit_mgeo/highway_overtake_escape.py
  src/control/purepursuit_mgeo/src/purepursuit_mgeo/highway_overtake_escape2.py
  src/control/purepursuit_mgeo/src/purepursuit_mgeo/highway_overtake_escape3.py
  src/control/purepursuit_mgeo/src/purepursuit_mgeo/highway_zone.py
  src/control/purepursuit_mgeo/scripts/highway_overtake_node.py
  src/control/purepursuit_mgeo/scripts/highway_overtake_escape_node.py
  src/control/purepursuit_mgeo/scripts/highway_overtake_escape2_node.py
  src/control/purepursuit_mgeo/scripts/highway_overtake_escape3_node.py
  src/control/purepursuit_mgeo/scripts/curvature_pp_overtake_node.py
  src/control/purepursuit_mgeo/scripts/maneuver_fusion_overtake_node.py
  src/control/purepursuit_mgeo/config/highway_overtake.yaml
  src/control/purepursuit_mgeo/config/highway_overtake_escape3.yaml
  src/control/purepursuit_mgeo/config/highway_lanes.json
  src/control/purepursuit_mgeo/config/route_mission_regions_overtake.json
  src/control/purepursuit_mgeo/tools/highway_overtake_sim.py
  src/control/purepursuit_mgeo/tools/make_highway_lanes.py
  src/control/purepursuit_mgeo/test/test_highway_overtake.py
  src/control/purepursuit_mgeo/test/test_highway_overtake_node.py
  src/control/purepursuit_mgeo/test/test_highway_overtake_escape.py
  src/control/purepursuit_mgeo/test/test_highway_overtake_escape2.py
  src/control/purepursuit_mgeo/test/test_highway_overtake_escape3.py
  src/control/purepursuit_mgeo/test/test_overtake_v61.py
  docker/final_ws/make_overtake_launch.py
  docker/final_ws/make_overtake_v61_launch.py
  "$MANIFEST"
)
SCRIPTS=(highway_overtake_node.py highway_overtake_escape_node.py highway_overtake_escape2_node.py
         highway_overtake_escape3_node.py
         curvature_pp_overtake_node.py
         maneuver_fusion_overtake_node.py)

die() { echo "ERROR: $*" >&2; exit 2; }
number() { [[ "$1" =~ ^[0-9]+([.][0-9]+)?$ && "$1" =~ [1-9] ]]; }
set_value() {   # NAME VALUE: replace NAME=... in the settings file or append it
  if grep -q "^$1=" "$ENV_FILE"; then sed -i "s|^$1=.*|$1=$2|" "$ENV_FILE"; else printf '%s=%s\n' "$1" "$2" >> "$ENV_FILE"; fi
}
DOCKER=(docker --context default)
if ! "${DOCKER[@]}" info >/dev/null 2>&1; then DOCKER=(sudo docker --context default); fi
exists() { "${DOCKER[@]}" container inspect "$1" >/dev/null 2>&1; }
running() { [[ "$("${DOCKER[@]}" inspect --format '{{.State.Running}}' "$1" 2>/dev/null || true)" == true ]]; }

if [[ "$ACTION" == help || "$ACTION" == -h || "$ACTION" == --help ]]; then
  sed -n '2,23p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  exit 0
fi

if [[ "$ACTION" == init ]]; then
  [[ ! -e "$ENV_FILE" ]] || die "$ENV_FILE already exists; edit it, or delete it to start over"
  TEAM="${1:-}"
  if [[ -z "$TEAM" ]]; then
    mapfile -t FOUND < <("${DOCKER[@]}" ps -a --format '{{.Names}}' | grep '^morai-highway-gpu' || true)
    [[ ${#FOUND[@]} -eq 1 ]] ||
      die "found ${#FOUND[@]} morai-highway-gpu* containers (${FOUND[*]:-none}); run: bash run_overtake_v61.sh init NAME"
    TEAM="${FOUND[0]}"
  fi
  exists "$TEAM" || die "container $TEAM not found (see: docker ps -a)"
  cat > "$ENV_FILE" <<EOF
# Settings of the overtaking copy only; the team's files are not read from here.
TEAM_CONTAINER=$TEAM
CONTAINER_NAME=morai-overtake-v61
IMAGE_NAME=morai-final:overtake-v61
ROS_IP=192.168.10.21
# Outside the highway zone (the team launch's max_speed_kph; its default is 43.2).
MAX_SPEED_KPH=43.2
HIGHWAY_CRUISE_KPH=54.0
HIGHWAY_MAX_KPH=79.2
HIGHWAY_LATERAL_MPS2=2.0
EOF
  echo "Wrote $ENV_FILE (team container $TEAM, copy morai-overtake-v61)."
  echo "Check ROS_IP there (the team uses 192.168.10.21), then stop the team container and run: bash run_overtake_v61.sh setup"
  exit 0
fi

[[ -f "$ENV_FILE" ]] || die "missing $ENV_FILE; run: bash run_overtake_v61.sh init"
source "$ENV_FILE"
: "${TEAM_CONTAINER:?}" "${CONTAINER_NAME:?}" "${IMAGE_NAME:?}" "${ROS_IP:?}"
[[ "$CONTAINER_NAME" != "$TEAM_CONTAINER" && "$CONTAINER_NAME" != morai-highway* ]] ||
  die "CONTAINER_NAME=$CONTAINER_NAME is the team's container; use morai-overtake-v61 in $ENV_FILE"
[[ "$IMAGE_NAME" != morai-final:highway* ]] ||
  die "IMAGE_NAME=$IMAGE_NAME is a team image; use morai-final:overtake-v61 in $ENV_FILE"

if [[ "$ACTION" == speed ]]; then
  number "${1:-}" || die 'Usage: bash run_overtake_v61.sh speed KMH, e.g. speed 43.2'
  set_value MAX_SPEED_KPH "$1"
  echo "Outside the zone: max_speed_kph $1. Restart the drive to apply."
  exit 0
fi
if [[ "$ACTION" == highway ]]; then
  number "${1:-}" || die 'Usage: bash run_overtake_v61.sh highway CRUISE_KMH [MAX_KMH], e.g. highway 54 79.2'
  MAX="${2:-$HIGHWAY_MAX_KPH}"
  number "$MAX" || die "MAX_KMH must be a positive number"
  awk -v c="$1" -v m="$MAX" 'BEGIN {exit !(m >= c)}' || die "MAX_KMH $MAX is below CRUISE_KMH $1"
  set_value HIGHWAY_CRUISE_KPH "$1"
  set_value HIGHWAY_MAX_KPH "$MAX"
  echo "Highway zone: cruise $1 km/h, overtaking up to $MAX km/h. Restart the drive to apply."
  exit 0
fi

ros() {   # run a command in the copy with the v61 workspace sourced
  "${DOCKER[@]}" exec "$@"
}
ros_env="source /opt/ros/noetic/setup.bash && source $V61/devel/setup.bash && unset ROS_HOSTNAME && export ROS_IP=$ROS_IP ROS_MASTER_URI=http://$ROS_IP:11311"
need_running() { running "$CONTAINER_NAME" || die "$CONTAINER_NAME is not running; bash run_overtake_v61.sh start"; }

check_copy() {
  need_running
  ros "$CONTAINER_NAME" bash -c "test -f $V61/devel/setup.bash" ||
    die "$CONTAINER_NAME has no built $V61; the team container may keep it outside the image"
  if ! ros "$CONTAINER_NAME" bash -c "cd $V61 && sha256sum --quiet -c $MANIFEST"; then
    die "team files in the copy differ from $MANIFEST (above): the team changed its workspace since the snapshot this branch was made from; send a new snapshot"
  fi
  echo "TEAM_FILES_OK $(grep -c . "$WS/$MANIFEST") team files match the snapshot"
  ros "$CONTAINER_NAME" bash -c "$ros_env && cd $V61/src/control/purepursuit_mgeo && python3 -m unittest discover -s test -p 'test_highway_overtake*.py' && python3 -m unittest discover -s test -p 'test_overtake_v61.py'"
  for script in "${SCRIPTS[@]}"; do
    ros "$CONTAINER_NAME" bash -c "$ros_env && python3 -c \"import roslib.packages as p; m = p.find_node('purepursuit_mgeo', '$script'); assert len(m) == 1, m; print('NODE_OK', m[0])\""
  done
  for file in "$LAUNCH" "$ESCAPE_LAUNCH" "$ESCAPE2_LAUNCH" "$ESCAPE3_LAUNCH"; do
    NODES="$(ros "$CONTAINER_NAME" bash -c "$ros_env && roslaunch --nodes morai_bringup $file workspace_path:=$V61")"
    for name in /highway_lane_strategy /curvature_speed_purepursuit /curvature_signal_controller; do
      grep -qx "$name" <<<"$NODES" || die "launch check failed ($file), $name missing: $NODES"
    done
  done
  echo "OVERTAKE_V61_CHECK_OK container=$CONTAINER_NAME"
}

install_files() {
  need_running
  for file in "${FILES[@]}"; do [[ -f "$WS/$file" ]] || die "missing $WS/$file"; done
  ros "$CONTAINER_NAME" mkdir -p "$V61/src/control/purepursuit_mgeo/tools"
  for file in "${FILES[@]}"; do
    "${DOCKER[@]}" cp "$WS/$file" "$CONTAINER_NAME:$V61/$file"
  done
  for script in "${SCRIPTS[@]}"; do
    ros "$CONTAINER_NAME" chmod 755 "$V61/src/control/purepursuit_mgeo/scripts/$script"
  done
  # Derived from the copy's own team launch: everything else stays identical.
  ros "$CONTAINER_NAME" python3 "$V61/docker/final_ws/make_overtake_v61_launch.py"
  check_copy
}

launch() {   # LAUNCH_FILE CONTROL [extra args]
  need_running
  ! running "$TEAM_CONTAINER" ||
    die "team container $TEAM_CONTAINER is running; stop it first (same ROS master and MORAI ports)"
  local file="$1" control="$2"
  shift 2
  echo "Launch=$file control=$control max_speed_kph=$MAX_SPEED_KPH $*"
  exec "${DOCKER[@]}" exec -it "$CONTAINER_NAME" bash -c "$ros_env && roslaunch morai_bringup $file workspace_path:=$V61 enable_control:=$control max_speed_kph:=$MAX_SPEED_KPH $*"
}
HIGHWAY_ARGS=("highway_cruise_kph:=$HIGHWAY_CRUISE_KPH" "highway_max_speed_kph:=$HIGHWAY_MAX_KPH"
              "highway_lateral_accel_mps2:=$HIGHWAY_LATERAL_MPS2")

case "$ACTION" in
  setup)
    exists "$TEAM_CONTAINER" || die "team container $TEAM_CONTAINER not found (see: docker ps -a)"
    ! running "$TEAM_CONTAINER" || die "stop the team container first: docker stop $TEAM_CONTAINER"
    ! exists "$CONTAINER_NAME" || die "$CONTAINER_NAME already exists; use update, or remove --yes to start over"
    echo "Snapshot $TEAM_CONTAINER -> image $IMAGE_NAME (the team container is not changed)"
    "${DOCKER[@]}" commit "$TEAM_CONTAINER" "$IMAGE_NAME"
    GPU_ARGS=()
    if [[ "$("${DOCKER[@]}" inspect --format '{{json .HostConfig.DeviceRequests}}' "$TEAM_CONTAINER")" != null ]]; then
      GPU_ARGS=(--gpus all)
    fi
    "${DOCKER[@]}" run -d --init --name "$CONTAINER_NAME" --network host --shm-size 1g \
      "${GPU_ARGS[@]}" -e "ROS_IP=$ROS_IP" -e "ROS_MASTER_URI=http://$ROS_IP:11311" \
      --entrypoint sleep "$IMAGE_NAME" infinity
    install_files
    ;;
  update) install_files ;;
  check) check_copy ;;
  start)
    exists "$CONTAINER_NAME" || die "$CONTAINER_NAME does not exist; bash run_overtake_v61.sh setup"
    "${DOCKER[@]}" start "$CONTAINER_NAME"
    ;;
  stop) "${DOCKER[@]}" stop --time 10 "$CONTAINER_NAME" ;;
  shell)
    need_running
    exec "${DOCKER[@]}" exec -it "$CONTAINER_NAME" bash -c "$ros_env && cd $V61 && exec bash"
    ;;
  status)
    for name in "$TEAM_CONTAINER" "$CONTAINER_NAME"; do
      echo "$name: $("${DOCKER[@]}" inspect --format '{{.State.Status}}' "$name" 2>/dev/null || echo missing)"
    done
    ;;
  monitor) launch "$LAUNCH" false "${HIGHWAY_ARGS[@]}" ;;
  drive) launch "$LAUNCH" true "${HIGHWAY_ARGS[@]}" ;;
  monitor-escape) launch "$ESCAPE_LAUNCH" false "${HIGHWAY_ARGS[@]}" ;;
  drive-escape) launch "$ESCAPE_LAUNCH" true "${HIGHWAY_ARGS[@]}" ;;
  monitor-escape2) launch "$ESCAPE2_LAUNCH" false "${HIGHWAY_ARGS[@]}" ;;
  drive-escape2) launch "$ESCAPE2_LAUNCH" true "${HIGHWAY_ARGS[@]}" ;;
  monitor-escape3) launch "$ESCAPE3_LAUNCH" false "${HIGHWAY_ARGS[@]}" ;;
  drive-escape3) launch "$ESCAPE3_LAUNCH" true "${HIGHWAY_ARGS[@]}" ;;
  team-monitor) launch "$TEAM_LAUNCH" false ;;
  team-drive) launch "$TEAM_LAUNCH" true ;;
  remove)
    [[ "${1:-}" == --yes ]] || die "this deletes $CONTAINER_NAME and $IMAGE_NAME; run: bash run_overtake_v61.sh remove --yes"
    if exists "$CONTAINER_NAME"; then "${DOCKER[@]}" rm -f "$CONTAINER_NAME"; fi
    "${DOCKER[@]}" image rm "$IMAGE_NAME" || true
    ;;
  *) die "unknown action $ACTION; bash run_overtake_v61.sh help" ;;
esac

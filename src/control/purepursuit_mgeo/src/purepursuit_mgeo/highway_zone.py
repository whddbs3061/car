"""Zone-only overrides of the team's nodes for the overtaking highway planner.

highway_overtake_node publishes /highway_lane_strategy/zone_active while the
ego is inside its fixed highway zone.  The wrapper nodes load a team script
unchanged (load_script) and apply their override only while that flag is
fresh and true; a missing, stale or false flag leaves the team's behaviour.
"""
import importlib.util
import os
import sys
import threading
import time


class ZoneFlag:
    """Latest zone_active sample; active only while fresh and true."""

    def __init__(self, timeout_s: float = 0.5) -> None:
        self.timeout_s = timeout_s
        self._lock = threading.Lock()
        self._value = False
        self._at = None

    def on_msg(self, msg) -> None:
        with self._lock:
            self._value, self._at = bool(msg.data), time.monotonic()

    def active(self) -> bool:
        with self._lock:
            return (self._value and self._at is not None
                    and 0.0 <= time.monotonic() - self._at <= self.timeout_s)


def load_script(package: str, script: str, module_name: str):
    """Import <package>/scripts/<script> as a module without running its main."""
    import rospkg
    path = os.path.join(rospkg.RosPack().get_path(package), "scripts", script)
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

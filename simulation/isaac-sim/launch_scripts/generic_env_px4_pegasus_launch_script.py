#!/usr/bin/env python
"""
Generic USD-environment PX4 Pegasus launcher.

This script intentionally contains no TEEX-specific scene logic. It loads any
local or omniverse:// environment USD, spawns robot_1..robot_N using an explicit
drone USD, and wires the drone camera graph expected by the DiffPhysDrone policy wrapper.

Required env:
 - ENV_USD_PATH: local path or omniverse:// URL for the environment USD
 - PEGASUS_DRONE_USD: local path or omniverse:// URL for the drone USD

Optional env:
 - NUM_ROBOTS (default 1)
 - ENABLE_LIDAR (default false)
 - PLAY_SIM_ON_START (default false)
 - GENERIC_ADD_COLLIDERS (default true): add collision APIs under /World/stage
 - GENERIC_ADD_DOME_LIGHT (default true)
 - GENERIC_STAGE_SCALE (default 1.0)
 - GENERIC_REMOVE_EXISTING_DRONES (default true): remove vehicle prims that
   came from the environment USD before spawning the requested drone
 - GENERIC_EXISTING_DRONE_ROOTS: optional comma-separated prim paths to remove
 - GENERIC_SPAWN_POSITIONS: semicolon-separated x,y,z positions, e.g.
   "0,0,1;2,0,1"
 - GENERIC_SPAWN_X/Y/Z: fallback first spawn position, default 0,0,1
 - GENERIC_SPAWN_SPACING_M (default 2.0): x spacing for multiple robots
 - GENERIC_CAMERA_MODE (default starling): starling, zed, or none
 - GENERIC_STARLING_RGB_CAMERA (default body/RGB_IMX412)
 - GENERIC_STARLING_DEPTH_CAMERA (default body/ToF_sensor)
 - GENERIC_CAMERA_OFFSET (ZED mode only, default 0.2,0,-0.05)
 - GENERIC_CAMERA_ROTATION_OFFSET (ZED mode only, default 0,0,0)
"""

import os
import sys
import time
from pathlib import Path

import carb
from isaacsim import SimulationApp


_headless = os.environ.get("ISAAC_SIM_HEADLESS", "false").lower() == "true"
simulation_app = SimulationApp({"headless": _headless})

import omni.kit.app
import omni.timeline
import omni.usd
from omni.isaac.core.world import World

from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface
from pegasus.simulator.ogn.api.spawn_multirotor import spawn_px4_multirotor_node
from pegasus.simulator.ogn.api.spawn_ouster_lidar import add_ouster_lidar_subgraph
from pegasus.simulator.ogn.api.spawn_zed_camera import (
    add_starling_camera_subgraph,
    add_zed_stereo_camera_subgraph,
)

sys.path.insert(
    0,
    os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "utils")
    ),
)
from scene_prep import add_colliders, add_dome_light, scale_stage_prim


def env_bool(name: str, default: bool) -> bool:
    return os.environ.get(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


def env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


def parse_vec3(value: str, *, name: str) -> list[float]:
    parts = [p.strip() for p in value.split(",")]
    if len(parts) != 3:
        raise ValueError(f"{name} must contain exactly three comma-separated values")
    return [float(p) for p in parts]


def require_usd_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} must be set explicitly")
    if value.startswith("omniverse://"):
        return value
    expanded = os.path.expanduser(value)
    if not Path(expanded).exists():
        raise FileNotFoundError(f"{name} does not exist: {expanded}")
    return expanded


ENV_USD_PATH = require_usd_env("ENV_USD_PATH")
DRONE_USD = require_usd_env("PEGASUS_DRONE_USD")

NUM_ROBOTS = int(os.environ.get("NUM_ROBOTS", "1"))
ENABLE_LIDAR = env_bool("ENABLE_LIDAR", False)
PLAY_SIM_ON_START = env_bool("PLAY_SIM_ON_START", False)
REMOVE_EXISTING_DRONES = env_bool("GENERIC_REMOVE_EXISTING_DRONES", True)
ADD_COLLIDERS = env_bool("GENERIC_ADD_COLLIDERS", True)
ADD_DOME_LIGHT = env_bool("GENERIC_ADD_DOME_LIGHT", True)
STAGE_SCALE = env_float("GENERIC_STAGE_SCALE", 1.0)
SPAWN_SPACING_M = env_float("GENERIC_SPAWN_SPACING_M", 2.0)
CAMERA_MODE = os.environ.get("GENERIC_CAMERA_MODE", "starling").strip().lower()
STARLING_RGB_CAMERA = os.environ.get("GENERIC_STARLING_RGB_CAMERA", "body/RGB_IMX412").strip()
STARLING_DEPTH_CAMERA = os.environ.get("GENERIC_STARLING_DEPTH_CAMERA", "body/ToF_sensor").strip()

CAMERA_OFFSET = parse_vec3(os.environ.get("GENERIC_CAMERA_OFFSET", "0.2,0.0,-0.05"), name="GENERIC_CAMERA_OFFSET")
CAMERA_ROTATION_OFFSET = parse_vec3(
    os.environ.get("GENERIC_CAMERA_ROTATION_OFFSET", "0.0,0.0,0.0"),
    name="GENERIC_CAMERA_ROTATION_OFFSET",
)


ext_manager = omni.kit.app.get_app().get_extension_manager()
for ext in [
    "omni.graph.core",
    "omni.graph.action",
    "omni.graph.action_nodes",
    "isaacsim.core.nodes",
    "omni.graph.ui",
    "omni.graph.visualization.nodes",
    "omni.graph.scriptnode",
    "omni.graph.window.action",
    "omni.graph.window.generic",
    "omni.graph.ui_nodes",
    "pegasus.simulator",
]:
    if not ext_manager.is_extension_enabled(ext):
        ext_manager.set_extension_enabled_immediate(ext, True)


def spawn_positions() -> list[list[float]]:
    configured = os.environ.get("GENERIC_SPAWN_POSITIONS", "").strip()
    if configured:
        positions = [
            parse_vec3(item.strip(), name="GENERIC_SPAWN_POSITIONS")
            for item in configured.split(";")
            if item.strip()
        ]
        if len(positions) < NUM_ROBOTS:
            raise ValueError("GENERIC_SPAWN_POSITIONS has fewer positions than NUM_ROBOTS")
        return positions[:NUM_ROBOTS]

    base = [
        env_float("GENERIC_SPAWN_X", 0.0),
        env_float("GENERIC_SPAWN_Y", 0.0),
        env_float("GENERIC_SPAWN_Z", 1.0),
    ]
    return [
        [base[0] + SPAWN_SPACING_M * (i - 1), base[1], base[2]]
        for i in range(1, NUM_ROBOTS + 1)
    ]


def wait_for_stage(stage, timeout_s: float = 20.0) -> bool:
    for _ in range(int(timeout_s / 0.1)):
        omni.kit.app.get_app().update()
        world_prim = stage.GetPrimAtPath("/World")
        if world_prim.IsValid():
            children = [child for child in world_prim.GetChildren() if child.GetName() != "PhysicsScene"]
            if children:
                return True
        time.sleep(0.1)
    return False


VEHICLE_ROOT_NAME_PREFIXES = ("drone", "iris", "starling", "quadrotor", "x500")
VEHICLE_DESCENDANT_NAMES = ("base_link", "PX4Multirotor", "PX4MultirotorGraph")
VEHICLE_DESCENDANT_NAME_FRAGMENTS = ("PX4Multirotor", "PegasusMultirotor")
KNOWN_EMBEDDED_VEHICLE_ROOTS = ("/World/stage/World",)


def configured_existing_drone_roots() -> list[str]:
    configured = os.environ.get("GENERIC_EXISTING_DRONE_ROOTS", "").strip()
    if not configured:
        return []
    return [path.strip() for path in configured.split(",") if path.strip()]


def prim_has_vehicle_marker(root_prim) -> bool:
    stack = list(root_prim.GetChildren())
    while stack:
        prim = stack.pop()
        name = prim.GetName()
        if name in VEHICLE_DESCENDANT_NAMES:
            return True
        if any(fragment in name for fragment in VEHICLE_DESCENDANT_NAME_FRAGMENTS):
            return True
        stack.extend(list(prim.GetChildren()))
    return False


def vehicle_root_for_marker(prim) -> str:
    path = prim.GetPath().pathString
    name = prim.GetName()
    if name == "base_link":
        return str(prim.GetParent().GetPath())
    if name == "PX4MultirotorGraph" or any(fragment in name for fragment in VEHICLE_DESCENDANT_NAME_FRAGMENTS):
        parent = prim.GetParent()
        if parent and parent.GetName() == "base_link":
            return str(parent.GetParent().GetPath())
        return str(parent.GetPath()) if parent else path
    return path


def find_existing_vehicle_roots(stage) -> list[str]:
    roots = set(configured_existing_drone_roots())
    world_prim = stage.GetPrimAtPath("/World")
    if not world_prim.IsValid():
        return sorted(roots)

    for known_root in KNOWN_EMBEDDED_VEHICLE_ROOTS:
        prim = stage.GetPrimAtPath(known_root)
        if prim.IsValid() and prim_has_vehicle_marker(prim):
            roots.add(known_root)

    stack = list(world_prim.GetChildren())
    while stack:
        prim = stack.pop()
        path = prim.GetPath().pathString
        name = prim.GetName()
        lower_name = name.lower()
        if path == "/World/PhysicsScene":
            continue

        if lower_name.startswith(VEHICLE_ROOT_NAME_PREFIXES):
            roots.add(path)
            continue
        if name in VEHICLE_DESCENDANT_NAMES or any(
            fragment in name for fragment in VEHICLE_DESCENDANT_NAME_FRAGMENTS
        ):
            roots.add(vehicle_root_for_marker(prim))
            continue

        stack.extend(list(prim.GetChildren()))

    # If both a parent vehicle root and a child marker path were detected, keep
    # only the parent so RemovePrim operates on the whole embedded vehicle.
    collapsed = set(roots)
    for root in roots:
        for other in roots:
            if root != other and root.startswith(other.rstrip("/") + "/"):
                collapsed.discard(root)
                break
    return sorted(collapsed)


def remove_existing_vehicle_prims(stage) -> None:
    if not REMOVE_EXISTING_DRONES:
        carb.log_warn("[generic_env] Existing drone cleanup disabled.")
        return

    vehicle_roots = find_existing_vehicle_roots(stage)
    if not vehicle_roots:
        carb.log_warn("[generic_env] No pre-existing drone prims found in the environment USD.")
        return

    carb.log_warn(
        "[generic_env] Removing pre-existing drone/PX4 prims from environment USD: "
        + ", ".join(vehicle_roots)
    )
    for prim_path in vehicle_roots:
        if stage.GetPrimAtPath(prim_path).IsValid():
            stage.RemovePrim(prim_path)
    for _ in range(5):
        omni.kit.app.get_app().update()


def spawn_drone(index: int, init_pos: list[float]):
    robot_name = f"robot_{index}"
    drone_prim = f"/World/drone{index}/base_link"

    graph_handle = spawn_px4_multirotor_node(
        pegasus_node_name=f"PX4Multirotor_{index}",
        drone_prim=drone_prim,
        robot_name=robot_name,
        vehicle_id=index,
        domain_id=index,
        usd_file=DRONE_USD,
        init_pos=init_pos,
        init_orient=[0.0, 0.0, 0.0, 1.0],
    )

    if CAMERA_MODE == "starling":
        add_starling_camera_subgraph(
            parent_graph_handle=graph_handle,
            drone_prim=drone_prim,
            robot_name=robot_name,
            rgb_camera_relative_path=STARLING_RGB_CAMERA,
            depth_camera_relative_path=STARLING_DEPTH_CAMERA,
        )
        carb.log_warn(
            f"[generic_env] {robot_name}: using Starling built-in cameras "
            f"rgb={drone_prim}/{STARLING_RGB_CAMERA}, "
            f"depth={drone_prim}/{STARLING_DEPTH_CAMERA}"
        )
    elif CAMERA_MODE == "zed":
        add_zed_stereo_camera_subgraph(
            parent_graph_handle=graph_handle,
            drone_prim=drone_prim,
            robot_name=robot_name,
            camera_name="ZEDCamera",
            camera_offset=CAMERA_OFFSET,
            camera_rotation_offset=CAMERA_ROTATION_OFFSET,
        )
        carb.log_warn(f"[generic_env] {robot_name}: using synthetic ZED camera graph")
    elif CAMERA_MODE in ("none", "off", "false"):
        carb.log_warn(f"[generic_env] {robot_name}: camera graph disabled")
    else:
        raise RuntimeError(f"Unsupported GENERIC_CAMERA_MODE={CAMERA_MODE!r}")

    if ENABLE_LIDAR:
        add_ouster_lidar_subgraph(
            parent_graph_handle=graph_handle,
            drone_prim=drone_prim,
            robot_name=robot_name,
            lidar_name="OS1_REV6_128_10hz___512_resolution",
            lidar_offset=[0.0, 0.0, 0.025],
            lidar_rotation_offset=[0.0, 0.0, 0.0],
            lidar_min_range=0.75,
        )

    carb.log_warn(
        f"[generic_env] {robot_name}: drone_prim={drone_prim}, "
        f"depth_topic=/{robot_name}/sensors/front_stereo/right/depth_ground_truth"
    )


class PegasusApp:
    def __init__(self):
        self.timeline = omni.timeline.get_timeline_interface()
        self.pg = PegasusInterface()
        self.pg._world = World(**self.pg._world_settings)
        self.world = self.pg.world
        self.timeline.stop()

        carb.log_warn(f"[generic_env] Loading environment USD: {ENV_USD_PATH}")
        carb.log_warn(f"[generic_env] Using drone USD: {DRONE_USD}")
        self.pg.load_environment(ENV_USD_PATH)

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            raise RuntimeError("Stage failed to load")
        if not wait_for_stage(stage):
            carb.log_warn("[generic_env] Stage load timed out; continuing anyway.")

        remove_existing_vehicle_prims(stage)

        stage_prim = stage.GetPrimAtPath("/World/stage")
        if stage_prim.IsValid():
            if STAGE_SCALE != 1.0:
                scale_stage_prim(stage, "/World/stage", STAGE_SCALE)
            if ADD_COLLIDERS:
                add_colliders(stage_prim)
                for _ in range(10):
                    omni.kit.app.get_app().update()
        else:
            carb.log_warn("[generic_env] /World/stage not found; skipping stage scale/colliders.")

        if ADD_DOME_LIGHT:
            add_dome_light(stage)

        positions = spawn_positions()
        carb.log_warn(
            f"[generic_env] Spawning {NUM_ROBOTS} drone(s), lidar={'on' if ENABLE_LIDAR else 'off'}, "
            f"positions={positions}"
        )
        for index, position in enumerate(positions, start=1):
            spawn_drone(index, position)

    def run(self):
        if PLAY_SIM_ON_START:
            self.timeline.play()
        else:
            self.timeline.stop()

        app = omni.kit.app.get_app()
        while simulation_app.is_running():
            world = World.instance()
            if world is not None and hasattr(world, "_scene"):
                world.step(render=True)
                if world is not self.world:
                    self.world = world
                    self.pg._world = world
            else:
                app.update()

        carb.log_warn("[generic_env] Closing simulation.")
        self.timeline.stop()
        simulation_app.close()


def main():
    app = PegasusApp()
    app.run()


if __name__ == "__main__":
    main()

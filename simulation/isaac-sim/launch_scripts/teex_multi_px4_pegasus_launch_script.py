#!/usr/bin/env python
"""
TEEX multi-drone PX4 Pegasus launcher.

This loads the generated TEEX USD scene and spawns robot_1..robot_N in the
same local metric frame used by the DTC planner map products.

Env:
 - NUM_ROBOTS (default 3): how many drones to spawn
 - ENABLE_LIDAR (default false): attach an Ouster lidar to each drone
 - PLAY_SIM_ON_START (default true): autoplay timeline
 - TEEX_USD_PATH: override the generated TEEX scene USD path
 - TEEX_MANIFEST_PATH: override the generated TEEX scene manifest path
 - TEEX_SPAWN_ALTITUDE_M (default 12.0): spawn altitude in TEEX local z
"""

import json
import os
import sys
import time
from pathlib import Path

import carb
from isaacsim import SimulationApp

# Must be created before any omni imports.
_headless = os.environ.get("ISAAC_SIM_HEADLESS", "false").lower() == "true"
simulation_app = SimulationApp({"headless": _headless})

import omni.kit.app
import omni.timeline
import omni.usd

from omni.isaac.core.world import World

from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface
from pegasus.simulator.ogn.api.spawn_multirotor import spawn_px4_multirotor_node
from pegasus.simulator.ogn.api.spawn_ouster_lidar import add_ouster_lidar_subgraph
from pegasus.simulator.ogn.api.spawn_zed_camera import add_zed_stereo_camera_subgraph

sys.path.insert(
    0,
    os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "utils")
    ),
)
from scene_prep import add_colliders, add_dome_light


TEEX_USD_PATH = os.environ.get(
    "TEEX_USD_PATH",
    "/isaac-sim/AirStack/simulation/isaac-sim/assets/teex/usd/teex_scene.usd",
)
TEEX_MANIFEST_PATH = os.environ.get(
    "TEEX_MANIFEST_PATH",
    "/isaac-sim/AirStack/simulation/isaac-sim/assets/teex/usd/teex_scene_manifest.json",
)
DRONE_USD = os.environ.get(
    "PEGASUS_DRONE_USD",
    "~/.local/share/ov/data/documents/Kit/shared/exts/pegasus.simulator/pegasus/simulator/assets/Robots/Iris/iris.usd",
)

NUM_ROBOTS = int(os.environ.get("NUM_ROBOTS", "3"))
ENABLE_LIDAR = os.environ.get("ENABLE_LIDAR", "false").lower() == "true"
SPAWN_ALTITUDE_M = float(os.environ.get("TEEX_SPAWN_ALTITUDE_M", "12.0"))

STAGE_PRIM_PATH = "/World/stage"
TEEX_PRIM_PATH = f"{STAGE_PRIM_PATH}/TEEX"


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


def load_manifest() -> dict:
    path = Path(TEEX_MANIFEST_PATH)
    if not path.exists():
        carb.log_warn(f"TEEX manifest not found at {path}; using conservative spawn defaults.")
        return {}
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def local_bounds(manifest: dict) -> tuple[float, float, float, float]:
    bounds = manifest.get("bounds_local_m")
    if isinstance(bounds, list) and len(bounds) == 4:
        return tuple(float(value) for value in bounds)
    return 0.0, 0.0, 40.0, 40.0


def wait_for_prim(stage, prim_path: str, timeout_s: float = 30.0) -> bool:
    app = omni.kit.app.get_app()
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        app.update()
        prim = stage.GetPrimAtPath(prim_path)
        if prim.IsValid():
            return True
        time.sleep(0.1)
    return False


def spawn_positions(manifest: dict) -> list[list[float]]:
    min_x, min_y, max_x, max_y = local_bounds(manifest)
    width = max_x - min_x
    height = max_y - min_y
    center_x = min_x + width / 2.0
    center_y = min_y + height / 2.0

    spacing = 6.0
    start_x = center_x - spacing * (NUM_ROBOTS - 1) / 2.0
    y = min_y + min(max(height * 0.15, 12.0), max(height - 12.0, 12.0))
    return [
        [start_x + spacing * index, y, SPAWN_ALTITUDE_M]
        for index in range(NUM_ROBOTS)
    ]


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

    add_zed_stereo_camera_subgraph(
        parent_graph_handle=graph_handle,
        drone_prim=drone_prim,
        robot_name=robot_name,
        camera_name="ZEDCamera",
        camera_offset=[0.2, 0.0, -0.05],
        camera_rotation_offset=[0.0, 0.0, 0.0],
    )

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


class PegasusApp:
    def __init__(self):
        self.timeline = omni.timeline.get_timeline_interface()
        self.pg = PegasusInterface()
        self.pg._world = World(**self.pg._world_settings)
        self.world = self.pg.world
        self.timeline.stop()

        manifest = load_manifest()
        carb.log_warn(f"[teex_multi] Loading TEEX scene: {TEEX_USD_PATH}")
        self.pg.load_environment(TEEX_USD_PATH)

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            raise RuntimeError("Stage failed to load")
        if not wait_for_prim(stage, TEEX_PRIM_PATH):
            raise RuntimeError(f"TEEX scene did not appear at {TEEX_PRIM_PATH}")

        stage_prim = stage.GetPrimAtPath(STAGE_PRIM_PATH)
        add_colliders(stage_prim)
        add_dome_light(stage)

        min_x, min_y, max_x, max_y = local_bounds(manifest)
        center_x = min_x + (max_x - min_x) / 2.0
        center_y = min_y + (max_y - min_y) / 2.0
        self.pg.set_viewport_camera(
            [center_x, center_y - max(max_x - min_x, max_y - min_y), 220.0],
            [center_x, center_y, 0.0],
        )

        positions = spawn_positions(manifest)
        print(
            f"[teex_multi] Spawning {NUM_ROBOTS} drone(s), "
            f"lidar={'on' if ENABLE_LIDAR else 'off'}, positions={positions}",
            flush=True,
        )
        for index, position in enumerate(positions, start=1):
            spawn_drone(index, position)

        self.play_on_start = os.environ.get("PLAY_SIM_ON_START", "true").lower() == "true"

    def run(self):
        if self.play_on_start:
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

        carb.log_warn("Closing TEEX multi-drone simulation.")
        self.timeline.stop()
        simulation_app.close()


def main():
    app = PegasusApp()
    app.run()


if __name__ == "__main__":
    main()

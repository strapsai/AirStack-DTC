#!/usr/bin/env python
"""Load a standalone RESCUE/GLB-derived USD scene and spawn Pegasus drones.

This launcher intentionally does not use TEEX map products or GPS alignment.
It is for inspecting the direct GLB -> USD reconstruction in Isaac Sim.

Env:
 - RESCUE_USD_PATH: converted GLB USD to load
 - NUM_ROBOTS (default 3): how many drones to spawn
 - ENABLE_LIDAR (default false): attach an Ouster lidar to each drone
 - PLAY_SIM_ON_START (default false): autoplay timeline
 - RESCUE_NORMALIZE_FLOOR (default true): shift scene so min z is 0
 - RESCUE_SPAWN_ALTITUDE_M (default 2.0): spawn above scene max z
"""

import os
import sys
import time

import carb
from isaacsim import SimulationApp

# Must be created before any omni imports.
_headless = os.environ.get("ISAAC_SIM_HEADLESS", "false").lower() == "true"
simulation_app = SimulationApp({"headless": _headless})

import omni.kit.app
import omni.timeline
import omni.usd

from omni.isaac.core.world import World
from pxr import Gf, UsdGeom

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


RESCUE_USD_PATH = os.environ.get(
    "RESCUE_USD_PATH",
    "/isaac-sim/AirStack/simulation/isaac-sim/assets/teex/rescue/reconstruction_mesh_simple.usd",
)
DRONE_USD = os.environ.get(
    "PEGASUS_DRONE_USD",
    "~/.local/share/ov/data/documents/Kit/shared/exts/pegasus.simulator/pegasus/simulator/assets/Robots/Iris/iris.usd",
)

NUM_ROBOTS = int(os.environ.get("NUM_ROBOTS", "3"))
ENABLE_LIDAR = os.environ.get("ENABLE_LIDAR", "false").lower() == "true"
NORMALIZE_FLOOR = os.environ.get("RESCUE_NORMALIZE_FLOOR", "true").lower() == "true"
SPAWN_ALTITUDE_M = float(os.environ.get("RESCUE_SPAWN_ALTITUDE_M", "2.0"))
STAGE_PRIM_PATH = "/World/stage"


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


def world_bounds(stage, prim_path: str) -> tuple[float, float, float, float, float, float]:
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"Prim not found for bounds: {prim_path}")
    cache = UsdGeom.BBoxCache(
        0.0,
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
        useExtentsHint=True,
    )
    box = cache.ComputeWorldBound(prim).ComputeAlignedBox()
    min_point = box.GetMin()
    max_point = box.GetMax()
    return (
        float(min_point[0]),
        float(min_point[1]),
        float(min_point[2]),
        float(max_point[0]),
        float(max_point[1]),
        float(max_point[2]),
    )


def normalize_floor(stage, prim_path: str) -> tuple[float, float, float, float, float, float]:
    bounds = world_bounds(stage, prim_path)
    if not NORMALIZE_FLOOR:
        return bounds

    min_z = bounds[2]
    prim = stage.GetPrimAtPath(prim_path)
    UsdGeom.XformCommonAPI(prim).SetTranslate(Gf.Vec3d(0.0, 0.0, -min_z))
    omni.kit.app.get_app().update()
    shifted_bounds = world_bounds(stage, prim_path)
    carb.log_warn(
        "[rescue_glb] Normalized scene floor to z=0 "
        f"(applied z shift {-min_z:.3f} m)."
    )
    return shifted_bounds


def spawn_positions(bounds: tuple[float, float, float, float, float, float]) -> list[list[float]]:
    min_x, min_y, _min_z, max_x, max_y, max_z = bounds
    center_x = min_x + (max_x - min_x) / 2.0
    center_y = min_y + (max_y - min_y) / 2.0
    z = max_z + SPAWN_ALTITUDE_M
    spacing = 3.0
    start_x = center_x - spacing * (NUM_ROBOTS - 1) / 2.0
    return [[start_x + spacing * index, center_y, z] for index in range(NUM_ROBOTS)]


def spawn_drone(index: int, init_pos: list[float]) -> None:
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

        carb.log_warn(f"[rescue_glb] Loading scene: {RESCUE_USD_PATH}")
        self.pg.load_environment(RESCUE_USD_PATH)

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            raise RuntimeError("Stage failed to load")
        if not wait_for_prim(stage, STAGE_PRIM_PATH):
            raise RuntimeError(f"Scene did not appear at {STAGE_PRIM_PATH}")

        bounds = normalize_floor(stage, STAGE_PRIM_PATH)
        stage_prim = stage.GetPrimAtPath(STAGE_PRIM_PATH)
        add_colliders(stage_prim)
        for _ in range(10):
            omni.kit.app.get_app().update()
        add_dome_light(stage)

        min_x, min_y, min_z, max_x, max_y, max_z = bounds
        center_x = min_x + (max_x - min_x) / 2.0
        center_y = min_y + (max_y - min_y) / 2.0
        self.pg.set_viewport_camera(
            [center_x, center_y - max(max_x - min_x, max_y - min_y), max_z + 80.0],
            [center_x, center_y, min_z],
        )

        positions = spawn_positions(bounds)
        print(
            f"[rescue_glb] Spawning {NUM_ROBOTS} drone(s), "
            f"lidar={'on' if ENABLE_LIDAR else 'off'}, positions={positions}",
            flush=True,
        )
        for index, position in enumerate(positions, start=1):
            spawn_drone(index, position)

        self.play_on_start = os.environ.get("PLAY_SIM_ON_START", "false").lower() == "true"

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

        carb.log_warn("Closing RESCUE GLB multi-drone simulation.")
        self.timeline.stop()
        simulation_app.close()


def main():
    app = PegasusApp()
    app.run()


if __name__ == "__main__":
    main()

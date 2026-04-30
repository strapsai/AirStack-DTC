#!/usr/bin/env python
"""
TEEX multi-drone PX4 Pegasus launcher.

This loads the generated TEEX USD scene and spawns robot_1..robot_N in the
same local metric frame used by the DTC planner map products.

Env:
 - NUM_ROBOTS (default 3): how many drones to spawn
 - ENABLE_LIDAR (default false): attach an Ouster lidar to each drone
 - PLAY_SIM_ON_START (default false): autoplay timeline
 - TEEX_HIGHLIGHT_DRONES (default true): attach colored visual markers to drones
 - TEEX_USD_PATH: override the generated TEEX scene USD path
 - TEEX_MANIFEST_PATH: override the generated TEEX scene manifest path
 - TEEX_SPAWN_REFERENCE (default agl): agl or map_z
 - TEEX_SPAWN_ALTITUDE_M (default 0.6): spawn altitude in chosen reference
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
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

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
from scene_prep import add_colliders, add_collision_box_floor, add_dome_light


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
SPAWN_REFERENCE = os.environ.get("TEEX_SPAWN_REFERENCE", "agl").strip().lower()
SPAWN_ALTITUDE_M = float(os.environ.get("TEEX_SPAWN_ALTITUDE_M", "0.6"))
HIGHLIGHT_DRONES = os.environ.get("TEEX_HIGHLIGHT_DRONES", "true").lower() == "true"
FLATTEN_FLOOR = os.environ.get("TEEX_FLATTEN_FLOOR", "true").lower() == "true"
ADD_PHYSICS_FLOOR = os.environ.get("TEEX_ADD_PHYSICS_FLOOR", "true").lower() == "true"

STAGE_PRIM_PATH = "/World/stage"
TEEX_PRIM_PATH = f"{STAGE_PRIM_PATH}/TEEX"
TEEX_TERRAIN_PRIM_PATH = f"{TEEX_PRIM_PATH}/Terrain"
TEEX_OBSTACLES_PRIM_PATH = f"{TEEX_PRIM_PATH}/Obstacles"


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


class TerrainHeightSampler:
    def __init__(self, points):
        self.points = points

    @classmethod
    def from_manifest(cls, manifest: dict):
        manifest_path = Path(TEEX_MANIFEST_PATH)
        terrain_name = manifest.get("generated_files", {}).get("terrain_usd", "teex_terrain.usd")
        terrain_path = manifest_path.parent / terrain_name
        if not terrain_path.exists():
            raise FileNotFoundError(f"terrain USD not found: {terrain_path}")

        stage = Usd.Stage.Open(str(terrain_path))
        if stage is None:
            raise RuntimeError(f"failed to open terrain USD: {terrain_path}")
        prim = stage.GetDefaultPrim()
        mesh = UsdGeom.Mesh(prim)
        points = mesh.GetPointsAttr().Get()
        if not points:
            raise RuntimeError(f"terrain USD has no mesh points: {terrain_path}")
        return cls(points)

    def height_at(self, x: float, y: float) -> float:
        best_point = min(
            self.points,
            key=lambda point: (float(point[0]) - x) ** 2 + (float(point[1]) - y) ** 2,
        )
        return float(best_point[2])


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


def _mesh_points(stage, prim_path: str):
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid() or not prim.IsA(UsdGeom.Mesh):
        carb.log_warn(f"[teex_multi] Mesh prim not found for floor normalization: {prim_path}")
        return None, []
    mesh = UsdGeom.Mesh(prim)
    points = mesh.GetPointsAttr().Get()
    if not points:
        carb.log_warn(f"[teex_multi] Mesh has no points for floor normalization: {prim_path}")
        return None, []
    return mesh, list(points)


def normalize_floor_to_zero(stage) -> None:
    terrain_mesh, terrain_points = _mesh_points(stage, TEEX_TERRAIN_PRIM_PATH)
    if terrain_mesh is None:
        return

    ground_z = [float(point[2]) for point in terrain_points]
    flat_terrain = [
        Gf.Vec3f(float(point[0]), float(point[1]), 0.0)
        for point in terrain_points
    ]
    terrain_mesh.GetPointsAttr().Set(flat_terrain)

    obstacles_mesh, obstacle_points = _mesh_points(stage, TEEX_OBSTACLES_PRIM_PATH)
    if obstacles_mesh is not None and len(obstacle_points) == len(terrain_points):
        normalized_obstacles = [
            Gf.Vec3f(
                float(point[0]),
                float(point[1]),
                max(float(point[2]) - ground_z[index], 0.0),
            )
            for index, point in enumerate(obstacle_points)
        ]
        obstacles_mesh.GetPointsAttr().Set(normalized_obstacles)
        carb.log_warn(
            "[teex_multi] Normalized TEEX floor to z=0 and converted obstacles "
            "to height above local ground."
        )
    elif obstacles_mesh is not None:
        carb.log_warn(
            "[teex_multi] Flattened terrain to z=0, but left obstacles unchanged "
            "because obstacle/terrain point counts differ."
        )


def resolve_spawn_z(sampler: TerrainHeightSampler | None, x: float, y: float) -> float:
    if FLATTEN_FLOOR and SPAWN_REFERENCE in {"agl", "above_ground", "terrain"}:
        return SPAWN_ALTITUDE_M
    if SPAWN_REFERENCE in {"agl", "above_ground", "terrain"} and sampler is not None:
        return sampler.height_at(x, y) + SPAWN_ALTITUDE_M
    if SPAWN_REFERENCE not in {"map_z", "local_z", "z"} and sampler is None:
        carb.log_warn(
            "[teex_multi] Terrain sampling unavailable; using TEEX_SPAWN_ALTITUDE_M "
            "as raw local map z."
        )
    return SPAWN_ALTITUDE_M


def spawn_positions(manifest: dict) -> list[list[float]]:
    min_x, min_y, max_x, max_y = local_bounds(manifest)
    width = max_x - min_x
    height = max_y - min_y
    center_x = min_x + width / 2.0
    center_y = min_y + height / 2.0

    spacing = 6.0
    start_x = center_x - spacing * (NUM_ROBOTS - 1) / 2.0
    y = min_y + min(max(height * 0.15, 12.0), max(height - 12.0, 12.0))
    sampler = None
    if SPAWN_REFERENCE in {"agl", "above_ground", "terrain"}:
        try:
            sampler = TerrainHeightSampler.from_manifest(manifest)
        except Exception as exc:
            carb.log_warn(
                "[teex_multi] Could not load terrain heights for AGL spawning: "
                f"{exc}"
            )
    return [
        [
            start_x + spacing * index,
            y,
            resolve_spawn_z(sampler, start_x + spacing * index, y),
        ]
        for index in range(NUM_ROBOTS)
    ]


def make_marker_material(stage, path: str, color: tuple[float, float, float]):
    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(*color)
    )
    shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(*color)
    )
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.25)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def add_drone_highlight(stage, index: int, drone_prim: str) -> None:
    colors = [
        (1.0, 0.1, 0.05),
        (0.1, 1.0, 0.2),
        (0.2, 0.45, 1.0),
        (1.0, 0.85, 0.1),
    ]
    color = colors[(index - 1) % len(colors)]
    root_path = f"{drone_prim}/TEEXHighlight"
    UsdGeom.Xform.Define(stage, root_path)
    material = make_marker_material(stage, f"{root_path}/MarkerMaterial", color)

    mast = UsdGeom.Cylinder.Define(stage, f"{root_path}/Mast")
    mast.CreateRadiusAttr(0.08)
    mast.CreateHeightAttr(3.0)
    mast.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    UsdGeom.XformCommonAPI(mast).SetTranslate((0.0, 0.0, 1.6))
    UsdShade.MaterialBindingAPI(mast).Bind(material)

    beacon = UsdGeom.Sphere.Define(stage, f"{root_path}/Beacon")
    beacon.CreateRadiusAttr(0.65)
    beacon.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    UsdGeom.XformCommonAPI(beacon).SetTranslate((0.0, 0.0, 3.25))
    UsdShade.MaterialBindingAPI(beacon).Bind(material)


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

    if HIGHLIGHT_DRONES:
        stage = omni.usd.get_context().get_stage()
        if stage is not None:
            add_drone_highlight(stage, index, drone_prim)


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

        if FLATTEN_FLOOR:
            normalize_floor_to_zero(stage)

        stage_prim = stage.GetPrimAtPath(STAGE_PRIM_PATH)
        add_colliders(stage_prim)

        min_x, min_y, max_x, max_y = local_bounds(manifest)
        center_x = min_x + (max_x - min_x) / 2.0
        center_y = min_y + (max_y - min_y) / 2.0
        if ADD_PHYSICS_FLOOR:
            add_collision_box_floor(
                stage,
                "/World/TEEXPhysicsFloor",
                (center_x, center_y),
                (max_x - min_x, max_y - min_y),
                top_z=0.0,
            )
            for _ in range(10):
                omni.kit.app.get_app().update()

        add_dome_light(stage)

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

        carb.log_warn("Closing TEEX multi-drone simulation.")
        self.timeline.stop()
        simulation_app.close()


def main():
    app = PegasusApp()
    app.run()


if __name__ == "__main__":
    main()

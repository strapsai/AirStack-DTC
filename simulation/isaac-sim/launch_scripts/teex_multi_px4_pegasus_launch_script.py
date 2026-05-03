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
 - TEEX_ADD_TERRAIN_COLLIDER (default auto): add an invisible DEM collider
   when the TEEX floor is not flattened
 - TEEX_TERRAIN_COLLIDER_MODE (default box_grid): box_grid, mesh, or both
 - TEEX_TERRAIN_COLLIDER_GRID (default 96): max samples per collider axis
 - TEEX_TERRAIN_BOX_GRID (default 48): box collider cells per terrain axis
 - TEEX_TERRAIN_COLLIDER_Z_OFFSET_M (default 0.05): lift collider above visual DEM
 - TEEX_TERRAIN_COLLIDER_THICKNESS_M (default 6.0): box-grid collider depth
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
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

try:
    from pxr import PhysxSchema
except ImportError:
    PhysxSchema = None

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
SPAWN_REFERENCE = os.environ.get("TEEX_SPAWN_REFERENCE", "agl").strip().lower()
SPAWN_ALTITUDE_M = float(os.environ.get("TEEX_SPAWN_ALTITUDE_M", "0.6"))
HIGHLIGHT_DRONES = os.environ.get("TEEX_HIGHLIGHT_DRONES", "true").lower() == "true"
FLATTEN_FLOOR = os.environ.get("TEEX_FLATTEN_FLOOR", "true").lower() == "true"
ADD_PHYSICS_FLOOR = os.environ.get("TEEX_ADD_PHYSICS_FLOOR", "true").lower() == "true"
ADD_TERRAIN_COLLIDER = os.environ.get("TEEX_ADD_TERRAIN_COLLIDER", "auto").strip().lower()
TERRAIN_COLLIDER_MODE = os.environ.get(
    "TEEX_TERRAIN_COLLIDER_MODE",
    "box_grid",
).strip().lower()
TERRAIN_COLLIDER_GRID = max(2, int(os.environ.get("TEEX_TERRAIN_COLLIDER_GRID", "96")))
TERRAIN_BOX_GRID = max(2, int(os.environ.get("TEEX_TERRAIN_BOX_GRID", "48")))
TERRAIN_COLLIDER_Z_OFFSET_M = float(os.environ.get("TEEX_TERRAIN_COLLIDER_Z_OFFSET_M", "0.05"))
TERRAIN_COLLIDER_THICKNESS_M = max(
    0.1,
    float(os.environ.get("TEEX_TERRAIN_COLLIDER_THICKNESS_M", "6.0")),
)
TERRAIN_COLLIDER_APPROXIMATION = os.environ.get(
    "TEEX_TERRAIN_COLLIDER_APPROXIMATION",
    "none",
).strip()

STAGE_PRIM_PATH = "/World/stage"
TEEX_PRIM_PATH = f"{STAGE_PRIM_PATH}/TEEX"
TEEX_TERRAIN_PRIM_PATH = f"{TEEX_PRIM_PATH}/Terrain"
TEEX_OBSTACLES_PRIM_PATH = f"{TEEX_PRIM_PATH}/Obstacles"
TEEX_TERRAIN_COLLIDER_PRIM_PATH = f"{STAGE_PRIM_PATH}/TEEXTerrainCollider"
TEEX_TERRAIN_BOX_COLLIDER_PRIM_PATH = "/World/TEEXTerrainCollisionGrid"


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


def add_collision_box_floor(
    stage,
    prim_path: str,
    center_xy: tuple[float, float],
    size_xy: tuple[float, float],
    *,
    top_z: float = 0.0,
    thickness: float = 0.2,
    margin: float = 20.0,
):
    """Add a simple invisible collider under the TEEX texture floor."""
    width = max(float(size_xy[0]) + 2.0 * float(margin), 1.0)
    depth = max(float(size_xy[1]) + 2.0 * float(margin), 1.0)
    height = max(float(thickness), 0.01)
    center_z = float(top_z) - height / 2.0

    cube = UsdGeom.Cube.Define(stage, prim_path)
    cube.CreateSizeAttr(1.0)
    UsdGeom.XformCommonAPI(cube).SetTranslate(
        Gf.Vec3d(float(center_xy[0]), float(center_xy[1]), center_z)
    )
    UsdGeom.XformCommonAPI(cube).SetScale(Gf.Vec3f(width, depth, height))

    prim = cube.GetPrim()
    if not prim.HasAPI(UsdPhysics.CollisionAPI):
        UsdPhysics.CollisionAPI.Apply(prim)
    if PhysxSchema is not None and not prim.HasAPI(PhysxSchema.PhysxCollisionAPI):
        PhysxSchema.PhysxCollisionAPI.Apply(prim)

    UsdGeom.Imageable(prim).MakeInvisible()
    carb.log_warn(
        "[teex_multi] Added invisible TEEX physics floor "
        f"{prim_path} size=({width:.2f}, {depth:.2f}, {height:.2f}) top_z={float(top_z):.2f}"
    )
    return prim


def add_static_collider(geom, approximation: str | None = None) -> None:
    prim = geom.GetPrim()
    if not prim.HasAPI(UsdPhysics.CollisionAPI):
        UsdPhysics.CollisionAPI.Apply(prim)
    if approximation is not None and prim.IsA(UsdGeom.Mesh) and hasattr(UsdPhysics, "MeshCollisionAPI"):
        mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(prim)
        mesh_collision.CreateApproximationAttr().Set(str(approximation))
    if PhysxSchema is not None and not prim.HasAPI(PhysxSchema.PhysxCollisionAPI):
        PhysxSchema.PhysxCollisionAPI.Apply(prim)


def add_static_mesh_collider(mesh, approximation: str = "none") -> None:
    add_static_collider(mesh, approximation)


def should_add_terrain_collider() -> bool:
    if ADD_TERRAIN_COLLIDER in {"1", "true", "yes", "on"}:
        return True
    if ADD_TERRAIN_COLLIDER in {"0", "false", "no", "off"}:
        return False
    if ADD_TERRAIN_COLLIDER not in {"", "auto"}:
        carb.log_warn(
            "[teex_multi] Unknown TEEX_ADD_TERRAIN_COLLIDER="
            f"{ADD_TERRAIN_COLLIDER!r}; using auto."
        )
    return not FLATTEN_FLOOR


def sample_axis_indices(count: int, max_samples: int) -> list[int]:
    count = int(count)
    max_samples = max(2, int(max_samples))
    if count <= max_samples:
        return list(range(max(count, 0)))

    last = count - 1
    values = {
        int(round(last * sample / float(max_samples - 1)))
        for sample in range(max_samples)
    }
    values.add(0)
    values.add(last)
    return sorted(values)


def terrain_grid_from_stage(stage, manifest: dict):
    terrain_mesh, terrain_points = _mesh_points(stage, TEEX_TERRAIN_PRIM_PATH)
    if terrain_mesh is None:
        return None, [], 0, 0

    mesh_sampling = manifest.get("mesh_sampling", {})
    row_count = int(mesh_sampling.get("sampled_rows", 0))
    col_count = int(mesh_sampling.get("sampled_cols", 0))
    if row_count <= 1 or col_count <= 1 or row_count * col_count != len(terrain_points):
        carb.log_warn(
            "[teex_multi] Could not derive DEM collider grid dimensions from "
            "manifest; leaving terrain mesh collision as-is."
        )
        return terrain_mesh, [], 0, 0
    return terrain_mesh, terrain_points, row_count, col_count


def add_dem_mesh_terrain_collider(stage, manifest: dict) -> None:
    """Add an invisible decimated terrain mesh for PhysX collision.

    The generated TEEX terrain already carries USD collision metadata, but in
    practice Pegasus drones can still tunnel through that referenced visual
    mesh. This runtime collider is authored directly into the active stage and
    intentionally stays invisible so the GES/TEEX visual scene is unchanged.
    """
    if not should_add_terrain_collider():
        return

    _terrain_mesh, terrain_points, row_count, col_count = terrain_grid_from_stage(
        stage,
        manifest,
    )
    if not terrain_points:
        return

    row_indices = sample_axis_indices(row_count, TERRAIN_COLLIDER_GRID)
    col_indices = sample_axis_indices(col_count, TERRAIN_COLLIDER_GRID)
    if len(row_indices) <= 1 or len(col_indices) <= 1:
        carb.log_warn("[teex_multi] DEM collider grid is too small; skipping.")
        return

    if stage.GetPrimAtPath(TEEX_TERRAIN_COLLIDER_PRIM_PATH).IsValid():
        stage.RemovePrim(TEEX_TERRAIN_COLLIDER_PRIM_PATH)

    collider_points = []
    for row in row_indices:
        for col in col_indices:
            point = terrain_points[row * col_count + col]
            collider_points.append(
                Gf.Vec3f(
                    float(point[0]),
                    float(point[1]),
                    float(point[2]) + TERRAIN_COLLIDER_Z_OFFSET_M,
                )
            )

    collider_col_count = len(col_indices)
    face_counts = []
    face_indices = []
    for row in range(len(row_indices) - 1):
        for col in range(len(col_indices) - 1):
            i00 = row * collider_col_count + col
            i01 = i00 + 1
            i10 = (row + 1) * collider_col_count + col
            i11 = i10 + 1
            face_counts.append(4)
            face_indices.extend([i00, i01, i11, i10])

    collider = UsdGeom.Mesh.Define(stage, TEEX_TERRAIN_COLLIDER_PRIM_PATH)
    collider.CreatePointsAttr(collider_points)
    collider.CreateFaceVertexCountsAttr(face_counts)
    collider.CreateFaceVertexIndicesAttr(face_indices)
    collider.CreateSubdivisionSchemeAttr("none")
    collider.CreateDoubleSidedAttr(True)
    add_static_mesh_collider(collider, TERRAIN_COLLIDER_APPROXIMATION)
    UsdGeom.Imageable(collider.GetPrim()).MakeInvisible()

    carb.log_warn(
        "[teex_multi] Added invisible DEM terrain collider "
        f"{TEEX_TERRAIN_COLLIDER_PRIM_PATH} "
        f"grid={len(col_indices)}x{len(row_indices)} "
        f"faces={len(face_counts)} "
        f"z_offset={TERRAIN_COLLIDER_Z_OFFSET_M:.2f} "
        f"approximation={TERRAIN_COLLIDER_APPROXIMATION}"
    )


def add_dem_box_terrain_collider(stage, manifest: dict) -> None:
    """Add an invisible terraced grid of cube colliders following the DEM."""
    if not should_add_terrain_collider():
        return

    _terrain_mesh, terrain_points, row_count, col_count = terrain_grid_from_stage(
        stage,
        manifest,
    )
    if not terrain_points:
        return

    row_edges = sample_axis_indices(row_count, TERRAIN_BOX_GRID + 1)
    col_edges = sample_axis_indices(col_count, TERRAIN_BOX_GRID + 1)
    if len(row_edges) <= 1 or len(col_edges) <= 1:
        carb.log_warn("[teex_multi] DEM box collider grid is too small; skipping.")
        return

    if stage.GetPrimAtPath(TEEX_TERRAIN_BOX_COLLIDER_PRIM_PATH).IsValid():
        stage.RemovePrim(TEEX_TERRAIN_BOX_COLLIDER_PRIM_PATH)

    UsdGeom.Xform.Define(stage, TEEX_TERRAIN_BOX_COLLIDER_PRIM_PATH)
    box_count = 0
    min_top_z = None
    max_top_z = None

    for row_index in range(len(row_edges) - 1):
        r0 = row_edges[row_index]
        r1 = row_edges[row_index + 1]
        if r1 <= r0:
            continue
        for col_index in range(len(col_edges) - 1):
            c0 = col_edges[col_index]
            c1 = col_edges[col_index + 1]
            if c1 <= c0:
                continue

            corners = [
                terrain_points[r0 * col_count + c0],
                terrain_points[r0 * col_count + c1],
                terrain_points[r1 * col_count + c0],
                terrain_points[r1 * col_count + c1],
            ]
            min_x = min(float(point[0]) for point in corners)
            max_x = max(float(point[0]) for point in corners)
            min_y = min(float(point[1]) for point in corners)
            max_y = max(float(point[1]) for point in corners)
            if max_x <= min_x or max_y <= min_y:
                continue

            top_z = max(float(point[2]) for point in corners) + TERRAIN_COLLIDER_Z_OFFSET_M
            center_x = min_x + (max_x - min_x) / 2.0
            center_y = min_y + (max_y - min_y) / 2.0
            center_z = top_z - TERRAIN_COLLIDER_THICKNESS_M / 2.0
            prim_path = (
                f"{TEEX_TERRAIN_BOX_COLLIDER_PRIM_PATH}/"
                f"Cell_{row_index:03d}_{col_index:03d}"
            )

            cube = UsdGeom.Cube.Define(stage, prim_path)
            cube.CreateSizeAttr(1.0)
            UsdGeom.XformCommonAPI(cube).SetTranslate(
                Gf.Vec3d(center_x, center_y, center_z)
            )
            UsdGeom.XformCommonAPI(cube).SetScale(
                Gf.Vec3f(max_x - min_x, max_y - min_y, TERRAIN_COLLIDER_THICKNESS_M)
            )
            add_static_collider(cube)
            UsdGeom.Imageable(cube.GetPrim()).MakeInvisible()
            box_count += 1
            min_top_z = top_z if min_top_z is None else min(min_top_z, top_z)
            max_top_z = top_z if max_top_z is None else max(max_top_z, top_z)

    if box_count == 0:
        carb.log_warn("[teex_multi] DEM box collider produced no boxes; skipping.")
        return

    carb.log_warn(
        "[teex_multi] Added invisible DEM box terrain collider "
        f"{TEEX_TERRAIN_BOX_COLLIDER_PRIM_PATH} "
        f"grid={len(col_edges) - 1}x{len(row_edges) - 1} "
        f"boxes={box_count} "
        f"top_z_range=({float(min_top_z):.2f}, {float(max_top_z):.2f}) "
        f"thickness={TERRAIN_COLLIDER_THICKNESS_M:.2f}"
    )


def add_dem_terrain_collider(stage, manifest: dict) -> None:
    mode = TERRAIN_COLLIDER_MODE
    if mode == "auto":
        mode = "box_grid"
    if mode in {"box", "boxes", "grid"}:
        mode = "box_grid"
    if mode not in {"box_grid", "mesh", "both"}:
        carb.log_warn(
            "[teex_multi] Unknown TEEX_TERRAIN_COLLIDER_MODE="
            f"{TERRAIN_COLLIDER_MODE!r}; using box_grid."
        )
        mode = "box_grid"

    if mode in {"box_grid", "both"}:
        add_dem_box_terrain_collider(stage, manifest)
    if mode in {"mesh", "both"}:
        add_dem_mesh_terrain_collider(stage, manifest)


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
        add_dem_terrain_collider(stage, manifest)

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

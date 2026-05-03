#!/usr/bin/env python3
"""Bake Google Earth Studio render colors onto metric TEEX USD meshes.

Geometry and collision remain TEEX-derived. The GES video/metadata are used
only to project RGB colors onto existing TEEX vertices for visual refinement.

This writes new USD layers and does not modify the original TEEX scene.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class GesBakeError(RuntimeError):
    """Raised for clear user-facing baking failures."""


@dataclass(frozen=True)
class Origin:
    easting: float
    northing: float
    z: float
    epsg: int


@dataclass(frozen=True)
class MeshData:
    prim_path: str
    points: Any
    face_counts: list[int]
    face_indices: list[int]


@dataclass(frozen=True)
class Deps:
    np: Any
    cv2: Any
    rasterio: Any
    Transformer: Any
    geodetic2enu: Any
    Rotation: Any
    Usd: Any
    UsdGeom: Any
    UsdPhysics: Any
    UsdLux: Any
    Sdf: Any
    Gf: Any


def require_dependencies() -> Deps:
    missing: list[str] = []
    try:
        import numpy as np
    except ImportError:
        missing.append("numpy")
        np = None
    try:
        import cv2
    except ImportError:
        missing.append("opencv-python")
        cv2 = None
    try:
        import rasterio
    except ImportError:
        rasterio = None
    try:
        from pyproj import Transformer
    except ImportError:
        missing.append("pyproj")
        Transformer = None
    try:
        from pymap3d.enu import geodetic2enu
    except ImportError:
        missing.append("pymap3d")
        geodetic2enu = None
    try:
        from scipy.spatial.transform import Rotation
    except ImportError:
        missing.append("scipy")
        Rotation = None
    try:
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics
    except ImportError:
        missing.append("pxr (install usd-core or use Isaac Sim's Python)")
        Gf = Sdf = Usd = UsdGeom = UsdLux = UsdPhysics = None

    if missing:
        raise GesBakeError(
            "Missing required dependencies: "
            + ", ".join(missing)
            + ". Run this from an environment with RESCUE/geospatial deps and USD Python."
        )

    return Deps(
        np=np,
        cv2=cv2,
        rasterio=rasterio,
        Transformer=Transformer,
        geodetic2enu=geodetic2enu,
        Rotation=Rotation,
        Usd=Usd,
        UsdGeom=UsdGeom,
        UsdPhysics=UsdPhysics,
        UsdLux=UsdLux,
        Sdf=Sdf,
        Gf=Gf,
    )


def default_paths() -> dict[str, Path]:
    isaac_sim_dir = Path(__file__).resolve().parents[1]
    autonomy_ws = Path(__file__).resolve().parents[6]
    return {
        "teex_usd_dir": isaac_sim_dir / "assets/teex/usd",
        "output_dir": isaac_sim_dir / "assets/teex/ges_visual",
        "video": autonomy_ws / "src/simulation/RESCUE-DTC/generated/renders/disaster_city.mp4",
        "ges_json": autonomy_ws / "src/simulation/RESCUE-DTC/generated/renders/disaster_city.json",
    }


def parse_args() -> argparse.Namespace:
    defaults = default_paths()
    parser = argparse.ArgumentParser(
        description="Bake GES video colors onto TEEX USD meshes while preserving TEEX geometry.",
    )
    parser.add_argument("--teex-usd-dir", type=Path, default=defaults["teex_usd_dir"])
    parser.add_argument("--output-dir", type=Path, default=defaults["output_dir"])
    parser.add_argument("--video", type=Path, default=defaults["video"])
    parser.add_argument("--ges-json", type=Path, default=defaults["ges_json"])
    parser.add_argument(
        "--colorize",
        choices=("obstacles", "terrain", "both"),
        default="obstacles",
        help="Which TEEX mesh layer(s) to colorize from the GES video.",
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=15,
        help="Use every Nth GES/video frame. 15 is about 2 fps for the default 30 fps render.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Optional cap on sampled frames. 0 means no cap.",
    )
    parser.add_argument(
        "--max-depth-m",
        type=float,
        default=600.0,
        help="Ignore projected points farther than this from the camera.",
    )
    parser.add_argument(
        "--fallback-color",
        nargs=3,
        type=float,
        default=(0.62, 0.60, 0.55),
        metavar=("R", "G", "B"),
        help="RGB fallback in [0,1] for vertices never seen by a GES camera.",
    )
    parser.add_argument(
        "--no-collider",
        action="store_true",
        help="Do not apply collider metadata to generated colored mesh layers.",
    )
    return parser.parse_args()


def load_manifest(teex_usd_dir: Path) -> dict[str, Any]:
    manifest_path = teex_usd_dir / "teex_scene_manifest.json"
    if not manifest_path.exists():
        raise GesBakeError(f"Missing TEEX manifest: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def load_origin(manifest: dict[str, Any]) -> Origin:
    origin = manifest.get("origin", {})
    crs = manifest.get("crs", {})
    for key in ("easting", "northing", "z_origin"):
        if key not in origin:
            raise GesBakeError(f"TEEX manifest origin is missing {key}")
    return Origin(
        easting=float(origin["easting"]),
        northing=float(origin["northing"]),
        z=float(origin["z_origin"]),
        epsg=int(crs.get("epsg", 32614)),
    )


def resolve_map_products_dir(teex_usd_dir: Path, manifest: dict[str, Any]) -> Path | None:
    source = manifest.get("source", {})
    raw = source.get("map_products_dir")
    if raw:
        path = Path(raw)
        if path.exists():
            return path
    sibling = teex_usd_dir.parent / "map_products"
    if sibling.exists():
        return sibling
    return None


def load_mesh_from_usd(deps: Deps, usd_path: Path, prim_path: str) -> MeshData:
    if not usd_path.exists():
        raise GesBakeError(f"Missing USD layer: {usd_path}")
    stage = deps.Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise GesBakeError(f"Failed to open USD layer: {usd_path}")
    prim = stage.GetDefaultPrim()
    if not prim or not prim.IsValid() or not prim.IsA(deps.UsdGeom.Mesh):
        prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid() or not prim.IsA(deps.UsdGeom.Mesh):
        raise GesBakeError(f"No mesh prim found at {prim_path} in {usd_path}")
    mesh = deps.UsdGeom.Mesh(prim)
    points = deps.np.asarray(mesh.GetPointsAttr().Get(), dtype="float64")
    face_counts = list(mesh.GetFaceVertexCountsAttr().Get())
    face_indices = list(mesh.GetFaceVertexIndicesAttr().Get())
    return MeshData(prim_path=prim_path, points=points, face_counts=face_counts, face_indices=face_indices)


def rot_ecef2enu(deps: Deps, lat: float, lon: float):
    np = deps.np
    lamb = np.deg2rad(lon)
    phi = np.deg2rad(lat)
    s_l = np.sin(lamb)
    s_p = np.sin(phi)
    c_l = np.cos(lamb)
    c_p = np.cos(phi)
    return np.array(
        [
            [-s_l, c_l, 0],
            [-s_p * c_l, -s_p * s_l, c_p],
            [c_p * c_l, c_p * s_l, s_p],
        ],
        dtype="float64",
    )


def ges_frame_to_c2w(deps: Deps, frame: dict[str, Any], ref: dict[str, Any], ecef_to_enu):
    np = deps.np
    coord = frame["coordinate"]
    ref_coord = ref["coordinate"]
    x, y, z = deps.geodetic2enu(
        coord["latitude"],
        coord["longitude"],
        coord["altitude"],
        ref_coord["latitude"],
        ref_coord["longitude"],
        ref_coord["altitude"],
    )
    rot = frame["rotation"]
    r_ecef = deps.Rotation.from_euler(
        "XYZ",
        [rot["x"], rot["y"], rot["z"]],
        degrees=True,
    ).as_matrix()
    r_enu = ecef_to_enu @ r_ecef
    c2w = np.eye(4, dtype="float64")
    c2w[:3, :3] = r_enu
    c2w[:3, 3] = np.array([x, y, z], dtype="float64")
    return c2w


def sample_dem_altitudes(
    deps: Deps,
    map_products_dir: Path | None,
    origin: Origin,
    points_local: Any,
) -> Any:
    np = deps.np
    fallback = np.full(points_local.shape[0], origin.z, dtype="float64")
    if deps.rasterio is None or map_products_dir is None:
        return fallback
    dem_path = map_products_dir / "dem_merged.tif"
    if not dem_path.exists():
        return fallback

    eastings = origin.easting + points_local[:, 0]
    northings = origin.northing + points_local[:, 1]
    coords = list(zip(eastings.tolist(), northings.tolist()))
    values = np.empty(points_local.shape[0], dtype="float64")
    with deps.rasterio.open(dem_path) as src:
        for start in range(0, len(coords), 65536):
            chunk = coords[start : start + 65536]
            sampled = [value[0] for value in src.sample(chunk)]
            values[start : start + len(chunk)] = np.asarray(sampled, dtype="float64")
    invalid = ~np.isfinite(values) | (values < -1000.0)
    values[invalid] = origin.z
    return values


def local_points_to_ges_enu(
    deps: Deps,
    points_local: Any,
    origin: Origin,
    dem_altitudes: Any,
    ges_ref_frame: dict[str, Any],
    *,
    ground_mode: str,
) -> Any:
    transformer = deps.Transformer.from_crs(
        f"EPSG:{origin.epsg}",
        "EPSG:4326",
        always_xy=True,
    )
    eastings = origin.easting + points_local[:, 0]
    northings = origin.northing + points_local[:, 1]
    lons, lats = transformer.transform(eastings, northings)
    if ground_mode == "dem":
        alts = origin.z + points_local[:, 2]
    else:
        alts = dem_altitudes + points_local[:, 2]
    ref = ges_ref_frame["coordinate"]
    x, y, z = deps.geodetic2enu(
        lats,
        lons,
        alts,
        ref["latitude"],
        ref["longitude"],
        ref["altitude"],
    )
    return deps.np.stack([x, y, z], axis=1).astype("float64", copy=False)


def read_video_frame(deps: Deps, cap: Any, index: int):
    cap.set(deps.cv2.CAP_PROP_POS_FRAMES, int(index))
    ok, frame_bgr = cap.read()
    if not ok:
        return None
    return deps.cv2.cvtColor(frame_bgr, deps.cv2.COLOR_BGR2RGB).astype("float32") / 255.0


def bilinear_sample(deps: Deps, image: Any, u: Any, v: Any):
    np = deps.np
    h, w = image.shape[:2]
    u0 = np.floor(u).astype("int64").clip(0, w - 2)
    v0 = np.floor(v).astype("int64").clip(0, h - 2)
    du = (u - u0)[:, None]
    dv = (v - v0)[:, None]
    return (
        image[v0, u0] * (1.0 - du) * (1.0 - dv)
        + image[v0, u0 + 1] * du * (1.0 - dv)
        + image[v0 + 1, u0] * (1.0 - du) * dv
        + image[v0 + 1, u0 + 1] * du * dv
    )


def project_colors(
    deps: Deps,
    points_enu: Any,
    ges_data: dict[str, Any],
    video_path: Path,
    *,
    frame_stride: int,
    max_frames: int,
    max_depth_m: float,
    fallback_color: tuple[float, float, float],
) -> tuple[Any, int, int]:
    np = deps.np
    frames = ges_data["cameraFrames"]
    width = int(ges_data["width"])
    height = int(ges_data["height"])
    ref = frames[0]
    ecef_to_enu = rot_ecef2enu(
        deps,
        ref["coordinate"]["latitude"],
        ref["coordinate"]["longitude"],
    )
    indices = list(range(0, min(len(frames), int(ges_data.get("numFrames", len(frames)))), max(1, frame_stride)))
    if max_frames > 0:
        indices = indices[: int(max_frames)]

    colors = np.tile(np.asarray(fallback_color, dtype="float32"), (points_enu.shape[0], 1))
    best_depth = np.full(points_enu.shape[0], np.inf, dtype="float64")

    cap = deps.cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise GesBakeError(f"Could not open video: {video_path}")

    used_frames = 0
    for frame_index in indices:
        image = read_video_frame(deps, cap, frame_index)
        if image is None:
            continue
        frame = frames[frame_index]
        c2w = ges_frame_to_c2w(deps, frame, ref, ecef_to_enu)
        r = c2w[:3, :3]
        t = c2w[:3, 3]
        points_cam = (points_enu - t) @ r
        z = points_cam[:, 2]
        f = height / (2.0 * math.tan(math.radians(float(frame["fovVertical"]) / 2.0)))
        u = f * (points_cam[:, 0] / z) + width / 2.0
        v = f * (points_cam[:, 1] / z) + height / 2.0
        valid = (
            (z > 0.5)
            & (z < float(max_depth_m))
            & (u >= 0.0)
            & (u < width - 1)
            & (v >= 0.0)
            & (v < height - 1)
            & (z < best_depth)
        )
        if not bool(np.any(valid)):
            continue
        sampled = bilinear_sample(deps, image, u[valid], v[valid])
        colors[valid] = sampled
        best_depth[valid] = z[valid]
        used_frames += 1

    cap.release()
    seen_vertices = int(np.count_nonzero(np.isfinite(best_depth)))
    return colors.astype("float32", copy=False), seen_vertices, used_frames


def apply_mesh_collider(deps: Deps, mesh: Any) -> None:
    prim = mesh.GetPrim()
    if not prim.HasAPI(deps.UsdPhysics.CollisionAPI):
        deps.UsdPhysics.CollisionAPI.Apply(prim)
    if hasattr(deps.UsdPhysics, "MeshCollisionAPI"):
        mesh_collision = deps.UsdPhysics.MeshCollisionAPI.Apply(prim)
        mesh_collision.CreateApproximationAttr().Set("meshSimplification")


def write_colored_mesh_usd(
    deps: Deps,
    output_path: Path,
    prim_path: str,
    mesh_data: MeshData,
    colors: Any,
    *,
    add_collider: bool,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stage = deps.Usd.Stage.CreateNew(str(output_path))
    deps.UsdGeom.SetStageUpAxis(stage, deps.UsdGeom.Tokens.z)
    deps.UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    mesh = deps.UsdGeom.Mesh.Define(stage, prim_path)
    stage.SetDefaultPrim(mesh.GetPrim())
    mesh.CreatePointsAttr(
        [
            deps.Gf.Vec3f(float(point[0]), float(point[1]), float(point[2]))
            for point in mesh_data.points
        ]
    )
    mesh.CreateFaceVertexCountsAttr(mesh_data.face_counts)
    mesh.CreateFaceVertexIndicesAttr(mesh_data.face_indices)
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateDoubleSidedAttr(True)
    color_values = [
        deps.Gf.Vec3f(float(color[0]), float(color[1]), float(color[2]))
        for color in colors
    ]
    color_primvar = mesh.CreateDisplayColorPrimvar(
        interpolation=deps.UsdGeom.Tokens.vertex
    )
    color_primvar.Set(color_values)
    if add_collider:
        apply_mesh_collider(deps, mesh)
    stage.GetRootLayer().Save()


def make_relative_asset_path(asset_path: Path, usd_path: Path) -> str:
    return os.path.relpath(asset_path.resolve(), usd_path.parent.resolve())


def write_root_usd(
    deps: Deps,
    output_path: Path,
    terrain_path: Path,
    obstacles_path: Path,
    manifest_path: Path,
) -> None:
    stage = deps.Usd.Stage.CreateNew(str(output_path))
    deps.UsdGeom.SetStageUpAxis(stage, deps.UsdGeom.Tokens.z)
    deps.UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    world = deps.UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    teex = deps.UsdGeom.Xform.Define(stage, "/World/TEEX")

    terrain = stage.DefinePrim("/World/TEEX/Terrain")
    terrain.GetReferences().AddReference(make_relative_asset_path(terrain_path, output_path))
    obstacles = stage.DefinePrim("/World/TEEX/Obstacles")
    obstacles.GetReferences().AddReference(make_relative_asset_path(obstacles_path, output_path))
    metadata = deps.UsdGeom.Xform.Define(stage, "/World/TEEX/Metadata").GetPrim()
    metadata.CreateAttribute("ges_visual_manifest", deps.Sdf.ValueTypeNames.Asset).Set(
        deps.Sdf.AssetPath(make_relative_asset_path(manifest_path, output_path))
    )
    teex.GetPrim().SetCustomDataByKey(
        "description",
        "TEEX metric scene with GES-render-baked visual colors",
    )
    sun = deps.UsdLux.DistantLight.Define(stage, "/World/TEEX/Sun")
    sun.CreateIntensityAttr(600.0)
    sun.CreateAngleAttr(0.53)
    deps.UsdGeom.XformCommonAPI(sun).SetRotate((45.0, 0.0, 35.0))
    stage.GetRootLayer().Save()


def bake_layer(
    deps: Deps,
    layer_name: str,
    mesh_data: MeshData,
    origin: Origin,
    map_products_dir: Path | None,
    ges_data: dict[str, Any],
    args: argparse.Namespace,
    output_path: Path,
    ground_mode: str,
) -> dict[str, Any]:
    dem_altitudes = sample_dem_altitudes(deps, map_products_dir, origin, mesh_data.points)
    points_enu = local_points_to_ges_enu(
        deps,
        mesh_data.points,
        origin,
        dem_altitudes,
        ges_data["cameraFrames"][0],
        ground_mode=ground_mode,
    )
    colors, seen_vertices, used_frames = project_colors(
        deps,
        points_enu,
        ges_data,
        args.video.expanduser().resolve(),
        frame_stride=int(args.frame_stride),
        max_frames=int(args.max_frames),
        max_depth_m=float(args.max_depth_m),
        fallback_color=tuple(float(value) for value in args.fallback_color),
    )
    write_colored_mesh_usd(
        deps,
        output_path,
        f"/{layer_name}",
        mesh_data,
        colors,
        add_collider=not bool(args.no_collider),
    )
    return {
        "vertices": int(mesh_data.points.shape[0]),
        "seen_vertices": seen_vertices,
        "seen_fraction": seen_vertices / max(int(mesh_data.points.shape[0]), 1),
        "used_frames": used_frames,
        "output": output_path.name,
    }


def generate(args: argparse.Namespace) -> dict[str, Any]:
    deps = require_dependencies()
    teex_usd_dir = args.teex_usd_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.video.expanduser().exists():
        raise GesBakeError(f"Missing GES video: {args.video}")
    if not args.ges_json.expanduser().exists():
        raise GesBakeError(f"Missing GES JSON: {args.ges_json}")

    manifest = load_manifest(teex_usd_dir)
    origin = load_origin(manifest)
    ground_mode = manifest.get("vertical_model", {}).get("ground_mode", "flat_zero")
    map_products_dir = resolve_map_products_dir(teex_usd_dir, manifest)
    ges_data = json.loads(args.ges_json.expanduser().read_text(encoding="utf-8"))

    terrain_src = teex_usd_dir / manifest.get("generated_files", {}).get("terrain_usd", "teex_terrain.usd")
    obstacles_src = teex_usd_dir / manifest.get("generated_files", {}).get("obstacles_usd", "teex_obstacles.usd")
    terrain_out = output_dir / "teex_terrain_ges_color.usd"
    obstacles_out = output_dir / "teex_obstacles_ges_color.usd"
    root_out = output_dir / "teex_scene_ges_visual.usd"
    bake_manifest_path = output_dir / "teex_ges_visual_manifest.json"

    results: dict[str, Any] = {}
    if args.colorize in {"terrain", "both"}:
        terrain_mesh = load_mesh_from_usd(deps, terrain_src, "/Terrain")
        results["terrain"] = bake_layer(
            deps,
            "Terrain",
            terrain_mesh,
            origin,
            map_products_dir,
            ges_data,
            args,
            terrain_out,
            ground_mode,
        )
        terrain_for_root = terrain_out
    else:
        terrain_for_root = terrain_src

    if args.colorize in {"obstacles", "both"}:
        obstacle_mesh = load_mesh_from_usd(deps, obstacles_src, "/Obstacles")
        results["obstacles"] = bake_layer(
            deps,
            "Obstacles",
            obstacle_mesh,
            origin,
            map_products_dir,
            ges_data,
            args,
            obstacles_out,
            ground_mode,
        )
        obstacles_for_root = obstacles_out
    else:
        obstacles_for_root = obstacles_src

    bake_manifest = {
        "source": {
            "teex_usd_dir": str(teex_usd_dir),
            "map_products_dir": str(map_products_dir) if map_products_dir else None,
            "ges_video": str(args.video.expanduser().resolve()),
            "ges_json": str(args.ges_json.expanduser().resolve()),
        },
        "origin": {
            "easting": origin.easting,
            "northing": origin.northing,
            "z_origin": origin.z,
            "epsg": origin.epsg,
        },
        "settings": {
            "colorize": args.colorize,
            "teex_ground_mode": ground_mode,
            "frame_stride": int(args.frame_stride),
            "max_frames": int(args.max_frames),
            "max_depth_m": float(args.max_depth_m),
            "fallback_color": [float(value) for value in args.fallback_color],
        },
        "layers": results,
        "generated_files": {
            "root_usd": root_out.name,
            "terrain_usd": terrain_for_root.name,
            "obstacles_usd": obstacles_for_root.name,
            "manifest": bake_manifest_path.name,
        },
    }
    bake_manifest_path.write_text(json.dumps(bake_manifest, indent=2) + "\n", encoding="utf-8")
    write_root_usd(
        deps,
        root_out,
        terrain_for_root,
        obstacles_for_root,
        bake_manifest_path,
    )
    return bake_manifest


def main() -> int:
    try:
        manifest = generate(parse_args())
    except GesBakeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"error: unexpected GES visual baking failure: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(manifest["generated_files"], indent=2))
    for name, layer in manifest["layers"].items():
        print(
            f"{name}: seen {layer['seen_vertices']}/{layer['vertices']} "
            f"({layer['seen_fraction']:.1%}) using {layer['used_frames']} frames"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

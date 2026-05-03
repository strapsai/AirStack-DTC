#!/usr/bin/env python3
"""Create a TEEX Isaac scene that uses RESCUE-DTC reconstruction geometry.

This tool keeps the existing TEEX scene intact. It writes a separate Rescue
building layer plus a root USD that references the existing TEEX terrain and
the generated Rescue mesh.

Run it from an environment that has:
  numpy, trimesh, safetensors, pyproj, open3d, and pxr/usd-core

The RESCUE-DTC repository is treated as an input artifact only.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class RescueUsdError(RuntimeError):
    """Raised for clear user-facing conversion failures."""


@dataclass(frozen=True)
class Origin:
    easting: float
    northing: float
    z: float
    epsg: int


@dataclass(frozen=True)
class Deps:
    np: Any
    trimesh: Any
    safe_open: Any
    Transformer: Any
    o3d: Any
    Usd: Any
    UsdGeom: Any
    UsdPhysics: Any
    UsdShade: Any
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
        import trimesh
    except ImportError:
        missing.append("trimesh")
        trimesh = None
    try:
        from safetensors.numpy import safe_open
    except ImportError:
        missing.append("safetensors")
        safe_open = None
    try:
        from pyproj import Transformer
    except ImportError:
        missing.append("pyproj")
        Transformer = None
    try:
        import open3d as o3d
    except ImportError:
        missing.append("open3d")
        o3d = None
    try:
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade
    except ImportError:
        missing.append("pxr (install usd-core or use Isaac Sim's Python)")
        Gf = Sdf = Usd = UsdGeom = UsdLux = UsdPhysics = UsdShade = None

    if missing:
        raise RescueUsdError(
            "Missing required dependencies: "
            + ", ".join(missing)
            + ". Run this from a conversion environment that has RESCUE's "
            "geometry dependencies plus USD Python bindings."
        )

    return Deps(
        np=np,
        trimesh=trimesh,
        safe_open=safe_open,
        Transformer=Transformer,
        o3d=o3d,
        Usd=Usd,
        UsdGeom=UsdGeom,
        UsdPhysics=UsdPhysics,
        UsdShade=UsdShade,
        UsdLux=UsdLux,
        Sdf=Sdf,
        Gf=Gf,
    )


def default_paths() -> dict[str, Path]:
    isaac_sim_dir = Path(__file__).resolve().parents[1]
    autonomy_ws = Path(__file__).resolve().parents[6]
    return {
        "rescue_run_dir": autonomy_ws
        / "src/simulation/RESCUE-DTC/generated/full_pipeline_runs/teex_disaster_city",
        "teex_usd_dir": isaac_sim_dir / "assets/teex/usd",
        "output_dir": isaac_sim_dir / "assets/teex/rescue",
    }


def parse_args() -> argparse.Namespace:
    defaults = default_paths()
    parser = argparse.ArgumentParser(
        description="Merge RESCUE-DTC GLB reconstruction into a TEEX Isaac USD layer.",
    )
    parser.add_argument(
        "--rescue-run-dir",
        type=Path,
        default=defaults["rescue_run_dir"],
        help="RESCUE run directory containing reconstruction_mesh.glb and georeg.safetensors.",
    )
    parser.add_argument(
        "--teex-usd-dir",
        type=Path,
        default=defaults["teex_usd_dir"],
        help="Directory containing the current TEEX USD assets and manifest.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=defaults["output_dir"],
        help="Directory where Rescue USD assets will be written.",
    )
    parser.add_argument(
        "--max-faces",
        type=int,
        default=500_000,
        help="Target maximum face count after clipping and decimation.",
    )
    parser.add_argument(
        "--clip-margin-m",
        type=float,
        default=20.0,
        help="Meters of margin around TEEX bounds kept from the RESCUE mesh.",
    )
    parser.add_argument(
        "--clip-face-mode",
        choices=("centroid", "any", "all"),
        default="centroid",
        help="How to decide whether a triangle is inside the TEEX map bounds.",
    )
    parser.add_argument(
        "--floor-percentile",
        type=float,
        default=2.0,
        help="Percentile of RESCUE z values treated as the local floor.",
    )
    parser.add_argument(
        "--z-scale",
        type=float,
        default=1.0,
        help="Scale applied to RESCUE relative heights.",
    )
    parser.add_argument(
        "--z-offset-m",
        type=float,
        default=0.0,
        help="Offset added after floor normalization.",
    )
    parser.add_argument(
        "--min-height-m",
        type=float,
        default=2.5,
        help=(
            "Only keep RESCUE faces whose highest vertex is at least this far "
            "above the reconstructed floor. This removes the full ground sheet "
            "so the GLB acts as an overlay on TEEX terrain/obstacles."
        ),
    )
    parser.add_argument(
        "--color-boost",
        type=float,
        default=1.0,
        help="Multiplier applied to GLB vertex RGB colors before writing USD.",
    )
    parser.add_argument(
        "--include-lidar-obstacles",
        action="store_true",
        help="Reference the old LiDAR obstacle layer in the Rescue root scene.",
    )
    parser.add_argument(
        "--reuse-existing-rescue-layer",
        action="store_true",
        help=(
            "Skip GLB/georeg processing when teex_rescue_buildings.usd already "
            "exists, and only rewrite the root Rescue scene."
        ),
    )
    return parser.parse_args()


def load_manifest(teex_usd_dir: Path) -> dict[str, Any]:
    manifest_path = teex_usd_dir / "teex_scene_manifest.json"
    if not manifest_path.exists():
        raise RescueUsdError(f"Missing TEEX manifest: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def load_origin(manifest: dict[str, Any]) -> Origin:
    origin = manifest.get("origin", {})
    crs = manifest.get("crs", {})
    missing = [
        name
        for name in ("easting", "northing", "z_origin")
        if name not in origin
    ]
    if missing:
        raise RescueUsdError(
            "TEEX manifest origin is missing fields: " + ", ".join(missing)
        )
    epsg = crs.get("epsg")
    if epsg is None:
        source = manifest.get("source", {})
        map_products_dir = source.get("map_products_dir")
        if map_products_dir:
            origin_yaml = Path(map_products_dir) / "site_origin.yaml"
            epsg = parse_epsg_from_origin_yaml(origin_yaml)
    if epsg is None:
        epsg = 32614
    return Origin(
        easting=float(origin["easting"]),
        northing=float(origin["northing"]),
        z=float(origin["z_origin"]),
        epsg=int(epsg),
    )


def parse_epsg_from_origin_yaml(path: Path) -> int | None:
    if not path.exists():
        return None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if line.startswith("utm_epsg:"):
            return int(line.split(":", 1)[1].strip())
    return None


def local_bounds(manifest: dict[str, Any], margin_m: float) -> tuple[float, float, float, float]:
    bounds = manifest.get("bounds_local_m")
    if not isinstance(bounds, list) or len(bounds) != 4:
        raise RescueUsdError("TEEX manifest does not contain bounds_local_m[4].")
    min_x, min_y, max_x, max_y = [float(value) for value in bounds]
    margin = float(margin_m)
    return min_x - margin, min_y - margin, max_x + margin, max_y + margin


def load_rescue_georeg(deps: Deps, georeg_path: Path):
    if not georeg_path.exists():
        raise RescueUsdError(f"Missing RESCUE georeg file: {georeg_path}")
    with deps.safe_open(georeg_path, framework="np") as tensors:
        keys = set(tensors.keys())
        if "latlon" not in keys or "world_points" not in keys:
            raise RescueUsdError(
                f"{georeg_path} must contain latlon and world_points tensors; found {sorted(keys)}"
            )
        latlon = tensors.get_tensor("latlon")
        world_points = tensors.get_tensor("world_points")
    if latlon.ndim != 2 or latlon.shape[1] != 2:
        raise RescueUsdError(f"latlon has unexpected shape: {latlon.shape}")
    if world_points.ndim != 2 or world_points.shape[1] != 3:
        raise RescueUsdError(f"world_points has unexpected shape: {world_points.shape}")
    return latlon, world_points


def load_rescue_mesh(deps: Deps, mesh_path: Path):
    if not mesh_path.exists():
        raise RescueUsdError(f"Missing RESCUE mesh: {mesh_path}")
    loaded = deps.trimesh.load(mesh_path, force="mesh", process=False)
    if not hasattr(loaded, "vertices") or not hasattr(loaded, "faces"):
        raise RescueUsdError(f"Could not load {mesh_path} as a triangle mesh.")
    if len(loaded.vertices) == 0 or len(loaded.faces) == 0:
        raise RescueUsdError(f"{mesh_path} contains no mesh geometry.")
    return loaded


def rescue_vertex_colors(deps: Deps, rescue_mesh: Any, *, color_boost: float):
    np = deps.np
    if not hasattr(rescue_mesh, "visual") or not hasattr(rescue_mesh.visual, "vertex_colors"):
        return None
    colors = np.asarray(rescue_mesh.visual.vertex_colors)
    if colors.ndim != 2 or colors.shape[0] != rescue_mesh.vertices.shape[0] or colors.shape[1] < 3:
        return None
    rgb = colors[:, :3].astype("float32") / 255.0
    if not np.isclose(float(color_boost), 1.0):
        rgb = np.clip(rgb * float(color_boost), 0.0, 1.0)
    return rgb.astype("float32", copy=False)


def rescue_vertices_to_teex_local(
    deps: Deps,
    latlon: Any,
    world_points: Any,
    origin: Origin,
    *,
    floor_percentile: float,
    z_scale: float,
    z_offset_m: float,
):
    np = deps.np
    transformer = deps.Transformer.from_crs(
        "EPSG:4326",
        f"EPSG:{origin.epsg}",
        always_xy=True,
    )
    lat = latlon[:, 0].astype("float64", copy=False)
    lon = latlon[:, 1].astype("float64", copy=False)
    easting, northing = transformer.transform(lon, lat)
    z_raw = world_points[:, 2].astype("float64", copy=False)
    floor_z = float(np.percentile(z_raw, float(floor_percentile)))

    local = np.empty((latlon.shape[0], 3), dtype="float32")
    local[:, 0] = np.asarray(easting, dtype="float64") - origin.easting
    local[:, 1] = np.asarray(northing, dtype="float64") - origin.northing
    local[:, 2] = np.maximum((z_raw - floor_z) * float(z_scale) + float(z_offset_m), 0.0)
    return local, floor_z


def filter_faces_to_bounds(
    deps: Deps,
    faces: Any,
    vertices: Any,
    bounds: tuple[float, float, float, float],
    *,
    mode: str,
    min_height_m: float,
    colors: Any | None = None,
    chunk_faces: int = 1_000_000,
):
    np = deps.np
    min_x, min_y, max_x, max_y = bounds
    inside_vertices = (
        (vertices[:, 0] >= min_x)
        & (vertices[:, 0] <= max_x)
        & (vertices[:, 1] >= min_y)
        & (vertices[:, 1] <= max_y)
    )

    chunks: list[Any] = []
    face_count = int(faces.shape[0])
    for start in range(0, face_count, int(chunk_faces)):
        stop = min(start + int(chunk_faces), face_count)
        tri = faces[start:stop]
        if mode == "any":
            keep = inside_vertices[tri].any(axis=1)
        elif mode == "all":
            keep = inside_vertices[tri].all(axis=1)
        else:
            cx = vertices[tri, 0].mean(axis=1)
            cy = vertices[tri, 1].mean(axis=1)
            keep = (cx >= min_x) & (cx <= max_x) & (cy >= min_y) & (cy <= max_y)
        if float(min_height_m) > 0.0:
            keep &= vertices[tri, 2].max(axis=1) >= float(min_height_m)
        if bool(np.any(keep)):
            chunks.append(tri[keep])

    if not chunks:
        raise RescueUsdError(
            "No RESCUE mesh faces overlap the TEEX bounds. Check georeg alignment."
        )
    kept_faces = np.concatenate(chunks, axis=0)
    used_indices = np.unique(kept_faces.reshape(-1))
    compact_vertices = vertices[used_indices]
    compact_colors = colors[used_indices] if colors is not None else None
    remap = np.full(vertices.shape[0], -1, dtype=np.int64)
    remap[used_indices] = np.arange(used_indices.shape[0], dtype=np.int64)
    compact_faces = remap[kept_faces].astype("int32", copy=False)
    return compact_vertices, compact_faces, compact_colors


def simplify_mesh(deps: Deps, vertices: Any, faces: Any, colors: Any | None, max_faces: int):
    if faces.shape[0] <= int(max_faces):
        return vertices, faces, colors, False

    mesh = deps.o3d.geometry.TriangleMesh()
    mesh.vertices = deps.o3d.utility.Vector3dVector(vertices.astype("float64", copy=False))
    mesh.triangles = deps.o3d.utility.Vector3iVector(faces.astype("int32", copy=False))
    if colors is not None:
        mesh.vertex_colors = deps.o3d.utility.Vector3dVector(colors.astype("float64", copy=False))
    mesh.remove_duplicated_vertices()
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_non_manifold_edges()
    mesh.remove_unreferenced_vertices()
    mesh = mesh.simplify_quadric_decimation(int(max_faces))
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_unreferenced_vertices()
    out_vertices = deps.np.asarray(mesh.vertices, dtype="float32")
    out_faces = deps.np.asarray(mesh.triangles, dtype="int32")
    out_colors = None
    if mesh.has_vertex_colors():
        out_colors = deps.np.asarray(mesh.vertex_colors, dtype="float32")
    return out_vertices, out_faces, out_colors, True


def make_relative_asset_path(asset_path: Path, usd_path: Path) -> str:
    return os.path.relpath(asset_path.resolve(), usd_path.parent.resolve())


def apply_mesh_collider(deps: Deps, mesh: Any) -> None:
    prim = mesh.GetPrim()
    if not prim.HasAPI(deps.UsdPhysics.CollisionAPI):
        deps.UsdPhysics.CollisionAPI.Apply(prim)
    if hasattr(deps.UsdPhysics, "MeshCollisionAPI"):
        mesh_collision = deps.UsdPhysics.MeshCollisionAPI.Apply(prim)
        mesh_collision.CreateApproximationAttr().Set("meshSimplification")


def write_rescue_mesh_usd(
    deps: Deps,
    output_path: Path,
    vertices: Any,
    faces: Any,
    colors: Any | None,
) -> None:
    stage = deps.Usd.Stage.CreateNew(str(output_path))
    deps.UsdGeom.SetStageUpAxis(stage, deps.UsdGeom.Tokens.z)
    deps.UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    mesh = deps.UsdGeom.Mesh.Define(stage, "/RescueBuildings")
    stage.SetDefaultPrim(mesh.GetPrim())
    mesh.CreatePointsAttr(
        [deps.Gf.Vec3f(float(x), float(y), float(z)) for x, y, z in vertices]
    )
    mesh.CreateFaceVertexCountsAttr([3] * int(faces.shape[0]))
    mesh.CreateFaceVertexIndicesAttr(faces.reshape(-1).astype("int64").tolist())
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateDoubleSidedAttr(True)
    apply_mesh_collider(deps, mesh)

    if colors is not None and int(colors.shape[0]) == int(vertices.shape[0]):
        color_values = [
            deps.Gf.Vec3f(float(color[0]), float(color[1]), float(color[2]))
            for color in colors
        ]
        color_primvar = mesh.CreateDisplayColorPrimvar(
            interpolation=deps.UsdGeom.Tokens.vertex
        )
        color_primvar.Set(color_values)
    else:
        material = deps.UsdShade.Material.Define(stage, "/RescueBuildings/Looks/Overlay")
        preview = deps.UsdShade.Shader.Define(
            stage,
            "/RescueBuildings/Looks/Overlay/PreviewSurface",
        )
        preview.CreateIdAttr("UsdPreviewSurface")
        preview.CreateInput("diffuseColor", deps.Sdf.ValueTypeNames.Color3f).Set(
            deps.Gf.Vec3f(0.2, 0.55, 0.95)
        )
        preview.CreateInput("emissiveColor", deps.Sdf.ValueTypeNames.Color3f).Set(
            deps.Gf.Vec3f(0.05, 0.12, 0.22)
        )
        preview.CreateInput("roughness", deps.Sdf.ValueTypeNames.Float).Set(0.88)
        material.CreateSurfaceOutput().ConnectToSource(preview.ConnectableAPI(), "surface")
        deps.UsdShade.MaterialBindingAPI(mesh).Bind(material)
    stage.GetRootLayer().Save()


def write_root_usd(
    deps: Deps,
    output_path: Path,
    teex_usd_dir: Path,
    rescue_mesh_usd: Path,
    manifest_path: Path,
    *,
    include_lidar_obstacles: bool,
) -> None:
    terrain_path = teex_usd_dir / "teex_terrain.usd"
    obstacles_path = teex_usd_dir / "teex_obstacles.usd"
    if not terrain_path.exists():
        raise RescueUsdError(f"Missing TEEX terrain layer: {terrain_path}")

    stage = deps.Usd.Stage.CreateNew(str(output_path))
    deps.UsdGeom.SetStageUpAxis(stage, deps.UsdGeom.Tokens.z)
    deps.UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    world = deps.UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    teex = deps.UsdGeom.Xform.Define(stage, "/World/TEEX")

    terrain = stage.DefinePrim("/World/TEEX/Terrain")
    terrain.GetReferences().AddReference(make_relative_asset_path(terrain_path, output_path))

    if include_lidar_obstacles:
        if not obstacles_path.exists():
            raise RescueUsdError(f"Missing TEEX obstacle layer: {obstacles_path}")
        obstacles = stage.DefinePrim("/World/TEEX/Obstacles")
        obstacles.GetReferences().AddReference(
            make_relative_asset_path(obstacles_path, output_path)
        )

    rescue = stage.DefinePrim("/World/TEEX/RescueBuildings")
    rescue.GetReferences().AddReference(make_relative_asset_path(rescue_mesh_usd, output_path))

    metadata = deps.UsdGeom.Xform.Define(stage, "/World/TEEX/Metadata").GetPrim()
    metadata.CreateAttribute("rescue_manifest", deps.Sdf.ValueTypeNames.Asset).Set(
        deps.Sdf.AssetPath(make_relative_asset_path(manifest_path, output_path))
    )
    teex.GetPrim().SetCustomDataByKey(
        "description",
        "TEEX scene with flat satellite floor and RESCUE-DTC reconstruction mesh",
    )

    sun = deps.UsdLux.DistantLight.Define(stage, "/World/TEEX/Sun")
    sun.CreateIntensityAttr(600.0)
    sun.CreateAngleAttr(0.53)
    deps.UsdGeom.XformCommonAPI(sun).SetRotate((45.0, 0.0, 35.0))
    stage.GetRootLayer().Save()


def generate(args: argparse.Namespace) -> dict[str, Any]:
    deps = require_dependencies()
    rescue_run_dir = args.rescue_run_dir.expanduser().resolve()
    teex_usd_dir = args.teex_usd_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rescue_mesh_path = rescue_run_dir / "reconstruction_mesh.glb"
    georeg_path = rescue_run_dir / "georeg.safetensors"
    rescue_mesh_usd = output_dir / "teex_rescue_buildings.usd"
    root_usd = output_dir / "teex_scene_rescue.usd"
    manifest_path = output_dir / "teex_rescue_buildings_manifest.json"

    if args.reuse_existing_rescue_layer:
        if not rescue_mesh_usd.exists():
            raise RescueUsdError(f"Missing existing Rescue USD layer: {rescue_mesh_usd}")
        if not manifest_path.exists():
            raise RescueUsdError(f"Missing existing Rescue manifest: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        write_root_usd(
            deps,
            root_usd,
            teex_usd_dir,
            rescue_mesh_usd,
            manifest_path,
            include_lidar_obstacles=bool(args.include_lidar_obstacles),
        )
        return manifest

    teex_manifest = load_manifest(teex_usd_dir)
    origin = load_origin(teex_manifest)
    bounds = local_bounds(teex_manifest, args.clip_margin_m)

    latlon, world_points = load_rescue_georeg(deps, georeg_path)
    rescue_mesh = load_rescue_mesh(deps, rescue_mesh_path)
    input_colors = rescue_vertex_colors(deps, rescue_mesh, color_boost=args.color_boost)
    if rescue_mesh.vertices.shape[0] != latlon.shape[0]:
        raise RescueUsdError(
            "RESCUE mesh vertex count does not match georeg tensors: "
            f"mesh={rescue_mesh.vertices.shape[0]}, georeg={latlon.shape[0]}. "
            "Refusing to guess a vertex mapping."
        )

    local_vertices, rescue_floor_z = rescue_vertices_to_teex_local(
        deps,
        latlon,
        world_points,
        origin,
        floor_percentile=args.floor_percentile,
        z_scale=args.z_scale,
        z_offset_m=args.z_offset_m,
    )
    clipped_vertices, clipped_faces, clipped_colors = filter_faces_to_bounds(
        deps,
        rescue_mesh.faces.astype("int64", copy=False),
        local_vertices,
        bounds,
        mode=args.clip_face_mode,
        min_height_m=args.min_height_m,
        colors=input_colors,
    )
    final_vertices, final_faces, final_colors, simplified = simplify_mesh(
        deps,
        clipped_vertices,
        clipped_faces,
        clipped_colors,
        args.max_faces,
    )
    if final_faces.shape[0] == 0:
        raise RescueUsdError("No faces remain after simplification.")

    write_rescue_mesh_usd(deps, rescue_mesh_usd, final_vertices, final_faces, final_colors)

    manifest = {
        "source": {
            "rescue_run_dir": str(rescue_run_dir),
            "rescue_mesh": str(rescue_mesh_path),
            "georeg": str(georeg_path),
            "teex_usd_dir": str(teex_usd_dir),
        },
        "alignment": {
            "utm_epsg": origin.epsg,
            "origin_easting": origin.easting,
            "origin_northing": origin.northing,
            "teex_bounds_local_m_with_margin": list(bounds),
            "rescue_floor_z_raw": rescue_floor_z,
            "floor_percentile": float(args.floor_percentile),
            "z_scale": float(args.z_scale),
            "z_offset_m": float(args.z_offset_m),
        },
        "mesh": {
            "input_vertices": int(rescue_mesh.vertices.shape[0]),
            "input_faces": int(rescue_mesh.faces.shape[0]),
            "clipped_vertices": int(clipped_vertices.shape[0]),
            "clipped_faces": int(clipped_faces.shape[0]),
            "output_vertices": int(final_vertices.shape[0]),
            "output_faces": int(final_faces.shape[0]),
            "max_faces": int(args.max_faces),
            "simplified": bool(simplified),
            "clip_face_mode": args.clip_face_mode,
            "clip_margin_m": float(args.clip_margin_m),
            "min_height_m": float(args.min_height_m),
            "vertex_colors": final_colors is not None,
            "color_boost": float(args.color_boost),
        },
        "generated_files": {
            "rescue_buildings_usd": rescue_mesh_usd.name,
            "root_usd": root_usd.name,
            "manifest": manifest_path.name,
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    write_root_usd(
        deps,
        root_usd,
        teex_usd_dir,
        rescue_mesh_usd,
        manifest_path,
        include_lidar_obstacles=bool(args.include_lidar_obstacles),
    )
    return manifest


def main() -> int:
    try:
        manifest = generate(parse_args())
    except RescueUsdError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"error: unexpected RESCUE-to-TEEX USD conversion failure: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(manifest["generated_files"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Generate Isaac Sim USD assets from TEEX map products.

The source of truth is the generated map-product directory, not raw LAZ files.
This script creates separate terrain and obstacle USD layers, a PNG texture,
and a root USD that references those generated assets.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REQUIRED_MAP_FILES = (
    "dem_merged.tif",
    "surface_merged.tif",
    "site_rgb.tif",
    "site_origin.yaml",
)


class TeexUsdError(RuntimeError):
    """Raised for clear user-facing generation failures."""


@dataclass(frozen=True)
class Origin:
    easting: float
    northing: float
    z: float
    epsg: int
    resolution_m: float | None
    frame_id: str | None


@dataclass(frozen=True)
class Deps:
    np: Any
    rasterio: Any
    Usd: Any
    UsdGeom: Any
    UsdShade: Any
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
        import rasterio
    except ImportError:
        missing.append("rasterio")
        rasterio = None
    try:
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade
    except ImportError:
        missing.append("pxr (install usd-core or use Isaac Sim's Python)")
        Gf = Sdf = Usd = UsdGeom = UsdShade = None

    if missing:
        raise TeexUsdError(
            "Missing required dependencies: "
            + ", ".join(missing)
            + ". Run this with an environment that provides rasterio, numpy, "
            "and pxr/usd-core."
        )

    return Deps(
        np=np,
        rasterio=rasterio,
        Usd=Usd,
        UsdGeom=UsdGeom,
        UsdShade=UsdShade,
        Sdf=Sdf,
        Gf=Gf,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate TEEX Isaac Sim USD assets from map products.",
    )
    parser.add_argument(
        "--map-products-dir",
        required=True,
        type=Path,
        help="Directory containing dem_merged.tif, surface_merged.tif, site_rgb.tif, and origin metadata.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory where USD, PNG texture, and manifest files will be written.",
    )
    parser.add_argument(
        "--max-grid-size",
        type=int,
        default=512,
        help="Maximum terrain samples per raster dimension. Lower values produce smaller USD files.",
    )
    parser.add_argument(
        "--obstacle-height-threshold-m",
        type=float,
        default=1.0,
        help="Minimum surface-minus-DEM height used to create obstacle faces.",
    )
    return parser.parse_args()


def parse_simple_yaml(path: Path) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, raw_value = line.split(":", 1)
        values[key.strip()] = parse_scalar(raw_value.strip())
    return values


def parse_scalar(raw: str) -> Any:
    if not raw:
        return ""
    if raw[0] in {"'", '"'} and raw[-1:] == raw[0]:
        return raw[1:-1]
    if raw.startswith("[") and raw.endswith("]"):
        items = [item.strip() for item in raw[1:-1].split(",") if item.strip()]
        return [parse_scalar(item) for item in items]
    try:
        if any(ch in raw.lower() for ch in (".", "e")):
            return float(raw)
        return int(raw)
    except ValueError:
        return raw


def load_origin(map_products_dir: Path) -> Origin:
    origin_path = map_products_dir / "site_origin.yaml"
    if not origin_path.exists():
        raise TeexUsdError(f"Missing required origin metadata: {origin_path}")

    data = parse_simple_yaml(origin_path)
    missing = [
        key
        for key in ("easting_origin", "northing_origin", "altitude_origin", "utm_epsg")
        if key not in data
    ]
    if missing:
        raise TeexUsdError(
            f"{origin_path} is missing required fields: {', '.join(missing)}"
        )

    return Origin(
        easting=float(data["easting_origin"]),
        northing=float(data["northing_origin"]),
        z=float(data["altitude_origin"]),
        epsg=int(data["utm_epsg"]),
        resolution_m=float(data["map_resolution"]) if "map_resolution" in data else None,
        frame_id=str(data["frame_id"]) if "frame_id" in data else None,
    )


def load_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def validate_inputs(map_products_dir: Path) -> None:
    if not map_products_dir.exists():
        raise TeexUsdError(f"Map products directory does not exist: {map_products_dir}")
    missing = [name for name in REQUIRED_MAP_FILES if not (map_products_dir / name).exists()]
    if missing:
        raise TeexUsdError(
            "Map products directory is missing required files: " + ", ".join(missing)
        )


def crs_epsg(dataset: Any, label: str) -> int:
    if dataset.crs is None:
        raise TeexUsdError(f"{label} has no CRS metadata; refusing to assume EPSG/CRS.")
    epsg = dataset.crs.to_epsg()
    if epsg is None:
        raise TeexUsdError(
            f"{label} CRS is present but cannot be resolved to an EPSG code: {dataset.crs}"
        )
    return int(epsg)


def read_band_filled(deps: Deps, dataset: Any, label: str, fill_value: float | None = None):
    arr = dataset.read(1, masked=True).astype("float64")
    if arr.mask is deps.np.ma.nomask or not deps.np.any(arr.mask):
        return deps.np.asarray(arr), None

    valid = deps.np.asarray(arr.compressed())
    if valid.size == 0:
        raise TeexUsdError(f"{label} contains no valid raster cells.")
    fill = float(fill_value) if fill_value is not None else float(valid.min())
    warning = (
        f"{label} contains nodata cells; filled them with {fill:.3f} m "
        "for USD mesh continuity."
    )
    return deps.np.asarray(arr.filled(fill)), warning


def sample_indices(length: int, max_size: int, deps: Deps):
    if length <= 1:
        return deps.np.array([0], dtype=int)
    target = max(2, min(int(max_size), length))
    return deps.np.unique(deps.np.linspace(0, length - 1, target).round().astype(int))


def raster_xy_arrays(deps: Deps, transform: Any, rows: Any, cols: Any):
    col_grid, row_grid = deps.np.meshgrid(cols.astype("float64"), rows.astype("float64"))
    x = (
        transform.c
        + (col_grid + 0.5) * transform.a
        + (row_grid + 0.5) * transform.b
    )
    y = (
        transform.f
        + (col_grid + 0.5) * transform.d
        + (row_grid + 0.5) * transform.e
    )
    return x, y


def build_grid_mesh(
    deps: Deps,
    elevation: Any,
    transform: Any,
    origin: Origin,
    max_grid_size: int,
):
    np = deps.np
    rows = sample_indices(elevation.shape[0], max_grid_size, deps)
    cols = sample_indices(elevation.shape[1], max_grid_size, deps)
    sampled = elevation[np.ix_(rows, cols)]
    easting, northing = raster_xy_arrays(deps, transform, rows, cols)

    x = easting - origin.easting
    y = northing - origin.northing
    z = sampled - origin.z

    points = [
        deps.Gf.Vec3f(float(x[r, c]), float(y[r, c]), float(z[r, c]))
        for r in range(z.shape[0])
        for c in range(z.shape[1])
    ]

    face_counts: list[int] = []
    face_indices: list[int] = []
    st_values: list[Any] = []
    rows_count, cols_count = z.shape
    for r in range(rows_count - 1):
        for c in range(cols_count - 1):
            i00 = r * cols_count + c
            i01 = r * cols_count + c + 1
            i11 = (r + 1) * cols_count + c + 1
            i10 = (r + 1) * cols_count + c
            face_counts.append(4)
            face_indices.extend([i00, i01, i11, i10])
            u0 = c / max(cols_count - 1, 1)
            u1 = (c + 1) / max(cols_count - 1, 1)
            v0 = 1.0 - r / max(rows_count - 1, 1)
            v1 = 1.0 - (r + 1) / max(rows_count - 1, 1)
            st_values.extend(
                [
                    deps.Gf.Vec2f(float(u0), float(v0)),
                    deps.Gf.Vec2f(float(u1), float(v0)),
                    deps.Gf.Vec2f(float(u1), float(v1)),
                    deps.Gf.Vec2f(float(u0), float(v1)),
                ]
            )

    return points, face_counts, face_indices, st_values, rows, cols


def build_obstacle_mesh(
    deps: Deps,
    dem: Any,
    surface: Any,
    transform: Any,
    origin: Origin,
    max_grid_size: int,
    threshold_m: float,
):
    np = deps.np
    rows = sample_indices(dem.shape[0], max_grid_size, deps)
    cols = sample_indices(dem.shape[1], max_grid_size, deps)
    surface_sampled = surface[np.ix_(rows, cols)]
    delta_sampled = surface_sampled - dem[np.ix_(rows, cols)]
    easting, northing = raster_xy_arrays(deps, transform, rows, cols)

    x = easting - origin.easting
    y = northing - origin.northing
    z = surface_sampled - origin.z

    points = [
        deps.Gf.Vec3f(float(x[r, c]), float(y[r, c]), float(z[r, c]))
        for r in range(z.shape[0])
        for c in range(z.shape[1])
    ]

    face_counts: list[int] = []
    face_indices: list[int] = []
    rows_count, cols_count = z.shape
    for r in range(rows_count - 1):
        for c in range(cols_count - 1):
            cell_delta = delta_sampled[r : r + 2, c : c + 2]
            if float(np.nanmax(cell_delta)) < threshold_m:
                continue
            i00 = r * cols_count + c
            i01 = r * cols_count + c + 1
            i11 = (r + 1) * cols_count + c + 1
            i10 = (r + 1) * cols_count + c
            face_counts.append(4)
            face_indices.extend([i00, i01, i11, i10])

    return points, face_counts, face_indices


def write_texture_png(deps: Deps, rgb_path: Path, output_path: Path) -> None:
    np = deps.np
    with deps.rasterio.open(rgb_path) as src:
        if src.count < 3:
            raise TeexUsdError(f"{rgb_path} must have at least 3 bands for RGB texture.")
        rgb = src.read([1, 2, 3], masked=True)

    if rgb.dtype != np.uint8:
        data = rgb.astype("float64")
        valid = data.compressed() if hasattr(data, "compressed") else data.reshape(-1)
        if valid.size == 0:
            raise TeexUsdError(f"{rgb_path} contains no valid RGB pixels.")
        min_val = float(valid.min())
        max_val = float(valid.max())
        if math.isclose(min_val, max_val):
            scaled = np.zeros(data.shape, dtype="uint8")
        else:
            scaled = np.clip((data.filled(min_val) - min_val) / (max_val - min_val), 0, 1)
            scaled = (scaled * 255).astype("uint8")
    else:
        scaled = np.asarray(rgb.filled(0) if hasattr(rgb, "filled") else rgb)

    with deps.rasterio.open(
        output_path,
        "w",
        driver="PNG",
        width=scaled.shape[2],
        height=scaled.shape[1],
        count=3,
        dtype="uint8",
    ) as dst:
        dst.write(scaled)


def make_relative_asset_path(asset_path: Path, usd_path: Path) -> str:
    return asset_path.resolve().relative_to(usd_path.parent.resolve()).as_posix()


def write_terrain_usd(
    deps: Deps,
    output_path: Path,
    texture_path: Path,
    points: list[Any],
    face_counts: list[int],
    face_indices: list[int],
    st_values: list[Any],
) -> None:
    stage = deps.Usd.Stage.CreateNew(str(output_path))
    deps.UsdGeom.SetStageUpAxis(stage, deps.UsdGeom.Tokens.z)
    deps.UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    mesh = deps.UsdGeom.Mesh.Define(stage, "/Terrain")
    stage.SetDefaultPrim(mesh.GetPrim())
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr(face_counts)
    mesh.CreateFaceVertexIndicesAttr(face_indices)
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateDoubleSidedAttr(True)

    st = deps.UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
        "st",
        deps.Sdf.ValueTypeNames.TexCoord2fArray,
        deps.UsdGeom.Tokens.faceVarying,
    )
    st.Set(st_values)

    material = deps.UsdShade.Material.Define(stage, "/Materials/SiteRgb")
    preview = deps.UsdShade.Shader.Define(stage, "/Materials/SiteRgb/PreviewSurface")
    preview.CreateIdAttr("UsdPreviewSurface")
    preview.CreateInput("roughness", deps.Sdf.ValueTypeNames.Float).Set(0.85)

    texture = deps.UsdShade.Shader.Define(stage, "/Materials/SiteRgb/DiffuseTexture")
    texture.CreateIdAttr("UsdUVTexture")
    texture.CreateInput("file", deps.Sdf.ValueTypeNames.Asset).Set(
        deps.Sdf.AssetPath(make_relative_asset_path(texture_path, output_path))
    )

    st_reader = deps.UsdShade.Shader.Define(stage, "/Materials/SiteRgb/StReader")
    st_reader.CreateIdAttr("UsdPrimvarReader_float2")
    st_reader.CreateInput("varname", deps.Sdf.ValueTypeNames.Token).Set("st")
    texture.CreateInput("st", deps.Sdf.ValueTypeNames.Float2).ConnectToSource(
        st_reader.ConnectableAPI(),
        "result",
    )
    preview.CreateInput("diffuseColor", deps.Sdf.ValueTypeNames.Color3f).ConnectToSource(
        texture.ConnectableAPI(),
        "rgb",
    )
    material.CreateSurfaceOutput().ConnectToSource(preview.ConnectableAPI(), "surface")
    deps.UsdShade.MaterialBindingAPI(mesh).Bind(material)

    stage.GetRootLayer().Save()


def write_obstacles_usd(
    deps: Deps,
    output_path: Path,
    points: list[Any],
    face_counts: list[int],
    face_indices: list[int],
) -> None:
    stage = deps.Usd.Stage.CreateNew(str(output_path))
    deps.UsdGeom.SetStageUpAxis(stage, deps.UsdGeom.Tokens.z)
    deps.UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    mesh = deps.UsdGeom.Mesh.Define(stage, "/Obstacles")
    stage.SetDefaultPrim(mesh.GetPrim())
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr(face_counts)
    mesh.CreateFaceVertexIndicesAttr(face_indices)
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateDoubleSidedAttr(True)

    material = deps.UsdShade.Material.Define(stage, "/Materials/Obstacle")
    preview = deps.UsdShade.Shader.Define(stage, "/Materials/Obstacle/PreviewSurface")
    preview.CreateIdAttr("UsdPreviewSurface")
    preview.CreateInput("diffuseColor", deps.Sdf.ValueTypeNames.Color3f).Set(
        deps.Gf.Vec3f(0.6, 0.55, 0.48)
    )
    preview.CreateInput("roughness", deps.Sdf.ValueTypeNames.Float).Set(0.9)
    material.CreateSurfaceOutput().ConnectToSource(preview.ConnectableAPI(), "surface")
    deps.UsdShade.MaterialBindingAPI(mesh).Bind(material)

    stage.GetRootLayer().Save()


def write_root_usd(
    deps: Deps,
    output_path: Path,
    terrain_path: Path,
    obstacles_path: Path,
    texture_path: Path,
    manifest_path: Path,
    origin: Origin,
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
    metadata.CreateAttribute("origin_easting", deps.Sdf.ValueTypeNames.Double).Set(origin.easting)
    metadata.CreateAttribute("origin_northing", deps.Sdf.ValueTypeNames.Double).Set(origin.northing)
    metadata.CreateAttribute("z_origin", deps.Sdf.ValueTypeNames.Double).Set(origin.z)
    metadata.CreateAttribute("utm_epsg", deps.Sdf.ValueTypeNames.Int).Set(origin.epsg)
    metadata.CreateAttribute("texture_file", deps.Sdf.ValueTypeNames.Asset).Set(
        deps.Sdf.AssetPath(make_relative_asset_path(texture_path, output_path))
    )
    metadata.CreateAttribute("manifest_file", deps.Sdf.ValueTypeNames.Asset).Set(
        deps.Sdf.AssetPath(make_relative_asset_path(manifest_path, output_path))
    )
    teex.GetPrim().SetCustomDataByKey("description", "TEEX generated map-products scene")

    stage.GetRootLayer().Save()


def bounds_list(bounds: Any) -> list[float]:
    return [float(bounds.left), float(bounds.bottom), float(bounds.right), float(bounds.top)]


def generate(args: argparse.Namespace) -> dict[str, Any]:
    deps = require_dependencies()
    map_products_dir = args.map_products_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    validate_inputs(map_products_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    origin = load_origin(map_products_dir)
    heightmap_meta = load_optional_json(map_products_dir / "heightmap_meta.json")
    quality_report = load_optional_json(map_products_dir / "map_quality_report.json")
    warnings: list[str] = []

    dem_path = map_products_dir / "dem_merged.tif"
    surface_path = map_products_dir / "surface_merged.tif"
    rgb_path = map_products_dir / "site_rgb.tif"
    terrain_usd = output_dir / "teex_terrain.usd"
    obstacles_usd = output_dir / "teex_obstacles.usd"
    root_usd = output_dir / "teex_scene.usd"
    texture_png = output_dir / "site_rgb.png"
    manifest_path = output_dir / "teex_scene_manifest.json"

    with deps.rasterio.open(dem_path) as dem_src, deps.rasterio.open(surface_path) as surface_src, deps.rasterio.open(rgb_path) as rgb_src:
        dem_epsg = crs_epsg(dem_src, "dem_merged.tif")
        surface_epsg = crs_epsg(surface_src, "surface_merged.tif")
        rgb_epsg = crs_epsg(rgb_src, "site_rgb.tif")
        if dem_epsg != origin.epsg:
            raise TeexUsdError(
                f"DEM CRS EPSG:{dem_epsg} does not match site_origin utm_epsg:{origin.epsg}."
            )
        if surface_epsg != dem_epsg:
            raise TeexUsdError(
                f"surface_merged.tif CRS EPSG:{surface_epsg} does not match DEM EPSG:{dem_epsg}."
            )
        if rgb_epsg != dem_epsg:
            raise TeexUsdError(
                f"site_rgb.tif CRS EPSG:{rgb_epsg} does not match DEM EPSG:{dem_epsg}."
            )
        if surface_src.shape != dem_src.shape or surface_src.transform != dem_src.transform:
            raise TeexUsdError("surface_merged.tif must be aligned to dem_merged.tif.")
        if rgb_src.bounds != dem_src.bounds:
            warnings.append("site_rgb.tif bounds differ from DEM bounds; UV texture is applied over full terrain extent.")

        dem, dem_warning = read_band_filled(deps, dem_src, "dem_merged.tif")
        if dem_warning:
            warnings.append(dem_warning)
        surface, surface_warning = read_band_filled(
            deps,
            surface_src,
            "surface_merged.tif",
            fill_value=float(deps.np.nanmin(dem)),
        )
        if surface_warning:
            warnings.append(surface_warning)

        terrain_points, terrain_counts, terrain_indices, st_values, sampled_rows, sampled_cols = build_grid_mesh(
            deps,
            dem,
            dem_src.transform,
            origin,
            args.max_grid_size,
        )
        obstacle_points, obstacle_counts, obstacle_indices = build_obstacle_mesh(
            deps,
            dem,
            surface,
            dem_src.transform,
            origin,
            args.max_grid_size,
            args.obstacle_height_threshold_m,
        )

        manifest = {
            "source": {
                "map_products_dir": str(map_products_dir),
                "dem": "dem_merged.tif",
                "surface": "surface_merged.tif",
                "rgb": "site_rgb.tif",
            },
            "crs": {
                "epsg": dem_epsg,
                "wkt": dem_src.crs.to_wkt(),
            },
            "origin": {
                "easting": origin.easting,
                "northing": origin.northing,
                "z_origin": origin.z,
                "frame_id": origin.frame_id,
            },
            "bounds_projected_m": bounds_list(dem_src.bounds),
            "bounds_local_m": [
                float(dem_src.bounds.left - origin.easting),
                float(dem_src.bounds.bottom - origin.northing),
                float(dem_src.bounds.right - origin.easting),
                float(dem_src.bounds.top - origin.northing),
            ],
            "resolution_m": [abs(float(dem_src.transform.a)), abs(float(dem_src.transform.e))],
            "raster_shape": {
                "width_px": int(dem_src.width),
                "height_px": int(dem_src.height),
            },
            "heightmap_meta": heightmap_meta,
            "map_quality_report_summary": {
                "warnings": quality_report.get("warnings", []) if quality_report else [],
                "dem": quality_report.get("dem", {}) if quality_report else {},
                "imagery": quality_report.get("imagery", {}) if quality_report else {},
                "occupancy": quality_report.get("occupancy", {}) if quality_report else {},
            },
            "mesh_sampling": {
                "max_grid_size": int(args.max_grid_size),
                "sampled_rows": int(len(sampled_rows)),
                "sampled_cols": int(len(sampled_cols)),
                "terrain_vertices": int(len(terrain_points)),
                "terrain_faces": int(len(terrain_counts)),
                "obstacle_vertices": int(len(obstacle_points)),
                "obstacle_faces": int(len(obstacle_counts)),
                "obstacle_height_threshold_m": float(args.obstacle_height_threshold_m),
            },
            "generated_files": {
                "root_usd": root_usd.name,
                "terrain_usd": terrain_usd.name,
                "obstacles_usd": obstacles_usd.name,
                "texture_png": texture_png.name,
                "manifest": manifest_path.name,
            },
            "warnings": warnings,
        }

    write_texture_png(deps, rgb_path, texture_png)
    write_terrain_usd(
        deps,
        terrain_usd,
        texture_png,
        terrain_points,
        terrain_counts,
        terrain_indices,
        st_values,
    )
    write_obstacles_usd(deps, obstacles_usd, obstacle_points, obstacle_counts, obstacle_indices)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    write_root_usd(
        deps,
        root_usd,
        terrain_usd,
        obstacles_usd,
        texture_png,
        manifest_path,
        origin,
    )
    return manifest


def main() -> int:
    try:
        manifest = generate(parse_args())
    except TeexUsdError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"error: unexpected TEEX USD generation failure: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(manifest["generated_files"], indent=2))
    if manifest["warnings"]:
        print("warnings:", file=sys.stderr)
        for warning in manifest["warnings"]:
            print(f"- {warning}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Convert a GLB mesh to a simple Isaac/USD mesh asset.

This is intentionally independent of TEEX map products. It preserves GLB
vertex colors when present, can decimate large meshes with Open3D, and can add
USD collision metadata for quick Isaac Sim inspection.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ConvertGlbError(RuntimeError):
    """Raised for clear user-facing conversion failures."""


@dataclass(frozen=True)
class Deps:
    np: Any
    trimesh: Any
    o3d: Any
    Usd: Any
    UsdGeom: Any
    UsdPhysics: Any
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
        import trimesh
    except ImportError:
        missing.append("trimesh")
        trimesh = None
    try:
        import open3d as o3d
    except ImportError:
        o3d = None
    try:
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade
    except ImportError:
        missing.append("pxr (install usd-core or use Isaac Sim's Python)")
        Gf = Sdf = Usd = UsdGeom = UsdPhysics = UsdShade = None

    if missing:
        raise ConvertGlbError(
            "Missing required dependencies: "
            + ", ".join(missing)
            + ". Run this from an environment with trimesh and USD Python bindings."
        )

    return Deps(
        np=np,
        trimesh=trimesh,
        o3d=o3d,
        Usd=Usd,
        UsdGeom=UsdGeom,
        UsdPhysics=UsdPhysics,
        UsdShade=UsdShade,
        Sdf=Sdf,
        Gf=Gf,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert a GLB mesh to USD.")
    parser.add_argument("--input-glb", required=True, type=Path, help="Source .glb file.")
    parser.add_argument("--output-usd", required=True, type=Path, help="Output .usd file.")
    parser.add_argument(
        "--prim-path",
        default="/GLBMesh",
        help="USD prim path for the generated mesh.",
    )
    parser.add_argument(
        "--max-faces",
        type=int,
        default=500_000,
        help="Decimate to this many faces. Use 0 to keep all faces.",
    )
    parser.add_argument("--scale", type=float, default=1.0, help="Uniform XYZ scale.")
    parser.add_argument(
        "--translate",
        nargs=3,
        type=float,
        default=(0.0, 0.0, 0.0),
        metavar=("X", "Y", "Z"),
        help="XYZ translation applied after scaling.",
    )
    parser.add_argument(
        "--rotate-z-deg",
        type=float,
        default=0.0,
        help="Optional yaw rotation about Z, in degrees, applied before translation.",
    )
    parser.add_argument(
        "--add-collider",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply UsdPhysics collision metadata to the generated mesh.",
    )
    return parser.parse_args()


def load_mesh(deps: Deps, input_glb: Path):
    if not input_glb.exists():
        raise ConvertGlbError(f"Input GLB does not exist: {input_glb}")
    mesh = deps.trimesh.load(input_glb, force="mesh", process=False)
    if not hasattr(mesh, "vertices") or not hasattr(mesh, "faces"):
        raise ConvertGlbError(f"Could not load {input_glb} as a triangle mesh.")
    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise ConvertGlbError(f"{input_glb} contains no mesh geometry.")
    return mesh


def vertex_colors(deps: Deps, mesh: Any):
    np = deps.np
    if not hasattr(mesh, "visual") or not hasattr(mesh.visual, "vertex_colors"):
        return None
    colors = np.asarray(mesh.visual.vertex_colors)
    if colors.ndim != 2 or colors.shape[0] != mesh.vertices.shape[0] or colors.shape[1] < 3:
        return None
    return (colors[:, :3].astype("float32") / 255.0).astype("float32", copy=False)


def transform_vertices(deps: Deps, vertices: Any, scale: float, translate: tuple[float, float, float], rotate_z_deg: float):
    np = deps.np
    out = np.asarray(vertices, dtype="float32").copy()
    out *= float(scale)
    yaw = np.deg2rad(float(rotate_z_deg))
    if abs(float(yaw)) > 1e-12:
        c = float(np.cos(yaw))
        s = float(np.sin(yaw))
        x = out[:, 0].copy()
        y = out[:, 1].copy()
        out[:, 0] = c * x - s * y
        out[:, 1] = s * x + c * y
    out[:, 0] += float(translate[0])
    out[:, 1] += float(translate[1])
    out[:, 2] += float(translate[2])
    return out


def simplify_mesh(deps: Deps, vertices: Any, faces: Any, colors: Any | None, max_faces: int):
    if int(max_faces) <= 0 or faces.shape[0] <= int(max_faces):
        return vertices, faces, colors, False
    if deps.o3d is None:
        raise ConvertGlbError(
            f"Mesh has {faces.shape[0]} faces and --max-faces={max_faces}, "
            "but open3d is not installed for decimation. Install open3d or pass --max-faces 0."
        )

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


def apply_collider(deps: Deps, mesh: Any) -> None:
    prim = mesh.GetPrim()
    if not prim.HasAPI(deps.UsdPhysics.CollisionAPI):
        deps.UsdPhysics.CollisionAPI.Apply(prim)
    if hasattr(deps.UsdPhysics, "MeshCollisionAPI"):
        mesh_collision = deps.UsdPhysics.MeshCollisionAPI.Apply(prim)
        mesh_collision.CreateApproximationAttr().Set("meshSimplification")


def write_usd(
    deps: Deps,
    output_usd: Path,
    prim_path: str,
    vertices: Any,
    faces: Any,
    colors: Any | None,
    *,
    add_collider: bool,
) -> None:
    output_usd.parent.mkdir(parents=True, exist_ok=True)
    stage = deps.Usd.Stage.CreateNew(str(output_usd))
    deps.UsdGeom.SetStageUpAxis(stage, deps.UsdGeom.Tokens.z)
    deps.UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    mesh = deps.UsdGeom.Mesh.Define(stage, prim_path)
    stage.SetDefaultPrim(mesh.GetPrim())
    mesh.CreatePointsAttr(
        [deps.Gf.Vec3f(float(x), float(y), float(z)) for x, y, z in vertices]
    )
    mesh.CreateFaceVertexCountsAttr([3] * int(faces.shape[0]))
    mesh.CreateFaceVertexIndicesAttr(faces.reshape(-1).astype("int64").tolist())
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateDoubleSidedAttr(True)

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
        material = deps.UsdShade.Material.Define(stage, f"{prim_path}/Looks/Fallback")
        preview = deps.UsdShade.Shader.Define(stage, f"{prim_path}/Looks/Fallback/PreviewSurface")
        preview.CreateIdAttr("UsdPreviewSurface")
        preview.CreateInput("diffuseColor", deps.Sdf.ValueTypeNames.Color3f).Set(
            deps.Gf.Vec3f(0.35, 0.55, 0.85)
        )
        preview.CreateInput("roughness", deps.Sdf.ValueTypeNames.Float).Set(0.85)
        material.CreateSurfaceOutput().ConnectToSource(preview.ConnectableAPI(), "surface")
        deps.UsdShade.MaterialBindingAPI(mesh).Bind(material)

    if add_collider:
        apply_collider(deps, mesh)

    stage.GetRootLayer().Save()


def main() -> int:
    try:
        deps = require_dependencies()
        args = parse_args()
        mesh = load_mesh(deps, args.input_glb.expanduser().resolve())
        vertices = transform_vertices(
            deps,
            mesh.vertices,
            args.scale,
            tuple(args.translate),
            args.rotate_z_deg,
        )
        faces = deps.np.asarray(mesh.faces, dtype="int32")
        colors = vertex_colors(deps, mesh)
        vertices, faces, colors, simplified = simplify_mesh(
            deps,
            vertices,
            faces,
            colors,
            args.max_faces,
        )
        write_usd(
            deps,
            args.output_usd.expanduser().resolve(),
            args.prim_path,
            vertices,
            faces,
            colors,
            add_collider=bool(args.add_collider),
        )
    except ConvertGlbError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"error: unexpected GLB-to-USD conversion failure: {exc}", file=sys.stderr)
        return 1

    print(
        f"wrote {args.output_usd} "
        f"vertices={int(vertices.shape[0])} faces={int(faces.shape[0])} "
        f"vertex_colors={colors is not None} simplified={simplified}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

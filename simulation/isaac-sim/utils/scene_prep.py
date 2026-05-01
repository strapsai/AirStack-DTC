"""
scene_prep.py — Utilities for preparing Isaac Sim / USD stages before simulation.

Functions:
    get_stage_meters_per_unit   — Read stage unit scale
    scale_stage_prim            — Apply a uniform scale transform to a prim
    add_colliders               — Recursively apply CollisionAPI to all meshes
    add_collision_box_floor     — Add an invisible static collision floor
    add_dome_light              — Add or update a dome light on the stage
    save_scene_as_contained_usd — Collect all assets into a self-contained directory
"""

import asyncio
import time as _time
import omni.kit.app
import omni.usd
from pxr import Gf, UsdGeom, UsdPhysics, UsdLux, Sdf

try:
    from pxr import PhysxSchema
except ImportError:
    PhysxSchema = None


# ---------------------------------------------------------------------------
# Stage units
# ---------------------------------------------------------------------------

def get_stage_meters_per_unit(stage) -> tuple:
    """Return (meters_per_unit, scene_scale_factor).

    scene_scale_factor is 1/mpu — multiply metric coordinates by this to get
    stage-space coordinates.
    """
    mpu = UsdGeom.GetStageMetersPerUnit(stage)
    if mpu is None or mpu <= 0:
        mpu = 1.0
    return mpu, 1.0 / mpu


# ---------------------------------------------------------------------------
# Scaling
# ---------------------------------------------------------------------------

def scale_stage_prim(stage, prim_path: str, scale_factor: float):
    """Apply a uniform XYZ scale to *prim_path*, clearing any prior xform ops.

    Args:
        stage:        Active USD stage.
        prim_path:    Stage path of the prim to scale (e.g. "/World/stage").
        scale_factor: Uniform scale to apply (e.g. 0.01 to convert cm → m).

    Returns:
        The scaled UsdPrim.

    Raises:
        ValueError: If the prim is not found at *prim_path*.
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"Prim not found at path: {prim_path}")

    xformable = UsdGeom.Xformable(prim)
    xformable.ClearXformOpOrder()

    translate_op = xformable.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
    translate_op.Set(Gf.Vec3d(0.0, 0.0, 0.0))

    scale_op = xformable.AddScaleOp(UsdGeom.XformOp.PrecisionDouble)
    scale_op.Set(Gf.Vec3d(scale_factor, scale_factor, scale_factor))

    print(f"[scene_prep] Scaled '{prim_path}' by {scale_factor}")
    return prim


# ---------------------------------------------------------------------------
# Collision
# ---------------------------------------------------------------------------

def add_colliders(prim):
    """Recursively apply UsdPhysics.CollisionAPI to every UsdGeom.Mesh under *prim*.

    Skips prims that already have the API applied.
    """
    if prim.IsA(UsdGeom.Mesh):
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI.Apply(prim)
            print(f"[scene_prep] Added collider: {prim.GetPath()}")
        if hasattr(UsdPhysics, "MeshCollisionAPI"):
            mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(prim)
            mesh_collision.CreateApproximationAttr().Set("meshSimplification")

    for child in prim.GetChildren():
        add_colliders(child)


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
    """Create an invisible static box collider whose top face is at *top_z*.

    This gives PhysX a simple floor even when a large textured terrain mesh is
    too complex or too thin to behave reliably as a collision surface.
    """
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

    # Render visibility is separate from physics collision; keep the satellite
    # imagery unobstructed while preserving the collider.
    UsdGeom.Imageable(prim).MakeInvisible()
    print(
        "[scene_prep] Added invisible collision floor: "
        f"{prim_path} center=({float(center_xy[0]):.2f}, {float(center_xy[1]):.2f}, {center_z:.2f}) "
        f"size=({width:.2f}, {depth:.2f}, {height:.2f}) top_z={float(top_z):.2f}"
    )
    return prim


# ---------------------------------------------------------------------------
# Lighting
# ---------------------------------------------------------------------------

def add_dome_light(stage, prim_path: str = "/World/DomeLight", intensity: float = 3500.0, exposure: float = -3.0):
    """Add a dome light to the stage, or update it if it already exists.

    Args:
        stage:     Active USD stage.
        prim_path: Stage path for the dome light prim.
        intensity: Light intensity value.
        exposure:  Light exposure value.
    """
    if stage.GetPrimAtPath(prim_path).IsValid():
        dome = UsdLux.DomeLight.Get(stage, prim_path)
    else:
        dome = UsdLux.DomeLight.Define(stage, Sdf.Path(prim_path))
    dome.CreateIntensityAttr(intensity)
    dome.CreateExposureAttr(exposure)
    print(f"[scene_prep] Dome light set at '{prim_path}' (intensity={intensity}, exposure={exposure})")


# ---------------------------------------------------------------------------
# Save as self-contained USD collection
# ---------------------------------------------------------------------------

def save_scene_as_contained_usd(source_usd_url: str, output_dir: str) -> bool:
    """Collect a USD stage and all its dependencies into a self-contained directory.

    Uses omni.kit.usd.collect.Collector (Isaac Sim 5.1 / Kit 107) which correctly
    handles Nucleus omniverse:// URLs, MDL materials, and texture files.

    The collected root USD will be written to output_dir, with all referenced
    assets copied alongside it using relative paths.

    Args:
        source_usd_url: Path or omniverse:// URL of the source USD to collect.
        output_dir:     Local directory to write the collected package into.

    Returns:
        True on success, False on failure.
    """
    # Enable the collect extension if not already active
    ext_manager = omni.kit.app.get_app().get_extension_manager()
    if not ext_manager.is_extension_enabled("omni.kit.usd.collect"):
        ext_manager.set_extension_enabled_immediate("omni.kit.usd.collect", True)

    from omni.kit.usd.collect import Collector

    collector = Collector(
        usd_path=source_usd_url,
        collect_dir=output_dir,
        usd_only=False,        # include textures, MDLs, etc.
        flat_collection=False, # preserve source folder hierarchy
        skip_existing=False,
    )

    done = [False]
    result = [False]

    def on_progress(current: int, total: int):
        print(f"[scene_prep] Collecting assets: {current}/{total}", flush=True)

    def on_finish():
        result[0] = True
        done[0] = True
        print("[scene_prep] Collection complete.", flush=True)

    # Schedule the collect coroutine on Kit's event loop so the app loop keeps ticking
    asyncio.ensure_future(
        collector.collect(progress_callback=on_progress, finish_callback=on_finish)
    )

    # Pump the Kit app loop until collection finishes
    app = omni.kit.app.get_app()
    while not done[0]:
        app.update()

    collector.destroy()
    return result[0]
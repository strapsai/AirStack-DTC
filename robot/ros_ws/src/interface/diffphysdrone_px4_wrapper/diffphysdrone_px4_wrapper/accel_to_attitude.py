from dataclasses import dataclass
import math
from typing import Iterable, Sequence, Tuple


Vector3 = Tuple[float, float, float]
Matrix3 = Tuple[Vector3, Vector3, Vector3]
QuaternionWXYZ = Tuple[float, float, float, float]


class ConversionError(ValueError):
    """Raised when an action cannot be converted into a finite command."""


@dataclass(frozen=True)
class AttitudeThrustCommand:
    quaternion_wxyz: QuaternionWXYZ
    thrust: float
    specific_force_enu: Vector3


def _as_vector3(values: Iterable[float], name: str) -> Vector3:
    data = tuple(float(v) for v in values)
    if len(data) != 3:
        raise ConversionError(f"{name} must have exactly 3 values")
    if not all(math.isfinite(v) for v in data):
        raise ConversionError(f"{name} contains NaN or Inf")
    return data  # type: ignore[return-value]


def _dot(a: Vector3, b: Vector3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a: Vector3, b: Vector3) -> Vector3:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _norm(a: Vector3) -> float:
    return math.sqrt(_dot(a, a))


def _normalize(a: Vector3, name: str) -> Vector3:
    n = _norm(a)
    if n < 1e-9:
        raise ConversionError(f"{name} is too small to normalize")
    return (a[0] / n, a[1] / n, a[2] / n)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _clamp_vector_norm(v: Vector3, max_norm: float) -> Vector3:
    if max_norm <= 0.0:
        return v
    n = _norm(v)
    if n <= max_norm or n < 1e-9:
        return v
    scale = max_norm / n
    return (v[0] * scale, v[1] * scale, v[2] * scale)


def _mat_vec(m: Matrix3, v: Vector3) -> Vector3:
    return (
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    )


def diffphys_yaw_frame_from_rotation(rotation_enu_flu: Matrix3) -> Matrix3:
    """Return the yaw-stabilized frame used by DiffPhysDrone training."""

    fwd = (rotation_enu_flu[0][0], rotation_enu_flu[1][0], 0.0)
    if _norm(fwd) < 1e-6:
        yaw = math.atan2(rotation_enu_flu[1][0], rotation_enu_flu[0][0])
        fwd = (math.cos(yaw), math.sin(yaw), 0.0)
    fwd = _normalize(fwd, "diffphys_fwd")
    up = (0.0, 0.0, 1.0)
    left = _normalize(_cross(up, fwd), "diffphys_left")
    return (
        (fwd[0], left[0], up[0]),
        (fwd[1], left[1], up[1]),
        (fwd[2], left[2], up[2]),
    )


def quaternion_wxyz_to_matrix(q: Sequence[float]) -> Matrix3:
    if len(q) != 4:
        raise ConversionError("quaternion must have exactly 4 values")
    w, x, y, z = (float(q[0]), float(q[1]), float(q[2]), float(q[3]))
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n < 1e-9:
        raise ConversionError("quaternion is too small to normalize")
    w, x, y, z = w / n, x / n, y / n, z / n
    return (
        (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)),
        (2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)),
        (2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)),
    )


def yaw_from_quaternion_wxyz(q: Sequence[float]) -> float:
    if len(q) != 4:
        raise ConversionError("quaternion must have exactly 4 values")
    w, x, y, z = (float(q[0]), float(q[1]), float(q[2]), float(q[3]))
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _matrix_to_quaternion_wxyz(r: Matrix3) -> QuaternionWXYZ:
    trace = r[0][0] + r[1][1] + r[2][2]
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (r[2][1] - r[1][2]) / s
        y = (r[0][2] - r[2][0]) / s
        z = (r[1][0] - r[0][1]) / s
    elif r[0][0] > r[1][1] and r[0][0] > r[2][2]:
        s = math.sqrt(1.0 + r[0][0] - r[1][1] - r[2][2]) * 2.0
        w = (r[2][1] - r[1][2]) / s
        x = 0.25 * s
        y = (r[0][1] + r[1][0]) / s
        z = (r[0][2] + r[2][0]) / s
    elif r[1][1] > r[2][2]:
        s = math.sqrt(1.0 + r[1][1] - r[0][0] - r[2][2]) * 2.0
        w = (r[0][2] - r[2][0]) / s
        x = (r[0][1] + r[1][0]) / s
        y = 0.25 * s
        z = (r[1][2] + r[2][1]) / s
    else:
        s = math.sqrt(1.0 + r[2][2] - r[0][0] - r[1][1]) * 2.0
        w = (r[1][0] - r[0][1]) / s
        x = (r[0][2] + r[2][0]) / s
        y = (r[1][2] + r[2][1]) / s
        z = 0.25 * s

    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n < 1e-9:
        raise ConversionError("computed quaternion is too small to normalize")
    return (w / n, x / n, y / n, z / n)


def _clamp_specific_force_tilt(specific_force: Vector3, max_tilt_rad: float) -> Vector3:
    if max_tilt_rad <= 0.0:
        return (0.0, 0.0, max(1e-6, specific_force[2]))
    if max_tilt_rad >= math.pi * 0.5:
        return specific_force

    fx, fy, fz = specific_force
    fz = max(fz, 1e-6)
    horizontal = math.hypot(fx, fy)
    max_horizontal = fz * math.tan(max_tilt_rad)
    if horizontal <= max_horizontal or horizontal < 1e-9:
        return (fx, fy, fz)
    scale = max_horizontal / horizontal
    return (fx * scale, fy * scale, fz)


def acceleration_to_attitude_thrust(
    accel_enu: Iterable[float],
    yaw_ref_rad: float,
    hover_thrust: float,
    *,
    gravity_mps2: float = 9.80665,
    input_is_net_acceleration: bool = True,
    thrust_min: float = 0.0,
    thrust_max: float = 1.0,
    max_tilt_rad: float = math.radians(35.0),
    max_net_accel_mps2: float = 0.0,
) -> AttitudeThrustCommand:
    """Convert an ENU acceleration command into FLU attitude and normalized thrust.

    If input_is_net_acceleration is True, accel_enu is the desired world-frame
    acceleration excluding gravity. Hover is therefore (0, 0, 0), and the
    desired specific force is accel_enu - g_enu.
    """

    if gravity_mps2 <= 0.0:
        raise ConversionError("gravity_mps2 must be positive")
    if hover_thrust <= 0.0:
        raise ConversionError("hover_thrust must be positive")
    if thrust_min > thrust_max:
        raise ConversionError("thrust_min cannot exceed thrust_max")

    accel = _clamp_vector_norm(_as_vector3(accel_enu, "accel_enu"), max_net_accel_mps2)

    if input_is_net_acceleration:
        specific_force = (accel[0], accel[1], accel[2] + gravity_mps2)
    else:
        specific_force = accel

    if _norm(specific_force) < 1e-9:
        specific_force = (0.0, 0.0, gravity_mps2)

    specific_force = _clamp_specific_force_tilt(specific_force, max_tilt_rad)
    thrust = _clamp(_norm(specific_force) / gravity_mps2 * hover_thrust, thrust_min, thrust_max)

    z_body = _normalize(specific_force, "specific_force")
    x_heading = (math.cos(yaw_ref_rad), math.sin(yaw_ref_rad), 0.0)
    y_body = _cross(z_body, x_heading)
    if _norm(y_body) < 1e-6:
        y_body = _cross(z_body, (0.0, 1.0, 0.0))
    y_body = _normalize(y_body, "y_body")
    x_body = _normalize(_cross(y_body, z_body), "x_body")

    # Columns are body FLU axes expressed in world ENU coordinates.
    r_enu_flu = (
        (x_body[0], y_body[0], z_body[0]),
        (x_body[1], y_body[1], z_body[1]),
        (x_body[2], y_body[2], z_body[2]),
    )

    return AttitudeThrustCommand(
        quaternion_wxyz=_matrix_to_quaternion_wxyz(r_enu_flu),
        thrust=thrust,
        specific_force_enu=specific_force,
    )


def diffphys_raw_action_to_net_acceleration(
    raw_action: Sequence[float],
    rotation_enu_flu: Matrix3,
    *,
    gravity_mps2: float = 9.80665,
    thrust_est_error: float = 1.0,
    layout: str = "interleaved",
) -> Vector3:
    """Apply the DiffPhysDrone action post-processing from training code.

    The model output has six values. In the training code, it is reshaped to
    (3, 2), rotated to world frame, then split into a_pred and v_pred. With the
    default PyTorch row-major reshape, that means action order is:
    [a_x, v_x, a_y, v_y, a_z, v_z].
    """

    if len(raw_action) < 6:
        raise ConversionError("raw_action must contain at least 6 values")
    data = [float(v) for v in raw_action[:6]]
    if not all(math.isfinite(v) for v in data):
        raise ConversionError("raw_action contains NaN or Inf")
    if not math.isfinite(thrust_est_error):
        raise ConversionError("thrust_est_error contains NaN or Inf")

    if layout == "interleaved":
        a_body = (data[0], data[2], data[4])
        v_body = (data[1], data[3], data[5])
    elif layout == "split":
        a_body = (data[0], data[1], data[2])
        v_body = (data[3], data[4], data[5])
    else:
        raise ConversionError("layout must be 'interleaved' or 'split'")

    a_pred = _mat_vec(rotation_enu_flu, a_body)
    v_pred = _mat_vec(rotation_enu_flu, v_body)
    gravity_enu = (0.0, 0.0, -gravity_mps2)
    return (
        (a_pred[0] - v_pred[0] - gravity_enu[0]) * thrust_est_error + gravity_enu[0],
        (a_pred[1] - v_pred[1] - gravity_enu[1]) * thrust_est_error + gravity_enu[1],
        (a_pred[2] - v_pred[2] - gravity_enu[2]) * thrust_est_error + gravity_enu[2],
    )

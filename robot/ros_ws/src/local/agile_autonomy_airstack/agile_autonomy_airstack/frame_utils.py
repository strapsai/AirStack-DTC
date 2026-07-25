import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Vector3:
    x: float
    y: float
    z: float

    def __add__(self, other):
        return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)

    def scale(self, scalar):
        return Vector3(self.x * scalar, self.y * scalar, self.z * scalar)

    def norm(self):
        return math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)


@dataclass(frozen=True)
class Quaternion:
    x: float
    y: float
    z: float
    w: float


@dataclass(frozen=True)
class Pose3:
    position: Vector3
    orientation: Quaternion


def is_finite_vector(vector):
    return math.isfinite(vector.x) and math.isfinite(vector.y) and math.isfinite(vector.z)


def normalize_quaternion(q):
    norm = math.sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w)
    if norm <= 0.0 or not math.isfinite(norm):
        return Quaternion(0.0, 0.0, 0.0, 1.0)
    return Quaternion(q.x / norm, q.y / norm, q.z / norm, q.w / norm)


def quaternion_multiply(a, b):
    return Quaternion(
        a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
        a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
        a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
        a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
    )


def quaternion_conjugate(q):
    return Quaternion(-q.x, -q.y, -q.z, q.w)


def rotate_vector(q, vector):
    qn = normalize_quaternion(q)
    vq = Quaternion(vector.x, vector.y, vector.z, 0.0)
    rotated = quaternion_multiply(quaternion_multiply(qn, vq), quaternion_conjugate(qn))
    return Vector3(rotated.x, rotated.y, rotated.z)


def transform_pose(transform, pose):
    position = transform.position + rotate_vector(transform.orientation, pose.position)
    orientation = quaternion_multiply(transform.orientation, pose.orientation)
    return Pose3(position, normalize_quaternion(orientation))


def transform_body_relative_point(odom_pose, body_point):
    return odom_pose.position + rotate_vector(odom_pose.orientation, body_point)


def yaw_from_quaternion(q):
    qn = normalize_quaternion(q)
    siny_cosp = 2.0 * (qn.w * qn.z + qn.x * qn.y)
    cosy_cosp = 1.0 - 2.0 * (qn.y * qn.y + qn.z * qn.z)
    return math.atan2(siny_cosp, cosy_cosp)


def yaw_to_quaternion(yaw):
    half = yaw * 0.5
    return Quaternion(0.0, 0.0, math.sin(half), math.cos(half))


def yaw_between(a, b, fallback_yaw=0.0):
    dx = b.x - a.x
    dy = b.y - a.y
    if abs(dx) < 1e-9 and abs(dy) < 1e-9:
        return fallback_yaw
    return math.atan2(dy, dx)

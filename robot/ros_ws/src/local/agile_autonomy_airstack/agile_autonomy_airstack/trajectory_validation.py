import math
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Tuple

from agile_autonomy_airstack.frame_utils import Vector3


@dataclass(frozen=True)
class CandidatePoint:
    time_from_start: float
    position: Vector3
    qx: float = 0.0
    qy: float = 0.0
    qz: float = 0.0
    qw: float = 1.0


@dataclass(frozen=True)
class CandidateTrajectoryData:
    stamp: float
    frame_id: str
    candidate_id: str
    cost: float
    frame_mode: str
    points: List[CandidatePoint]


@dataclass(frozen=True)
class ValidationConfig:
    candidate_timeout_sec: float = 0.3
    odom_timeout_sec: float = 0.3
    max_speed_mps: float = 2.0
    max_accel_mps2: float = 3.0
    max_vertical_speed_mps: float = 1.0
    min_points: int = 3
    min_horizon_sec: float = 0.2


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    reason: str
    max_speed_mps: float = 0.0
    max_accel_mps2: float = 0.0
    max_vertical_speed_mps: float = 0.0
    horizon_sec: float = 0.0
    details: dict = field(default_factory=dict)


def _finite(value):
    return math.isfinite(value)


def _finite_point(point):
    return (
        _finite(point.time_from_start)
        and _finite(point.position.x)
        and _finite(point.position.y)
        and _finite(point.position.z)
        and _finite(point.qx)
        and _finite(point.qy)
        and _finite(point.qz)
        and _finite(point.qw)
    )


def validate_candidate(
    candidate: CandidateTrajectoryData,
    now_sec: float,
    candidate_received_sec: Optional[float],
    odom_received_sec: Optional[float],
    config: ValidationConfig,
) -> ValidationResult:
    if candidate_received_sec is None:
        return ValidationResult(False, "no_candidate")
    if now_sec - candidate_received_sec > config.candidate_timeout_sec:
        return ValidationResult(False, "candidate_stale")
    if odom_received_sec is None:
        return ValidationResult(False, "no_odometry")
    if now_sec - odom_received_sec > config.odom_timeout_sec:
        return ValidationResult(False, "odometry_stale")
    if len(candidate.points) < config.min_points:
        return ValidationResult(False, "too_few_points")
    if any(not _finite_point(point) for point in candidate.points):
        return ValidationResult(False, "non_finite_point")

    times = [point.time_from_start for point in candidate.points]
    if times[0] < 0.0:
        return ValidationResult(False, "negative_start_time")
    for prev, curr in zip(times, times[1:]):
        if curr <= prev:
            return ValidationResult(False, "non_increasing_time")

    horizon = times[-1] - times[0]
    if horizon < config.min_horizon_sec:
        return ValidationResult(False, "horizon_too_short", horizon_sec=horizon)

    velocities = []
    max_speed = 0.0
    max_vertical_speed = 0.0
    for first, second in zip(candidate.points, candidate.points[1:]):
        dt = second.time_from_start - first.time_from_start
        delta = second.position - first.position
        velocity = delta.scale(1.0 / dt)
        speed = velocity.norm()
        vertical_speed = abs(velocity.z)
        max_speed = max(max_speed, speed)
        max_vertical_speed = max(max_vertical_speed, vertical_speed)
        if speed > config.max_speed_mps:
            return ValidationResult(
                False,
                "speed_limit_exceeded",
                max_speed_mps=speed,
                max_vertical_speed_mps=max_vertical_speed,
                horizon_sec=horizon,
            )
        if vertical_speed > config.max_vertical_speed_mps:
            return ValidationResult(
                False,
                "vertical_speed_limit_exceeded",
                max_speed_mps=max_speed,
                max_vertical_speed_mps=vertical_speed,
                horizon_sec=horizon,
            )
        velocities.append((first.time_from_start, second.time_from_start, velocity))

    max_accel = 0.0
    for prev, curr in zip(velocities, velocities[1:]):
        prev_mid = (prev[0] + prev[1]) * 0.5
        curr_mid = (curr[0] + curr[1]) * 0.5
        dt = curr_mid - prev_mid
        if dt <= 0.0:
            return ValidationResult(False, "invalid_acceleration_dt")
        accel = (curr[2] - prev[2]).scale(1.0 / dt)
        accel_mag = accel.norm()
        max_accel = max(max_accel, accel_mag)
        if accel_mag > config.max_accel_mps2:
            return ValidationResult(
                False,
                "accel_limit_exceeded",
                max_speed_mps=max_speed,
                max_accel_mps2=accel_mag,
                max_vertical_speed_mps=max_vertical_speed,
                horizon_sec=horizon,
            )

    return ValidationResult(
        True,
        "valid",
        max_speed_mps=max_speed,
        max_accel_mps2=max_accel,
        max_vertical_speed_mps=max_vertical_speed,
        horizon_sec=horizon,
    )


def select_best_valid_candidate(
    candidates: Iterable[CandidateTrajectoryData],
    now_sec: float,
    candidate_received_sec: Optional[float],
    odom_received_sec: Optional[float],
    config: ValidationConfig,
) -> Tuple[Optional[CandidateTrajectoryData], Optional[ValidationResult], List[ValidationResult]]:
    valid = []
    rejected = []
    for index, candidate in enumerate(candidates):
        result = validate_candidate(candidate, now_sec, candidate_received_sec, odom_received_sec, config)
        if result.valid:
            cost = candidate.cost if math.isfinite(candidate.cost) else math.inf
            valid.append((cost, index, candidate, result))
        else:
            rejected.append(result)

    if not valid:
        return None, None, rejected

    valid.sort(key=lambda item: (item[0], item[1]))
    _, _, candidate, result = valid[0]
    return candidate, result, rejected

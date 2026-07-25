from agile_autonomy_airstack.frame_utils import Vector3
from agile_autonomy_airstack.trajectory_validation import (
    CandidatePoint,
    CandidateTrajectoryData,
    ValidationConfig,
    select_best_valid_candidate,
    validate_candidate,
)


def make_candidate(points):
    return CandidateTrajectoryData(
        stamp=0.0,
        frame_id="body_relative",
        candidate_id="test",
        cost=0.0,
        frame_mode="body_relative",
        points=[
            CandidatePoint(time_from_start=t, position=Vector3(x, y, z))
            for t, x, y, z in points
        ],
    )


def test_rejects_stale_candidate():
    candidate = make_candidate([(0.0, 0.0, 0.0, 0.0), (0.1, 0.05, 0.0, 0.0), (0.2, 0.1, 0.0, 0.0)])
    result = validate_candidate(
        candidate,
        now_sec=1.0,
        candidate_received_sec=0.0,
        odom_received_sec=1.0,
        config=ValidationConfig(candidate_timeout_sec=0.3),
    )

    assert not result.valid
    assert result.reason == "candidate_stale"


def test_rejects_too_fast_candidate():
    candidate = make_candidate([(0.0, 0.0, 0.0, 0.0), (0.1, 10.0, 0.0, 0.0), (0.2, 20.0, 0.0, 0.0)])
    result = validate_candidate(
        candidate,
        now_sec=0.1,
        candidate_received_sec=0.1,
        odom_received_sec=0.1,
        config=ValidationConfig(max_speed_mps=2.0),
    )

    assert not result.valid
    assert result.reason == "speed_limit_exceeded"


def test_accepts_simple_valid_candidate():
    candidate = make_candidate([(0.0, 0.0, 0.0, 0.0), (0.1, 0.05, 0.0, 0.0), (0.2, 0.1, 0.0, 0.0)])
    result = validate_candidate(
        candidate,
        now_sec=0.1,
        candidate_received_sec=0.1,
        odom_received_sec=0.1,
        config=ValidationConfig(max_speed_mps=2.0, max_accel_mps2=3.0),
    )

    assert result.valid
    assert result.reason == "valid"


def test_selects_lowest_cost_valid_candidate():
    expensive = make_candidate([(0.0, 0.0, 0.0, 0.0), (0.1, 0.05, 0.0, 0.0), (0.2, 0.1, 0.0, 0.0)])
    cheap = CandidateTrajectoryData(
        stamp=0.0,
        frame_id="body_relative",
        candidate_id="cheap",
        cost=-1.0,
        frame_mode="body_relative",
        points=expensive.points,
    )
    best, result, rejected = select_best_valid_candidate(
        [expensive, cheap],
        now_sec=0.1,
        candidate_received_sec=0.1,
        odom_received_sec=0.1,
        config=ValidationConfig(),
    )

    assert result.valid
    assert rejected == []
    assert best.candidate_id == "cheap"

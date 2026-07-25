import math
import unittest
from pathlib import Path

import yaml

from agile_autonomy_airstack.frame_utils import (
    Pose3,
    Vector3,
    transform_body_relative_point,
    yaw_to_quaternion,
)
from agile_autonomy_airstack.trajectory_validation import (
    CandidatePoint,
    CandidateTrajectoryData,
    ValidationConfig,
    select_best_valid_candidate,
    validate_candidate,
)


def make_candidate(points, candidate_id="test", cost=0.0):
    return CandidateTrajectoryData(
        stamp=0.0,
        frame_id="body_relative",
        candidate_id=candidate_id,
        cost=cost,
        frame_mode="body_relative",
        points=[
            CandidatePoint(time_from_start=t, position=Vector3(x, y, z))
            for t, x, y, z in points
        ],
    )


class TestAgileAutonomyAirstack(unittest.TestCase):
    def test_body_relative_point_uses_odometry_yaw(self):
        odom_pose = Pose3(Vector3(1.0, 2.0, 3.0), yaw_to_quaternion(math.pi / 2.0))
        point = transform_body_relative_point(odom_pose, Vector3(1.0, 0.0, 0.0))

        self.assertAlmostEqual(point.x, 1.0)
        self.assertAlmostEqual(point.y, 3.0)
        self.assertAlmostEqual(point.z, 3.0)

    def test_rejects_stale_candidate(self):
        candidate = make_candidate(
            [(0.0, 0.0, 0.0, 0.0), (0.1, 0.05, 0.0, 0.0), (0.2, 0.1, 0.0, 0.0)]
        )
        result = validate_candidate(
            candidate,
            now_sec=1.0,
            candidate_received_sec=0.0,
            odom_received_sec=1.0,
            config=ValidationConfig(candidate_timeout_sec=0.3),
        )

        self.assertFalse(result.valid)
        self.assertEqual(result.reason, "candidate_stale")

    def test_rejects_too_fast_candidate(self):
        candidate = make_candidate(
            [(0.0, 0.0, 0.0, 0.0), (0.1, 10.0, 0.0, 0.0), (0.2, 20.0, 0.0, 0.0)]
        )
        result = validate_candidate(
            candidate,
            now_sec=0.1,
            candidate_received_sec=0.1,
            odom_received_sec=0.1,
            config=ValidationConfig(max_speed_mps=2.0),
        )

        self.assertFalse(result.valid)
        self.assertEqual(result.reason, "speed_limit_exceeded")

    def test_accepts_simple_valid_candidate(self):
        candidate = make_candidate(
            [(0.0, 0.0, 0.0, 0.0), (0.1, 0.05, 0.0, 0.0), (0.2, 0.1, 0.0, 0.0)]
        )
        result = validate_candidate(
            candidate,
            now_sec=0.1,
            candidate_received_sec=0.1,
            odom_received_sec=0.1,
            config=ValidationConfig(max_speed_mps=2.0, max_accel_mps2=3.0),
        )

        self.assertTrue(result.valid)
        self.assertEqual(result.reason, "valid")

    def test_selects_lowest_cost_valid_candidate(self):
        expensive = make_candidate(
            [(0.0, 0.0, 0.0, 0.0), (0.1, 0.05, 0.0, 0.0), (0.2, 0.1, 0.0, 0.0)],
            candidate_id="expensive",
            cost=1.0,
        )
        cheap = make_candidate(
            [(0.0, 0.0, 0.0, 0.0), (0.1, 0.05, 0.0, 0.0), (0.2, 0.1, 0.0, 0.0)],
            candidate_id="cheap",
            cost=-1.0,
        )
        best, result, rejected = select_best_valid_candidate(
            [expensive, cheap],
            now_sec=0.1,
            candidate_received_sec=0.1,
            odom_received_sec=0.1,
            config=ValidationConfig(),
        )

        self.assertTrue(result.valid)
        self.assertEqual(rejected, [])
        self.assertEqual(best.candidate_id, "cheap")

    def test_config_loads(self):
        config_path = Path(__file__).parents[1] / "config" / "agile_autonomy_airstack.yaml"
        data = yaml.safe_load(config_path.read_text())
        params = data["/**"]["ros__parameters"]

        self.assertFalse(params["enabled_default"])
        self.assertEqual(params["backend_type"], "mock")
        self.assertLessEqual(params["max_speed_mps"], 2.0)


if __name__ == "__main__":
    unittest.main()

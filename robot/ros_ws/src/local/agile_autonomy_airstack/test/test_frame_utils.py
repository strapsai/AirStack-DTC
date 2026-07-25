import math

from agile_autonomy_airstack.frame_utils import (
    Pose3,
    Vector3,
    transform_body_relative_point,
    yaw_to_quaternion,
)


def test_body_relative_point_uses_odometry_yaw():
    odom_pose = Pose3(Vector3(1.0, 2.0, 3.0), yaw_to_quaternion(math.pi / 2.0))
    point = transform_body_relative_point(odom_pose, Vector3(1.0, 0.0, 0.0))

    assert abs(point.x - 1.0) < 1e-6
    assert abs(point.y - 3.0) < 1e-6
    assert abs(point.z - 3.0) < 1e-6

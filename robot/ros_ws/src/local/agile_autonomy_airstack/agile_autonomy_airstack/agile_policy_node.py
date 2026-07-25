import os

import rclpy
from builtin_interfaces.msg import Duration
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Bool, String

from agile_autonomy_airstack.backend import create_backend
from agile_autonomy_airstack.status import make_status
from agile_autonomy_airstack_msgs.msg import (
    CandidateTrajectory,
    CandidateTrajectoryArray,
    CandidateTrajectoryPoint,
)


def _now_sec(node):
    return node.get_clock().now().nanoseconds * 1e-9


def _duration_from_sec(value):
    sec = int(value)
    nanosec = int(round((value - sec) * 1e9))
    if nanosec >= 1000000000:
        sec += 1
        nanosec -= 1000000000
    return Duration(sec=sec, nanosec=nanosec)


class AgilePolicyNode(Node):
    def __init__(self):
        super().__init__("agile_policy_node")

        self.enabled = self.declare_parameter("enabled_default", False).value
        self.backend_type = self.declare_parameter("backend_type", "mock").value
        self.publish_rate_hz = float(self.declare_parameter("publish_rate_hz", 15.0).value)
        self.input_timeout_sec = float(self.declare_parameter("input_timeout_sec", 0.5).value)
        self.trajectory_horizon_sec = float(
            self.declare_parameter("trajectory_horizon_sec", 1.0).value
        )
        self.trajectory_dt_sec = float(self.declare_parameter("trajectory_dt_sec", 0.1).value)
        self.mock_forward_speed_mps = float(
            self.declare_parameter("mock_forward_speed_mps", 0.5).value
        )
        self.mock_speed_limit_mps = float(
            self.declare_parameter("mock_speed_limit_mps", 2.0).value
        )
        self.frame_mode = self.declare_parameter("frame_mode", "body_relative").value
        self.robot_name = self.declare_parameter(
            "robot_name", os.environ.get("ROBOT_NAME", "")
        ).value
        self.external_backend_config = self.declare_parameter(
            "external_backend_config", ""
        ).value
        self.debug = bool(self.declare_parameter("debug", False).value)

        self.last_odom_time = None
        self.last_depth_time = None
        self.last_camera_info_time = None
        self.last_global_plan_time = None
        self.last_odom = None
        self.last_depth = None
        self.last_camera_info = None
        self.last_global_plan = None

        sensor_qos = QoSProfile(depth=1)
        sensor_qos.reliability = ReliabilityPolicy.BEST_EFFORT

        self.odom_sub = self.create_subscription(
            Odometry, "odometry", self._odom_callback, sensor_qos
        )
        self.depth_sub = self.create_subscription(
            Image, "depth", self._depth_callback, sensor_qos
        )
        self.camera_info_sub = self.create_subscription(
            CameraInfo, "camera_info", self._camera_info_callback, sensor_qos
        )
        self.global_plan_sub = self.create_subscription(
            Path, "global_plan", self._global_plan_callback, 1
        )
        self.enable_sub = self.create_subscription(Bool, "enable", self._enable_callback, 1)

        self.candidate_pub = self.create_publisher(
            CandidateTrajectoryArray, "trajectory_candidates", 1
        )
        self.status_pub = self.create_publisher(String, "status", 1)

        self.backend = create_backend(
            self.backend_type,
            self.trajectory_horizon_sec,
            self.trajectory_dt_sec,
            self.mock_forward_speed_mps,
            self.mock_speed_limit_mps,
            self.frame_mode,
            self.external_backend_config,
        )

        period = 1.0 / max(self.publish_rate_hz, 1e-6)
        self.timer = self.create_timer(period, self._timer_callback)
        self.get_logger().info(
            f"Agile policy node started with backend='{self.backend_type}', enabled={self.enabled}"
        )

    def _odom_callback(self, msg):
        self.last_odom = msg
        self.last_odom_time = _now_sec(self)

    def _depth_callback(self, msg):
        self.last_depth = msg
        self.last_depth_time = _now_sec(self)

    def _camera_info_callback(self, msg):
        self.last_camera_info = msg
        self.last_camera_info_time = _now_sec(self)

    def _global_plan_callback(self, msg):
        self.last_global_plan = msg
        self.last_global_plan_time = _now_sec(self)

    def _enable_callback(self, msg):
        self.enabled = bool(msg.data)

    def _age(self, stamp_sec):
        if stamp_sec is None:
            return None
        return max(0.0, _now_sec(self) - stamp_sec)

    def _inputs_fresh(self):
        odom_age = self._age(self.last_odom_time)
        depth_age = self._age(self.last_depth_time)
        if odom_age is None:
            return False, "waiting_for_odometry"
        if depth_age is None:
            return False, "waiting_for_depth"
        if odom_age > self.input_timeout_sec:
            return False, "odometry_stale"
        if depth_age > self.input_timeout_sec:
            return False, "depth_stale"
        return True, "ok"

    def _candidate_to_msg(self, candidate, stamp):
        msg = CandidateTrajectory()
        msg.header.stamp = stamp
        msg.header.frame_id = self.frame_mode
        msg.candidate_id = candidate.candidate_id
        msg.cost = float(candidate.cost)
        msg.frame_mode = candidate.frame_mode
        for point in candidate.points:
            point_msg = CandidateTrajectoryPoint()
            point_msg.time_from_start = _duration_from_sec(point.time_from_start)
            point_msg.pose.position.x = point.x
            point_msg.pose.position.y = point.y
            point_msg.pose.position.z = point.z
            point_msg.pose.orientation.w = 1.0
            msg.points.append(point_msg)
        return msg

    def _publish_status(self, healthy, reason):
        status = String()
        status.data = make_status(
            self.enabled,
            healthy,
            reason,
            odom_age_sec=self._age(self.last_odom_time),
            depth_age_sec=self._age(self.last_depth_time),
            camera_info_age_sec=self._age(self.last_camera_info_time),
            global_plan_age_sec=self._age(self.last_global_plan_time),
            backend_type=self.backend_type,
            publish_rate_hz=self.publish_rate_hz,
            robot_name=self.robot_name,
        )
        self.status_pub.publish(status)

    def _timer_callback(self):
        if not self.enabled:
            self._publish_status(True, "disabled")
            return

        fresh, reason = self._inputs_fresh()
        if not fresh:
            self._publish_status(False, reason)
            return

        result = self.backend.generate()
        if not result.healthy:
            self._publish_status(False, result.reason)
            return

        stamp = self.get_clock().now().to_msg()
        array_msg = CandidateTrajectoryArray()
        array_msg.header.stamp = stamp
        array_msg.header.frame_id = self.frame_mode
        for candidate in result.candidates:
            array_msg.candidates.append(self._candidate_to_msg(candidate, stamp))

        if array_msg.candidates:
            self.candidate_pub.publish(array_msg)
        self._publish_status(True, result.reason)


def main(args=None):
    rclpy.init(args=args)
    node = AgilePolicyNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

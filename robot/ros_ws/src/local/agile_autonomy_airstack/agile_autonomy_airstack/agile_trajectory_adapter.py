import math

import rclpy
import tf2_ros
from airstack_msgs.msg import TrajectoryXYZVYaw, WaypointXYZVYaw
from nav_msgs.msg import Odometry
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool, String

from agile_autonomy_airstack.frame_utils import (
    Pose3,
    Quaternion,
    Vector3,
    transform_body_relative_point,
    transform_pose,
    yaw_between,
    yaw_from_quaternion,
)
from agile_autonomy_airstack.status import make_status
from agile_autonomy_airstack.trajectory_validation import (
    CandidatePoint,
    CandidateTrajectoryData,
    ValidationConfig,
    select_best_valid_candidate,
)
from agile_autonomy_airstack_msgs.msg import CandidateTrajectoryArray


def _now_sec(node):
    return node.get_clock().now().nanoseconds * 1e-9


def _time_msg_to_sec(stamp):
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def _duration_msg_to_sec(duration):
    return float(duration.sec) + float(duration.nanosec) * 1e-9


def _pose_from_ros(pose):
    return Pose3(
        Vector3(pose.position.x, pose.position.y, pose.position.z),
        Quaternion(
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        ),
    )


def _pose_from_transform(transform):
    return Pose3(
        Vector3(
            transform.transform.translation.x,
            transform.transform.translation.y,
            transform.transform.translation.z,
        ),
        Quaternion(
            transform.transform.rotation.x,
            transform.transform.rotation.y,
            transform.transform.rotation.z,
            transform.transform.rotation.w,
        ),
    )


class AgileTrajectoryAdapter(Node):
    def __init__(self):
        super().__init__("agile_trajectory_adapter")

        self.enabled = self.declare_parameter("enabled_default", False).value
        self.candidate_timeout_sec = float(
            self.declare_parameter("candidate_timeout_sec", 0.3).value
        )
        self.odom_timeout_sec = float(self.declare_parameter("odom_timeout_sec", 0.3).value)
        self.max_speed_mps = float(self.declare_parameter("max_speed_mps", 2.0).value)
        self.max_accel_mps2 = float(self.declare_parameter("max_accel_mps2", 3.0).value)
        self.max_vertical_speed_mps = float(
            self.declare_parameter("max_vertical_speed_mps", 1.0).value
        )
        self.min_points = int(self.declare_parameter("min_points", 3).value)
        self.min_horizon_sec = float(self.declare_parameter("min_horizon_sec", 0.2).value)
        self.target_frame = self.declare_parameter("target_frame", "map").value
        self.candidate_frame_mode = self.declare_parameter(
            "candidate_frame_mode", "body_relative"
        ).value
        self.yaw_mode = self.declare_parameter("yaw_mode", "current").value
        self.debug = bool(self.declare_parameter("debug", False).value)

        self.validation_config = ValidationConfig(
            candidate_timeout_sec=self.candidate_timeout_sec,
            odom_timeout_sec=self.odom_timeout_sec,
            max_speed_mps=self.max_speed_mps,
            max_accel_mps2=self.max_accel_mps2,
            max_vertical_speed_mps=self.max_vertical_speed_mps,
            min_points=self.min_points,
            min_horizon_sec=self.min_horizon_sec,
        )

        self.last_candidates = None
        self.last_candidates_time = None
        self.last_odom = None
        self.last_odom_time = None
        self.last_status_reason = "starting"

        sensor_qos = QoSProfile(depth=1)
        sensor_qos.reliability = ReliabilityPolicy.BEST_EFFORT

        self.candidate_sub = self.create_subscription(
            CandidateTrajectoryArray,
            "trajectory_candidates",
            self._candidate_callback,
            1,
        )
        self.odom_sub = self.create_subscription(
            Odometry, "odometry", self._odom_callback, sensor_qos
        )
        self.enable_sub = self.create_subscription(Bool, "enable", self._enable_callback, 1)

        self.trajectory_pub = self.create_publisher(
            TrajectoryXYZVYaw, "trajectory_segment", 1
        )
        self.status_pub = self.create_publisher(String, "adapter_status", 1)

        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=5.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.status_timer = self.create_timer(0.2, self._publish_status_timer)
        self.get_logger().info(f"Agile trajectory adapter started, enabled={self.enabled}")

    def _odom_callback(self, msg):
        self.last_odom = msg
        self.last_odom_time = _now_sec(self)

    def _enable_callback(self, msg):
        self.enabled = bool(msg.data)

    def _candidate_callback(self, msg):
        self.last_candidates = msg
        self.last_candidates_time = _now_sec(self)
        self._process_latest_candidates()

    def _age(self, stamp_sec):
        if stamp_sec is None:
            return None
        return max(0.0, _now_sec(self) - stamp_sec)

    def _candidate_msg_to_data(self, msg):
        points = []
        for point in msg.points:
            points.append(
                CandidatePoint(
                    time_from_start=_duration_msg_to_sec(point.time_from_start),
                    position=Vector3(
                        point.pose.position.x,
                        point.pose.position.y,
                        point.pose.position.z,
                    ),
                    qx=point.pose.orientation.x,
                    qy=point.pose.orientation.y,
                    qz=point.pose.orientation.z,
                    qw=point.pose.orientation.w,
                )
            )
        return CandidateTrajectoryData(
            stamp=_time_msg_to_sec(msg.header.stamp),
            frame_id=msg.header.frame_id,
            candidate_id=msg.candidate_id,
            cost=msg.cost,
            frame_mode=msg.frame_mode or self.candidate_frame_mode,
            points=points,
        )

    def _frame_matches(self, frame):
        return frame == self.target_frame or not self.target_frame or not frame

    def _odom_pose_in_target_frame(self):
        if self.last_odom is None:
            raise RuntimeError("no_odometry")
        odom_pose = _pose_from_ros(self.last_odom.pose.pose)
        source_frame = self.last_odom.header.frame_id
        if self._frame_matches(source_frame):
            return odom_pose

        transform = self.tf_buffer.lookup_transform(
            self.target_frame,
            source_frame,
            rclpy.time.Time(),
            timeout=Duration(seconds=0.05),
        )
        return transform_pose(_pose_from_transform(transform), odom_pose)

    def _points_in_target_frame(self, candidate):
        mode = candidate.frame_mode or self.candidate_frame_mode
        if mode == "body_relative":
            odom_pose = self._odom_pose_in_target_frame()
            return [
                transform_body_relative_point(odom_pose, point.position)
                for point in candidate.points
            ]

        if mode in ("odom", "map", "target_frame"):
            if self._frame_matches(candidate.frame_id) or mode == "target_frame":
                return [point.position for point in candidate.points]

            transform = self.tf_buffer.lookup_transform(
                self.target_frame,
                candidate.frame_id,
                rclpy.time.Time(),
                timeout=Duration(seconds=0.05),
            )
            transform_pose_msg = _pose_from_transform(transform)
            return [
                transform_pose(
                    transform_pose_msg,
                    Pose3(
                        point.position,
                        Quaternion(point.qx, point.qy, point.qz, point.qw),
                    ),
                ).position
                for point in candidate.points
            ]

        raise RuntimeError("unsupported_candidate_frame_mode")

    def _segment_velocities(self, points, times):
        velocities = []
        for first, second, first_time, second_time in zip(
            points, points[1:], times, times[1:]
        ):
            dt = max(second_time - first_time, 1e-6)
            velocities.append((second - first).scale(1.0 / dt))
        if not velocities:
            return [Vector3(0.0, 0.0, 0.0)]
        velocities.append(velocities[-1])
        return velocities

    def _segment_accelerations(self, velocities, times):
        accelerations = [Vector3(0.0, 0.0, 0.0)]
        for index in range(1, len(velocities)):
            dt = max(times[index] - times[index - 1], 1e-6)
            accelerations.append((velocities[index] - velocities[index - 1]).scale(1.0 / dt))
        return accelerations

    def _build_air_segment(self, candidate):
        points = self._points_in_target_frame(candidate)
        times = [point.time_from_start for point in candidate.points]
        velocities = self._segment_velocities(points, times)
        accelerations = self._segment_accelerations(velocities, times)

        current_yaw = 0.0
        if self.last_odom is not None:
            current_yaw = yaw_from_quaternion(_pose_from_ros(self.last_odom.pose.pose).orientation)

        msg = TrajectoryXYZVYaw()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.target_frame

        for index, point in enumerate(points):
            wp = WaypointXYZVYaw()
            wp.position.x = point.x
            wp.position.y = point.y
            wp.position.z = point.z
            wp.velocity = velocities[index].norm()
            if self.yaw_mode == "path" and index + 1 < len(points):
                wp.yaw = yaw_between(point, points[index + 1], current_yaw)
            else:
                wp.yaw = current_yaw
            wp.acceleration.x = accelerations[index].x
            wp.acceleration.y = accelerations[index].y
            wp.acceleration.z = accelerations[index].z
            msg.waypoints.append(wp)
        return msg

    def _process_latest_candidates(self):
        now = _now_sec(self)
        if not self.enabled:
            self.last_status_reason = "disabled"
            return
        if self.last_candidates is None:
            self.last_status_reason = "no_candidates"
            return

        candidate_data = [
            self._candidate_msg_to_data(candidate)
            for candidate in self.last_candidates.candidates
        ]
        candidate, result, rejected = select_best_valid_candidate(
            candidate_data,
            now,
            self.last_candidates_time,
            self.last_odom_time,
            self.validation_config,
        )
        if candidate is None:
            reason = rejected[0].reason if rejected else "no_valid_candidate"
            self.last_status_reason = reason
            self.get_logger().warn(
                f"Rejected Agile candidates: {reason}",
                throttle_duration_sec=1.0,
            )
            return

        try:
            segment = self._build_air_segment(candidate)
        except Exception as exc:
            self.last_status_reason = "frame_transform_unavailable"
            self.get_logger().warn(
                f"Unable to transform Agile candidate '{candidate.candidate_id}': {exc}",
                throttle_duration_sec=1.0,
            )
            return

        if any(
            not math.isfinite(wp.position.x)
            or not math.isfinite(wp.position.y)
            or not math.isfinite(wp.position.z)
            or not math.isfinite(wp.velocity)
            for wp in segment.waypoints
        ):
            self.last_status_reason = "non_finite_output_segment"
            return

        self.trajectory_pub.publish(segment)
        self.last_status_reason = f"published:{candidate.candidate_id}:{result.reason}"

    def _publish_status_timer(self):
        healthy = self.enabled and self.last_status_reason.startswith("published:")
        status = String()
        status.data = make_status(
            self.enabled,
            healthy,
            self.last_status_reason,
            candidate_age_sec=self._age(self.last_candidates_time),
            odom_age_sec=self._age(self.last_odom_time),
            target_frame=self.target_frame,
            candidate_frame_mode=self.candidate_frame_mode,
            yaw_mode=self.yaw_mode,
            max_speed_mps=self.max_speed_mps,
            max_accel_mps2=self.max_accel_mps2,
            max_vertical_speed_mps=self.max_vertical_speed_mps,
        )
        self.status_pub.publish(status)


def main(args=None):
    rclpy.init(args=args)
    node = AgileTrajectoryAdapter()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

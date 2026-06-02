import math
from typing import Optional, Tuple

from geometry_msgs.msg import Vector3Stamped
from mav_msgs.msg import AttitudeThrust
from nav_msgs.msg import Odometry
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from std_msgs.msg import Float32MultiArray

from diffphysdrone_px4_wrapper.accel_to_attitude import (
    ConversionError,
    Matrix3,
    acceleration_to_attitude_thrust,
    diffphys_raw_action_to_net_acceleration,
    diffphys_yaw_frame_from_rotation,
    quaternion_wxyz_to_matrix,
    yaw_from_quaternion_wxyz,
)


class DiffPhysDronePx4Wrapper(Node):
    """Convert DiffPhysDrone policy actions into AirStack PX4 attitude/thrust commands."""

    def __init__(self):
        super().__init__('diffphysdrone_px4_wrapper')

        self.declare_parameter('accel_topic', 'diffphysdrone/accel_cmd')
        self.declare_parameter('raw_action_topic', 'diffphysdrone/raw_action')
        self.declare_parameter('odom_topic', 'odometry')
        self.declare_parameter('output_topic', 'cmd_attitude_thrust')
        self.declare_parameter('hover_thrust', 0.5)
        self.declare_parameter('thrust_min', 0.05)
        self.declare_parameter('thrust_max', 0.9)
        self.declare_parameter('max_tilt_deg', 35.0)
        self.declare_parameter('max_net_accel_mps2', 20.0)
        self.declare_parameter('gravity_mps2', 9.80665)
        self.declare_parameter('input_is_net_acceleration', True)
        self.declare_parameter('use_current_yaw', True)
        self.declare_parameter('yaw_ref_rad', 0.0)
        self.declare_parameter('publish_rate_hz', 50.0)
        self.declare_parameter('command_timeout_s', 0.2)
        self.declare_parameter('publish_stale_hover', False)
        self.declare_parameter('thrust_est_error', 1.0)
        self.declare_parameter('raw_action_layout', 'interleaved')

        self._hover_thrust = float(self.get_parameter('hover_thrust').value)
        self._thrust_min = float(self.get_parameter('thrust_min').value)
        self._thrust_max = float(self.get_parameter('thrust_max').value)
        self._max_tilt_rad = math.radians(float(self.get_parameter('max_tilt_deg').value))
        self._max_net_accel = float(self.get_parameter('max_net_accel_mps2').value)
        self._gravity = float(self.get_parameter('gravity_mps2').value)
        self._input_is_net_accel = bool(self.get_parameter('input_is_net_acceleration').value)
        self._use_current_yaw = bool(self.get_parameter('use_current_yaw').value)
        self._yaw_ref = float(self.get_parameter('yaw_ref_rad').value)
        self._timeout_s = float(self.get_parameter('command_timeout_s').value)
        self._publish_stale_hover = bool(self.get_parameter('publish_stale_hover').value)
        self._thrust_est_error = float(self.get_parameter('thrust_est_error').value)
        self._raw_action_layout = str(self.get_parameter('raw_action_layout').value)

        self._last_accel: Optional[Tuple[float, float, float]] = None
        self._last_command_time = None
        self._current_yaw = self._yaw_ref
        self._rotation_enu_flu: Optional[Matrix3] = None
        self._diffphys_frame: Optional[Matrix3] = None
        self._odom_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)

        self._pub = self.create_publisher(
            AttitudeThrust,
            str(self.get_parameter('output_topic').value),
            1,
        )
        self.create_subscription(
            Vector3Stamped,
            str(self.get_parameter('accel_topic').value),
            self._on_accel,
            1,
        )
        self.create_subscription(
            Float32MultiArray,
            str(self.get_parameter('raw_action_topic').value),
            self._on_raw_action,
            1,
        )
        self.create_subscription(
            Odometry,
            str(self.get_parameter('odom_topic').value),
            self._on_odom,
            self._odom_qos,
        )

        publish_rate = max(1.0, float(self.get_parameter('publish_rate_hz').value))
        self.create_timer(1.0 / publish_rate, self._publish_latest)

        self.get_logger().info(
            'DiffPhysDrone PX4 wrapper publishing AttitudeThrust on '
            f'{self.get_parameter("output_topic").value}'
        )

    def _now(self):
        return self.get_clock().now()

    def _on_odom(self, msg: Odometry):
        q = msg.pose.pose.orientation
        q_wxyz = (q.w, q.x, q.y, q.z)
        try:
            self._current_yaw = yaw_from_quaternion_wxyz(q_wxyz)
            self._rotation_enu_flu = quaternion_wxyz_to_matrix(q_wxyz)
            self._diffphys_frame = diffphys_yaw_frame_from_rotation(self._rotation_enu_flu)
        except ConversionError as exc:
            self.get_logger().warn(f'Ignoring invalid odometry quaternion: {exc}', throttle_duration_sec=1.0)

    def _on_accel(self, msg: Vector3Stamped):
        self._last_accel = (msg.vector.x, msg.vector.y, msg.vector.z)
        self._last_command_time = self._now()

    def _on_raw_action(self, msg: Float32MultiArray):
        if self._diffphys_frame is None:
            self.get_logger().warn('Raw DiffPhys action received before odometry; ignoring.', throttle_duration_sec=1.0)
            return
        try:
            self._last_accel = diffphys_raw_action_to_net_acceleration(
                msg.data,
                self._diffphys_frame,
                gravity_mps2=self._gravity,
                thrust_est_error=self._thrust_est_error,
                layout=self._raw_action_layout,
            )
            self._last_command_time = self._now()
        except ConversionError as exc:
            self.get_logger().warn(f'Ignoring invalid raw DiffPhys action: {exc}', throttle_duration_sec=1.0)

    def _command_is_fresh(self) -> bool:
        if self._last_command_time is None:
            return False
        age = (self._now() - self._last_command_time).nanoseconds * 1e-9
        return age <= self._timeout_s

    def _publish_latest(self):
        accel = self._last_accel
        if accel is None:
            return
        if not self._command_is_fresh():
            if not self._publish_stale_hover:
                return
            accel = (0.0, 0.0, 0.0)

        yaw_ref = self._current_yaw if self._use_current_yaw else self._yaw_ref

        try:
            command = acceleration_to_attitude_thrust(
                accel,
                yaw_ref,
                self._hover_thrust,
                gravity_mps2=self._gravity,
                input_is_net_acceleration=self._input_is_net_accel,
                thrust_min=self._thrust_min,
                thrust_max=self._thrust_max,
                max_tilt_rad=self._max_tilt_rad,
                max_net_accel_mps2=self._max_net_accel,
            )
        except ConversionError as exc:
            self.get_logger().warn(f'Not publishing invalid command: {exc}', throttle_duration_sec=1.0)
            return

        msg = AttitudeThrust()
        msg.header.stamp = self._now().to_msg()
        msg.header.frame_id = 'map'
        msg.attitude.w = command.quaternion_wxyz[0]
        msg.attitude.x = command.quaternion_wxyz[1]
        msg.attitude.y = command.quaternion_wxyz[2]
        msg.attitude.z = command.quaternion_wxyz[3]
        msg.thrust.x = 0.0
        msg.thrust.y = 0.0
        msg.thrust.z = command.thrust
        self._pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = DiffPhysDronePx4Wrapper()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

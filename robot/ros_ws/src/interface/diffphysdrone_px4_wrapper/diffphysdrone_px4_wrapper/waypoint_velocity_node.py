import math
from typing import Optional

from geometry_msgs.msg import PointStamped, Vector3Stamped
from nav_msgs.msg import Odometry
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy


class WaypointVelocityNode(Node):
    """Convert a waypoint into a bounded ENU target velocity for DiffPhysDrone."""

    def __init__(self):
        super().__init__('diffphysdrone_waypoint_velocity')

        self.declare_parameter('odom_topic', 'odometry')
        self.declare_parameter('goal_topic', 'diffphysdrone/goal')
        self.declare_parameter('target_velocity_topic', 'diffphysdrone/target_velocity')
        self.declare_parameter('max_speed_mps', 0.3)
        self.declare_parameter('goal_radius_m', 0.4)
        self.declare_parameter('publish_rate_hz', 10.0)
        self.declare_parameter('hold_z', True)
        self.declare_parameter('max_vertical_speed_mps', 0.2)

        self._max_speed = float(self.get_parameter('max_speed_mps').value)
        self._goal_radius = float(self.get_parameter('goal_radius_m').value)
        self._hold_z = bool(self.get_parameter('hold_z').value)
        self._max_vertical_speed = float(self.get_parameter('max_vertical_speed_mps').value)
        self._odom: Optional[Odometry] = None
        self._goal: Optional[PointStamped] = None

        odom_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(
            Odometry,
            str(self.get_parameter('odom_topic').value),
            self._on_odom,
            odom_qos,
        )
        self.create_subscription(
            PointStamped,
            str(self.get_parameter('goal_topic').value),
            self._on_goal,
            1,
        )
        self._pub = self.create_publisher(
            Vector3Stamped,
            str(self.get_parameter('target_velocity_topic').value),
            1,
        )

        publish_rate = max(1.0, float(self.get_parameter('publish_rate_hz').value))
        self.create_timer(1.0 / publish_rate, self._publish)
        self.get_logger().info(
            'Waypoint velocity node publishing to '
            f'{self.get_parameter("target_velocity_topic").value}'
        )

    def _on_odom(self, msg: Odometry):
        self._odom = msg

    def _on_goal(self, msg: PointStamped):
        self._goal = msg
        self.get_logger().info(
            f'New goal: x={msg.point.x:.2f}, y={msg.point.y:.2f}, z={msg.point.z:.2f}'
        )

    def _publish(self):
        if self._odom is None or self._goal is None:
            return

        pos = self._odom.pose.pose.position
        dx = float(self._goal.point.x - pos.x)
        dy = float(self._goal.point.y - pos.y)
        dz = 0.0 if self._hold_z else float(self._goal.point.z - pos.z)

        dist_xy = math.hypot(dx, dy)
        dist = math.sqrt(dx * dx + dy * dy + dz * dz)

        msg = Vector3Stamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self._goal.header.frame_id or self._odom.header.frame_id or 'map'

        if dist_xy <= self._goal_radius and abs(dz) <= self._goal_radius:
            msg.vector.x = 0.0
            msg.vector.y = 0.0
            msg.vector.z = 0.0
            self._pub.publish(msg)
            return

        if dist < 1e-6:
            self._pub.publish(msg)
            return

        speed = min(self._max_speed, max(0.05, dist))
        msg.vector.x = speed * dx / dist
        msg.vector.y = speed * dy / dist
        msg.vector.z = speed * dz / dist
        msg.vector.z = max(-self._max_vertical_speed, min(self._max_vertical_speed, msg.vector.z))
        self._pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = WaypointVelocityNode()
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

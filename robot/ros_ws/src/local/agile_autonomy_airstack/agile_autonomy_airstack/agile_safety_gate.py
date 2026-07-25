import json

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, String

from agile_autonomy_airstack.status import make_status


def _now_sec(node):
    return node.get_clock().now().nanoseconds * 1e-9


class AgileSafetyGate(Node):
    def __init__(self):
        super().__init__("agile_safety_gate")

        self.require_depth = bool(self.declare_parameter("require_depth", True).value)
        self.require_odom = bool(self.declare_parameter("require_odom", True).value)
        self.min_policy_rate_hz = float(
            self.declare_parameter("min_policy_rate_hz", 10.0).value
        )
        self.input_timeout_sec = float(
            self.declare_parameter("input_timeout_sec", 0.5).value
        )
        self.fail_closed = bool(self.declare_parameter("fail_closed", True).value)
        self.debug = bool(self.declare_parameter("debug", False).value)

        self.external_enable = not self.fail_closed
        self.last_odom_time = None
        self.last_depth_time = None
        self.last_policy_status_time = None
        self.last_adapter_status_time = None
        self.last_policy_status = {}
        self.last_adapter_status = {}
        self.current_gate = False

        sensor_qos = QoSProfile(depth=1)
        sensor_qos.reliability = ReliabilityPolicy.BEST_EFFORT

        self.policy_status_sub = self.create_subscription(
            String, "policy_status", self._policy_status_callback, 10
        )
        self.adapter_status_sub = self.create_subscription(
            String, "adapter_status", self._adapter_status_callback, 10
        )
        self.odom_sub = self.create_subscription(
            Odometry, "odometry", self._odom_callback, sensor_qos
        )
        self.depth_sub = self.create_subscription(
            Image, "depth", self._depth_callback, sensor_qos
        )
        self.external_enable_sub = self.create_subscription(
            Bool, "external_enable", self._external_enable_callback, 1
        )

        self.policy_enable_pub = self.create_publisher(Bool, "policy_enable", 1)
        self.adapter_enable_pub = self.create_publisher(Bool, "adapter_enable", 1)
        self.status_pub = self.create_publisher(String, "status", 10)

        self.timer = self.create_timer(0.1, self._timer_callback)
        self.get_logger().info(f"Agile safety gate started, fail_closed={self.fail_closed}")

    def _parse_status(self, msg):
        try:
            return json.loads(msg.data)
        except json.JSONDecodeError:
            return {"raw": msg.data}

    def _policy_status_callback(self, msg):
        self.last_policy_status_time = _now_sec(self)
        self.last_policy_status = self._parse_status(msg)

    def _adapter_status_callback(self, msg):
        self.last_adapter_status_time = _now_sec(self)
        self.last_adapter_status = self._parse_status(msg)

    def _odom_callback(self, msg):
        self.last_odom_time = _now_sec(self)

    def _depth_callback(self, msg):
        self.last_depth_time = _now_sec(self)

    def _external_enable_callback(self, msg):
        self.external_enable = bool(msg.data)

    def _age(self, stamp_sec):
        if stamp_sec is None:
            return None
        return max(0.0, _now_sec(self) - stamp_sec)

    def _fresh(self, stamp_sec):
        return stamp_sec is not None and self._age(stamp_sec) <= self.input_timeout_sec

    def _policy_status_rate_ok(self):
        if self.last_policy_status_time is None or self.min_policy_rate_hz <= 0.0:
            return False
        max_age = 2.0 / self.min_policy_rate_hz
        return self._age(self.last_policy_status_time) <= max_age

    def _compute_gate(self):
        if not self.external_enable:
            return False, "external_enable_false"
        if self.require_odom and not self._fresh(self.last_odom_time):
            return False, "odometry_unavailable_or_stale"
        if self.require_depth and not self._fresh(self.last_depth_time):
            return False, "depth_unavailable_or_stale"
        return True, "enabled"

    def _timer_callback(self):
        gate, reason = self._compute_gate()
        self.current_gate = gate

        enable_msg = Bool()
        enable_msg.data = gate
        self.policy_enable_pub.publish(enable_msg)
        self.adapter_enable_pub.publish(enable_msg)

        status = String()
        status.data = make_status(
            gate,
            gate and self._policy_status_rate_ok(),
            reason,
            external_enable=self.external_enable,
            odom_age_sec=self._age(self.last_odom_time),
            depth_age_sec=self._age(self.last_depth_time),
            policy_status_age_sec=self._age(self.last_policy_status_time),
            adapter_status_age_sec=self._age(self.last_adapter_status_time),
            policy_status_rate_ok=self._policy_status_rate_ok(),
            min_policy_rate_hz=self.min_policy_rate_hz,
            policy_status=self.last_policy_status,
            adapter_status=self.last_adapter_status,
            require_depth=self.require_depth,
            require_odom=self.require_odom,
            fail_closed=self.fail_closed,
        )
        self.status_pub.publish(status)


def main(args=None):
    rclpy.init(args=args)
    node = AgileSafetyGate()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

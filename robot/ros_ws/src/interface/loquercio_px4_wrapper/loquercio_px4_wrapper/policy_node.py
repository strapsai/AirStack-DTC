from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import numpy as np
from geometry_msgs.msg import PoseStamped, Vector3Stamped
from nav_msgs.msg import Odometry, Path as NavPath
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray

from loquercio_px4_wrapper.inference import (
    TensorFlowLoquercioBackend,
    TfliteLoquercioBackend,
)
from loquercio_px4_wrapper.model import LoquercioModelConfig


Vector3 = tuple[float, float, float]
Matrix3 = tuple[Vector3, Vector3, Vector3]


def _norm(v: Vector3) -> float:
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def _limit_norm(v: Vector3, max_norm: float) -> Vector3:
    n = _norm(v)
    if max_norm <= 0.0 or n <= max_norm or n < 1e-9:
        return v
    scale = max_norm / n
    return (v[0] * scale, v[1] * scale, v[2] * scale)


def _mat_vec(m: Matrix3, v: Vector3) -> Vector3:
    return (
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    )


def _transpose_mat_vec(m: Matrix3, v: Vector3) -> Vector3:
    return (
        m[0][0] * v[0] + m[1][0] * v[1] + m[2][0] * v[2],
        m[0][1] * v[0] + m[1][1] * v[1] + m[2][1] * v[2],
        m[0][2] * v[0] + m[1][2] * v[1] + m[2][2] * v[2],
    )


def _mat_mul(a: Matrix3, b: Matrix3) -> Matrix3:
    cols = (
        (b[0][0], b[1][0], b[2][0]),
        (b[0][1], b[1][1], b[2][1]),
        (b[0][2], b[1][2], b[2][2]),
    )
    rows = []
    for row in a:
        rows.append(tuple(row[0] * col[0] + row[1] * col[1] + row[2] * col[2] for col in cols))
    return tuple(rows)  # type: ignore[return-value]


def _quaternion_xyzw_to_matrix(x: float, y: float, z: float, w: float) -> Matrix3:
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n < 1e-9:
        raise ValueError('zero-norm quaternion')
    w, x, y, z = w / n, x / n, y / n, z / n
    return (
        (
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - z * w),
            2.0 * (x * z + y * w),
        ),
        (
            2.0 * (x * y + z * w),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - x * w),
        ),
        (
            2.0 * (x * z - y * w),
            2.0 * (y * z + x * w),
            1.0 - 2.0 * (x * x + y * y),
        ),
    )


def _pitch_rotation(rad: float) -> Matrix3:
    c = math.cos(rad)
    s = math.sin(rad)
    return ((c, 0.0, s), (0.0, 1.0, 0.0), (-s, 0.0, c))


class LoquercioPolicy(Node):
    """Run Loquercio agile_autonomy inference and publish AirStack commands."""

    def __init__(self):
        super().__init__('loquercio_policy')

        self.declare_parameter('checkpoint_path', '/airlab-storage/chiron/models/loquercio/ckpt-50')
        self.declare_parameter('backend', 'tensorflow')
        self.declare_parameter('tflite_path', '/airlab-storage/chiron/models/loquercio/ckpt-50_float32.tflite')
        self.declare_parameter('depth_topic', 'front_stereo/depth')
        self.declare_parameter('depth_reliability', 'reliable')
        self.declare_parameter('odom_topic', 'odometry')
        self.declare_parameter('target_velocity_topic', 'loquercio/target_velocity')
        self.declare_parameter('goal_topic', 'loquercio/goal')
        self.declare_parameter('accel_output_topic', 'loquercio/accel_cmd')
        self.declare_parameter('raw_prediction_topic', 'loquercio/raw_prediction')
        self.declare_parameter('selected_trajectory_topic', 'loquercio/selected_trajectory')
        self.declare_parameter('debug_depth_topic', 'loquercio/debug_depth_input')
        self.declare_parameter('image_width', 224)
        self.declare_parameter('image_height', 224)
        self.declare_parameter('out_seq_len', 10)
        self.declare_parameter('modes', 3)
        self.declare_parameter('lookahead_step', 4)
        self.declare_parameter('target_timeout_s', 0.5)
        self.declare_parameter('default_target_speed_mps', 2.0)
        self.declare_parameter('max_net_accel_mps2', 3.0)
        self.declare_parameter('camera_pitch_deg', 0.0)
        self.declare_parameter('odom_velocity_is_world_frame', True)
        self.declare_parameter('odom_angular_velocity_is_world_frame', False)
        self.declare_parameter('frame_id', 'map')
        self.declare_parameter('debug_depth_max_m', 20.0)
        self.declare_parameter('min_inference_period_s', 0.0)

        self._config = LoquercioModelConfig(
            img_width=int(self.get_parameter('image_width').value),
            img_height=int(self.get_parameter('image_height').value),
            out_seq_len=int(self.get_parameter('out_seq_len').value),
            modes=int(self.get_parameter('modes').value),
        )
        backend = str(self.get_parameter('backend').value).strip().lower()
        if backend in ('tensorflow', 'tf', ''):
            self._backend = TensorFlowLoquercioBackend(
                str(self.get_parameter('checkpoint_path').value),
                self._config,
            )
            backend_label = 'tensorflow'
            model_label = self._backend.checkpoint_prefix
        elif backend in ('tflite', 'lite'):
            self._backend = TfliteLoquercioBackend(
                str(self.get_parameter('tflite_path').value),
                self._config,
            )
            backend_label = 'tflite'
            model_label = self._backend.model_path
        else:
            raise ValueError(f"Unsupported Loquercio backend: {backend!r}")

        self._target_timeout_s = float(self.get_parameter('target_timeout_s').value)
        self._default_target_speed = float(self.get_parameter('default_target_speed_mps').value)
        self._max_accel = float(self.get_parameter('max_net_accel_mps2').value)
        self._lookahead_step = max(0, min(int(self.get_parameter('lookahead_step').value), self._config.out_seq_len - 1))
        self._camera_pitch = math.radians(float(self.get_parameter('camera_pitch_deg').value))
        self._odom_velocity_is_world = bool(self.get_parameter('odom_velocity_is_world_frame').value)
        self._odom_angular_is_world = bool(self.get_parameter('odom_angular_velocity_is_world_frame').value)
        self._frame_id = str(self.get_parameter('frame_id').value)
        self._debug_depth_max_m = max(0.1, float(self.get_parameter('debug_depth_max_m').value))
        self._min_inference_period_s = max(0.0, float(self.get_parameter('min_inference_period_s').value))

        self._position_enu: Optional[Vector3] = None
        self._velocity_enu: Optional[Vector3] = None
        self._angular_velocity_world_or_body: Optional[Vector3] = None
        self._rotation_enu_cam: Optional[Matrix3] = None
        self._target_velocity_enu: Optional[Vector3] = None
        self._target_velocity_time = None
        self._goal_enu: Optional[Vector3] = None
        self._last_inference_time = None

        self._accel_pub = self.create_publisher(
            Vector3Stamped,
            str(self.get_parameter('accel_output_topic').value),
            1,
        )
        self._raw_pub = self.create_publisher(
            Float32MultiArray,
            str(self.get_parameter('raw_prediction_topic').value),
            1,
        )
        self._path_pub = self.create_publisher(
            NavPath,
            str(self.get_parameter('selected_trajectory_topic').value),
            1,
        )
        self._debug_depth_pub = self.create_publisher(
            Image,
            str(self.get_parameter('debug_depth_topic').value),
            1,
        )

        self.create_subscription(
            Odometry,
            str(self.get_parameter('odom_topic').value),
            self._on_odom,
            QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT),
        )
        self.create_subscription(
            Vector3Stamped,
            str(self.get_parameter('target_velocity_topic').value),
            self._on_target_velocity,
            1,
        )
        self.create_subscription(
            PoseStamped,
            str(self.get_parameter('goal_topic').value),
            self._on_goal,
            1,
        )
        self.create_subscription(
            Image,
            str(self.get_parameter('depth_topic').value),
            self._on_depth,
            self._make_depth_qos(str(self.get_parameter('depth_reliability').value)),
        )

        self.get_logger().info(
            'Loquercio policy loaded '
            f'{model_label} using backend={backend_label}; '
            f'publishing accel on {self.get_parameter("accel_output_topic").value}'
        )

    def _make_depth_qos(self, reliability: str) -> QoSProfile:
        reliability = reliability.strip().lower()
        if reliability in ('best_effort', 'besteffort', 'best-effort'):
            policy = ReliabilityPolicy.BEST_EFFORT
        else:
            policy = ReliabilityPolicy.RELIABLE
        return QoSProfile(depth=10, reliability=policy)

    def _on_odom(self, msg: Odometry) -> None:
        q = msg.pose.pose.orientation
        try:
            rotation_enu_flu = _quaternion_xyzw_to_matrix(q.x, q.y, q.z, q.w)
        except ValueError as exc:
            self.get_logger().warn(f'Ignoring invalid odometry quaternion: {exc}', throttle_duration_sec=1.0)
            return

        body_to_cam = _pitch_rotation(-self._camera_pitch)
        self._rotation_enu_cam = _mat_mul(rotation_enu_flu, body_to_cam)
        self._position_enu = (
            float(msg.pose.pose.position.x),
            float(msg.pose.pose.position.y),
            float(msg.pose.pose.position.z),
        )
        velocity = (
            float(msg.twist.twist.linear.x),
            float(msg.twist.twist.linear.y),
            float(msg.twist.twist.linear.z),
        )
        self._velocity_enu = velocity if self._odom_velocity_is_world else _mat_vec(rotation_enu_flu, velocity)
        self._angular_velocity_world_or_body = (
            float(msg.twist.twist.angular.x),
            float(msg.twist.twist.angular.y),
            float(msg.twist.twist.angular.z),
        )

    def _on_target_velocity(self, msg: Vector3Stamped) -> None:
        self._target_velocity_enu = (
            float(msg.vector.x),
            float(msg.vector.y),
            float(msg.vector.z),
        )
        self._target_velocity_time = self.get_clock().now()

    def _on_goal(self, msg: PoseStamped) -> None:
        self._goal_enu = (
            float(msg.pose.position.x),
            float(msg.pose.position.y),
            float(msg.pose.position.z),
        )

    def _on_depth(self, msg: Image) -> None:
        if self._rotation_enu_cam is None or self._velocity_enu is None or self._position_enu is None:
            self.get_logger().warn('Depth received before odometry; waiting.', throttle_duration_sec=1.0)
            return
        now = self.get_clock().now()
        if self._last_inference_time is not None and self._min_inference_period_s > 0.0:
            age = (now - self._last_inference_time).nanoseconds * 1e-9
            if age < self._min_inference_period_s:
                return
        depth_model, debug_depth = self._depth_to_model_input(msg)
        if depth_model is None:
            return

        imu = self._state_to_model_input()
        alphas, trajectories = self._backend.infer(depth_model, imu)
        self._last_inference_time = now
        self._publish_debug_depth(msg, debug_depth)
        self._publish_raw_prediction(alphas, trajectories)
        self._publish_selected_path_and_accel(msg.header.stamp, trajectories[0])

    def _state_to_model_input(self) -> np.ndarray:
        assert self._rotation_enu_cam is not None
        assert self._velocity_enu is not None
        assert self._position_enu is not None
        rotation = self._rotation_enu_cam
        local_velocity = _transpose_mat_vec(rotation, self._velocity_enu)

        angular = self._angular_velocity_world_or_body or (0.0, 0.0, 0.0)
        local_angular = _transpose_mat_vec(rotation, angular) if self._odom_angular_is_world else angular
        goal_world = self._target_direction_world()
        local_goal = _transpose_mat_vec(rotation, goal_world)

        state = [
            self._position_enu[0],
            self._position_enu[1],
            self._position_enu[2],
            rotation[0][0],
            rotation[0][1],
            rotation[0][2],
            rotation[1][0],
            rotation[1][1],
            rotation[1][2],
            rotation[2][0],
            rotation[2][1],
            rotation[2][2],
            local_velocity[0],
            local_velocity[1],
            local_velocity[2],
            local_angular[0],
            local_angular[1],
            local_angular[2],
            local_goal[0],
            local_goal[1],
            local_goal[2],
        ]
        return np.asarray(state, dtype=np.float32).reshape((1, self._config.seq_len, self._config.raw_state_dim))

    def _target_direction_world(self) -> Vector3:
        now = self.get_clock().now()
        if self._target_velocity_enu is not None and self._target_velocity_time is not None:
            age = (now - self._target_velocity_time).nanoseconds * 1e-9
            if age <= self._target_timeout_s and _norm(self._target_velocity_enu) > 1e-6:
                n = _norm(self._target_velocity_enu)
                return (
                    self._target_velocity_enu[0] / n,
                    self._target_velocity_enu[1] / n,
                    self._target_velocity_enu[2] / n,
                )
        if self._goal_enu is not None and self._position_enu is not None:
            delta = (
                self._goal_enu[0] - self._position_enu[0],
                self._goal_enu[1] - self._position_enu[1],
                self._goal_enu[2] - self._position_enu[2],
            )
            n = _norm(delta)
            if n > 1e-6:
                return (delta[0] / n, delta[1] / n, delta[2] / n)
        assert self._rotation_enu_cam is not None
        return (
            self._rotation_enu_cam[0][0],
            self._rotation_enu_cam[1][0],
            self._rotation_enu_cam[2][0],
        )

    def _depth_to_model_input(self, msg: Image) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if msg.is_bigendian:
            self.get_logger().warn('Big-endian depth images are not supported.', throttle_duration_sec=1.0)
            return None, None

        encoding = msg.encoding.upper()
        if encoding in ('32FC1', '32FC'):
            dtype = np.float32
            item_size = 4
            scale_to_mm = 1000.0
        elif encoding in ('16UC1', 'MONO16'):
            dtype = np.uint16
            item_size = 2
            scale_to_mm = 1.0
        else:
            self.get_logger().warn(f'Unsupported depth encoding: {msg.encoding}', throttle_duration_sec=1.0)
            return None, None

        step = int(msg.step) if msg.step else int(msg.width) * item_size
        elems_per_row = step // item_size
        data = np.frombuffer(msg.data, dtype=dtype)
        try:
            depth = data.reshape((int(msg.height), elems_per_row))[:, : int(msg.width)].astype(np.float32)
        except ValueError:
            self.get_logger().warn('Depth image buffer shape does not match metadata.', throttle_duration_sec=1.0)
            return None, None
        depth_mm = np.nan_to_num(depth * scale_to_mm, nan=20000.0, posinf=20000.0, neginf=0.0)
        depth_mm = np.clip(depth_mm, 0.0, 20000.0)
        resized = self._resize_nearest(depth_mm, self._config.img_height, self._config.img_width)
        normalized = resized / 80.0
        model_depth = np.repeat(normalized[..., None], 3, axis=-1)
        model_depth = model_depth.reshape(
            (1, self._config.seq_len, self._config.img_height, self._config.img_width, 3)
        ).astype(np.float32)
        debug = np.clip((resized / 1000.0) / self._debug_depth_max_m * 255.0, 0.0, 255.0).astype(np.uint8)
        return model_depth, debug

    def _resize_nearest(self, image: np.ndarray, height: int, width: int) -> np.ndarray:
        src_h, src_w = image.shape[:2]
        if src_h == height and src_w == width:
            return image
        ys = np.linspace(0, src_h - 1, height).astype(np.int64)
        xs = np.linspace(0, src_w - 1, width).astype(np.int64)
        return image[ys[:, None], xs[None, :]]

    def _publish_debug_depth(self, src: Image, debug: Optional[np.ndarray]) -> None:
        if debug is None:
            return
        msg = Image()
        msg.header = src.header
        msg.height = int(debug.shape[0])
        msg.width = int(debug.shape[1])
        msg.encoding = 'mono8'
        msg.is_bigendian = False
        msg.step = int(debug.shape[1])
        msg.data = debug.tobytes()
        self._debug_depth_pub.publish(msg)

    def _publish_raw_prediction(self, alphas: np.ndarray, trajectories: np.ndarray) -> None:
        msg = Float32MultiArray()
        payload = [
            float(self._config.modes),
            float(self._config.out_seq_len),
            float(self._config.state_dim),
        ]
        for alpha, traj in zip(alphas, trajectories):
            payload.append(float(alpha))
            payload.extend(float(x) for x in traj.reshape((-1,)))
        msg.data = payload
        self._raw_pub.publish(msg)

    def _publish_selected_path_and_accel(self, stamp, flat_trajectory: np.ndarray) -> None:
        assert self._rotation_enu_cam is not None
        assert self._velocity_enu is not None
        assert self._position_enu is not None

        local_xyz = flat_trajectory.reshape((self._config.state_dim, self._config.out_seq_len))
        path = NavPath()
        path.header.stamp = stamp
        path.header.frame_id = self._frame_id
        world_points: list[Vector3] = []
        for idx in range(self._config.out_seq_len):
            local_point = (
                float(local_xyz[0, idx]),
                float(local_xyz[1, idx]),
                float(local_xyz[2, idx]),
            )
            delta_world = _mat_vec(self._rotation_enu_cam, local_point)
            world = (
                self._position_enu[0] + delta_world[0],
                self._position_enu[1] + delta_world[1],
                self._position_enu[2] + delta_world[2],
            )
            world_points.append(world)
            pose = PoseStamped()
            pose.header = path.header
            pose.pose.position.x = world[0]
            pose.pose.position.y = world[1]
            pose.pose.position.z = world[2]
            pose.pose.orientation.w = 1.0
            path.poses.append(pose)
        self._path_pub.publish(path)

        lookahead = self._lookahead_step
        target_delta = (
            world_points[lookahead][0] - self._position_enu[0],
            world_points[lookahead][1] - self._position_enu[1],
            world_points[lookahead][2] - self._position_enu[2],
        )
        dt = max(0.1, 0.1 * float(lookahead + 1))
        accel = (
            2.0 * (target_delta[0] - self._velocity_enu[0] * dt) / (dt * dt),
            2.0 * (target_delta[1] - self._velocity_enu[1] * dt) / (dt * dt),
            2.0 * (target_delta[2] - self._velocity_enu[2] * dt) / (dt * dt),
        )
        accel = _limit_norm(accel, self._max_accel)
        msg = Vector3Stamped()
        msg.header.stamp = stamp
        msg.header.frame_id = self._frame_id
        msg.vector.x = accel[0]
        msg.vector.y = accel[1]
        msg.vector.z = accel[2]
        self._accel_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = LoquercioPolicy()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

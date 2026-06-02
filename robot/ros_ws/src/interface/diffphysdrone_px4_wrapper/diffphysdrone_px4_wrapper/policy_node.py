import math
from pathlib import Path
from typing import Optional, Tuple

from geometry_msgs.msg import Vector3Stamped
from nav_msgs.msg import Odometry
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
import torch
from torch.nn import functional as F

from diffphysdrone_px4_wrapper.accel_to_attitude import (
    ConversionError,
    Matrix3,
    Vector3,
    diffphys_raw_action_to_net_acceleration,
    diffphys_yaw_frame_from_rotation,
    quaternion_wxyz_to_matrix,
)
from diffphysdrone_px4_wrapper.inference import (
    OnnxPolicyBackend,
    PolicyBackend,
    TorchPolicyBackend,
    checkpoint_dimensions,
)


def _dot(a: Vector3, b: Vector3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _norm(v: Vector3) -> float:
    return math.sqrt(_dot(v, v))


def _mat_vec(m: Matrix3, v: Vector3) -> Vector3:
    return (
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    )


def _world_to_frame(v: Vector3, frame_enu_local: Matrix3) -> Vector3:
    return (
        v[0] * frame_enu_local[0][0] + v[1] * frame_enu_local[1][0] + v[2] * frame_enu_local[2][0],
        v[0] * frame_enu_local[0][1] + v[1] * frame_enu_local[1][1] + v[2] * frame_enu_local[2][1],
        v[0] * frame_enu_local[0][2] + v[1] * frame_enu_local[1][2] + v[2] * frame_enu_local[2][2],
    )


def _limit_norm(v: Vector3, max_norm: float) -> Vector3:
    n = _norm(v)
    if max_norm <= 0.0 or n <= max_norm or n < 1e-9:
        return v
    scale = max_norm / n
    return (v[0] * scale, v[1] * scale, v[2] * scale)


class DiffPhysDronePolicy(Node):
    """Run the DiffPhysDrone checkpoint and publish policy actions."""

    def __init__(self):
        super().__init__('diffphysdrone_policy')

        self.declare_parameter(
            'checkpoint_path',
            '/airlab-storage/chiron/models/diffphysdrone/checkpoint0004.pth',
        )
        self.declare_parameter('depth_topic', 'front_stereo/depth')
        self.declare_parameter('odom_topic', 'odometry')
        self.declare_parameter('target_velocity_topic', 'diffphysdrone/target_velocity')
        self.declare_parameter('accel_output_topic', 'diffphysdrone/accel_cmd')
        self.declare_parameter('raw_action_output_topic', 'diffphysdrone/raw_action')
        self.declare_parameter('debug_depth_topic', 'diffphysdrone/debug_depth_input')
        self.declare_parameter('device', 'cpu')
        self.declare_parameter('backend', 'torch')
        self.declare_parameter('onnx_path', '')
        self.declare_parameter('onnx_provider', '')
        self.declare_parameter('target_speed_mps', 3.0)
        self.declare_parameter('target_timeout_s', 0.5)
        self.declare_parameter('margin_m', 0.2)
        self.declare_parameter('image_width', 64)
        self.declare_parameter('image_height', 48)
        self.declare_parameter('gravity_mps2', 9.80665)
        self.declare_parameter('thrust_est_error', 1.0)
        self.declare_parameter('raw_action_layout', 'interleaved')
        self.declare_parameter('odom_velocity_is_world_frame', True)
        self.declare_parameter('depth_reliability', 'reliable')

        self._target_speed = float(self.get_parameter('target_speed_mps').value)
        self._target_timeout_s = float(self.get_parameter('target_timeout_s').value)
        self._margin = float(self.get_parameter('margin_m').value)
        self._image_width = int(self.get_parameter('image_width').value)
        self._image_height = int(self.get_parameter('image_height').value)
        self._gravity = float(self.get_parameter('gravity_mps2').value)
        self._thrust_est_error = float(self.get_parameter('thrust_est_error').value)
        self._raw_action_layout = str(self.get_parameter('raw_action_layout').value)
        self._odom_velocity_is_world_frame = bool(
            self.get_parameter('odom_velocity_is_world_frame').value
        )

        self._rotation_enu_flu: Optional[Matrix3] = None
        self._diffphys_frame: Optional[Matrix3] = None
        self._velocity_enu: Optional[Vector3] = None
        self._target_velocity_enu: Optional[Vector3] = None
        self._target_velocity_time = None
        self._hidden: Optional[torch.Tensor] = None
        self._odom_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self._depth_qos = self._make_depth_qos(str(self.get_parameter('depth_reliability').value))

        self._device = self._select_device(str(self.get_parameter('device').value))
        self._backend, self._dim_obs = self._load_backend(
            Path(str(self.get_parameter('checkpoint_path').value))
        )

        self._accel_pub = self.create_publisher(
            Vector3Stamped,
            str(self.get_parameter('accel_output_topic').value),
            1,
        )
        self._raw_pub = self.create_publisher(
            Float32MultiArray,
            str(self.get_parameter('raw_action_output_topic').value),
            1,
        )
        self._debug_depth_pub = self.create_publisher(
            Image,
            str(self.get_parameter('debug_depth_topic').value),
            1,
        )
        self.create_subscription(
            Image,
            str(self.get_parameter('depth_topic').value),
            self._on_depth,
            self._depth_qos,
        )
        self.create_subscription(
            Odometry,
            str(self.get_parameter('odom_topic').value),
            self._on_odom,
            self._odom_qos,
        )
        self.create_subscription(
            Vector3Stamped,
            str(self.get_parameter('target_velocity_topic').value),
            self._on_target_velocity,
            1,
        )

        self.get_logger().info(
            'DiffPhysDrone policy loaded '
            f'{self.get_parameter("checkpoint_path").value} '
            f'using backend={self.get_parameter("backend").value} '
            f'on {self._device} with state dim {self._dim_obs}'
        )


    def _make_depth_qos(self, reliability: str) -> QoSProfile:
        reliability = reliability.strip().lower()
        if reliability in ('best_effort', 'besteffort', 'best-effort'):
            policy = ReliabilityPolicy.BEST_EFFORT
        elif reliability in ('reliable', ''):
            policy = ReliabilityPolicy.RELIABLE
        else:
            self.get_logger().warn(
                f'Unknown depth_reliability={reliability!r}; using reliable.'
            )
            policy = ReliabilityPolicy.RELIABLE
        return QoSProfile(depth=10, reliability=policy)

    def _select_device(self, requested: str) -> torch.device:
        if requested.startswith('cuda') and not torch.cuda.is_available():
            self.get_logger().warn('CUDA requested but unavailable; using CPU.')
            requested = 'cpu'
        return torch.device(requested)

    def _load_backend(self, checkpoint_path: Path) -> Tuple[PolicyBackend, int]:
        backend = str(self.get_parameter('backend').value).strip().lower()
        if backend in ('torch', 'pytorch', ''):
            policy_backend = TorchPolicyBackend(checkpoint_path, self._device)
            return policy_backend, policy_backend.dim_obs
        if backend == 'onnx':
            onnx_path = str(self.get_parameter('onnx_path').value).strip()
            if not onnx_path:
                raise ValueError("backend='onnx' requires onnx_path")
            dim_obs, dim_action = checkpoint_dimensions(checkpoint_path)
            if dim_action != 6:
                raise ValueError(f'Expected a 6-action DiffPhysDrone checkpoint, got {dim_action}')
            provider = str(self.get_parameter('onnx_provider').value).strip()
            providers = [provider] if provider else None
            policy_backend = OnnxPolicyBackend(Path(onnx_path), dim_obs, providers=providers)
            return policy_backend, policy_backend.dim_obs
        raise ValueError(f"Unknown DiffPhysDrone policy backend: {backend!r}")

    def _now(self):
        return self.get_clock().now()

    def _on_odom(self, msg: Odometry):
        q = msg.pose.pose.orientation
        q_wxyz = (q.w, q.x, q.y, q.z)
        try:
            rotation = quaternion_wxyz_to_matrix(q_wxyz)
            self._rotation_enu_flu = rotation
            self._diffphys_frame = diffphys_yaw_frame_from_rotation(rotation)
        except ConversionError as exc:
            self.get_logger().warn(f'Ignoring invalid odometry quaternion: {exc}', throttle_duration_sec=1.0)
            return

        velocity = (
            float(msg.twist.twist.linear.x),
            float(msg.twist.twist.linear.y),
            float(msg.twist.twist.linear.z),
        )
        self._velocity_enu = velocity if self._odom_velocity_is_world_frame else _mat_vec(rotation, velocity)

    def _on_target_velocity(self, msg: Vector3Stamped):
        self._target_velocity_enu = (
            float(msg.vector.x),
            float(msg.vector.y),
            float(msg.vector.z),
        )
        self._target_velocity_time = self._now()

    def _target_velocity(self, diffphys_frame: Matrix3) -> Vector3:
        if self._target_velocity_enu is not None and self._target_velocity_time is not None:
            age = (self._now() - self._target_velocity_time).nanoseconds * 1e-9
            if age <= self._target_timeout_s:
                return _limit_norm(self._target_velocity_enu, self._target_speed)

        fwd = (diffphys_frame[0][0], diffphys_frame[1][0], diffphys_frame[2][0])
        return (
            fwd[0] * self._target_speed,
            fwd[1] * self._target_speed,
            fwd[2] * self._target_speed,
        )

    def _depth_to_model_input(self, msg: Image) -> Optional[torch.Tensor]:
        if msg.is_bigendian:
            self.get_logger().warn('Big-endian depth images are not supported.', throttle_duration_sec=1.0)
            return None

        encoding = msg.encoding.upper()
        if encoding in ('32FC1', '32FC'):
            dtype = torch.float32
            item_size = 4
            scale = 1.0
        elif encoding in ('16UC1', 'MONO16'):
            dtype = torch.uint16
            item_size = 2
            scale = 0.001
        else:
            self.get_logger().warn(f'Unsupported depth encoding: {msg.encoding}', throttle_duration_sec=1.0)
            return None

        step = int(msg.step) if msg.step else int(msg.width) * item_size
        step_elems = step // item_size
        required = int(msg.height) * step_elems
        raw = torch.frombuffer(bytearray(msg.data), dtype=dtype)
        if raw.numel() < required or step_elems < int(msg.width):
            self.get_logger().warn('Depth image buffer is smaller than expected.', throttle_duration_sec=1.0)
            return None

        depth = raw[:required].reshape(int(msg.height), step_elems)[:, :int(msg.width)]
        depth = depth.to(dtype=torch.float32, device=self._device) * scale
        depth = torch.nan_to_num(depth, nan=24.0, posinf=24.0, neginf=0.3)
        depth = depth[None, None]
        if depth.shape[-2:] != (self._image_height, self._image_width):
            depth = F.interpolate(
                depth,
                size=(self._image_height, self._image_width),
                mode='nearest',
            )
        depth = torch.clamp(depth, 0.3, 24.0)
        return F.max_pool2d(3.0 / depth - 0.6, 4, 4)


    def _publish_debug_depth(self, depth: torch.Tensor, header) -> None:
        if self._debug_depth_pub.get_subscription_count() == 0:
            return

        image = depth[0, 0].detach().float().cpu()
        image = torch.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0)
        image_min = float(image.min())
        image_max = float(image.max())
        if image_max > image_min + 1e-6:
            image = (image - image_min) / (image_max - image_min)
        else:
            image = torch.zeros_like(image)

        image_u8 = (image * 255.0).clamp(0, 255).to(torch.uint8).contiguous()
        msg = Image()
        msg.header = header
        msg.height = int(image_u8.shape[0])
        msg.width = int(image_u8.shape[1])
        msg.encoding = 'mono8'
        msg.is_bigendian = False
        msg.step = msg.width
        msg.data = image_u8.numpy().tobytes()
        self._debug_depth_pub.publish(msg)

    def _state_tensor(self) -> Optional[torch.Tensor]:
        if self._rotation_enu_flu is None or self._diffphys_frame is None or self._velocity_enu is None:
            return None

        local_target = _world_to_frame(self._target_velocity(self._diffphys_frame), self._diffphys_frame)
        attitude_feature = (
            self._rotation_enu_flu[2][0],
            self._rotation_enu_flu[2][1],
            self._rotation_enu_flu[2][2],
        )

        if self._dim_obs == 10:
            local_velocity = _world_to_frame(self._velocity_enu, self._diffphys_frame)
            values = local_velocity + local_target + attitude_feature + (self._margin,)
        else:
            values = local_target + attitude_feature + (self._margin,)

        return torch.tensor(values, dtype=torch.float32, device=self._device)[None]

    def _on_depth(self, msg: Image):
        if self._diffphys_frame is None:
            self.get_logger().warn('Depth received before odometry; waiting.', throttle_duration_sec=1.0)
            return

        depth = self._depth_to_model_input(msg)
        state = self._state_tensor()
        if depth is None or state is None:
            return

        self._publish_debug_depth(depth, msg.header)

        with torch.inference_mode():
            action, self._hidden = self._backend.infer(depth, state)

        raw_action = [float(v) for v in action[0].detach().cpu().tolist()]
        raw_msg = Float32MultiArray()
        raw_msg.data = raw_action
        self._raw_pub.publish(raw_msg)

        try:
            accel = diffphys_raw_action_to_net_acceleration(
                raw_action,
                self._diffphys_frame,
                gravity_mps2=self._gravity,
                thrust_est_error=self._thrust_est_error,
                layout=self._raw_action_layout,
            )
        except ConversionError as exc:
            self.get_logger().warn(f'Policy produced invalid action: {exc}', throttle_duration_sec=1.0)
            return

        accel_msg = Vector3Stamped()
        accel_msg.header = msg.header
        accel_msg.vector.x = accel[0]
        accel_msg.vector.y = accel[1]
        accel_msg.vector.z = accel[2]
        self._accel_pub.publish(accel_msg)


def main(args=None):
    rclpy.init(args=args)
    node = DiffPhysDronePolicy()
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

"""Agile Flight (Loquercio / agile_autonomy) velocity-command policy wrapper.

Sibling of ``diffaero/diffaero_vel_core.py``: exposes the same
``compute(obs) -> cmd`` seam (``cmd.vel_cmd_enu``, ``cmd.vel_norm``,
``cmd.desired_yaw_enu``) so ``agile_velocity_commander.py`` can drive it exactly
like the DiffAero velocity commander drives ``DiffAeroVelPolicy``.

Unlike DiffAero (whose exported actor emits a velocity setpoint directly), the
Loquercio net outputs a *position* trajectory (``modes`` candidate curves of
``out_seq_len`` waypoints). A velocity setpoint that is dynamically feasible and
reliable in real flight is recovered by tracking the selected trajectory with the
acados RPG-MPC (``agile/mpc_acados.py``) and reading the velocity straight out of
the solved **MPC state** ``x = [px,py,pz, qw,qx,qy,qz, vx,vy,vz]``:

    vel_cmd_enu = mpc.solver.get(1, 'x')[7:10]   # world-ENU velocity, 0.1 s ahead

The flight controller then owns attitude (the commander publishes this velocity as
a TwistStamped setpoint). The MPC's attitude/thrust output (``u0``, ``q_pred``) is
intentionally discarded — velocity commands are far more reliable on hardware.

The heavy Loquercio TF/TFLite graph is reused from the ``loquercio_px4_wrapper``
package (a declared dependency) rather than duplicated; only the pure numpy glue
(depth->input, state->input, trajectory local->world) is ported here so the
``agile/`` subpackage stays self-contained like ``diffaero/``.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field

import numpy as np

from loquercio_px4_wrapper.inference import (
    TensorFlowLoquercioBackend,
    TfliteLoquercioBackend,
)
from loquercio_px4_wrapper.model import LoquercioModelConfig

from svg_ground_control.agile.mpc_acados import MPC, state_x0


# ---------------------------------------------------------------------------
# obs / cmd
# ---------------------------------------------------------------------------

@dataclass
class AgileObs:
    position_enu: np.ndarray                 # (3,) world ENU
    velocity_enu: np.ndarray                 # (3,) world ENU
    R_enu: np.ndarray                        # (3,3) body FLU -> world ENU
    goal_enu: np.ndarray                     # (3,) world ENU
    depth_planar: np.ndarray | None = None   # (H,W) metric depth (distance_to_image_plane), meters
    angular_velocity: np.ndarray = field(    # (3,) body rates (rad/s); optional
        default_factory=lambda: np.zeros(3))


@dataclass
class AgileVelCmd:
    vel_cmd_enu: np.ndarray         # (3,) world-frame velocity setpoint [m/s]
    vel_norm: float                 # magnitude of vel_cmd_enu
    desired_yaw_enu: float          # ENU yaw (rad, CCW from +x/East) of travel


# ---------------------------------------------------------------------------
# pure numpy helpers (ported from loquercio_px4_wrapper/policy_node.py)
# ---------------------------------------------------------------------------

def _pitch_rotation(rad: float) -> np.ndarray:
    c, s = math.cos(rad), math.sin(rad)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _resize_nearest(image: np.ndarray, height: int, width: int) -> np.ndarray:
    src_h, src_w = image.shape[:2]
    if src_h == height and src_w == width:
        return image
    ys = np.linspace(0, src_h - 1, height).astype(np.int64)
    xs = np.linspace(0, src_w - 1, width).astype(np.int64)
    return image[ys[:, None], xs[None, :]]


# ---------------------------------------------------------------------------
# policy
# ---------------------------------------------------------------------------

class AgileVelPolicy:
    """Loquercio net + acados MPC -> world-ENU velocity setpoint."""

    # The commander checks getattr(policy, 'planar', False) to decide goal-arrival
    # geometry. The MPC path controls full 3D, so this is never planar.
    planar = False

    def __init__(
        self,
        checkpoint_path: str,
        backend: str = 'tensorflow',
        tflite_path: str = '',
        img_width: int = 224,
        img_height: int = 224,
        out_seq_len: int = 10,
        modes: int = 3,
        camera_pitch_deg: float = 0.0,
        cruise_alt: float = 1.2,
        max_vel: float = 3.0,
        alt_hold: bool = True,
        lookahead_step: int = 4,
        depth_max_m: float = 20.0,
        mpc_dt_wp: float = 0.1,
    ):
        self._config = LoquercioModelConfig(
            img_width=int(img_width), img_height=int(img_height),
            out_seq_len=int(out_seq_len), modes=int(modes))

        backend = str(backend).strip().lower()
        if backend in ('tensorflow', 'tf', ''):
            self._backend = TensorFlowLoquercioBackend(checkpoint_path, self._config)
            self.model_label = self._backend.checkpoint_prefix
        elif backend in ('tflite', 'lite'):
            self._backend = TfliteLoquercioBackend(tflite_path, self._config)
            self.model_label = self._backend.model_path
        else:
            raise ValueError(f'Unsupported Agile/Loquercio backend: {backend!r}')

        self._camera_pitch = math.radians(float(camera_pitch_deg))
        self._body_to_cam = _pitch_rotation(-self._camera_pitch)
        self.cruise_alt = float(cruise_alt)
        self.max_vel = float(max_vel)
        self.alt_hold = bool(alt_hold)
        self._lookahead = max(0, min(int(lookahead_step), self._config.out_seq_len - 1))
        self._depth_max_m = max(0.1, float(depth_max_m))
        self._mpc_dt_wp = float(mpc_dt_wp)
        self._last_desired_yaw: float | None = None

        # acados MPC (compiles generated C on first build; identical to the
        # superfly loquercio_offboard MPC). Obstacle avoidance is left to the
        # policy, so the MPC only tracks (obstacles_xy_r=None each solve).
        self.mpc = MPC()

    def reset(self) -> None:
        self._last_desired_yaw = None

    # -- input builders ------------------------------------------------------

    def _depth_to_model_input(self, depth_planar: np.ndarray | None) -> np.ndarray:
        """Metric planar depth (H,W, meters) -> (1, seq, H, W, 3), matching the
        loquercio_px4_wrapper preprocessing (mm, clip 20 m, nearest resize, /80,
        3-channel tile). ``None`` -> an all-clear (far) frame."""
        H, W = self._config.img_height, self._config.img_width
        if depth_planar is None:
            resized = np.full((H, W), 20000.0, dtype=np.float32)
        else:
            depth_mm = np.nan_to_num(
                np.asarray(depth_planar, dtype=np.float32) * 1000.0,
                nan=20000.0, posinf=20000.0, neginf=0.0)
            depth_mm = np.clip(depth_mm, 0.0, 20000.0)
            resized = _resize_nearest(depth_mm, H, W)
        normalized = resized / 80.0
        model_depth = np.repeat(normalized[..., None], 3, axis=-1)
        return model_depth.reshape(
            (1, self._config.seq_len, H, W, 3)).astype(np.float32)

    def _state_to_model_input(self, obs: AgileObs, R_cam: np.ndarray) -> np.ndarray:
        """Build the 21-D IMU/state vector the net consumes (obs in the yaw-free
        camera frame): [pos(3), R_cam row-major(9), v_local(3), w_local(3),
        goal_dir_local(3)]."""
        v_local = R_cam.T @ np.asarray(obs.velocity_enu, dtype=np.float64)
        # bodyrates already in body frame -> leave as-is (matches wrapper default
        # odom_angular_velocity_is_world_frame=False).
        w_local = np.asarray(obs.angular_velocity, dtype=np.float64)
        goal_dir_world = self._goal_direction_world(obs, R_cam)
        goal_local = R_cam.T @ goal_dir_world
        pos = np.asarray(obs.position_enu, dtype=np.float64)
        state = np.concatenate([pos, R_cam.reshape(-1), v_local, w_local, goal_local])
        return state.astype(np.float32).reshape(
            (1, self._config.seq_len, self._config.raw_state_dim))

    def _goal_direction_world(self, obs: AgileObs, R_cam: np.ndarray) -> np.ndarray:
        rel = np.asarray(obs.goal_enu, dtype=np.float64) - np.asarray(
            obs.position_enu, dtype=np.float64)
        n = float(np.linalg.norm(rel))
        if n > 1e-6:
            return rel / n
        return R_cam[:, 0].copy()   # forward camera axis when already at goal

    # -- main ----------------------------------------------------------------

    def compute(self, obs: AgileObs) -> AgileVelCmd:
        R_enu = np.asarray(obs.R_enu, dtype=np.float64)
        R_cam = R_enu @ self._body_to_cam
        pos = np.asarray(obs.position_enu, dtype=np.float64)

        depth_model = self._depth_to_model_input(obs.depth_planar)
        imu = self._state_to_model_input(obs, R_cam)
        alphas, trajectories = self._backend.infer(depth_model, imu)

        # The backend sorts modes by |alpha| ascending; index 0 = lowest-cost
        # (best) trajectory, matching loquercio_px4_wrapper's selection.
        local_xyz = trajectories[0].reshape(
            (self._config.state_dim, self._config.out_seq_len))
        world_pts = [pos + R_cam @ local_xyz[:, i]
                     for i in range(self._config.out_seq_len)]

        # Heading reference for the MPC flatness map = direction toward a lookahead
        # waypoint (the net's intended travel direction).
        look = world_pts[self._lookahead] - pos
        yaw_des = (math.atan2(look[1], look[0]) if float(np.hypot(look[0], look[1])) > 1e-3
                   else self._yaw_from_R(R_enu))

        x0 = state_x0(pos, R_enu, np.asarray(obs.velocity_enu, dtype=np.float64))
        # Track only (policy owns avoidance). Read the velocity from the solved
        # MPC state one step ahead — this is the setpoint we command.
        self.mpc.compute(
            x0, world_pts, self.cruise_alt, yaw_des,
            dt_wp=self._mpc_dt_wp, max_vel=self.max_vel,
            obstacles_xy_r=None, alt_hold=self.alt_hold)
        x1 = np.asarray(self.mpc.solver.get(1, 'x'), dtype=np.float64)
        vel_cmd_enu = x1[7:10].copy()
        vel_norm = float(np.linalg.norm(vel_cmd_enu))

        # Desired yaw = travel direction; hold last heading when near-still so the
        # nose does not spin at hover/goal.
        if float(np.hypot(vel_cmd_enu[0], vel_cmd_enu[1])) >= 0.3:
            desired_yaw = math.atan2(vel_cmd_enu[1], vel_cmd_enu[0])
            self._last_desired_yaw = desired_yaw
        elif self._last_desired_yaw is not None:
            desired_yaw = self._last_desired_yaw
        else:
            desired_yaw = yaw_des

        return AgileVelCmd(vel_cmd_enu=vel_cmd_enu, vel_norm=vel_norm,
                           desired_yaw_enu=desired_yaw)

    @staticmethod
    def _yaw_from_R(R_enu: np.ndarray) -> float:
        fwd = R_enu[:, 0]
        return float(math.atan2(fwd[1], fwd[0]))

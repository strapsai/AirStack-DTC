import math
import unittest

from diffphysdrone_px4_wrapper.accel_to_attitude import (
    acceleration_to_attitude_thrust,
    diffphys_raw_action_to_net_acceleration,
    diffphys_yaw_frame_from_rotation,
    quaternion_wxyz_to_matrix,
)


class AccelToAttitudeTest(unittest.TestCase):

    def test_hover_is_identity_attitude_and_hover_thrust(self):
        command = acceleration_to_attitude_thrust((0.0, 0.0, 0.0), 0.0, 0.5)
        self.assertAlmostEqual(command.thrust, 0.5)
        self.assertAlmostEqual(abs(command.quaternion_wxyz[0]), 1.0)
        self.assertAlmostEqual(command.quaternion_wxyz[1], 0.0)
        self.assertAlmostEqual(command.quaternion_wxyz[2], 0.0)
        self.assertAlmostEqual(command.quaternion_wxyz[3], 0.0)

    def test_forward_acceleration_tilts_body_z_forward(self):
        command = acceleration_to_attitude_thrust((1.0, 0.0, 0.0), 0.0, 0.5)
        rotation = quaternion_wxyz_to_matrix(command.quaternion_wxyz)
        z_body_world = (rotation[0][2], rotation[1][2], rotation[2][2])
        self.assertGreater(z_body_world[0], 0.0)
        self.assertGreater(z_body_world[2], 0.9)

    def test_diffphys_yaw_frame_projects_body_x_to_world_xy(self):
        s = math.sqrt(0.5)
        rotation = (
            (s, 0.0, 0.0),
            (s, 0.0, 0.0),
            (0.25, 0.0, 1.0),
        )
        frame = diffphys_yaw_frame_from_rotation(rotation)
        self.assertAlmostEqual(frame[0][0], s)
        self.assertAlmostEqual(frame[1][0], s)
        self.assertAlmostEqual(frame[2][0], 0.0)
        self.assertAlmostEqual(frame[0][1], -s)
        self.assertAlmostEqual(frame[1][1], s)
        self.assertEqual((frame[0][2], frame[1][2], frame[2][2]), (0.0, 0.0, 1.0))

    def test_raw_diffphys_interleaved_action_matches_training_reshape(self):
        rotation = (
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        )
        accel = diffphys_raw_action_to_net_acceleration(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            rotation,
            gravity_mps2=9.80665,
            layout='interleaved',
        )
        self.assertEqual(accel, (-1.0, -1.0, -1.0))

    def test_tilt_limit_is_enforced(self):
        command = acceleration_to_attitude_thrust(
            (100.0, 0.0, 0.0),
            0.0,
            0.5,
            max_tilt_rad=math.radians(20.0),
        )
        rotation = quaternion_wxyz_to_matrix(command.quaternion_wxyz)
        z_body_world = (rotation[0][2], rotation[1][2], rotation[2][2])
        tilt = math.acos(max(-1.0, min(1.0, z_body_world[2])))
        self.assertLessEqual(tilt, math.radians(20.0) + 1e-6)


if __name__ == '__main__':
    unittest.main()

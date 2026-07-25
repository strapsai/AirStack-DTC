# Loquercio Agile Autonomy Starling PX4 Runbook

This runbook starts from a raw host terminal and runs the Loquercio et al.
`agile_autonomy` checkpoint through the AirStack/PX4 Starling simulation path.

The AirStack-side signal flow is intentionally the same shape as the
DiffPhysDrone wrapper:

```text
AirStack depth + odometry
  -> loquercio_px4_wrapper
  -> geometry_msgs/Vector3Stamped net acceleration
  -> diffphysdrone_px4_bridge attitude/thrust adapter
  -> /robot_1/interface/cmd_attitude_thrust
  -> AirStack robot_interface / PX4 offboard
  -> Pegasus / Isaac Sim Starling
```

TFLite is not required for this simulation run. The current baseline uses the
original TensorFlow checkpoint. TFLite can be added later as a second backend
without changing the ROS topic contract.

## Important Runtime Rule

`ros2 topic list` can show a topic even when nobody is publishing it. For policy
debugging, check publisher counts:

```bash
ros2 topic info -v /robot_1/sensors/front_stereo/right/depth_ground_truth
ros2 topic info -v /robot_1/odometry_conversion/odometry
```

The Loquercio wrapper will not publish `debug_depth_input`, `raw_prediction`,
or `accel_cmd` until both raw depth and odometry have active publishers and
fresh messages.

Also, `ros2 topic hz ...` blocks until you stop it. Run the checks one at a
time and press `Ctrl-C` before running the next one.

## Terminal 0: Host, Start AirStack

From a raw host terminal:

```bash
cd /home/ubuntu/volume/home/ubuntu/dtc/airlab_ws/autonomy_ws/src/simulation/AirStack-DTC

./airstack.sh down
```

Start the full AirStack stack with Isaac Sim, Starling, and a simple floor-only
scene. This is the first sanity check before testing Loquercio in clutter:

```bash
AUTOLAUNCH=true \
ENV_USD_PATH=/isaac-sim/AirStack/simulation/isaac-sim/assets/scenes/starling_empty_env.usda \
PEGASUS_DRONE_USD=/isaac-sim/AirStack/simulation/isaac-sim/extensions/PegasusSimulator/extensions/pegasus.simulator/pegasus/simulator/assets/Robots/Starling/starling2_preliminary.usd \
ISAAC_SIM_USE_STANDALONE=true \
ISAAC_SIM_SCRIPT_NAME=generic_env_px4_pegasus_launch_script.py \
NUM_ROBOTS=1 \
PLAY_SIM_ON_START=false \
GENERIC_REMOVE_EXISTING_DRONES=true \
GENERIC_SPAWN_X=0.0 \
GENERIC_SPAWN_Y=0.0 \
GENERIC_SPAWN_Z=0.4 \
./airstack.sh up
```

After the floor-only run works, use the small obstacle scene:

```bash
ENV_USD_PATH=/isaac-sim/AirStack/simulation/isaac-sim/assets/scenes/starling_obstacle_env.usda
```

After the obstacle scene works, switch to the TEEX construction-style
reconstruction scene:

```bash
ENV_USD_PATH=/isaac-sim/AirStack/simulation/isaac-sim/assets/teex/rescue/reconstruction_mesh_simple.usd
```

Wait for Isaac to load. Because `PLAY_SIM_ON_START=false`, press Play in Isaac
after confirming only one Starling was spawned.

Check Isaac logs from the host:

```bash
docker exec -it isaac-sim bash -lc 'tmux capture-pane -t isaac:0.0 -p -S -1000 | grep -E "\[generic_env\]|Using drone USD|Creating multirotor|Iris|iris|Starling|starling|drone_prim|PX4 node|Address already in use|poll timeout|Traceback|Error|ERROR|commander|arming|failsafe" | tail -200'
```

Good signs include:

```text
[generic_env] Loading environment USD: ...
[generic_env] Using drone USD: .../Starling/starling2_preliminary.usd
PX4 node at /World/drone1/base_link/PX4MultirotorGraph/PX4Multirotor_1
[generic_env] robot_1: drone_prim=/World/drone1/base_link
```

A bad sign is any active Iris USD spawn such as:

```text
Creating multirotor with USD file: .../Iris/iris.usd
```

## Terminal 1: Robot Container Setup

Open the AirStack robot container:

```bash
docker exec -it airstack-dtc-robot-desktop-1 bash
```

Source AirStack:

```bash
source /root/AirStack/robot/ros_ws/install/setup.bash
```

Build the wrapper if this container has not built it yet. Do not use the
TensorFlow venv for this build.

```bash
cd /root/AirStack/robot/ros_ws
colcon build --packages-select loquercio_px4_wrapper --symlink-install
source install/setup.bash
```

## Terminal 1: TensorFlow Runtime Setup

The AirStack container uses externally managed system Python, so the Loquercio
TensorFlow runtime lives in a venv. This is a one-time setup.

```bash
apt-get update
apt-get install -y python3.12-venv

python3 -m venv --system-site-packages /root/tf_loquercio_venv
source /root/tf_loquercio_venv/bin/activate

python3 -m pip install --upgrade pip wheel
python3 -m pip install --force-reinstall tensorflow-cpu==2.16.2

deactivate
```

For every Loquercio launch terminal, expose the venv packages to the ROS 2
process:

```bash
source /root/AirStack/robot/ros_ws/install/setup.bash

export TF_ENABLE_ONEDNN_OPTS=0
export PYTHONPATH=/root/tf_loquercio_venv/lib/python3.12/site-packages:$PYTHONPATH
```

Verify TensorFlow, ROS 2 Python, and the checkpoint:

```bash
python3 - <<'PY'
import numpy as np
import tensorflow as tf
import rclpy
from loquercio_px4_wrapper.inference import TensorFlowLoquercioBackend
from loquercio_px4_wrapper.model import LoquercioModelConfig

backend = TensorFlowLoquercioBackend(
    "/airlab-storage/chiron/models/loquercio/ckpt-50",
    LoquercioModelConfig(),
)
depth = np.zeros((1, 1, 224, 224, 3), dtype=np.float32)
imu = np.zeros((1, 1, 21), dtype=np.float32)
alphas, traj = backend.infer(depth, imu)
print("tensorflow", tf.__version__)
print("checkpoint", backend.checkpoint_prefix)
print("alphas", alphas.shape)
print("traj", traj.shape)
print("rclpy ok")
PY
```

Expected output includes:

```text
tensorflow 2.16.2
checkpoint /airlab-storage/chiron/models/loquercio/ckpt-50
alphas (3,)
traj (3, 30)
rclpy ok
```

If the checkpoint is missing, copy it from the ROS 1 agile autonomy container
from the host:

```bash
mkdir -p /home/ubuntu/volume/home/ubuntu/media/airlab-storage/chiron/models/loquercio

docker cp funny_archimedes:/workspace/dtc/agile_autonomy_ws/catkin_aa/src/agile_autonomy/planner_learning/models/. \
  /home/ubuntu/volume/home/ubuntu/media/airlab-storage/chiron/models/loquercio/
```

## Terminal 2: Confirm AirStack Inputs

Open a second robot-container terminal:

```bash
docker exec -it airstack-dtc-robot-desktop-1 bash
source /root/AirStack/robot/ros_ws/install/setup.bash
```

After Isaac is loaded and playing, these must have publishers:

```bash
ros2 topic info -v /robot_1/odometry_conversion/odometry
ros2 topic info -v /robot_1/sensors/front_stereo/right/depth_ground_truth
```

Then confirm fresh messages:

```bash
ros2 topic echo /robot_1/interface/mavros/state --once
ros2 topic echo /robot_1/odometry_conversion/odometry --once
ros2 topic echo /robot_1/sensors/front_stereo/right/depth_ground_truth --once --field header
```

If either raw depth or odometry has `Publisher count: 0`, do not debug the
policy yet. Start or restart AirStack/Isaac first.

## Terminal 1: Launch Loquercio Debug Mode

Debug mode publishes policy outputs but does not command PX4.

```bash
source /root/AirStack/robot/ros_ws/install/setup.bash

export TF_ENABLE_ONEDNN_OPTS=0
export PYTHONPATH=/root/tf_loquercio_venv/lib/python3.12/site-packages:$PYTHONPATH

ros2 launch loquercio_px4_wrapper loquercio_px4_wrapper.launch.xml \
  checkpoint_path:=/airlab-storage/chiron/models/loquercio/ckpt-50 \
  backend:=tensorflow
```

Expected startup:

```text
Loquercio policy loaded /airlab-storage/chiron/models/loquercio/ckpt-50 using backend=tensorflow
DiffPhysDrone C++ attitude bridge publishing to /robot_1/loquercio/debug_attitude_thrust
```

## Terminal 2: Check Loquercio Debug Outputs

Run these one at a time. Stop each `hz` check with `Ctrl-C`.

```bash
source /root/AirStack/robot/ros_ws/install/setup.bash

ros2 topic hz /robot_1/loquercio/debug_depth_input
ros2 topic hz /robot_1/loquercio/raw_prediction
ros2 topic hz /robot_1/loquercio/selected_trajectory
ros2 topic hz /robot_1/loquercio/accel_cmd
ros2 topic echo /robot_1/loquercio/debug_attitude_thrust --once
```

Optional depth visualization:

```bash
ros2 run rqt_image_view rqt_image_view /robot_1/loquercio/debug_depth_input
```

If Loquercio has publishers but no messages, check the upstream publishers:

```bash
ros2 topic info -v /robot_1/sensors/front_stereo/right/depth_ground_truth
ros2 topic info -v /robot_1/odometry_conversion/odometry
```

## Terminal 2: Publish A Gentle Target Direction

The wrapper defaults to the vehicle forward direction if no target is provided,
but publishing a target velocity makes the comparison with DiffPhysDrone
explicit.

```bash
ros2 topic pub -r 10 /robot_1/loquercio/target_velocity geometry_msgs/msg/Vector3Stamped \
"{header: {frame_id: map}, vector: {x: 0.2, y: 0.0, z: 0.0}}"
```

Increase the forward value only after stable hover and motion:

```text
x: 0.2 first
x: 0.5 after stable behavior
```

## Terminal 2: Take Off Before Policy Takeover

Loquercio's original ROS 1 pipeline takes off and hovers before the learned
planner starts sending trajectory commands. The policy wrapper is not a ground
takeoff controller. On the floor-only AirStack scene, first verify PX4 can lift
the Starling using AirStack's robot command service.

Stop the Loquercio debug or command launch if it is still running, then use the
AirStack takeoff action. Do not use `robot_command {command: 3}` for PX4 here;
the current `mavros_interface` implementation does not perform a PX4 takeoff for
that command.

```bash
source /root/AirStack/robot/ros_ws/install/setup.bash

ros2 action send_goal /robot_1/tasks/takeoff task_msgs/action/TakeoffTask \
  "{target_altitude_m: 1.0, velocity_m_s: 0.3}" \
  --feedback
```

Watch odometry or Isaac until the vehicle is airborne. If this takeoff action
does not move the drone, debug AirStack/PX4 takeoff before judging Loquercio.

After the vehicle is airborne, launch Loquercio command mode. Use a slightly
higher hover thrust for the first Starling test if `hover_thrust:=0.5` cannot
maintain altitude.

## Terminal 1: Launch Real PX4 Command Mode

Stop debug mode with `Ctrl-C`, then relaunch with the AirStack command topic.

```bash
source /root/AirStack/robot/ros_ws/install/setup.bash

export TF_ENABLE_ONEDNN_OPTS=0
export PYTHONPATH=/root/tf_loquercio_venv/lib/python3.12/site-packages:$PYTHONPATH

ros2 launch loquercio_px4_wrapper loquercio_px4_wrapper.launch.xml \
  checkpoint_path:=/airlab-storage/chiron/models/loquercio/ckpt-50 \
  backend:=tensorflow \
  output_topic:=/robot_1/interface/cmd_attitude_thrust \
  max_tilt_deg:=5.0 \
  thrust_max:=0.55 \
  max_net_accel_mps2:=3.0
```

Before arming, verify the command stream with `topic info`. Do not use
`ros2 topic hz` or `ros2 topic echo` on this topic in the current container; the
Python CLI can segfault when importing `mav_msgs/msg/AttitudeThrust`.

```bash
ros2 topic info -v /robot_1/interface/cmd_attitude_thrust
```

Expected:

```text
Publisher count: 1
Node name: loquercio_attitude_bridge
Subscription count: 1
Node name: robot_interface
```

## Terminal 2: Request Control And Arm

Only arm after `/robot_1/interface/cmd_attitude_thrust` is publishing.

```bash
source /root/AirStack/robot/ros_ws/install/setup.bash

ros2 service call /robot_1/interface/robot_command airstack_msgs/srv/RobotCommand "{command: 0}"
sleep 1
ros2 topic echo /robot_1/interface/has_control --once

ros2 service call /robot_1/interface/robot_command airstack_msgs/srv/RobotCommand "{command: 1}"
sleep 2
ros2 topic echo /robot_1/interface/is_armed --once
ros2 topic echo --qos-durability volatile --once /robot_1/interface/mavros/state
```

Expected healthy state:

```yaml
connected: true
armed: true
mode: OFFBOARD
```

If PX4 rejects arming, check the status text:

```bash
ros2 topic echo /robot_1/interface/mavros/statustext/recv --once
```

## Stop

Disarm:

```bash
ros2 service call /robot_1/interface/robot_command airstack_msgs/srv/RobotCommand "{command: 2}"
```

Stop the Loquercio launch with `Ctrl-C`.

From the host, stop AirStack:

```bash
cd /home/ubuntu/volume/home/ubuntu/dtc/airlab_ws/autonomy_ws/src/simulation/AirStack-DTC
./airstack.sh down
```

## Isaac Autolaunch Disappears

If Isaac briefly opens and then disappears, or the tmux log command prints:

```text
no server running on /tmp/tmux-0/default
```

then the Isaac container is alive but the tmux-launched Isaac process died. The
standalone autolaunch path should use `/isaac-sim/python.sh` directly, not the
`run_isaac_python` shell helper, because that helper may not exist inside the
fresh tmux shell. Check the compose line in
`simulation/isaac-sim/docker/docker-compose.yaml`; it should contain:

```text
PYTHONPATH="$$ISAAC_SIM_PYTHONPATH" /isaac-sim/python.sh ...generic_env_px4_pegasus_launch_script.py
```

After fixing it, restart AirStack with `./airstack.sh down` and `./airstack.sh up`.

## Quick Failure Map

`/robot_1/loquercio/debug_depth_input` has a publisher but no messages:

```text
Loquercio is running, but no depth/odom callback has completed.
Check raw depth and odometry publisher counts.
```

Raw depth has `Publisher count: 0`:

```text
Isaac/Pegasus is not publishing the camera stream.
Check Isaac is loaded, the timeline is playing, and the Starling camera graph exists.
```

Odometry has `Publisher count: 0`:

```text
AirStack/PX4/MAVROS or odometry conversion is not running.
Check the robot container bringup and Isaac/PX4 link.
```

Loquercio launch crashes on TensorFlow import:

```text
Do not use tensorflow-cpu 2.21.0 in this container.
Use tensorflow-cpu==2.16.2 in /root/tf_loquercio_venv.
```

Policy publishes debug output but the drone does not move:

```text
You are probably still in debug mode.
Relaunch with output_topic:=/robot_1/interface/cmd_attitude_thrust,
then request control and arm.
```

`ros2 topic hz /robot_1/interface/cmd_attitude_thrust` segfaults:

```text
This is a ROS 2 Python CLI/type-support issue with mav_msgs/msg/AttitudeThrust
in this container. The command topic can still be healthy. Use
ros2 topic info -v /robot_1/interface/cmd_attitude_thrust instead.
```

The drone arms but behaves too aggressively:

```text
Lower max_tilt_deg, thrust_max, max_net_accel_mps2, and target velocity.
Start with x: 0.2 m/s.
```

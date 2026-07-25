# Agile Autonomy AirStack

Experimental AirStack-native wrapper for adding Agile Autonomy as an optional local planner plugin.

This package does not command PX4 directly. It accepts or generates short-horizon trajectory candidates, validates them, converts the selected candidate into AirStack's `airstack_msgs/msg/TrajectoryXYZVYaw`, and publishes only to the existing trajectory controller input.

```text
AirStack behavior / mission logic
  -> optional Agile Autonomy local planner plugin
  -> trajectory adapter + safety gate
  -> AirStack trajectory controller
  -> existing AirStack controller/interface path
  -> PX4
```

## What It Does

- Starts disabled by default.
- Provides a mock backend that emits a deterministic 1 second forward candidate for simulation testing.
- Defines an internal candidate message contract with candidate IDs, costs, frame mode, timestamps, and ordered timed pose points.
- Validates freshness, point count, finite values, speed, acceleration, vertical speed, and frame transform availability before publishing.
- Converts valid candidates to `airstack_msgs/msg/TrajectoryXYZVYaw` for the AirStack trajectory controller.

## What It Does Not Do

- It does not publish raw neural-network commands to PX4.
- It does not publish MAVROS or `/fmu/*` setpoints.
- It does not arm, disarm, take off, land, or switch PX4 modes.
- It does not automatically switch the trajectory controller into `ADD_SEGMENT`; existing mission logic should own controller mode.
- It does not include GPL Agile Autonomy source or a neural-network model.

## Topics

Policy node subscriptions:

- `odometry` -> `nav_msgs/msg/Odometry`
- `depth` -> `sensor_msgs/msg/Image`
- `camera_info` -> `sensor_msgs/msg/CameraInfo`
- `global_plan` -> `nav_msgs/msg/Path`
- `enable` -> `std_msgs/msg/Bool`

Policy publications:

- `trajectory_candidates` -> `agile_autonomy_airstack_msgs/msg/CandidateTrajectoryArray`
- `status` -> `std_msgs/msg/String`

Adapter subscriptions:

- `trajectory_candidates`
- `odometry`
- `enable`

Adapter publications:

- `trajectory_segment` -> `airstack_msgs/msg/TrajectoryXYZVYaw`
- `adapter_status` -> `std_msgs/msg/String`

Safety gate subscriptions:

- `policy_status`
- `adapter_status`
- `odometry`
- `depth`
- `external_enable`

Safety gate publications:

- `policy_enable`
- `adapter_enable`
- `status`

The default local bringup remaps the adapter output to:

```text
/$(env ROBOT_NAME)/trajectory_controller/trajectory_segment_to_add
```

## Parameters

Safe defaults live in `config/agile_autonomy_airstack.yaml`.

Important defaults:

- `enabled_default: false`
- `backend_type: mock`
- `publish_rate_hz: 15.0`
- `input_timeout_sec: 0.5`
- `trajectory_horizon_sec: 1.0`
- `trajectory_dt_sec: 0.1`
- `mock_forward_speed_mps: 0.5`
- `max_speed_mps: 2.0`
- `max_accel_mps2: 3.0`
- `max_vertical_speed_mps: 1.0`
- `target_frame: map`
- `candidate_frame_mode: body_relative`
- `yaw_mode: current`
- `require_depth: true`
- `require_odom: true`
- `fail_closed: true`

## Build

Inside the robot container:

```bash
cd /root/AirStack/robot/ros_ws
bws --packages-select agile_autonomy_airstack_msgs agile_autonomy_airstack
bws --packages-select local_bringup
sws
```

Equivalent plain colcon:

```bash
cd /root/AirStack/robot/ros_ws
colcon build --packages-select agile_autonomy_airstack_msgs agile_autonomy_airstack --symlink-install
colcon build --packages-select local_bringup --symlink-install
source install/setup.bash
```

## Launch

Local layer only:

```bash
ros2 launch local_bringup local.launch.xml use_agile_autonomy:=true
```

Full autonomy launch:

```bash
ros2 launch autonomy_bringup robot.launch.xml role:=full sim:=true use_agile_autonomy:=true
```

Useful overrides:

```bash
ros2 launch local_bringup local.launch.xml \
  use_agile_autonomy:=true \
  agile_depth_topic:=/robot_1/sensors/front_stereo/right/depth_ground_truth \
  agile_odometry_topic:=/robot_1/odometry_conversion/odometry
```

## Test With Mock Backend

In separate shells after launch:

```bash
ros2 node list | grep agile_autonomy
ros2 topic echo /robot_1/agile_autonomy/status
ros2 topic echo /robot_1/agile_autonomy/policy/status
ros2 topic echo /robot_1/agile_autonomy/adapter/status
```

Enable:

```bash
ros2 topic pub --once /robot_1/agile_autonomy/enable std_msgs/msg/Bool "{data: true}"
```

Watch candidates and controller-facing output:

```bash
ros2 topic echo /robot_1/agile_autonomy/trajectory_candidates
ros2 topic echo /robot_1/trajectory_controller/trajectory_segment_to_add
```

Disable:

```bash
ros2 topic pub --once /robot_1/agile_autonomy/enable std_msgs/msg/Bool "{data: false}"
```

If you want the trajectory controller to consume stitched segments during manual testing, set controller mode through the existing AirStack service:

```bash
ros2 service call /robot_1/trajectory_controller/set_trajectory_mode airstack_msgs/srv/TrajectoryMode "{mode: 3}"
```

## Unit Tests

```bash
cd /root/AirStack/robot/ros_ws/src/local/agile_autonomy_airstack
python3 setup.py test
# or
python3 -m pytest test -q
```

The unit tests cover stale candidate rejection, speed-limit rejection, simple valid candidate acceptance, lowest-cost selection, body-relative frame conversion, and config loading.

## Real Backend Extension

Add the real Agile backend behind `backend.py` without changing AirStack topics:

1. Keep `agile_policy_node` subscriptions and candidate publication unchanged.
2. Convert model output into `CandidateTrajectoryArray`.
3. Keep candidate points short-horizon and conservative.
4. Let `agile_trajectory_adapter` validate and convert to `TrajectoryXYZVYaw`.
5. Continue using the existing trajectory controller and PX4 interface path.

## Safety Limitations And TODOs

- Collision, geofence, and ESDF checks are TODO hooks; this first pass only validates kinematics and frame availability.
- Behavior-tree node plugins are not added because this branch does not contain the expected BT packages or registration pattern.
- The mock backend is for plumbing tests, not obstacle avoidance.
- The adapter assumes body-relative candidates are x-forward, y-left, z-up in the odometry body frame.
- The controller must be in `ADD_SEGMENT` mode for stitched segments to affect flight.

Example future BT pseudo XML:

```xml
<Fallback>
  <Sequence>
    <AgileAutonomyHealthyCondition/>
    <EnableAgileAutonomyAction/>
  </Sequence>
  <DisableAgileAutonomyAction/>
</Fallback>
```

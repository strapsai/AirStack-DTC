# Agile Autonomy AirStack Integration Notes

These notes were collected from the AirStack-DTC checkout at
`autonomy_ws/src/simulation/AirStack-DTC`.

## Controller Contract

- `/$(env ROBOT_NAME)/trajectory_controller/trajectory_segment_to_add` is remapped from the trajectory controller's relative `trajectory_segment_to_add` subscription in `robot/ros_ws/src/local/local_bringup/launch/local.launch.xml`.
- The trajectory controller subscribes to `airstack_msgs/msg/TrajectoryXYZVYaw` on relative topic `trajectory_segment_to_add` in `robot/ros_ws/src/local/controls/trajectory_controller/src/trajectory_controller.cpp`.
- `airstack_msgs/msg/TrajectoryXYZVYaw` is:
  - `std_msgs/Header header`
  - `WaypointXYZVYaw[] waypoints`
- `airstack_msgs/msg/WaypointXYZVYaw` is:
  - `geometry_msgs/Point position`
  - `float64 velocity`
  - `float64 yaw`
  - `geometry_msgs/Vector3 acceleration`
  - `geometry_msgs/Vector3 jerk`
- `trajectory_library` removes consecutive duplicate waypoints, derives vector velocity direction from adjacent waypoint positions and scalar `velocity`, and generates waypoint timing from segment distance divided by average waypoint speed. Acceleration and jerk are passed through.

## Existing Local Planners

- `droan_gl` publishes `airstack_msgs/msg/TrajectoryXYZVYaw` on relative topic `trajectory_segment_to_add`; `local.launch.xml` remaps it to `/$(env ROBOT_NAME)/trajectory_controller/trajectory_segment_to_add`.
- `droan_local_planner` has the same intended output topic and message type, but it is inside an ignored XML block in the current `local.launch.xml`.
- `takeoff_landing_planner` and `fixed_trajectory_task` publish `trajectory_override`, not stitched local-planner segments.

## Odometry

- The standard local odometry topic is `/$(env ROBOT_NAME)/odometry_conversion/odometry`.
- `local.launch.xml` declares `local_odometry_in_topic` with that default and remaps local consumers to it.
- `trajectory_controller` subscribes to `nav_msgs/msg/Odometry` on relative topic `odometry`.
- `interface_bringup/launch/interface.launch.py` runs `robot_interface/odometry_conversion`, which overwrites odometry frame IDs to `map` and `base_link`, publishes the canonical odometry topic, and can publish `base_link_stabilized`.

## Depth And Camera Topics

- Stereo RGB/camera info convention:
  - `/$(env ROBOT_NAME)/sensors/front_stereo/left/image_rect`
  - `/$(env ROBOT_NAME)/sensors/front_stereo/right/image_rect`
  - `/$(env ROBOT_NAME)/sensors/front_stereo/left/camera_info`
  - `/$(env ROBOT_NAME)/sensors/front_stereo/right/camera_info`
- Depth ground-truth topics appear as:
  - `/$(env ROBOT_NAME)/sensors/front_stereo/left/depth_ground_truth`
  - `/$(env ROBOT_NAME)/sensors/front_stereo/right/depth_ground_truth`
- Current `local.launch.xml` uses disparity by default for DROAN:
  - `/$(env ROBOT_NAME)/perception/stereo_image_proc/disparity`
- This package defaults to the right depth ground-truth topic because existing policy-wrapper docs and launch files use the right depth stream, while keeping `agile_depth_topic` overridable.

## Frames

- The trajectory controller target frame is `map` in `local.launch.xml`.
- Common frames found in this branch include `map`, `world`, `base_link`, `base_link_stabilized`, `look_ahead_point_stabilized`, and optical camera frames.
- The PX4 interface code converts ENU AirStack setpoints into PX4 NED/uXRCE-DDS setpoints internally. This package does not publish PX4 setpoints.
- The Agile adapter assumes body-relative candidates are expressed with x-forward, y-left, z-up relative to the odometry pose orientation. It transforms those points into `target_frame` using current odometry and tf2 when the odometry frame differs.
- Candidates already in `map`, `odom`, or `target_frame` pass through only when they match the adapter target frame, or when tf2 can transform them.

## Launch And Package Style

- Launch XML uses `<arg>`, `<group if="$(var ...)">`, `<push-ros-namespace>`, `<include>`, `<node>`, `<param from="..." allow_substs="true" />`, and explicit `<remap>` tags.
- Python ROS 2 packages in this repo use `ament_python`, `setup.py`, `setup.cfg`, `resource/<package>`, `package.xml`, and console-script entry points.
- C++ ROS 2 packages use `ament_cmake`, `find_package(...)`, `ament_target_dependencies(...)`, `install(TARGETS ...)`, and `install(DIRECTORY launch ...)`.
- Message packages use `ament_cmake`, `rosidl_default_generators`, and `rosidl_generate_interfaces(...)`.

## Behavior Tree Status

- The requested `robot/ros_ws/src/behavior/behavior_tree`, `behavior_executive`, and `behavior_tree_example` packages are not present in this branch.
- Existing behavior packages are `behavior_bringup`, `drone_safety_monitor`, and `rqt_behavior_tree_command`.
- `rqt_behavior_tree_command` publishes `behavior_tree_msgs/msg/BehaviorTreeCommands`; no BehaviorTree.CPP node registration pattern was found in `robot/ros_ws/src/behavior`.
- BT integration is therefore left as a README TODO. The plugin exposes `/$(env ROBOT_NAME)/agile_autonomy/enable` and status topics so future BT nodes can control it without changing the local planner contract.

## Safety Boundary

- This package does not publish to MAVROS, `/fmu/*`, PX4 messages, `cmd_attitude_thrust`, `cmd_roll_pitch_yawrate_thrust`, or any arming/landing service.
- The only AirStack controller-facing command topic published by this package is the adapter's `trajectory_segment` output, remapped by launch to `/$(env ROBOT_NAME)/trajectory_controller/trajectory_segment_to_add`.

"""Launch the Agile Flight (Loquercio) single-drone VELOCITY-command commander.

Agile-Flight analog of ``diffaero_velocity_single.launch.py``: the Loquercio net
is tracked by the acados MPC and the solved MPC-state velocity is published as the
PX4 setpoint. See ``config/agile_vel_sim.yaml`` and ``agile/agile_core.py``.

    # Simulation (default):
    ros2 launch svg_ground_control agile_velocity_single.launch.py

    # Override scenario:
    ros2 launch svg_ground_control agile_velocity_single.launch.py scenario:=goal

    # Hardware (px4_interface + mocap bridge):
    ros2 launch svg_ground_control agile_velocity_single.launch.py \\
        config:=<share>/config/agile_vel_real.yaml use_mocap:=true
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def launch_setup(context, *args, **kwargs):
    config = LaunchConfiguration('config')
    scenario = LaunchConfiguration('scenario').perform(context)

    commander_params = [config]
    if scenario:
        commander_params.append({'scenario': scenario})

    return [
        Node(
            package='svg_ground_control',
            executable='agile_velocity_commander',
            name='agile_velocity_commander',
            output='screen',
            parameters=commander_params,
        ),
        Node(
            package='svg_ground_control',
            executable='mocap_bridge',
            name='mocap_bridge',
            output='screen',
            parameters=[config],
            condition=IfCondition(LaunchConfiguration('use_mocap')),
        ),
    ]


def generate_launch_description():
    default_config = PathJoinSubstitution(
        [FindPackageShare('svg_ground_control'), 'config', 'agile_vel_sim.yaml'])

    return LaunchDescription([
        DeclareLaunchArgument(
            'config', default_value=default_config,
            description='Parameter YAML for agile_velocity_commander'),
        DeclareLaunchArgument(
            'scenario', default_value='',
            description='Override scenario from config: hover, goal, random_walk, ...'),
        DeclareLaunchArgument(
            'use_mocap', default_value='false',
            description='Start the mocap bridge (hardware only)'),
        OpaqueFunction(function=launch_setup),
    ])

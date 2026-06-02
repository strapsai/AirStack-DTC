from setuptools import find_packages, setup

package_name = 'diffphysdrone_px4_wrapper'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/diffphysdrone_px4_wrapper.launch.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='todo',
    maintainer_email='todo@todo.todo',
    description='Convert DiffPhysDrone acceleration actions to PX4-compatible attitude/thrust commands.',
    license='TODO',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'diffphysdrone_px4_wrapper = diffphysdrone_px4_wrapper.node:main',
            'diffphysdrone_policy = diffphysdrone_px4_wrapper.policy_node:main',
            'diffphysdrone_waypoint_velocity = diffphysdrone_px4_wrapper.waypoint_velocity_node:main',
            'diffphysdrone_export_onnx = diffphysdrone_px4_wrapper.export_onnx:main',
            'diffphysdrone_check_onnx_parity = diffphysdrone_px4_wrapper.check_onnx_parity:main',
            'diffphysdrone_voxl_policy_smoke = diffphysdrone_px4_wrapper.voxl_policy_smoke:main',
        ],
    },
)

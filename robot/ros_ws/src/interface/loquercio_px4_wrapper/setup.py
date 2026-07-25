from setuptools import find_packages, setup

package_name = 'loquercio_px4_wrapper'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/loquercio_px4_wrapper.launch.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='todo',
    maintainer_email='todo@todo.todo',
    description='Run a Loquercio agile_autonomy trajectory policy behind the AirStack PX4 interface.',
    license='TODO',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'loquercio_policy = loquercio_px4_wrapper.policy_node:main',
        ],
    },
)

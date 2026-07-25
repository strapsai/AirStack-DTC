from glob import glob
import os

from setuptools import Command, find_packages, setup


package_name = "agile_autonomy_airstack"


class PyTestCommand(Command):
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        import pytest

        raise SystemExit(pytest.main(["test"]))

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml", "README.md", "NOTES.md"]),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.xml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="todo",
    maintainer_email="todo@todo.todo",
    description="Experimental AirStack-native Agile Autonomy local planner wrapper.",
    license="TODO",
    cmdclass={"test": PyTestCommand},
    entry_points={
        "console_scripts": [
            "agile_policy_node = agile_autonomy_airstack.agile_policy_node:main",
            "agile_trajectory_adapter = agile_autonomy_airstack.agile_trajectory_adapter:main",
            "agile_safety_gate = agile_autonomy_airstack.agile_safety_gate:main",
        ],
    },
)

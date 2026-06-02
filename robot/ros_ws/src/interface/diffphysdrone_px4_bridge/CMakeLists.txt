cmake_minimum_required(VERSION 3.14)
project(diffphysdrone_px4_bridge)

find_package(ament_cmake REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(mav_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(rclcpp REQUIRED)

add_executable(diffphysdrone_attitude_bridge src/diffphysdrone_attitude_bridge.cpp)
ament_target_dependencies(
  diffphysdrone_attitude_bridge
  geometry_msgs
  mav_msgs
  nav_msgs
  rclcpp
)

install(
  TARGETS diffphysdrone_attitude_bridge
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()

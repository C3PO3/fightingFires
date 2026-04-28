cmake_minimum_required(VERSION 3.8)
project(frame_inspector)

if(POLICY CMP0074)
    cmake_policy(SET CMP0074 NEW)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(PCL REQUIRED)
add_definitions(${PCL_DEFINITIONS})
find_package(OpenCV REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})

# --- Executables ---
add_executable(pointcloud_printer src/pointcloud_printer.cpp)

# --- Link ROS 2 deps
ament_target_dependencies(pointcloud_printer
  rclcpp
  sensor_msgs
  pcl_conversions
)

# --- Install
install(TARGETS
  pointcloud_printer
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()

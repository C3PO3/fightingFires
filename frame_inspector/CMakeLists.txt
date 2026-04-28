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
add_executable(wall_mapper src/wall_map_fix.cpp)
add_executable(NDT src/NDT_code.cpp)

# --- Link ROS 2 deps for both
ament_target_dependencies(pointcloud_printer
  rclcpp
  sensor_msgs
  pcl_conversions
)

ament_target_dependencies(wall_mapper
  rclcpp
  sensor_msgs
  pcl_conversions
)

ament_target_dependencies(NDT
  rclcpp
  sensor_msgs
  pcl_conversions
)

# --- Link non-ROS libraries
target_link_libraries(wall_mapper ${PCL_LIBRARIES} ${OpenCV_LIBS})
target_link_libraries(NDT ${PCL_LIBRARIES})

# --- Install
install(TARGETS
  pointcloud_printer
  wall_mapper
  NDT
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()

# Install Python scripts
install(PROGRAMS
  frame_inspector/ICP_code.py
  DESTINATION lib/${PROJECT_NAME}
)
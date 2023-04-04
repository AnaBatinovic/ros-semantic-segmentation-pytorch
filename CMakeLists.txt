cmake_minimum_required(VERSION 2.8.3)
project(semantic_segmentation_ros)

find_package(catkin REQUIRED COMPONENTS
    rospy
    std_msgs
    geometry_msgs
    sensor_msgs
    message_generation
)

add_message_files(
    FILES
    SegmentationNameMask.msg
    SegmentationNameMaskArray.msg
    SegmentationObject.msg
    SegmentationObjectArray.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
  sensor_msgs
)

catkin_package( 
	CATKIN_DEPENDS message_runtime
)

include_directories(include ${catkin_INCLUDE_DIRS})
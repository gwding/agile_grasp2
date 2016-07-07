cmake_minimum_required(VERSION 2.8.3)
project(agile_grasp2)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS cmake_modules cv_bridge eigen_conversions geometry_msgs 
message_generation pcl_conversions roscpp sensor_msgs std_msgs visualization_msgs)

## System dependencies are found with CMake's conventions
find_package(Eigen REQUIRED)
find_package(OpenCV)
find_package(PCL REQUIRED)
#~ find_package(VTK 6.0 REQUIRED NO_MODULE) # required in Fedora Core

IF(NOT EXISTS ${CAFFE_DIR})
  SET(CAFFE_DIR "$ENV{CAFFE_DIR}")
ENDIF()
MESSAGE("CAFFE_DIR: " ${CAFFE_DIR})

# CAFFE
find_package(Caffe)
include_directories(${Caffe_INCLUDE_DIRS})
add_definitions(${Caffe_DEFINITIONS})    # ex. -DCPU_ONLY

## Set compiler optimization flags
set(CMAKE_CXX_FLAGS "-DNDEBUG -O3 -fopenmp -Wno-deprecated -Wenum-compare")

## Uncomment this if the package has a setup.py. This macro ensures
## modules and global scripts declared therein get installed
## See http://ros.org/doc/api/catkin/html/user_guide/setup_dot_py.html
# catkin_python_setup()

################################################
## Declare ROS messages, services and actions ##
################################################

## To declare and build messages, services or actions from within this
## package, follow these steps:
## * Let MSG_DEP_SET be the set of packages whose message types you use in
##   your messages/services/actions (e.g. std_msgs, actionlib_msgs, ...).
## * In the file package.xml:
##   * add a build_depend and a run_depend tag for each package in MSG_DEP_SET
##   * If MSG_DEP_SET isn't empty the following dependencies might have been
##     pulled in transitively but can be declared for certainty nonetheless:
##     * add a build_depend tag for "message_generation"
##     * add a run_depend tag for "message_runtime"
## * In this file (CMakeLists.txt):
##   * add "message_generation" and every package in MSG_DEP_SET to
##     find_package(catkin REQUIRED COMPONENTS ...)
##   * add "message_runtime" and every package in MSG_DEP_SET to
##     catkin_package(CATKIN_DEPENDS ...)
##   * uncomment the add_*_files sections below as needed
##     and list every .msg/.srv/.action file to be processed
##   * uncomment the generate_messages entry below
##   * add every package in MSG_DEP_SET to generate_messages(DEPENDENCIES ...)

## Generate services in the 'srv' folder
add_service_files(FILES FindGrasps.srv Classify.srv)

## Generate messages in the 'msg' folder
add_message_files(FILES CloudIndexed.msg CloudSized.msg DensityGraspMsg.msg DensityGraspListMsg.msg GraspMsg.msg 
  GraspListMsg.msg SamplesMsg.msg)

## Generate added messages and services with any dependencies listed here
generate_messages(DEPENDENCIES geometry_msgs sensor_msgs std_msgs)

###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if you package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
 INCLUDE_DIRS 
  include
 LIBRARIES
  antipodal
  caffe_classifier
  cloud_camera
  finger_hand
  grasp_detection_node
  grasp_hypothesis
  hand_search
  handle
  handle_search
  importance_sampling
  learning
  local_frame
  plot
 CATKIN_DEPENDS
	cmake_modules
	cv_bridge
  eigen_conversions 
  geometry_msgs 
  message_runtime 
  pcl_conversions
  roscpp 
  sensor_msgs
  std_msgs
  visualization_msgs
 DEPENDS
  Caffe
  Eigen
  OpenCV
  PCL  
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(include ${catkin_INCLUDE_DIRS} ${PCL_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
#~ include(${VTK_USE_FILE}) # required in Fedora Core

## Declare a cpp library
add_library(antipodal src/${PROJECT_NAME}/antipodal.cpp)
add_library(caffe_classifier src/${PROJECT_NAME}/caffe_classifier.cpp)
add_library(classification_node src/nodes/classification_node.cpp)
add_library(cloud_camera src/${PROJECT_NAME}/cloud_camera.cpp)
add_library(finger_hand src/${PROJECT_NAME}/finger_hand.cpp)
add_library(grasp_detector src/${PROJECT_NAME}/grasp_detector.cpp)
add_library(grasp_detection_node src/nodes/grasp_detection_node.cpp)
add_library(grasp_hypothesis src/${PROJECT_NAME}/grasp_hypothesis.cpp)
add_library(hand_search src/${PROJECT_NAME}/hand_search.cpp)
add_library(handle src/${PROJECT_NAME}/handle.cpp)
add_library(handle_search src/${PROJECT_NAME}/handle_search.cpp)
add_library(importance_sampling src/${PROJECT_NAME}/importance_sampling.cpp)
add_library(learning src/${PROJECT_NAME}/learning.cpp)
add_library(local_frame src/${PROJECT_NAME}/local_frame.cpp)
add_library(plot src/${PROJECT_NAME}/plot.cpp)

## Declare a cpp executable
add_executable(classify_grasps src/nodes/classify_grasps.cpp)
add_executable(detect_grasps src/nodes/detect_grasps.cpp)
add_executable(detect_grasps_file src/nodes/detect_grasps_file.cpp)
add_executable(gpd_classification src/caffe/gpd_classification.cpp)
add_executable(test_cnn src/tests/test_cnn.cpp)

## Specify libraries to link a library or executable target against
target_link_libraries(caffe_classifier ${Caffe_LIBRARIES})
target_link_libraries(classification_node caffe_classifier cloud_camera hand_search handle_search learning plot 
  ${PCL_LIBRARIES})
target_link_libraries(classify_grasps classification_node ${catkin_LIBRARIES})
target_link_libraries(cloud_camera ${catkin_LIBRARIES} ${PCL_LIBRARIES})
target_link_libraries(detect_grasps_file grasp_detection_node ${PCL_LIBRARIES})
target_link_libraries(detect_grasps grasp_detection_node ${PCL_LIBRARIES})
target_link_libraries(gpd_classification ${Caffe_LIBRARIES})
target_link_libraries(grasp_detection_node caffe_classifier cloud_camera grasp_detector hand_search handle_search
 importance_sampling learning plot ${PCL_LIBRARIES})
target_link_libraries(grasp_detector caffe_classifier cloud_camera hand_search handle_search learning plot 
  ${PCL_LIBRARIES})
target_link_libraries(grasp_hypothesis ${catkin_LIBRARIES})
target_link_libraries(hand_search antipodal cloud_camera finger_hand grasp_hypothesis plot local_frame ${PCL_LIBRARIES})
target_link_libraries(handle grasp_hypothesis)
target_link_libraries(learning grasp_hypothesis handle ${OpenCV_LIBRARIES})
target_link_libraries(handle_search handle ${catkin_LIBRARIES} ${PCL_LIBRARIES})
target_link_libraries(plot grasp_hypothesis handle local_frame ${catkin_LIBRARIES} ${PCL_LIBRARIES})
target_link_libraries(local_frame ${PCL_LIBRARIES})
target_link_libraries(importance_sampling grasp_detector grasp_hypothesis ${catkin_LIBRARIES} ${PCL_LIBRARIES})
target_link_libraries(test_cnn caffe_classifier cloud_camera hand_search handle_search learning plot ${PCL_LIBRARIES})


#############
## Install ##
#############

## Mark cpp header files for installation
install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
  FILES_MATCHING PATTERN "*.h"
  PATTERN ".git" EXCLUDE)

## install launch files
install(DIRECTORY launch/
	DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch)


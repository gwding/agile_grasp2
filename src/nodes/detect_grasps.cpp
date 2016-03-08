#include <ros/ros.h>

#include <nodes/grasp_detection_node.h>


int main(int argc, char** argv)
{
  // initialize ROS
  ros::init(argc, argv, "detect_grasps");
  ros::NodeHandle node("~");
  
  GraspDetectionNode grasp_detection(node);
  grasp_detection.run();
  
	return 0;
}

#include <ros/ros.h>

#include <nodes/classification_node.h>


int main(int argc, char** argv)
{
  // initialize ROS
  ros::init(argc, argv, "classify_grasps");
  ros::NodeHandle node("~");
  
  ClassificationNode classification_node(node);
  classification_node.run();
  
	return 0;
}

#include <ros/ros.h>

#include <nodes/grasp_detection_node.h>


int main(int argc, char** argv)
{
  // initialize ROS
  ros::init(argc, argv, "detect_grasps");
  ros::NodeHandle node("~");
  
  // read filename from parameter
  std::string cloud_file_name;
  node.getParam("cloud_file_name", cloud_file_name);
  std::string file_name_left;
  std::string file_name_right;
  if (cloud_file_name.find(".pcd") == std::string::npos)
  {
    file_name_left = cloud_file_name + "l_reg.pcd";
    file_name_right = cloud_file_name + "r_reg.pcd";
  }
  else
  {
    file_name_left = cloud_file_name;
    file_name_right = "";
  }

  GraspDetectionNode grasp_detection(node);
  grasp_detection.detectGraspPosesInFile(file_name_left, file_name_right);
  
  return 0;
}

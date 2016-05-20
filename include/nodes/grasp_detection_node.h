/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2015, Andreas ten Pas
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef GRASP_DETECTION_NODE_H_
#define GRASP_DETECTION_NODE_H_

// system
#include <algorithm>
#include <vector>

// ROS
#include <eigen_conversions/eigen_msg.h>
#include <pcl_conversions/pcl_conversions.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>

// PCL
#include <pcl/common/common.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

// project-specific
#include <agile_grasp2/caffe_classifier.h>
#include <agile_grasp2/cloud_camera.h>
#include <agile_grasp2/hand_search.h>
#include <agile_grasp2/handle_search.h>
#include <agile_grasp2/learning.h>
#include <agile_grasp2/plot.h>

// custom messages
#include <agile_grasp2/CloudIndexed.h>
#include <agile_grasp2/CloudSized.h>
#include <agile_grasp2/GraspMsg.h>
#include <agile_grasp2/GraspListMsg.h>
#include <agile_grasp2/SamplesMsg.h>

// custom services
#include <agile_grasp2/FindGrasps.h>


typedef pcl::PointCloud<pcl::PointXYZRGBA> PointCloudRGBA;
typedef pcl::PointCloud<pcl::PointNormal> PointCloudNormal;


class GraspDetectionNode
{
public:
  
  GraspDetectionNode(ros::NodeHandle& node);

  ~GraspDetectionNode()
  {
    delete classifier_;
    delete learning_;
  }

  std::vector<GraspHypothesis> detectGraspPosesInFile(const std::string& file_name_left,
    const std::string& file_name_right);

  std::vector<GraspHypothesis> detectGraspPosesInTopic();

  std::vector<GraspHypothesis> detectGraspPoses(CloudCamera& cloud_cam);

  void run();

  static bool isScoreGreater(const GraspHypothesis& hypothesis1, const GraspHypothesis& hypothesis2);


private:

  std::vector<int> getSamplesInBall(const PointCloudRGBA::Ptr& cloud, const pcl::PointXYZRGBA& centroid,
    float radius);

  void preprocessPointCloud(const std::vector<double>& workspace, double voxel_size, int num_samples,
    CloudCamera& cloud_cam);

  bool graspsCallback(agile_grasp2::FindGrasps::Request& req, agile_grasp2::FindGrasps::Response& resp);

  std::vector<GraspHypothesis> pruneGraspsOnHandParameters(const std::vector<GraspHypothesis>& hands, float min_x,
    float max_x, float min_y, float max_y, float min_z);

  /**
   * \brief Callback function for the ROS topic that contains the input point cloud.
   * \param msg the incoming ROS message
  */
  void cloud_callback(const sensor_msgs::PointCloud2ConstPtr& msg);

  /**
   * \brief Callback function for the ROS topic that contains the input point cloud.
   * \param msg the incoming ROS message
  */
  void cloud_sized_callback(const agile_grasp2::CloudSized& msg);

  /**
   * \brief Callback function for the ROS topic that contains the input point cloud.
   * \param msg the incoming ROS message (of type agile_grasp2/CloudSized)
  */
  void cloud_indexed_callback(const agile_grasp2::CloudIndexed& msg);
  
  void samples_callback(const agile_grasp2::SamplesMsg& msg);

  agile_grasp2::GraspListMsg createGraspListMsg(const std::vector<Handle>& handles);

  agile_grasp2::GraspListMsg createGraspListMsg(const std::vector<GraspHypothesis>& hands);

  PointCloudRGBA::Ptr cloud_;
  PointCloudNormal::Ptr cloud_normals_;
  int size_left_cloud_;
  bool has_cloud_, has_normals_;
  ros::Subscriber cloud_sub_;
  ros::Subscriber samples_sub_;
  ros::Publisher grasps_pub_;
  ros::ServiceServer grasps_service_;
  std::vector<int> indices_;

  Classifier* classifier_;
  Learning* learning_;
  HandSearch hand_search_;
  HandleSearch handle_search_;

  /** Parameters from launch file */
  bool filter_half_grasps_;
  bool voxelize_;
  int antipodal_mode_;
  bool save_hypotheses_;
  bool only_plot_output_;
  int plot_mode_;
  std::vector<double> workspace_;
  std::vector<double> camera_pose_;
  double voxel_size_;
  int num_samples_;
  std::string grasp_image_dir_;
  double min_score_diff_;
  int num_selected_;
  double min_aperture_, max_aperture_;
  double outer_diameter_;

  /** constants for antipodal mode */
  static const int NONE; ///< no prediction/calculation of antipodal grasps, uses grasp hypotheses
  static const int PREDICTION; ///< predicts antipodal grasps
  static const int GEOMETRIC; ///< calculates antipodal grasps

  /** constants for plotting */
  static const int NO_PLOTTING; ///< nothing is plotted
  static const int PCL; ///< everything is plotted in pcl-visualizer
  static const int RVIZ; ///< everything is plotted in rviz

  /** constants for input point cloud types */
  static const int PCD_FILE; ///< *.pcd file
  static const int POINT_CLOUD_2; ///< sensor_msgs/PointCloud2
  static const int CLOUD_SIZED; ///< agile_grasp2/CloudSized
  static const int CLOUD_INDEXED; ///< agile_grasp2/CloudIndexed

  /** constants for ROS service */
  static const int ALL_POINTS; ///< service uses all points in the cloud
  static const int RADIUS; ///< service uses all points within a radius given in the request
  static const int INDICES; ///< service uses all points which are contained in an index list given in the request
};

#endif /* GRASP_DETECTION_NODE_H_ */

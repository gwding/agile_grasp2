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
#include <agile_grasp2/grasp_detector.h>
#include <agile_grasp2/hand_search.h>
#include <agile_grasp2/handle_search.h>
#include <agile_grasp2/importance_sampling.h>
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


/** GraspDetectionNode class
 *
 * \brief A ROS node that can detect grasp poses in a point cloud.
 *
 * This class is a ROS node that handles all the ROS topics.
 *
*/
class GraspDetectionNode
{
public:
  
  /**
   * \brief Constructor.
   * \param node the ROS node
  */
  GraspDetectionNode(ros::NodeHandle& node);

  /**
   * \brief Destructor.
  */
  ~GraspDetectionNode()
  {
    delete grasp_detector_;
    delete importance_sampling_;
  }

  /**
   * \brief Run the ROS node. Loops while waiting for incoming ROS messages.
  */
  void run();

  /**
   * \brief Detect grasp poses in a point cloud loaded from two *.pcd files.
   * \param file_name_left the location of the file that contains the point cloud from the left camera
   * \param file_name_left the location of the file that contains the point cloud from the right camera
   * \return the list of grasp poses
  */
  std::vector<GraspHypothesis> detectGraspPosesInFile(const std::string& file_name_left,
    const std::string& file_name_right);

  /**
   * \brief Detect grasp poses in a point cloud received from a ROS topic.
   * \return the list of grasp poses
  */
  std::vector<GraspHypothesis> detectGraspPosesInTopic();


private:

  /**
   * \brief Find the indices of the points within a ball around a given point in the cloud.
   * \param cloud the point cloud
   * \param centroid the centroid of the ball
   * \param radius the radius of the ball
   * \return the indices of the points in the point cloud that lie within the ball
  */
  std::vector<int> getSamplesInBall(const PointCloudRGBA::Ptr& cloud, const pcl::PointXYZRGBA& centroid, float radius);

  /** Callback for the grasps ROS service.
   * \param req the service request
   * \param resp the service response
   * \return true if grasps were found, false otherwise
  */
  bool graspsServiceCallback(agile_grasp2::FindGrasps::Request& req, agile_grasp2::FindGrasps::Response& resp);

  /**
   * \brief Callback function for the ROS topic that contains the input point cloud.
   * \param msg the incoming ROS message
  */
  void cloud_callback(const sensor_msgs::PointCloud2ConstPtr& msg);

  /**
   * \brief Callback function for the ROS topic that contains the input point cloud and the size of the left cloud.
   * \param msg the incoming ROS message
  */
  void cloud_sized_callback(const agile_grasp2::CloudSized& msg);

  /**
   * \brief Callback function for the ROS topic that contains the input point cloud and a list of indices.
   * \param msg the incoming ROS message
  */
  void cloud_indexed_callback(const agile_grasp2::CloudIndexed& msg);
  
  /**
   * \brief Callback function for the ROS topic that contains the input samples.
   * \param msg the incoming ROS message
  */
  void samples_callback(const agile_grasp2::SamplesMsg& msg);

  /**
   * \brief Create a ROS message that contains a list of grasp poses from a list of handles.
   * \param handles the list of handles
   * \return the ROS message that contains the grasp poses
  */
  agile_grasp2::GraspListMsg createGraspListMsg(const std::vector<Handle>& handles);

  /**
   * \brief Create a ROS message that contains a list of grasp poses from a list of handles.
   * \param hands the list of grasps
   * \return the ROS message that contains the grasp poses
  */
  agile_grasp2::GraspListMsg createGraspListMsg(const std::vector<GraspHypothesis>& hands);

  PointCloudRGBA::Ptr cloud_; ///< input point cloud without normals
  PointCloudNormal::Ptr cloud_normals_; ///< input point cloud with normals
  int size_left_cloud_; ///< size of the left point cloud (when using two point clouds as input)
  bool has_cloud_, has_normals_, has_samples_; ///< status variables for received messages
  ros::Subscriber cloud_sub_; ///< ROS subscriber for point cloud messages
  ros::Subscriber samples_sub_; ///< ROS publisher for samples messages
  ros::Publisher grasps_pub_; ///< ROS publisher for grasp list messages
  ros::ServiceServer grasps_service_; ///< grasps ROS service (untested)

  bool use_importance_sampling_; ///< if importance sampling is used

  GraspDetector* grasp_detector_; ///< used to run the grasp pose detection
  ImportanceSampling* importance_sampling_; ///< sequential importance sampling variation of grasp pose detection

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

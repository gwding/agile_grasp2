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

#ifndef GRASP_DETECTOR_H_
#define GRASP_DETECTOR_H_


// system
#include <algorithm>
#include <vector>

// PCL
#include <pcl/common/common.h>
//#include <pcl/point_cloud.h>
//#include <pcl/point_types.h>

// ROS
#include <ros/ros.h>

// project-specific
#include <agile_grasp2/caffe_classifier.h>
#include <agile_grasp2/cloud_camera.h>
#include <agile_grasp2/grasp_hypothesis.h>
#include <agile_grasp2/hand_search.h>
#include <agile_grasp2/handle_search.h>
#include <agile_grasp2/learning.h>
#include <agile_grasp2/plot.h>

// custom messages
#include <agile_grasp2/CloudIndexed.h>
#include <agile_grasp2/SamplesMsg.h>


/** GraspDetector class
 *
 * \brief Detect grasp poses in point clouds.
 *
 * This class detects grasps in a point cloud by first creating a large set of grasp hypotheses, and then
 * classifying each of them as a grasp or not. It also contains a function to preprocess the point cloud.
 *
*/
class GraspDetector
{
public:

  GraspDetector(ros::NodeHandle& node);

  ~GraspDetector()
  {
    delete classifier_;
    delete learning_;
  }

  std::vector<GraspHypothesis> detectGraspPoses(const CloudCamera& cloud_cam);

  void preprocessPointCloud(CloudCamera& cloud_cam);

  static bool isScoreGreater(const GraspHypothesis& hypothesis1, const GraspHypothesis& hypothesis2)
  {
    return hypothesis1.getScore() > hypothesis2.getScore();
  }

  bool getUseIncomingSamples() const
  {
    return use_incoming_samples_;
  }

  void setUseIncomingSamples(bool use_incoming_samples)
  {
    use_incoming_samples_ = use_incoming_samples;
  }

  const std::vector<double>& getWorkspace() const
  {
    return workspace_;
  }

  int getNumSamples() const
  {
    return num_samples_;
  }

  void setIndicesFromMsg(const agile_grasp2::CloudIndexed& msg);

  void setSamplesMsg(const agile_grasp2::SamplesMsg& samples_msg)
  {
    samples_msg_= samples_msg;
  }

  void setNumSamples(int num_samples)
  {
    num_samples_ = num_samples;
  }


private:

  std::vector<GraspHypothesis> pruneGraspsOnHandParameters(const std::vector<GraspHypothesis>& hands, float min_x,
      float max_x, float min_y, float max_y, float min_z);

  Classifier* classifier_;
  Learning* learning_;
  HandSearch hand_search_;
  HandleSearch handle_search_;
  agile_grasp2::SamplesMsg samples_msg_;
  bool use_incoming_samples_;
  double voxel_size_;

  /** Parameters from launch file */
  bool filter_half_grasps_;
  bool voxelize_;
  int antipodal_mode_;
  bool only_plot_output_;
  int plot_mode_;
  std::vector<double> workspace_;
  std::vector<double> camera_pose_;
  int num_samples_;
  double min_score_diff_;
  int num_selected_;
  double min_aperture_, max_aperture_;
  double outer_diameter_;
  std::vector<int> indices_;

  /** constants for antipodal mode */
  static const int NONE; ///< no prediction/calculation of antipodal grasps, uses grasp hypotheses
  static const int PREDICTION; ///< predicts antipodal grasps
  static const int GEOMETRIC; ///< calculates antipodal grasps

  /** constants for plotting */
  static const int NO_PLOTTING; ///< nothing is plotted
  static const int PCL; ///< everything is plotted in pcl-visualizer
  static const int RVIZ; ///< everything is plotted in rviz
};

#endif /* GRASP_DETECTOR_H_ */

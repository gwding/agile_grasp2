/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2016, Andreas ten Pas
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

#ifndef HAND_SEARCH_H
#define HAND_SEARCH_H


#include <Eigen/Dense>

#include <pcl/filters/random_sample.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/point_cloud.h>

#include <omp.h>

#include <agile_grasp2/antipodal.h>
#include <agile_grasp2/cloud_camera.h>
#include <agile_grasp2/finger_hand.h>
#include <agile_grasp2/grasp_hypothesis.h>
#include <agile_grasp2/local_frame.h>
#include <agile_grasp2/plot.h>


typedef pcl::PointCloud<pcl::PointXYZRGBA> PointCloudRGBA;


/** HandSearch class
 *
 * \brief Search for grasp hypotheses.
 * 
 * This class searches for grasp hypotheses in a point cloud by first calculating a local reference frame for a small
 * point neighborhood, and then finding geometrically feasible grasp hypotheses for a larger point neighborhood. It
 * can also estimate whether the grasp is antipodal from the normals of the point neighborhood.
 * 
*/
class HandSearch
{
  public:

    /**
     * \brief Parameters for the robot hand model.
    */
    struct Parameters
    {
      /** local reference frame estimation parameters */
      double nn_radius_taubin_; ///< local reference frame radius for point neighborhood search

      /** grasp hypotheses generation */
      int num_threads_; ///< the number of CPU threads to be used for the hypothesis generation
      int num_samples_; ///< the number of samples drawn from the point clouds;
      Eigen::Matrix4d cam_tf_left_; ///< pose of the left camera
      Eigen::Matrix4d cam_tf_right_; ///< pose of the right camera
      double nn_radius_hands_; ///< grasp hypothesis radius for point neighborhood search
      int num_orientations_; ///< number of hand orientations to evaluate

      /** robot hand geometry */
      double finger_width_; ///< the width of the robot hand fingers
      double hand_outer_diameter_; ///< the maximum robot hand aperture
      double hand_depth_; ///< the hand depth (length of fingers)
      double hand_height_; ///< the hand extends plus/minus this value along the hand axis
      double init_bite_; ///< the minimum object height
    };
    
    HandSearch() : plots_camera_sources_(false), plots_local_axes_(false)
      { }

    HandSearch(Parameters params) : cam_tf_left_(params.cam_tf_left_), cam_tf_right_(params.cam_tf_right_),
      finger_width_(params.finger_width_), hand_outer_diameter_(params.hand_outer_diameter_),
      hand_depth_(params.hand_depth_), hand_height_(params.hand_height_), init_bite_(params.init_bite_),
      num_threads_(params.num_threads_), num_samples_(params.num_samples_),
      nn_radius_taubin_(params.nn_radius_taubin_), nn_radius_hands_(params.nn_radius_hands_),
      num_orientations_(params.num_orientations_), plots_samples_(false), plots_local_axes_(false),
      plots_camera_sources_(false) { }

    /**
		 * \brief Constructor.
		 * \param finger_width the width of the robot hand fingers
		 * \param hand_outer_diameter the maximum robot hand aperture
		 * \param hand_depth the hand depth (length of fingers)
		 * \param hand_height the hand extends plus/minus this value along the hand axis
		 * \param init_bite the minimum object height
		 * \param num_threads the number of CPU threads to be used for the search
		 * \param num_samples the number of samples drawn from the point cloud
		*/
    HandSearch(double finger_width, double hand_outer_diameter, double hand_depth, double hand_height, double init_bite,
      int num_threads, int num_samples) : finger_width_(finger_width), hand_outer_diameter_(hand_outer_diameter), 
      hand_depth_(hand_depth), hand_height_(hand_height), init_bite_(init_bite), num_threads_(num_threads), 
      num_samples_(num_samples), plots_samples_(false), plots_local_axes_(false), nn_radius_taubin_(0.03),
      nn_radius_hands_(0.08) { }
    
    std::vector<GraspHypothesis> generateHypotheses(const CloudCamera& cloud_cam, int antipodal_mode, bool use_samples,
      bool forces_PSD = false, bool plots_normals = false, bool plots_samples = false);

    void setParameters(const Parameters& params);

    void setCamTfLeft(const Eigen::Matrix4d& cam_tf_left)
    {
      cam_tf_left_ = cam_tf_left;
    }

    void setCamTfRight(const Eigen::Matrix4d& cam_tf_right)
    {
      cam_tf_right_ = cam_tf_right;
    }
      
  
  private:
    
    void calculateNormals(const CloudCamera& cloud_cam, const pcl::KdTreeFLANN<pcl::PointXYZRGBA>& kdtree,
      bool plots_normals);

    void calculateNormalsOMP(const CloudCamera& cloud_cam);

    std::vector<LocalFrame> calculateLocalFrames(const CloudCamera& cloud_cam, const std::vector<int>& indices,
      double radius, const pcl::KdTreeFLANN<pcl::PointXYZRGBA>& kdtree);

    std::vector<LocalFrame> calculateLocalFrames(const CloudCamera& cloud_cam, const Eigen::Matrix3Xd& samples,
      double radius, const pcl::KdTreeFLANN<pcl::PointXYZRGBA>& kdtree);

    std::vector<GraspHypothesis> evaluateHands(const CloudCamera& cloud_cam, const std::vector<LocalFrame>& frames,
      double radius, const pcl::KdTreeFLANN<pcl::PointXYZRGBA>& kdtree);

    std::vector<GraspHypothesis> calculateHand(const Eigen::Matrix3Xd& points, const Eigen::Matrix3Xd& normals,
      const Eigen::MatrixXi& cam_source, const LocalFrame& local_frame, const Eigen::VectorXd& angles);

    Eigen::Matrix4d cam_tf_left_, cam_tf_right_; ///< camera poses
    
    /** hand geometry parameters */
    double finger_width_; ///< the finger width
    double hand_outer_diameter_; ///< the maximum robot hand aperture
    double hand_depth_; ///< the finger length
    double hand_height_; ///< the hand extends plus/minus this value along the hand axis
    double init_bite_; ///< the minimum object height

    /** hand search parameters */
    int num_orientations_; ///< the number of hand orientations to consider
    int num_threads_; ///< the number of threads used in the search
    int num_samples_; ///< the number of samples used in the search
    double nn_radius_taubin_; ///< the radius for the neighborhood search for the quadratic surface fit
    double nn_radius_hands_; ///< the radius for the neighborhood search for the hand search
    int antipodal_method_; ///< the antipodal calculation method (see the constants below)
    
    Eigen::Matrix3Xd cloud_normals_; ///< a 3xn matrix containing the normals for points in the point cloud
    Plot plot_; ///< plot object for visualization of search results
    
    /** plotting parameters (optional, not read in from ROS launch file) **/
    bool plots_samples_; ///< are the samples drawn from the point cloud plotted?
    bool plots_camera_sources_; ///< is the camera source for each point in the point cloud plotted?
    bool plots_local_axes_; ///< are the local axes estimated for each point neighborhood plotted?
};

#endif /* HAND_SEARCH_H */ 

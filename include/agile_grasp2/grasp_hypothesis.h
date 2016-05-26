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


#ifndef GRASP_HYPOTHESIS_H_
#define GRASP_HYPOTHESIS_H_


// Boost
#include <boost/lexical_cast.hpp>

// Eigen
#include <Eigen/Dense>

// System
#include <iostream>
#include <fstream>
#include <vector>

// ROS
#include <eigen_conversions/eigen_msg.h>

// custom messages
#include <agile_grasp2/GraspMsg.h>


/** GraspHypothesis class
 *
 * \brief Grasp hypothesis data structure
 * 
 * This class stores a single grasp hypothesis.
 * 
*/
class GraspHypothesis
{
public:

	/**
	 * \brief Default constructor.
	*/
	GraspHypothesis() : cam_source_(-1) {}

	GraspHypothesis(const Eigen::Vector3d& axis, const Eigen::Vector3d& approach, const Eigen::Vector3d& binormal,
    const Eigen::Vector3d& surface, const Eigen::Vector3d& bottom, const Eigen::Vector3d& top, double width,
    const Eigen::Matrix3Xd& points_for_learning, const Eigen::Matrix3Xd& normals_for_learning,
    const Eigen::MatrixXi& camera_source_for_learning)
	  : axis_(axis), approach_(approach), binormal_(binormal),  grasp_surface_(surface), grasp_bottom_(bottom),
	    grasp_top_(top), grasp_width_(width), points_for_learning_(points_for_learning),
	    normals_for_learning_(normals_for_learning), camera_source_for_learning_(camera_source_for_learning),
	    half_antipodal_(false), full_antipodal_(false), score_(0.0)
	{ }

	/**
	 * \brief Constructor.
	 * \param axis the hand axis
	 * \param approach the grasp approach vector
	 * \param binormal the binormal
	 * \param bottom the position between the end of the finger tips
	 * \param surface the position between the end of the finger tips projected onto the back of the hand
	 * \param points_for_learning the points used by the SVM for training/prediction
	 * \param indices_cam1 the point indices that belong to camera #1
	 * \param indices_cam2 the point indices that belong to camera #2
	 * \param cam_source the camera source of the sample
	*/
	GraspHypothesis(const Eigen::Vector3d& axis, const Eigen::Vector3d& approach, const Eigen::Vector3d& binormal,
	  const Eigen::Vector3d& bottom, const Eigen::Vector3d& surface, double width,
	  const Eigen::Matrix3Xd& points_for_learning, const Eigen::Matrix3Xd& normals_for_learning,
	  const std::vector<int>& indices_cam1,	const std::vector<int>& indices_cam2, int cam_source) :
			axis_(axis), approach_(approach), binormal_(binormal), grasp_bottom_(bottom), grasp_surface_(surface),
				grasp_width_(width), points_for_learning_(points_for_learning), normals_for_learning_(normals_for_learning),
				indices_points_for_learning_cam1_(indices_cam1), indices_points_for_learning_cam2_(indices_cam2),
				cam_source_(cam_source), half_antipodal_(false), full_antipodal_(false), score_(0.0)
	{	}

	void writeHandsToFile(const std::string& filename, const std::vector<GraspHypothesis>& hands);

	agile_grasp2::GraspMsg convertToGraspMsg() const;

	/**
	 * \brief Print a description of the grasp hypothesis to the systen's standard output.
	*/
	void print();
	
	/**
	 * \brief Return the approach vector of the grasp.
	 * \return 3x1 grasp approach vector
	*/
	const Eigen::Vector3d& getApproach() const
	{
		return approach_;
	}
	
	/**
	 * \brief Return the hand axis of the grasp.
	 * \return 3x1 hand axis
	*/
	const Eigen::Vector3d& getAxis() const
	{
		return axis_;
	}
	
	/**
	 * \brief Return the binormal of the grasp.
	 * \return 3x1 binormal
	*/
	const Eigen::Vector3d& getBinormal() const
	{
		return binormal_;
	}
	
	/**
	 * \brief Return whether the grasp is antipodal.
	 * \return true if the grasp is antipodal, false otherwise
	*/
	bool isFullAntipodal() const
	{
		return full_antipodal_;
	}
	
	/**
	 * \brief Return the the centered grasp position at the base of the robot hand.
	 * \return 3x1 grasp position at the base of the robot hand
	*/
	const Eigen::Vector3d& getGraspBottom() const
	{
		return grasp_bottom_;
	}

	/**
	 * \brief Return the grasp position between the end of the finger tips projected onto the back of the hand.
	 * \return 3x1 grasp position between the end of the finger tips projected onto the back of the hand
	*/
	const Eigen::Vector3d& getGraspSurface() const
	{
		return grasp_surface_;
	}
	
	/**
	 * \brief Return the width of the object contained in the grasp.
	 * \return the width of the object contained in the grasp
	*/
	double getGraspWidth() const
	{
		return grasp_width_;
	}
	
	/**
	 * \brief Return whether the grasp is indeterminate.
	 * \return true if the grasp is indeterminate, false otherwise
	*/
	bool isHalfAntipodal() const
	{
		return half_antipodal_;
	}
	
	/**
	 * \brief Return the points used for training/prediction by the SVM.
	 * \return the list of points used for training/prediction by the SVM
	*/
	const std::vector<int>& getIndicesPointsForLearningCam1() const
	{
		return indices_points_for_learning_cam1_;
	}
	
	/**
	 * \brief Return the points used for training/prediction by the SVM that belong to camera #1.
	 * \return the list of points used for training/prediction by the SVM that belong to camera #1
	*/
	const std::vector<int>& getIndicesPointsForLearningCam2() const
	{
		return indices_points_for_learning_cam2_;
	}
	
	/**
	 * \brief Return the points used for training/prediction by the SVM that belong to camera #2.
	 * \return the list of points used for training/prediction by the SVM that belong to camera #2
	*/
	const Eigen::Matrix3Xd& getPointsForLearning() const
	{
		return points_for_learning_;
	}
	
	/**
   * \brief Return the normals used for training/prediction by the SVM that belong to camera #2.
   * \return the list of normals used for training/prediction by the SVM that belong to camera #2
  */
  const Eigen::Matrix3Xd& getNormalsForLearning() const
  {
    return normals_for_learning_;
  }

	/**
	 * \brief Return the camera source of the sample
	 * \return the camera source of the sample
	*/
	int getCamSource() const
	{
		return cam_source_;
	}

	/**
	 * \brief Set whether the grasp is antipodal.
	 * \param b whether the grasp is antipodal
	*/
	void setFullAntipodal(bool b)
	{
		full_antipodal_ = b;
	}
	
	/**
	 * \brief Set whether the grasp is indeterminate.
	 * \param b whether the grasp is indeterminate
	*/
	void setHalfAntipodal(bool b)
	{
		half_antipodal_ = b;
	}
	
	/**
	 * \brief Set the width of the object contained in the grasp.
	 * \param w the width of the object contained in the grasp
	*/
	void setGraspWidth(double w)
	{
		grasp_width_ = w;
	}

	/**
	* \brief Return the the centered grasp position between the fingertips of the robot hand.
	* \return 3x1 grasp position between the fingertips
	*/
	const Eigen::Vector3d& getGraspTop() const
  {
    return grasp_top_;
  }

  void setGraspBottom(const Eigen::Vector3d& grasp_bottom)
  {
    grasp_bottom_ = grasp_bottom;
  }

  void setGraspSurface(const Eigen::Vector3d& grasp_surface)
  {
    grasp_surface_ = grasp_surface;
  }

  void setGraspTop(const Eigen::Vector3d& grasp_top)
  {
    grasp_top_ = grasp_top;
  }

  double getScore() const
  {
    return score_;
  }

  void setScore(double score)
  {
    score_ = score;
  }

private:

  std::string vectorToString(const Eigen::VectorXd& v);


protected:

	int cam_source_; ///< the camera source of the sample
	Eigen::Vector3d axis_; ///< the hand axis
  Eigen::Vector3d approach_; ///< the grasp approach vector (orthogonal to the hand axis)
  Eigen::Vector3d binormal_; ///< the binormal (orthogonal to the hand axis and the approach vector)
  Eigen::Vector3d grasp_surface_; ///< the centered grasp position on the object surface
  Eigen::Vector3d grasp_bottom_; ///< the centered grasp position at the base of the robot hand
  Eigen::Vector3d grasp_top_; ///< the centered grasp position between the fingertips of the robot hand
  double grasp_width_; ///< the width of object enclosed by the fingers
  double score_; ///< the score given by the classifier
  bool full_antipodal_; ///< whether the grasp hypothesis is antipodal
  bool half_antipodal_; ///< whether the grasp hypothesis is indeterminate
	Eigen::Matrix3Xd points_for_learning_; ///< the points used by the classifier
	Eigen::Matrix3Xd normals_for_learning_; ///< the normals used by the classifier
	Eigen::MatrixXi camera_source_for_learning_; ///< the camera source for each point used by the classifier
	std::vector<int> indices_points_for_learning_cam1_; ///< indices into the above points that originate from camera #1
	std::vector<int> indices_points_for_learning_cam2_; ///< indices into the above points that originate from camera #2
};

#endif /* GRASP_HYPOTHESIS_H_ */

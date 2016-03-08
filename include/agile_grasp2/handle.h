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

#ifndef HANDLE_H_
#define HANDLE_H_

#include <Eigen/Dense>
#include <iostream>
#include <vector>

#include <agile_grasp2/grasp_hypothesis.h>

/** HandleSearch class
 *
 * \brief Handle data structure
 * 
 * This class stores a single handle. The handle represents (a) a list of grasp hypotheses that 
 * are part of the handle, and (b) a grasp that is an "average" grasp over these grasp hypotheses.
 * 
*/
class Handle : public GraspHypothesis
{
public:
	
	Handle() {};

  /**
	 * \brief Constructor.
	 * \param hand_list the list of grasp hypotheses
	 * \param indices the list of indices of grasp hypotheses that are part of the handle
	*/
	Handle(const std::vector<GraspHypothesis>& hand_list, const std::vector<int>& inliers);

	Handle(const std::vector<GraspHypothesis>& hand_list, const std::vector<int>& inliers, int idx);

	/**
	 * \brief Return the list of grasp hypotheses.
	 * \return the list of grasp hypotheses
	*/
	const std::vector<GraspHypothesis>& getHandList() const
	{
		return hand_list_;
	}
	
	/**
	 * \brief Return the list of indices of grasp hypotheses that are part of the handle.
	 * \return the list of indices of grasp hypotheses that are part of the handle
	*/
	const std::vector<int>& getInliers() const
	{
		return inliers_;
	}

	void setNumAntipodalInliers(int num_antipodal_inliers)
	{
	  num_antipodal_inliers_ = num_antipodal_inliers;
	}

	bool operator<(const Handle& handle) const
	{
	  return num_antipodal_inliers_ > handle.num_antipodal_inliers_;
  }

  int getNumAntipodalInliers() const
  {
    return num_antipodal_inliers_;
  }

private:
	
  void setGraspVariablesMean();

	/**
	 * \brief Set the variables of the grasp.
	*/
	void setGraspVariables();
	
	/**
	 * \brief Set the hand axis of the grasp.
	*/
	void setAxis();

	/**
	 * \brief Set the distance along the handle's axis for each grasp hypothesis.
	*/
	void setDistAlongHandle();
	
	/**
	 * \brief Set the width of the object contained in the handle grasp.
	*/
	void setGraspWidth();

	void setCameraSource();

	void setPtsForLearning();

	int num_antipodal_inliers_;
	std::vector<int> inliers_; ///< the list of indices of grasp hypotheses that are part of the handle
	std::vector<GraspHypothesis> hand_list_; ///< the list of grasp hypotheses
	Eigen::VectorXd dist_along_handle_; ///< the 1xn vector of distances along the handle's axis for each grasp hypothesis
};

#endif /* HANDLE_H_ */

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

#ifndef HANDLE_SEARCH_H_
#define HANDLE_SEARCH_H_

#include <math.h>
#include <set>
#include <vector>

#include <pcl/point_cloud.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <ros/ros.h>

#include <agile_grasp2/grasp_hypothesis.h>
#include <agile_grasp2/handle.h>


typedef pcl::PointCloud<pcl::PointXYZRGBA> PointCloudRGBA;


/** HandleSearch class
 *
 * \brief Search handles given grasp hypotheses
 * 
 * This class searches for handles. Handles are clusters of grasps that are geometrically aligned.
 * 
*/
class HandleSearch
{
public:

  /**
   * \brief Search for handles given a list of grasp hypotheses.
   * \param hand_list the list of grasp hypotheses
  */
  std::vector<GraspHypothesis> findClusters(const std::vector<GraspHypothesis>& hand_list,
    bool remove_inliers = false);

	/**
	 * \brief Search for handles given a list of grasp hypotheses.
	 * \param hand_list the list of grasp hypotheses
	 * \param min_inliers the minimum number of grasp hypothesis contained in a handle
	 * \param min_length the minimum length of a handle
	 * \return the list of handles found
	*/
	std::vector<Handle> findHandles(const std::vector<GraspHypothesis>& hand_list);

	int getMinInliers() const
	{
	  return min_inliers_;
	}

  void setMinInliers(int min_inliers)
  {
    min_inliers_ = min_inliers;
  }

  void setMinLength(double min_length)
  {
    min_length_ = min_length;
  }

  void setReuseInliers(bool reuse_inliers)
  {
    reuse_inliers_ = reuse_inliers;
  }


private:
	
	/**
	 * \brief Shorten a handle to a continuous piece.
	 * 
	 * This function finds continuous handles by searching for a gap that is larger than @p gap_threshold.
	 * 
	 * \param inliers the list of grasp hypotheses that are part of the handle to be shortened
	 * \param gap_threshold the maximum gap size
	 * \return the shortened handle
	*/
	bool shortenHandle(std::vector<Eigen::Vector2d> &inliers, double gap_threshold);
	
	/**
	 * \brief Safe version of the acos(x) function.
	 * \param x the value whose arc cosine is computed
	 * \return the arc cosine of x, expressed in radians
	*/
	double safeAcos(double x);
	
	/**
	 * \brief Comparator for equality of the first two elements of two 3D-vectors.
	*/ 
	struct VectorFirstTwoElementsComparator
	{
		/**
		 * \brief Compare the first two elements of two 3D-vectors for equality.
	   * \param a the first 3D-vector to be compared
	   * \param b the second 3D-vector to be compared
	   * \return true if the first element of both vectors is equal or the second element of both 
	   * vectors is equal, false otherwise
		*/
		bool operator ()(const Eigen::Vector3d& a, const Eigen::Vector3d& b)
		{
			for (int i = 0; i < 2; i++)
			{
				if (a(i) != b(i))
				{
					return a(i) < b(i);
				}
			}

			return false;
		}
	};
	
	/**
	 * \brief Comparator for equality of the last element of two 2D-vectors.
	*/ 
	struct LastElementComparator
	{
		/**
		 * \brief Compare the last element of two 2D-vectors for equality.
	   * \param a the first 2D-vector to be compared
	   * \param b the second 2D-vector to be compared
	   * \return true if the last element of both vectors is equal, false otherwise
		*/
		bool operator ()(const Eigen::Vector2d& a, const Eigen::Vector2d& b)
		{
			if (a(1) != b(1))
			{
				return a(1) < b(1);
			}

			return false;
		}
	};

	bool reuse_inliers_; ///< if the inliers of a handle are reused for other clusters
	int min_inliers_; ///< minimum number of inliers in a handle
	double min_length_; ///< minimum length of a handle
};

#endif /* HANDLE_SEARCH_H_ */

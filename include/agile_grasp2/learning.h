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

#ifndef LEARNING_H_
#define LEARNING_H_

#include <fstream>
#include <iostream>
#include <set>
#include <sys/stat.h>
#include <vector>

#include <Eigen/Dense>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2/objdetect/objdetect.hpp>

#include <agile_grasp2/grasp_hypothesis.h>
#include <agile_grasp2/handle.h>


/** Learning class
 *
 * \brief Train and use an SVM to predict antipodal grasps  
 * 
 * This class trains an SVM classifier to predict antipodal grasps. Once trained, it can be used to 
 * predict antipodal grasps. The classifier is trained with HOG features obtained from grasp images. 
 * A grasp image is a 2D image representation of a grasp hypothesis.
 * 
*/
class Learning
{
public:

	/**
	 * \brief Constructor.
	*/
	Learning() : num_horizontal_cells_(60), num_vertical_cells_(60), num_threads_(1) {	}

	/**
   * \brief Constructor. Set the size of the grasp image and the number of threads to be used for
   * prediction.
  */
  Learning(int size, int num_threads) : num_horizontal_cells_(size), num_vertical_cells_(size),
    num_threads_(num_threads)
  {
  }

  std::vector<cv::Mat> createGraspImages(const std::vector<GraspHypothesis>& hands_list,
    const Eigen::Matrix3Xd& cam_pos, bool is_plotting = false, bool is_storing = false);

	/**
	 * \brief Store a given list of grasp hypotheses as grasp images.
	 * \param hands_list the set of grasp hypotheses to be used for training
	 * \param cam_pos the camera poses
   * \param root_dir the directory where the grasp images are stored
	 * \param is_plotting whether the grasp images are plotted
	 */
  std::vector<cv::Mat> storeGraspImages(const std::vector<GraspHypothesis>& hands_list, const Eigen::Matrix3Xd& cam_pos,
    const std::string& root_dir, bool is_plotting = false);

  cv::Mat createGraspImage(const Eigen::Matrix3Xd& points, const Eigen::Matrix3Xd& normals,
    const agile_grasp2::GraspMsg& grasp, int idx = -1, bool is_plotting = false);


private:

  /**
   * \brief Learning instance representing a grasp hypothesis.
  */
  struct Instance
  {
    Eigen::Matrix3Xd pts; ///< the points from the point cloud that make up the hypothesis
    Eigen::Matrix3Xd normals; ///< the points' normals from the point cloud that make up the hypothesis (optional)
    Eigen::Vector3d binormal; ///< the binormal of the grasp
    Eigen::Vector3d source_to_center; ///< the vector from the center of the grasp to the camera position
    bool label; ///< the label of the instance (true: antipodal, false: not antipodal)
  };

  /**
   * \brief Comparator for 2D vectors.
  */
	struct UniqueVectorComparator
	{
    /**
     * \brief Compare two vectors.
     * \param a the first vector to be compared
     * \param b the second vector to be compared
     * \return true if no elements of @p a and @p b are equal, false otherwise 
    */
		bool operator ()(const Eigen::Vector2i& a, const Eigen::Vector2i& b)
		{
			for (int i = 0; i < a.size(); i++)
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
   * \brief Create a learning instance from a grasp hypothesis.
   * \param h the grasp hypothesis from which the learning instance is created
   * \param cam_pos the camera poses
   * \param cam the index of the camera which produced the points in the grasp hypothesis
   * \return the created learning instance
  */
  Instance createInstance(const GraspHypothesis& h, const Eigen::Matrix3Xd& cam_pos, int cam = -1);

  Instance createInstanceFromPointsNormals(const Eigen::Matrix3Xd& points, const Eigen::Matrix3Xd& normals,
    const agile_grasp2::GraspMsg& grasp);

  /**
   * \brief Create a learning instance from a handle's centered grasp hypothesis.
   * \param h the grasp hypothesis from which the learning instance is created
   * \param cam_pos the camera poses
   * \param cam the index of the camera which produced the points in the grasp hypothesis
   * \return the created learning instance
  */
  Instance createInstanceFromHandle(const Handle& h, const Eigen::Matrix3Xd& cam_pos, int cam = -1);

  /**
   * \brief Convert a learning instance to a grasp image.
   * \param ins the learning instance to be converted
   * \return the created image
  */
	cv::Mat convertToImage(const Instance& ins);

	/**
	 * \brief Convert a learning instance to a RGB grasp image.
	 * @param ins the learning instance to be converted
   * \return the created image
	 */
	cv::Mat convertToImageRGB(const Instance& ins, bool aligns = true, bool plots = false);
  
  /**
   * \brief Round a vector's elements down to the closest, smaller integers.
   * \param a the vector whose elements are rounded down
   * \return the vector containing the rounded elements
  */
	Eigen::VectorXi floorVector(const Eigen::VectorXd& a);

	int num_horizontal_cells_; ///< the width of a grasp image
	int num_vertical_cells_; ///<  the height of a grasp image 
	int num_threads_; ///< the number of threads used for prediction
};

#endif /* LEARNING_H_ */

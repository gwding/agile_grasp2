#include <agile_grasp2/handle.h>

Handle::Handle(const std::vector<GraspHypothesis>& hand_list, const std::vector<int>& inliers) :
  hand_list_(hand_list), inliers_(inliers)
{
	setAxis();
	setDistAlongHandle();
	setGraspVariables();
	setGraspWidth();
	setCameraSource();
	setPtsForLearning();
}


Handle::Handle(const std::vector<GraspHypothesis>& hand_list, const std::vector<int>& inliers, int idx) :
  hand_list_(hand_list), inliers_(inliers), num_antipodal_inliers_(inliers.size())
{
  axis_ = hand_list[idx].getAxis();
  approach_ = hand_list[idx].getApproach();
  binormal_ = hand_list[idx].getBinormal();
  grasp_surface_ = hand_list[idx].getGraspSurface();
  grasp_bottom_ = hand_list[idx].getGraspBottom();
  grasp_top_ = hand_list[idx].getGraspTop();
  grasp_width_ = hand_list[idx].getGraspWidth();
  score_ = hand_list[idx].getScore();
}


void Handle::setGraspVariablesMean()
{
  Eigen::Matrix3Xd axis_mat(3, inliers_.size());
  Eigen::Matrix3Xd approach_mat(3, inliers_.size());
  grasp_bottom_ << 0.0, 0.0, 0.0;
  grasp_surface_ << 0.0, 0.0, 0.0;
  axis_ << 0.0, 0.0, 0.0;
  for (int i = 0; i < inliers_.size(); i++)
  {
//    axis_mat.col(i) = hand_list_[inliers_[i]].getAxis();
//    approach_mat.col(i) = hand_list_[inliers_[i]].getApproach();
    grasp_bottom_ += hand_list_[inliers_[i]].getGraspBottom();
    grasp_surface_ += hand_list_[inliers_[i]].getGraspSurface();
    axis_ += hand_list_[inliers_[i]].getAxis();
    approach_ += hand_list_[inliers_[i]].getApproach();
  }
//
//  Eigen::FullPivLU<Eigen::Matrix3d> lu_decomp(axis_mat);
//  std::cout << "The rank of axis_mat is " << lu_decomp.rank() << std::endl;

  grasp_bottom_ /= (double) inliers_.size();
  grasp_surface_ /= (double) inliers_.size();
  axis_ /= (double) inliers_.size();
  approach_ /= (double) inliers_.size();
  axis_.normalize();
  approach_.normalize();

  double closest = -1.0;
  int idx = -1;
  for (int i = 0; i < inliers_.size(); i++)
  {
    double dot = approach_.transpose() * hand_list_[inliers_[i]].getApproach();
    if (dot > closest)
    {
      closest = dot;
      idx = i;
    }
  }
  std::cout << "closest: " << closest << ", idx: " << idx << std::endl;
  axis_ = hand_list_[inliers_[idx]].getAxis();
  approach_ = hand_list_[inliers_[idx]].getApproach();
  binormal_ = hand_list_[inliers_[idx]].getBinormal();

//  Eigen::EigenSolver<Eigen::Matrix3d> eigen_solver(axis_mat * axis_mat.transpose());
//  Eigen::Vector3d eigen_values = eigen_solver.eigenvalues().real();
//  Eigen::Matrix3d eigen_vectors = eigen_solver.eigenvectors().real();
//  std::cout << "eigen_values: " << eigen_values.transpose() << std::endl;
//  std::cout << "eigen_vectors:\n" << eigen_vectors << std::endl;
//  int max_idx;
//  eigen_values.maxCoeff(&max_idx);
//  axis_ = eigen_vectors.col(max_idx);
//
//  Eigen::EigenSolver<Eigen::Matrix3d> eigen_solver2(approach_mat * approach_mat.transpose());
//  eigen_values = eigen_solver2.eigenvalues().real();
//  eigen_vectors = eigen_solver2.eigenvectors().real();
//  std::cout << "eigen_values: " << eigen_values.transpose() << std::endl;
//  std::cout << "eigen_vectors:\n" << eigen_vectors << std::endl;
//  eigen_values.maxCoeff(&max_idx);
//  approach_  = eigen_vectors.col(max_idx);
//  binormal_= axis_.cross(approach_);
//
  std::cout << "axis_.norm(): " << axis_.norm() << std::endl;
  std::cout << "approach_.norm(): " << approach_.norm() << std::endl;
  std::cout << "axis dot approach: " << axis_.transpose() * approach_ << std::endl;

//  double closest = -1.0;
//  int idx = -1;
//  for (int i = 0; i < inliers_.size(); i++)
//  {
//    double dot = axis_.transpose() * hand_list_[inliers_[i]].getAxis();
//    if (dot < closest)
//    {
//      closest = dot;
//      idx = i;
//    }
//  }
//  approach_ = hand_list_[inliers_[idx]].getApproach();


//  int min_idx;
//  eigen_values.minCoeff(&min_idx);
//  approach_ = eigen_vectors.col(min_idx);
}


void Handle::setAxis()
{
	Eigen::Matrix3Xd axis_mat(3, inliers_.size());
	for (int i = 0; i < inliers_.size(); i++)
	{
		axis_mat.col(i) = hand_list_[inliers_[i]].getAxis();
	}

	Eigen::EigenSolver<Eigen::Matrix3d> eigen_solver(axis_mat * axis_mat.transpose());
	Eigen::Vector3d eigen_values = eigen_solver.eigenvalues().real();
	Eigen::Matrix3d eigen_vectors = eigen_solver.eigenvectors().real();
	int max_idx;
	eigen_values.maxCoeff(&max_idx);
	axis_ = eigen_vectors.col(max_idx);
}


void Handle::setDistAlongHandle()
{
	dist_along_handle_.resize(inliers_.size());
	for (int i = 0; i < inliers_.size(); i++)
	{
		dist_along_handle_(i) = axis_.transpose()	* hand_list_[inliers_[i]].getGraspBottom();
	}
}


void Handle::setGraspVariables()
{
	double center_dist = (dist_along_handle_.maxCoeff() + dist_along_handle_.minCoeff()) / 2.0;
	double min_dist = 10000000;
	int min_idx = -1;
	for (int i = 0; i < dist_along_handle_.size(); i++)
	{
		double dist = fabs(dist_along_handle_(i) - center_dist);
		if (dist < min_dist)
		{
			min_dist = dist;
			min_idx = i;
		}
	}

	grasp_bottom_ = hand_list_[inliers_[min_idx]].getGraspBottom();
	axis_ = hand_list_[inliers_[min_idx]].getAxis();
	approach_ = hand_list_[inliers_[min_idx]].getApproach();
	grasp_surface_ = hand_list_[inliers_[min_idx]].getGraspSurface();
	binormal_ = approach_.cross(axis_);
//	std::cout << "axis_.norm(): " << axis_.norm() << std::endl;
//  std::cout << "approach_.norm(): " << approach_.norm() << std::endl;
//  std::cout << "binormal_.norm(): " << binormal_.norm() << std::endl;
//  std::cout << "axis dot approach: " << axis_.transpose() * approach_ << std::endl;
}


void Handle::setGraspWidth()
{
  grasp_width_ = 0.0;
	for (int i = 0; i < inliers_.size(); i++)
	{
	  grasp_width_ += hand_list_[inliers_[i]].getGraspWidth();
	}
	grasp_width_ /= (double) inliers_.size();
}


void Handle::setCameraSource()
{
  cam_source_ = 0;
  for (int i = 0; i < inliers_.size(); i++)
  {
    cam_source_ += hand_list_[inliers_[i]].getCamSource();
  }
  cam_source_ /= (double) inliers_.size();
  cam_source_ = int(round(cam_source_));
}


void Handle::setPtsForLearning()
{
  double min_dist = 100000;
  int idx = -1;
  for (int i = 0; i < inliers_.size(); i++)
  {
    double dist = (hand_list_[inliers_[i]].getGraspBottom() - grasp_bottom_).squaredNorm();
    if (dist < min_dist)
    {
      min_dist = dist;
      idx = i;
    }
  }

  points_for_learning_ = hand_list_[inliers_[idx]].getPointsForLearning();
  indices_points_for_learning_cam1_ = hand_list_[inliers_[idx]].getIndicesPointsForLearningCam1();
  indices_points_for_learning_cam2_ = hand_list_[inliers_[idx]].getIndicesPointsForLearningCam2();
}

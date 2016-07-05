#include <agile_grasp2/local_frame.h>


LocalFrame::LocalFrame(const std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d> >& T_cams,
  const Eigen::Vector3d& sample, int majority_cam_source)
  : sample_(sample), majority_cam_source_(majority_cam_source)
{
  cam_origins_.resize(3, T_cams.size());
  for (int i = 0; i < cam_origins_.cols(); i++)
  {
    cam_origins_.col(i) = T_cams.at(i).block<3,1>(0,3);
  }
}


void LocalFrame::print()
{
	std::cout << "sample: " << sample_.transpose() << std::endl;
	std::cout << "normals_ratio: " << normals_ratio_ << std::endl;
	std::cout << "curvature_axis: " << curvature_axis_.transpose() << std::endl;
	std::cout << "normal: " << normal_.transpose() << std::endl;
	std::cout << "binormal: " << binormal_.transpose() << std::endl;
}


void LocalFrame::findAverageNormalAxis(const Eigen::MatrixXd &normals)
{
  // 1. Calculate curvature axis.
	Eigen::Matrix3d M = normals * normals.transpose();
	Eigen::EigenSolver<Eigen::MatrixXd> eigen_solver(M);
	Eigen::Vector3d eigen_values = eigen_solver.eigenvalues().real();
	Eigen::Matrix3d eigen_vectors = eigen_solver.eigenvectors().real();
	Eigen::Vector3d sorted_eigen_values = eigen_values;
	std::sort(sorted_eigen_values.data(), sorted_eigen_values.data() + sorted_eigen_values.size());
	normals_ratio_ = sorted_eigen_values(1) / sorted_eigen_values(2);
	int min_index;
	eigen_values.minCoeff(&min_index);
	curvature_axis_ = eigen_vectors.col(min_index);

	// 2. Calculate surface normal.
	int max_index;
	(normals.transpose() * normals).array().pow(6).colwise().sum().maxCoeff(&max_index);
	Eigen::Vector3d normpartial = (Eigen::MatrixXd::Identity(3, 3)
    - curvature_axis_ * curvature_axis_.transpose()) * normals.col(max_index);
	normal_ = normpartial / normpartial.norm();

	// 3. Create binormal.
	binormal_ = curvature_axis_.cross(normal_);

	// 4. Require normal and binormal to be oriented towards source.
	Eigen::Vector3d source_to_sample = sample_ - cam_origins_.col(majority_cam_source_);
	if (normal_.dot(source_to_sample) > 0) // normal points away from source
		normal_ *= -1.0;
	if (binormal_.dot(source_to_sample) > 0) // binormal points away from source
		binormal_ *= -1.0;

	// adjust curvature axis to new frame
	curvature_axis_ = normal_.cross(binormal_);
}


void LocalFrame::plotAxes(void* viewer_void, int id) const
{
	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer = *static_cast<boost::shared_ptr<
			pcl::visualization::PCLVisualizer> *>(viewer_void);

	pcl::PointXYZ p, q, r;
	p.x = sample_(0);
	p.y = sample_(1);
	p.z = sample_(2);
	q.x = p.x + 0.02 * curvature_axis_(0);
	q.y = p.y + 0.02 * curvature_axis_(1);
	q.z = p.z + 0.02 * curvature_axis_(2);
	r.x = p.x + 0.02 * normal_(0);
	r.y = p.y + 0.02 * normal_(1);
	r.z = p.z + 0.02 * normal_(2);
	viewer->addLine<pcl::PointXYZ>(p, q, 0, 0, 255, "curvature_axis_" + boost::lexical_cast<std::string>(id));
	viewer->addLine<pcl::PointXYZ>(p, r, 255, 0, 0, "normal_axis_" + boost::lexical_cast<std::string>(id));
}

#include <agile_grasp2/hand_search.h>


std::vector<GraspHypothesis> HandSearch::generateHypotheses(const CloudCamera& cloud_cam, int antipodal_mode,
  bool forces_PSD, bool plots_normals, bool plots_samples)
{
  double t0_total = omp_get_wtime();

  // create KdTree for neighborhood search
  const PointCloudRGB::Ptr& cloud = cloud_cam.getCloudProcessed();
  pcl::KdTreeFLANN<pcl::PointXYZRGBA> kdtree;
  kdtree.setInputCloud(cloud);

  cloud_normals_.resize(3, cloud->size());
  cloud_normals_.setZero(3, cloud->size());

  // 1. Calculate surface normals for all points (optional).
  bool has_normals = false;
  double t0_normals = omp_get_wtime();
  if (cloud_cam.getNormals().cols() == 0)
  {
    std::cout << "Estimating normals for all points ";
    if (normal_estimation_method_ == NORMAL_ESTIMATION_QUADRICS)
    {
      std::cout << " using quadrics\n";
      calculateNormals(cloud_cam, kdtree, false);
    }
    else if (normal_estimation_method_ == NORMAL_ESTIMATION_OMP)
    {
      std::cout << " using pcl::NormalEstimationOMP\n";
      calculateNormalsOMP(cloud_cam);
    }
  }
  else
  {
    std::cout << "Using precomputed normals for all points\n";
    cloud_normals_ = cloud_cam.getNormals();
  }
  std::cout << " normals computation time: " << omp_get_wtime() - t0_normals << std::endl;

  if (plots_normals)
  {
    ROS_INFO("Plotting normals ...");
    plot_.plotNormals(cloud, cloud_normals_);
  }

  if (plots_samples)
  {
    ROS_INFO("Plotting samples ...");
    plot_.plotSamples(cloud_cam.getSampleIndices(), cloud_cam.getCloudProcessed());
  }

  // 2. Fit quadrics.
  std::cout << "Fitting quadrics ...\n";
  std::vector<Quadric> quadrics = fitQuadrics(cloud_cam, cloud_cam.getSampleIndices(), nn_radius_taubin_, kdtree);
//  std::vector<Quadric> quadrics = calculateLocalFrames(cloud_cam, cloud_cam.getSampleIndices(), nn_radius_taubin_,
//    kdtree);
  if (plots_local_axes_)
    plot_.plotLocalAxes(quadrics, cloud_cam.getCloudOriginal());

  // 3. Generate grasp hypotheses.
  std::cout << "Finding hand poses ...\n";
  std::vector<GraspHypothesis> hypotheses = evaluateHands(cloud_cam, quadrics, nn_radius_hands_, kdtree);

  std::cout << "====> HAND SEARCH TIME: " << omp_get_wtime() - t0_total << std::endl;

  return hypotheses;
}


void HandSearch::setParameters(const Parameters& params)
{
  finger_width_ = params.finger_width_;
  hand_outer_diameter_= params.hand_outer_diameter_;
  hand_depth_ = params.hand_depth_;
  hand_height_ = params.hand_height_;
  init_bite_ = params.init_bite_;

  num_threads_ = params.num_threads_;
  num_samples_ = params.num_samples_;
  nn_radius_taubin_ = params.nn_radius_taubin_;
  nn_radius_hands_ = params.nn_radius_hands_;
  num_orientations_ = params.num_orientations_;

  antipodal_method_ = params.antipodal_mode_;
  normal_estimation_method_ = params.normal_estimation_method_;
}


void HandSearch::calculateNormals(const CloudCamera& cloud_cam, const pcl::KdTreeFLANN<pcl::PointXYZRGBA>& kdtree,
  bool plots_normals)
{
  std::vector<int> indices(cloud_cam.getCloudProcessed()->size());
  for (int i = 0; i < indices.size(); i++)
    indices[i] = i;
  fitQuadrics(cloud_cam, indices, 0.01, kdtree, true, false);
}


void HandSearch::calculateNormalsOMP(const CloudCamera& cloud_cam)
{
  pcl::NormalEstimationOMP<pcl::PointXYZRGBA, pcl::Normal> estimator(num_threads_);
  pcl::PointCloud<pcl::Normal>::Ptr cloud_normals_omp(new pcl::PointCloud<pcl::Normal>);
  pcl::search::KdTree<pcl::PointXYZRGBA>::Ptr tree_ptr(new pcl::search::KdTree<pcl::PointXYZRGBA>);
  estimator.setInputCloud(cloud_cam.getCloudProcessed());
  estimator.setSearchMethod(tree_ptr);
  estimator.setRadiusSearch(0.01);
  estimator.compute(*cloud_normals_omp);
  cloud_normals_ = cloud_normals_omp->getMatrixXfMap().cast<double>();
}


std::vector<Quadric> HandSearch::fitQuadrics(const CloudCamera& cloud_cam, const std::vector<int>& indices,
  double radius, const pcl::KdTreeFLANN<pcl::PointXYZRGBA>& kdtree, bool calculates_normals, bool forces_PSD)
{
  double t1 = omp_get_wtime();
  std::vector<Quadric*> quadric_list(indices.size());
  std::vector<int> nn_indices;
  std::vector<float> nn_dists;
  std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d> > T_cams;
  T_cams.push_back(cam_tf_left_);
  T_cams.push_back(cam_tf_right_);

  const PointCloudRGB::Ptr& cloud = cloud_cam.getCloudProcessed();
  const Eigen::MatrixXi& camera_source = cloud_cam.getCameraSource();

#ifdef _OPENMP // parallelization using OpenMP
#pragma omp parallel for private(nn_indices, nn_dists) num_threads(num_threads_)
#endif
  for (int i = 0; i < indices.size(); i++)
  {
    const pcl::PointXYZRGBA& sample = cloud->points[indices[i]];

    if (kdtree.radiusSearch(sample, radius, nn_indices, nn_dists) > 0)
    {
      Eigen::MatrixXi nn_cam_source(camera_source.rows(), nn_indices.size());
      for (int j = 0; j < nn_indices.size(); j++)
        nn_cam_source.col(j) = camera_source.col(nn_indices[j]);

      Eigen::Vector3d sample_eigen = sample.getVector3fMap().cast<double>();
      Quadric* q = new Quadric(T_cams, cloud, sample_eigen, uses_determinstic_normal_estimation_);
      q->fitQuadric(nn_indices, forces_PSD);
      q->findTaubinNormalAxis(nn_indices, nn_cam_source);
      quadric_list[i] = q;
      if (calculates_normals)
        cloud_normals_.col(indices[i]) = q->getNormal();
    }
  }

  double delt1 = omp_get_wtime();

  std::vector<Quadric> quadrics;
  for (int i = 0; i < quadric_list.size(); i++)
  {
    if (quadric_list[i])
      quadrics.push_back(*quadric_list[i]);
    delete quadric_list[i];
  }
  quadric_list.clear();

  double t2 = omp_get_wtime();
  std::cout << "Deletion done in " << t2 - delt1 << " sec.\n";
  std::cout << "Fitted " << quadrics.size() << " quadrics in " << t2 - t1 << " sec.\n";

//  quadric_list[0].print(); // debugging
//  plot_.plotLocalAxes(quadric_list, cloud);

  return quadrics;
}


std::vector<Quadric> HandSearch::calculateLocalFrames(const CloudCamera& cloud_cam,
  const std::vector<int>& indices, double radius, const pcl::KdTreeFLANN<pcl::PointXYZRGBA>& kdtree)
{
  const int MIN_NEIGHBORS = 20;

  double t1 = omp_get_wtime();
  std::vector<Quadric*> quadric_list(indices.size());
  std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d> > T_cams;
  std::vector<int> nn_indices;
  std::vector<float> nn_dists;
  Eigen::Matrix3Xd nn_normals;
  Eigen::VectorXi num_source(cloud_cam.getCameraSource().rows());
  T_cams.push_back(cam_tf_left_);
  T_cams.push_back(cam_tf_right_);

  const PointCloudRGB::Ptr& cloud = cloud_cam.getCloudProcessed();
  const Eigen::MatrixXi& camera_source = cloud_cam.getCameraSource();

#ifdef _OPENMP // parallelization using OpenMP
#pragma omp parallel for private(nn_indices, nn_dists, nn_normals) num_threads(num_threads_)
#endif
  for (int i = 0; i < indices.size(); i++)
  {
    const pcl::PointXYZRGBA& sample = cloud->points[indices[i]];

    if (kdtree.radiusSearch(sample, radius, nn_indices, nn_dists) > 0)
    {
      int nn_num_samples = 50;
      nn_normals.setZero(3, std::min(nn_num_samples, (int) nn_indices.size()));
      num_source.setZero();

      for (int j = 0; j < nn_normals.cols(); j++)
      {
        int r = rand() % nn_indices.size();
        while (isnan(cloud->points[nn_indices[r]].x))
        {
          r = rand() % indices.size();
        }
        nn_normals.col(j) = cloud_normals_.col(nn_indices[r]);

        for (int cam_idx = 0; cam_idx < camera_source.rows(); cam_idx++)
        {
          if (camera_source(cam_idx, nn_indices[r]) == 1)
            num_source(cam_idx)++;
        }
      }

      // calculate camera source for majority of points
      int majority_cam_source;
      num_source.maxCoeff(&majority_cam_source);

      Eigen::MatrixXd gradient_magnitude = ((nn_normals.cwiseProduct(nn_normals)).colwise().sum()).cwiseSqrt();
      nn_normals = nn_normals.cwiseQuotient(gradient_magnitude.replicate(3, 1));
      Quadric* quadric = new Quadric(T_cams, sample.getVector3fMap().cast<double>(), majority_cam_source);
      quadric->findAverageNormalAxis(nn_normals);
      quadric_list[i] = quadric;
    }
  }

  double delt1 = omp_get_wtime();
  std::vector<Quadric> quadrics;
  for (int i = 0; i < quadric_list.size(); i++)
  {
    if (quadric_list[i])
      quadrics.push_back(*quadric_list[i]);
    delete quadric_list[i];
  }
  quadric_list.clear();

  double t2 = omp_get_wtime();
  std::cout << "Deletion done in " << t2 - delt1 << " sec.\n";
  std::cout << "Fitted " << quadrics.size() << " quadrics in " << t2 - t1 << " sec.\n";

  return quadrics;
}


std::vector<GraspHypothesis> HandSearch::evaluateHands(const CloudCamera& cloud_cam,
  const std::vector<Quadric>& quadric_list, double radius, const pcl::KdTreeFLANN<pcl::PointXYZRGBA>& kdtree)
{
  double t1 = omp_get_wtime();

  // possible angles describing hand orientations
  Eigen::VectorXd angles0 = Eigen::VectorXd::LinSpaced(num_orientations_ + 1, -1.0 * M_PI/2.0, M_PI/2.0);
  Eigen::VectorXd angles = angles0.block(0, 0, num_orientations_, 1);

  std::vector<int> nn_indices;
  std::vector<float> nn_dists;
  Eigen::Matrix3Xd nn_normals;
  Eigen::MatrixXi nn_cam_source;
  Eigen::Matrix3Xd centered_neighborhood;
  const PointCloudRGB::Ptr& cloud = cloud_cam.getCloudProcessed();
  const Eigen::MatrixXi& camera_source = cloud_cam.getCameraSource();
  std::vector< std::vector<GraspHypothesis> > hand_lists(quadric_list.size(), std::vector<GraspHypothesis>(0));

#ifdef _OPENMP // parallelization using OpenMP
#pragma omp parallel for private(nn_indices, nn_dists, nn_normals, nn_cam_source, centered_neighborhood) num_threads(num_threads_)
#endif
  for (std::size_t i = 0; i < quadric_list.size(); i++)
  {
    // std::cout << i << "\n";
    pcl::PointXYZRGBA sample;
    sample.x = quadric_list[i].getSample()(0);
    sample.y = quadric_list[i].getSample()(1);
    sample.z = quadric_list[i].getSample()(2);

    if (kdtree.radiusSearch(sample, radius, nn_indices, nn_dists) > 0)
    {
      nn_normals.setZero(3, nn_indices.size());
      nn_cam_source.setZero(camera_source.rows(), nn_indices.size());
      centered_neighborhood.setZero(3, nn_indices.size());
      for (int j = 0; j < nn_indices.size(); j++)
      {
        nn_cam_source.col(j) = camera_source.col(nn_indices[j]);
        centered_neighborhood.col(j) = (cloud->points[nn_indices[j]].getVector3fMap()
          - sample.getVector3fMap()).cast<double>();
        nn_normals.col(j) = cloud_normals_.col(nn_indices[j]);
      }

      std::vector<GraspHypothesis> hands = calculateHand(centered_neighborhood, nn_normals, nn_cam_source,
        quadric_list[i], angles);
      if (hands.size() > 0)
        hand_lists[i] = hands;
    }
  }

  // concatenate the grasp lists
  double t1_concat = omp_get_wtime();
  std::vector<GraspHypothesis> hypotheses;
  for (std::size_t i = 0; i < hand_lists.size(); i++)
  {
    if (hand_lists[i].size() > 0)
      hypotheses.insert(hypotheses.end(), hand_lists[i].begin(), hand_lists[i].end());
  }

  double t2 = omp_get_wtime();
  std::cout << " Concatenation runtime: " << t2 - t1_concat << " sec.\n";
  std::cout << " Found " << hypotheses.size() << " robot hand poses in " << t2 - t1 << " sec.\n";

  return hypotheses;
}


std::vector<GraspHypothesis> HandSearch::calculateHand(const Eigen::Matrix3Xd& points, const Eigen::Matrix3Xd& normals,
  const Eigen::MatrixXi& cam_source, const Quadric& quadric, const Eigen::VectorXd& angles)
{
  FingerHand finger_hand(finger_width_, hand_outer_diameter_, hand_depth_);

  // local quadric frame
  Eigen::Matrix3d frame;
  frame << quadric.getNormal(), quadric.getCurvatureAxis().cross(quadric.getNormal()), quadric.getCurvatureAxis();

  // transform points into quadric frame and crop them based on <hand_height>
  Eigen::Matrix3Xd points_frame = frame.transpose() * points;
  std::vector<int> indices(points_frame.cols());
  int k = 0;
  for (int i = 0; i < points_frame.cols(); i++)
  {
    if (points_frame(2, i) > -1.0 * hand_height_ && points_frame(2, i) < hand_height_)
    {
      indices[k] = i;
      k++;
    }
  }

  Eigen::Matrix3Xd points_cropped(3, k);
  Eigen::Matrix3Xd normals_cropped(3, k);
  Eigen::MatrixXi cam_source_cropped(cam_source.rows(), k);
  for (int i = 0; i < k; i++)
  {
    points_cropped.col(i) = points.col(indices[i]);
    normals_cropped.col(i) = normals.col(indices[i]);
    cam_source_cropped.col(i) = cam_source.col(indices[i]);
  }

  // calculate grasp hypotheses
  std::vector<GraspHypothesis> hand_list;
  for (int i = 0; i < angles.rows(); i++)
  {
    // rotate points into this hand orientation
    Eigen::Matrix3d rot;
    rot << cos(angles(i)), -1.0 * sin(angles(i)), 0.0, sin(angles(i)), cos(angles(i)), 0.0, 0.0, 0.0, 1.0;
    Eigen::Matrix3d frame_rot = frame * rot;
    Eigen::Matrix3Xd points_rot = frame_rot.transpose() * points_cropped;
    Eigen::Matrix3Xd normals_rot = frame_rot.transpose() * normals_cropped;

    // evaluate finger locations for this orientation
    finger_hand.evaluateFingers(points_rot, init_bite_);

    // check that there are at least two corresponding finger placements
    if (finger_hand.getFingers().cast<int>().sum() > 2)
    {
      finger_hand.evaluateHand();

      if (finger_hand.getHand().cast<int>().sum() > 0)
      {
        // try to move the hand as deep as possible onto the object
        finger_hand.deepenHand(points_rot, init_bite_, hand_depth_);

        // calculate grasp parameters
        std::vector<int> indices_learning = finger_hand.computePointsInClosingRegion(points_rot);
        if (indices_learning.size() == 0)
        {
          std::cout << "#pts_rot: " << points_rot.size() << ", #indices_learning: " << indices_learning.size() << "\n";
          continue;
        }
        Eigen::MatrixXd grasp_pos = finger_hand.calculateGraspParameters(frame_rot, quadric.getSample());
        Eigen::Vector3d binormal = frame_rot.col(0);
        Eigen::Vector3d approach = frame_rot.col(1);
        Eigen::Vector3d axis = frame_rot.col(2);

        // extract data for classification
        Eigen::Matrix3Xd points_in_box(3, indices_learning.size());
        Eigen::Matrix3Xd normals_in_box(3, indices_learning.size());
        Eigen::MatrixXi cam_source_in_box(cam_source.rows(), indices_learning.size());
        for (int j = 0; j < indices_learning.size(); j++)
        {
          points_in_box.col(j) = points_rot.col(indices_learning[j]);
          normals_in_box.col(j) = normals_rot.col(indices_learning[j]);
          cam_source_in_box.col(j) = cam_source.col(indices_learning[j]);
        }
        double grasp_width = points_in_box.row(0).maxCoeff() - points_in_box.row(0).minCoeff();

        // scale <ptsInBox> to fit into unit square
        double left = finger_hand.getLeft();
        double right = finger_hand.getRight();
        double top = finger_hand.getTop();
        double bottom = finger_hand.getBottom();
        double baseline_const = 0.1;
        double left_const = left - 0.5 * (baseline_const - (right - left));
        Eigen::Vector3d lower, scales;
        lower << left_const, bottom, -1.0 * hand_height_;
        scales << 1.0 / baseline_const, 1.0 / ((double) top - bottom), 1.0 / (2.0 * hand_height_);
        points_in_box = scales.replicate(1, points_in_box.cols()).array() * (points_in_box - lower.replicate(1, points_in_box.cols())).array();

        GraspHypothesis hand(axis, approach, binormal, grasp_pos.col(0), grasp_pos.col(1), grasp_pos.col(2),
          grasp_width, points_in_box, normals_in_box, cam_source_in_box);

        // use the "new" antipodal condition
        if (antipodal_method_ == NEW_ANTIPODAL)
        {
          Antipodal antipodal;
          hand.setFullAntipodal(antipodal.evaluateGrasp(points_in_box, normals_in_box, 0.002));
          hand.setHalfAntipodal(antipodal.evaluateGrasp(points_in_box, normals_in_box, 0.003));
        }
        // use the "old" antipodal condition
        else if (antipodal_method_ == OLD_ANTIPODAL)
        {
          Antipodal antipodal;
          int antipodal_type = antipodal.evaluateGrasp(normals_in_box, 20, 20);
          if (antipodal_type == Antipodal::FULL_GRASP)
          {
            hand.setHalfAntipodal(true);
            hand.setFullAntipodal(true);
          }
          else if (antipodal_type == Antipodal::HALF_GRASP)
            hand.setHalfAntipodal(true);
        }
        else
        {
          hand.setHalfAntipodal(false);
          hand.setFullAntipodal(false);
        }

        hand_list.push_back(hand);
      }
    }
  }

  return hand_list;
}

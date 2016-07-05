#include <agile_grasp2/hand_search.h>


std::vector<GraspHypothesis> HandSearch::generateHypotheses(const CloudCamera& cloud_cam, int antipodal_mode,
  bool use_samples, bool forces_PSD, bool plots_normals, bool plots_samples)
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
    std::cout << "Estimating normals for all points using pcl::NormalEstimationOMP\n";
    calculateNormalsOMP(cloud_cam);
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

  // 2. Estimate local reference frames.
  std::cout << "Estimating local reference frames ...\n";
  std::vector<LocalFrame> frames;
  if (use_samples)
    frames = calculateLocalFrames(cloud_cam, cloud_cam.getSamples(), nn_radius_taubin_, kdtree);
  else
    frames = calculateLocalFrames(cloud_cam, cloud_cam.getSampleIndices(), nn_radius_taubin_, kdtree);
  if (plots_local_axes_)
    plot_.plotLocalAxes(frames, cloud_cam.getCloudOriginal());

  // 3. Generate grasp hypotheses.
  std::cout << "Finding hand poses ...\n";
  std::vector<GraspHypothesis> hypotheses = evaluateHands(cloud_cam, frames, nn_radius_hands_, kdtree);

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

  cam_tf_left_ = params.cam_tf_left_;
  cam_tf_right_ = params.cam_tf_right_;
}


void HandSearch::calculateNormalsOMP(const CloudCamera& cloud_cam)
{
  pcl::NormalEstimationOMP<pcl::PointXYZRGBA, pcl::Normal> estimator(num_threads_);
  pcl::PointCloud<pcl::Normal>::Ptr cloud_normals_omp(new pcl::PointCloud<pcl::Normal>);
  pcl::search::KdTree<pcl::PointXYZRGBA>::Ptr tree_ptr(new pcl::search::KdTree<pcl::PointXYZRGBA>);
  estimator.setViewPoint(0,0,0);
  estimator.setInputCloud(cloud_cam.getCloudProcessed());
  estimator.setSearchMethod(tree_ptr);
  estimator.setRadiusSearch(0.01);
  estimator.compute(*cloud_normals_omp);
  cloud_normals_ = cloud_normals_omp->getMatrixXfMap().cast<double>();
}


std::vector<LocalFrame> HandSearch::calculateLocalFrames(const CloudCamera& cloud_cam,
  const std::vector<int>& indices, double radius, const pcl::KdTreeFLANN<pcl::PointXYZRGBA>& kdtree)
{
  const int MIN_NEIGHBORS = 20;

  double t1 = omp_get_wtime();
  std::vector<LocalFrame*> frames(indices.size());
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
          r = rand() % nn_indices.size();
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
      LocalFrame* frame = new LocalFrame(T_cams, sample.getVector3fMap().cast<double>(), majority_cam_source);
      frame->findAverageNormalAxis(nn_normals);
      frames[i] = frame;
//      frames[i]->print();
    }
  }

  std::vector<LocalFrame> frames_out;
  for (int i = 0; i < frames.size(); i++)
  {
    if (frames[i])
      frames_out.push_back(*frames[i]);
    delete frames[i];
  }
  frames.clear();

  double t2 = omp_get_wtime();
  std::cout << "Fitted " << frames_out.size() << " local reference frames in " << t2 - t1 << " sec.\n";

  return frames_out;
}


std::vector<GraspHypothesis> HandSearch::evaluateHands(const CloudCamera& cloud_cam,
  const std::vector<LocalFrame>& frames, double radius, const pcl::KdTreeFLANN<pcl::PointXYZRGBA>& kdtree)
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
  std::vector< std::vector<GraspHypothesis> > hand_lists(frames.size(), std::vector<GraspHypothesis>(0));

#ifdef _OPENMP // parallelization using OpenMP
#pragma omp parallel for private(nn_indices, nn_dists, nn_normals, nn_cam_source, centered_neighborhood) num_threads(num_threads_)
#endif
  for (std::size_t i = 0; i < frames.size(); i++)
  {
    pcl::PointXYZRGBA sample;
    sample.x = frames[i].getSample()(0);
    sample.y = frames[i].getSample()(1);
    sample.z = frames[i].getSample()(2);

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
        frames[i], angles);
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


std::vector<LocalFrame> HandSearch::calculateLocalFrames(const CloudCamera& cloud_cam, const Eigen::Matrix3Xd& samples,
  double radius, const pcl::KdTreeFLANN<pcl::PointXYZRGBA>& kdtree)
{
  const int MIN_NEIGHBORS = 20;

    double t1 = omp_get_wtime();
    std::vector<LocalFrame*> frames(samples.cols());
    std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d> > T_cams;
    std::vector<int> nn_indices;
    std::vector<float> nn_dists;
    Eigen::Matrix3Xd nn_normals;
    Eigen::VectorXi num_source(cloud_cam.getCameraSource().rows());
    T_cams.push_back(cam_tf_left_);
    T_cams.push_back(cam_tf_right_);

    const PointCloudRGB::Ptr& cloud = cloud_cam.getCloudProcessed();
    const Eigen::MatrixXi& camera_source = cloud_cam.getCameraSource();

//  #ifdef _OPENMP // parallelization using OpenMP
//  #pragma omp parallel for private(nn_indices, nn_dists, nn_normals) num_threads(num_threads_)
//  #endif
    for (int i = 0; i < samples.cols(); i++)
    {
      pcl::PointXYZRGBA sample;
      sample.x = samples(0,i);
      sample.y = samples(1,i);
      sample.z = samples(2,i);
//      std::cout << "samples: " << samples.col(i).transpose() << std::endl;
//      std::cout << "sample: " << sample << std::endl;

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
            r = rand() % nn_indices.size();
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
        LocalFrame* frame = new LocalFrame(T_cams, sample.getVector3fMap().cast<double>(), majority_cam_source);
        frame->findAverageNormalAxis(nn_normals);
        frames[i] = frame;
  //      frame[i]->print();
      }
    }

    double delt1 = omp_get_wtime();
    std::vector<LocalFrame> frames_out;
    for (int i = 0; i < frames.size(); i++)
    {
      if (frames[i])
        frames_out.push_back(*frames[i]);
      delete frames[i];
    }
    frames.clear();

    double t2 = omp_get_wtime();
    std::cout << "Fitted " << frames_out.size() << " local reference frames in " << t2 - t1 << " sec.\n";

    return frames_out;
}

std::vector<GraspHypothesis> HandSearch::calculateHand(const Eigen::Matrix3Xd& points, const Eigen::Matrix3Xd& normals,
  const Eigen::MatrixXi& cam_source, const LocalFrame& local_frame, const Eigen::VectorXd& angles)
{
  FingerHand finger_hand(finger_width_, hand_outer_diameter_, hand_depth_);

  // extract axes of local reference frame
  Eigen::Matrix3d frame;
  frame << local_frame.getNormal(), local_frame.getBinormal(), local_frame.getCurvatureAxis();

  // transform points into local reference frame and crop them based on <hand_height>
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
        Eigen::MatrixXd grasp_pos = finger_hand.calculateGraspParameters(frame_rot, local_frame.getSample());
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

        // evaluate if the grasp is antipodal
        Antipodal antipodal;
        int antipodal_result = antipodal.evaluateGrasp(points_in_box, normals_in_box, 0.003);
        hand.setHalfAntipodal(antipodal_result == Antipodal::HALF_GRASP || antipodal_result == Antipodal::FULL_GRASP);
        hand.setFullAntipodal(antipodal_result == Antipodal::FULL_GRASP);

        hand_list.push_back(hand);
      }
    }
  }

  return hand_list;
}

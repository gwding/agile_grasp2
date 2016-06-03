#include <agile_grasp2/grasp_detector.h>


/** constants for antipodal mode */
const int GraspDetector::NONE = 0; ///< no prediction/calculation of antipodal grasps, uses grasp hypotheses
const int GraspDetector::PREDICTION = 1; ///< predicts antipodal grasps
const int GraspDetector::GEOMETRIC = 2; ///< calculates antipodal grasps

/** constants for plotting */
const int GraspDetector::NO_PLOTTING = 0; ///< nothing is plotted
const int GraspDetector::PCL = 1; ///< everything is plotted in pcl-visualizer
const int GraspDetector::RVIZ = 2; ///< everything is plotted in rviz


GraspDetector::GraspDetector(ros::NodeHandle& node) : use_incoming_samples_(false), voxel_size_(0.003)
{
  // read hand search parameters
  HandSearch::Parameters params;
  node.getParam("workspace", workspace_);
  node.getParam("camera_pose", camera_pose_);
  node.param("num_samples", params.num_samples_, 1000);
  num_samples_ = params.num_samples_;
  node.getParam("sample_indices", indices_);
  if (indices_.size() > 0)
  {
    num_samples_ = indices_.size();
    params.num_samples_ = num_samples_;
    ROS_INFO_STREAM("params.num_samples_: " << params.num_samples_ << ", num_samples: " << num_samples_);
  }
  node.param("num_threads", params.num_threads_, 1);
  node.param("nn_radius_taubin", params.nn_radius_taubin_, 0.01);
  node.param("nn_radius_hands", params.nn_radius_hands_, 0.1);
  node.param("num_orientations", params.num_orientations_, 8);
  node.param("normal_estimation_method", params.normal_estimation_method_, 0);
  node.param("voxelize", voxelize_, true);
  node.param("filter_half_grasps", filter_half_grasps_, true);

  // read hand geometry parameters
  node.param("finger_width", params.finger_width_, 0.01);
  node.param("hand_outer_diameter", params.hand_outer_diameter_, 0.09);
  outer_diameter_ = params.hand_outer_diameter_;
  node.param("hand_depth", params.hand_depth_, 0.06);
  node.param("hand_height", params.hand_height_, 0.02);
  node.param("init_bite", params.init_bite_, 0.015);

  hand_search_.setParameters(params);

  // read classification parameters
  std::string model_file, trained_file, label_file;
  node.param("antipodal_mode", antipodal_mode_, PREDICTION);
  node.param("model_file", model_file, std::string(""));
  node.param("trained_file", trained_file, std::string(""));
  node.param("label_file", label_file, std::string(""));
  node.param("min_score_diff", min_score_diff_, 500.0);
  node.param("batch_size", batch_size_, 10);
  classifier_ = new Classifier(model_file, trained_file, label_file);
  learning_  = new Learning(60, params.num_threads_);

  // read handle search parameters
  int min_inliers;
  double min_handle_length;
  bool reuse_inliers;
  node.param("min_inliers", min_inliers, 0);
  node.param("min_length", min_handle_length, 0.005);
  node.param("reuse_inliers", reuse_inliers, true);
  handle_search_.setMinInliers(min_inliers);
  handle_search_.setMinLength(min_handle_length);
  handle_search_.setReuseInliers(reuse_inliers);

  // read grasp selection parameters
  std::vector<double> gripper_width_range(2);
  gripper_width_range[0] = 0.03;
  gripper_width_range[1] = 0.07;
  node.param("num_selected", num_selected_, 50);
  node.getParam("gripper_width_range", gripper_width_range);
  min_aperture_ = gripper_width_range[0];
  max_aperture_ = gripper_width_range[1];

  // read plotting parameters
  node.param("plot_mode", plot_mode_, PCL);
  node.param("only_plot_output", only_plot_output_, true);
}


std::vector<GraspHypothesis> GraspDetector::detectGraspPoses(const CloudCamera& cloud_cam)
{
  if (cloud_cam.getCloudOriginal()->size() == 0)
  {
    ROS_INFO("Point cloud is empty!");
    std::vector<GraspHypothesis> empty(0);
    return empty;
  }

  double t0_total = omp_get_wtime();

  Plot plotter;

  // plot the indices/samples
  if (!only_plot_output_ && plot_mode_ == PCL && indices_.size() > 0)
  {
    plotter.plotSamples(indices_, cloud_cam.getCloudOriginal());
  }
  else if (use_incoming_samples_ && plot_mode_ == PCL)
  {
    plotter.plotSamples(cloud_cam.getSamples(), cloud_cam.getCloudOriginal());
  }

  // camera poses for 2-camera Baxter setup
  Eigen::Matrix4d cam_tf_left, cam_tf_right; // camera poses
  if (camera_pose_.size() == 0)
  {
    Eigen::Matrix4d base_tf, sqrt_tf;

    base_tf << 0, 0.445417, 0.895323, 0.215,
               1, 0, 0, -0.015,
               0, 0.895323, -0.445417, 0.23,
               0, 0, 0, 1;

    sqrt_tf << 0.9366, -0.0162, 0.3500, -0.2863,
               0.0151, 0.9999, 0.0058, 0.0058,
               -0.3501, -0.0002, 0.9367, 0.0554,
               0, 0, 0, 1;


    cam_tf_left = base_tf * sqrt_tf.inverse();
    cam_tf_right = base_tf * sqrt_tf;
    hand_search_.setCamTfLeft(cam_tf_left);
    hand_search_.setCamTfRight(cam_tf_right);
  }
  // camera pose from launch file
  else
  {
    cam_tf_left <<  camera_pose_[0], camera_pose_[1], camera_pose_[2], camera_pose_[3],
                    camera_pose_[4], camera_pose_[5], camera_pose_[6], camera_pose_[7],
                    camera_pose_[8], camera_pose_[9], camera_pose_[10], camera_pose_[11],
                    camera_pose_[12], camera_pose_[13], camera_pose_[14], camera_pose_[15];
    hand_search_.setCamTfLeft(cam_tf_left);
  }


  // 1. Generate grasp hypotheses.
  std::vector<GraspHypothesis> hands = hand_search_.generateHypotheses(cloud_cam, 0, use_incoming_samples_);
  ROS_INFO_STREAM("# grasp hypotheses: " << hands.size());
  if (!only_plot_output_ && plot_mode_ == PCL)
  {
    plotter.plotFingers(hands, cloud_cam.getCloudProcessed(), "Grasp Hypotheses Within Workspace");
  }

  // 2. Prune on aperture and fingers below table surface.
  std::vector<GraspHypothesis> hands_filtered;
  if (indices_.size() == 0)
  {
    Eigen::Vector4f min_bound, max_bound;
    pcl::getMinMax3D(*cloud_cam.getCloudProcessed(), min_bound, max_bound);
    hands_filtered = pruneGraspsOnHandParameters(hands, workspace_[0], workspace_[1], workspace_[2], workspace_[3], min_bound(2));
    ROS_INFO_STREAM("# grasps within gripper width range and workspace: " << hands_filtered.size());
  }
  else
  {
    hands_filtered = hands;
  }

  if (!only_plot_output_ && plot_mode_ == PCL)
  {
    plotter.plotFingers(hands_filtered, cloud_cam.getCloudProcessed(),
                        "Grasp Hypotheses Within Gripper Width Limits And After Finger Pruning");
  }

  // 3. Predict or extract antipodal grasps (or do nothing).
  std::vector<GraspHypothesis> antipodal_hands;
  if (antipodal_mode_ == NONE)
  {
    std::cout << "No antipodal prediction/calculation (using grasp hypotheses) ...\n";
    ROS_INFO_STREAM("# grasp hypotheses: " << hands_filtered.size());
    std::cout << "====> TOTAL TIME: " << omp_get_wtime() - t0_total << std::endl;
    return hands_filtered;
  }
  else if (antipodal_mode_ == PREDICTION)
  {
    std::cout << "Creating grasp images for antipodal prediction ...\n";
    Eigen::Matrix<double, 3, 2> cams_mat;
    cams_mat.col(0) = cam_tf_left.block<3, 1>(0, 3);
    cams_mat.col(1) = cam_tf_right.block<3, 1>(0, 3);
    std::vector<cv::Mat> image_list = learning_->createGraspImages(hands_filtered, cams_mat, false, false);
    std::cout << " done\n";
    double t0_prediction = omp_get_wtime();
    int num_iterations = (int) ceil(image_list.size() / (double) batch_size_);
    std::cout << "num_iterations: " << num_iterations << "\n";
    for (int i = 0; i < num_iterations; i++)
    {
      std::vector<cv::Mat>::iterator end_it;
      if (i < num_iterations - 1)
        end_it = image_list.begin() + (i+1)*100;
      else
        end_it =  image_list.end();
      std::vector<cv::Mat> sub_image_list(image_list.begin() + i*100, end_it);
      std::vector< std::vector<Prediction> > predictions = classifier_->ClassifyBatch(sub_image_list, 2);

      for (int j = 0; j < predictions.size(); j++)
      {
        double score = predictions[j][1].second - predictions[j][0].second;
//        std::cout << i*100 + j << ", " <<  score << "\n";
        if (score >= min_score_diff_)
        {
          antipodal_hands.push_back(hands_filtered[i*100+j]);
          antipodal_hands[antipodal_hands.size()-1].setFullAntipodal(true);
          antipodal_hands[antipodal_hands.size()-1].setScore(score);
        }
      }
    }
    std::cout << "batch prediction time: " << omp_get_wtime() - t0_prediction << std::endl;
  }
  else if (antipodal_mode_ == GEOMETRIC)
  {
    std::cout << "Calculating antipodal grasps ...\n";

    for (int i = 0; i < hands_filtered.size(); i++)
    {
      if (hands_filtered[i].isFullAntipodal())
        antipodal_hands.push_back(hands_filtered[i]);
    }
  }
  ROS_INFO_STREAM("# antipodal grasps: " << antipodal_hands.size());
  if (!only_plot_output_ && plot_mode_ == PCL)
  {
    plotter.plotFingers(antipodal_hands, cloud_cam.getCloudOriginal(), "Antipodal Grasps");
  }

  // 4. Find grasp clusters.
  if (handle_search_.getMinInliers() > 0)
  {
    antipodal_hands = handle_search_.findClusters(antipodal_hands);
    plotter.plotFingers(antipodal_hands, cloud_cam.getCloudOriginal(), "Clusters");
  }

  // 5. Select the <num_selected_> highest ranking grasps.
  std::vector<GraspHypothesis> hands_selected;
  if (antipodal_hands.size() > num_selected_)
  {
    std::cout << "Partial Sorting the grasps based on their score ... \n";
    std::partial_sort(antipodal_hands.begin(), antipodal_hands.begin() + num_selected_, antipodal_hands.end(),
      isScoreGreater);
    hands_selected.assign(antipodal_hands.begin(), antipodal_hands.begin() + num_selected_);
  }
  else
  {
    std::cout << "Sorting the grasps based on their score ... \n";
    std::sort(antipodal_hands.begin(), antipodal_hands.end(), isScoreGreater);
    hands_selected = antipodal_hands;
  }
  ROS_INFO_STREAM("# highest-ranking antipodal grasps: " << hands_selected.size());
  std::cout << "====> TOTAL TIME: " << omp_get_wtime() - t0_total << std::endl;
  if (plot_mode_ == PCL)
  {
    plotter.plotFingers(hands_selected, cloud_cam.getCloudOriginal(), "Highest-Ranking Antipodal Grasps");
  }

  return hands_selected;
}


void GraspDetector::preprocessPointCloud(CloudCamera& cloud_cam)
{
  std::cout << "Processing cloud with: " << cloud_cam.getCloudOriginal()->size() << " points.\n";

  if (indices_.size() == 0)
  {
    // 1. Workspace filtering
    Eigen::VectorXd ws(6);
    ws << workspace_[0], workspace_[1], workspace_[2], workspace_[3], workspace_[4], workspace_[5];
    cloud_cam.filterWorkspace(ws);
    std::cout << "After workspace filtering: " << cloud_cam.getCloudProcessed()->size() << " points left.\n";

    // 2. Voxelization
    if (voxelize_)
    {
      cloud_cam.voxelizeCloud(voxel_size_);
      std::cout << "After voxelization: " << cloud_cam.getCloudProcessed()->size() << " points left.\n";
    }

    // 3. Subsampling
    if (use_incoming_samples_)
    {
      // remove samples outside of the workspace
      agile_grasp2::SamplesMsg filtered_samples_msg;
      for (int i = 0; i < samples_msg_.samples.size(); ++i)
      {
        const geometry_msgs::Point& p = samples_msg_.samples[i];
        if (p.x > ws(0) && p.x < ws(1) && p.y > ws(2) && p.y < ws(3) && p.z > ws(4) && p.z < ws(5))
          filtered_samples_msg.samples.push_back(p);
      }
      std::cout << "Workspace filtering removed " << samples_msg_.samples.size() - filtered_samples_msg.samples.size()
        << " samples.\n";

      cloud_cam.subsampleSamples(filtered_samples_msg, num_samples_);
      std::cout << "Using " << filtered_samples_msg.samples.size() << " samples from external source.\n";
    }
    else if (num_samples_ > cloud_cam.getCloudProcessed()->size())
    {
      std::vector<int> indices_all(cloud_cam.getCloudProcessed()->size());
      for (int i=0; i < cloud_cam.getCloudProcessed()->size(); i++)
        indices_all[i] = i;
      cloud_cam.setSampleIndices(indices_all);
      std::cout << "Cloud is smaller than num_samples. Subsampled all " << cloud_cam.getCloudProcessed()->size()
        << " points.\n";
    }
    else
    {
      cloud_cam.subsampleUniformly(num_samples_);
      std::cout << "Subsampled " <<  num_samples_ << " at random uniformly.\n";
    }
  }
  else
  {
    if (num_samples_ != indices_.size() && num_samples_ < cloud_cam.getCloudOriginal()->size())
    {
      std::vector<int> indices_rand(num_samples_);
      for (int i=0; i < num_samples_; i++)
        indices_rand[i] = indices_[rand() % indices_.size()];
      cloud_cam.setSampleIndices(indices_rand);
    }
    else
    {
      cloud_cam.setSampleIndices(indices_);
    }
  }
}


void GraspDetector::setIndicesFromMsg(const agile_grasp2::CloudIndexed& msg)
{
  indices_.resize(msg.indices.size());
  for (int i=0; i < indices_.size(); i++)
  {
    indices_[i] = msg.indices[i].data;
  }
}


std::vector<GraspHypothesis> GraspDetector::pruneGraspsOnHandParameters(const std::vector<GraspHypothesis>& hands,
  float min_x, float max_x, float min_y, float max_y, float min_z)
{
  std::vector<GraspHypothesis> hands_filtered(0);

  for (int i = 0; i < hands.size(); ++i)
  {
    if (!filter_half_grasps_ || hands[i].isHalfAntipodal())
    {
      double half_width = 0.5 * outer_diameter_;
      Eigen::Vector3d left_bottom = hands[i].getGraspBottom() + half_width * hands[i].getBinormal();
      Eigen::Vector3d right_bottom = hands[i].getGraspBottom() - half_width * hands[i].getBinormal();
      Eigen::Vector3d left_top = hands[i].getGraspTop() + half_width * hands[i].getBinormal();
      Eigen::Vector3d right_top = hands[i].getGraspTop() - half_width * hands[i].getBinormal();
      Eigen::Vector3d approach = hands[i].getGraspBottom() - 0.10 * hands[i].getApproach();
      Eigen::VectorXd x(5), y(5), z(5);
      x << left_bottom(0), right_bottom(0), left_top(0), right_top(0), approach(0);
      y << left_bottom(1), right_bottom(1), left_top(1), right_top(1), approach(1);
      z << left_bottom(2), right_bottom(2), left_top(2), right_top(2), approach(2);
      double aperture = hands[i].getGraspWidth();

      if (aperture >= min_aperture_ && aperture <= max_aperture_ // make sure the object fits into the hand
          && z.minCoeff() >= min_z // avoid grasping below the table
          && y.minCoeff() >= min_y && y.maxCoeff() <= max_y // avoid grasping outside the y-workspace
          && x.minCoeff() >= min_x && x.maxCoeff() <= max_x) // avoid grasping outside the x-workspace
      {
        hands_filtered.push_back(hands[i]);
      }
    }
  }

  return hands_filtered;
}

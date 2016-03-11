#include <nodes/grasp_detection_node.h>


GraspDetectionNode::GraspDetectionNode(ros::NodeHandle& node) : has_cloud_(false), has_normals_(false),
  cloud_(new PointCloudRGBA), cloud_normals_(new PointCloudNormal), voxel_size_(0.003), size_left_cloud_(0)
{
  indices_.resize(0);

  // read environment parameters
  node.getParam("workspace", workspace_);
  node.getParam("camera_pose", camera_pose_);

  int cloud_type;
  node.param("cloud_type", cloud_type, POINT_CLOUD_2);

  // read point cloud from ROS topic
  if (cloud_type != PCD_FILE)
  {
    std::string cloud_topic;
    node.param("cloud_topic", cloud_topic, std::string("/camera/depth_registered/points"));

    // subscribe to input point cloud ROS topic
    if (cloud_type == POINT_CLOUD_2)
      cloud_sub_ = node.subscribe(cloud_topic, 1, &GraspDetectionNode::cloud_callback, this);
    else if (cloud_type == CLOUD_SIZED)
      cloud_sub_ = node.subscribe(cloud_topic, 1, &GraspDetectionNode::cloud_sized_callback, this);
    else if (cloud_type == CLOUD_INDEXED)
      cloud_sub_ = node.subscribe(cloud_topic, 1, &GraspDetectionNode::cloud_indexed_callback, this);

    bool use_service;
    node.param("use_service", use_service, false);

    // uses a ROS service callback to provide grasps
    if (use_service)
    {
      grasps_service_ = node.advertiseService("find_grasps", &GraspDetectionNode::graspsCallback, this);
    }
    // uses a ROS topic to publish grasps
    else
    {
      grasps_pub_ = node.advertise<agile_grasp2::GraspListMsg>("grasps", 10);
    }
  }

  // read hand search parameters
  HandSearch::Parameters params;
  node.getParam("workspace", workspace_);
  node.getParam("camera_pose", camera_pose_);
  node.param("num_samples", params.num_samples_, 1000);
  num_samples_ = params.num_samples_;
  node.param("num_threads", params.num_threads_, 1);
  node.param("nn_radius_taubin", params.nn_radius_taubin_, 0.01);
  node.param("nn_radius_hands", params.nn_radius_hands_, 0.1);
  node.param("num_orientations", params.num_orientations_, 8);
  node.param("antipodal_method", params.antipodal_mode_, 0);
  node.param("normal_estimation_method", params.normal_estimation_method_, 0);

  // read hand geometry parameters
  node.param("finger_width", params.finger_width_, 0.01);
  node.param("hand_outer_diameter", params.hand_outer_diameter_, 0.09);
  node.param("hand_depth", params.hand_depth_, 0.06);
  node.param("hand_height", params.hand_height_, 0.02);
  node.param("init_bite", params.init_bite_, 0.015);

  hand_search_.setParameters(params);

  // read CAFFE classification parameters
  std::string model_file, trained_file, label_file;
  node.param("model_file", model_file, std::string(""));
  node.param("trained_file", trained_file, std::string(""));
  node.param("label_file", label_file, std::string(""));
  node.param("min_score_diff", min_score_diff_, 500.0);
  classifier_ = new Classifier(model_file, trained_file, label_file);
  learning_  = new Learning(60, params.num_threads_);

  // read handle search parameters
  int min_inliers;
  double min_handle_length;
  bool reuse_inliers;
  node.param("min_inliers", min_inliers, 5);
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

  // read optional general parameters
  node.param("plot_mode", plot_mode_, PCL);
  node.param("only_plot_output", only_plot_output_, true);
  node.param("save_hypotheses", save_hypotheses_, false);
  node.param("images_directory", grasp_image_dir_, std::string("/home/andreas/data/grasp_images/"));
}


void GraspDetectionNode::run()
{
  ros::Rate rate(1);
  ROS_INFO("Waiting for point cloud to arrive ...");

  while (ros::ok())
  {
    if (has_cloud_)
    {
      // detect grasps in point cloud
      std::vector<GraspHypothesis> grasps = detectGraspPosesInTopic();

      // output grasps as ROS message
      agile_grasp2::GraspListMsg grasps_msg = createGraspListMsg(grasps);
      grasps_pub_.publish(grasps_msg);
      ROS_INFO_STREAM("Published " << grasps_msg.grasps.size() << " grasps.");

      // reset the system
      has_cloud_ = false;
      ROS_INFO("Waiting for point cloud to arrive ...");
    }

    ros::spinOnce();
    rate.sleep();
  }
}


std::vector<GraspHypothesis> GraspDetectionNode::detectGraspPosesInFile(const std::string& file_name_left,
  const std::string& file_name_right)
{
  CloudCamera* cloud_cam;
  if (file_name_right.length() == 0)
    cloud_cam = new CloudCamera(file_name_left);
  else
    cloud_cam = new CloudCamera(file_name_left, file_name_right);

  preprocessPointCloud(workspace_, voxel_size_, num_samples_, *cloud_cam);
  std::vector<GraspHypothesis> handles = detectGraspPoses(*cloud_cam);

  delete cloud_cam;

  return handles;
}


std::vector<GraspHypothesis> GraspDetectionNode::detectGraspPosesInTopic()
{
  CloudCamera cloud_cam(cloud_, size_left_cloud_);
  preprocessPointCloud(workspace_, voxel_size_, num_samples_, cloud_cam);
  return detectGraspPoses(cloud_cam);
}


std::vector<GraspHypothesis> GraspDetectionNode::detectGraspPoses(CloudCamera& cloud_cam)
{
  if (cloud_cam.getCloudOriginal()->size() == 0)
  {
    ROS_INFO("Point cloud is empty!");
    std::vector<GraspHypothesis> empty(0);
    return empty;
  }

  double t0_total = omp_get_wtime();

  Plot plotter;

  if (!only_plot_output_ && plot_mode_ == PCL && indices_.size() > 0)
  {
    plotter.plotSamples(indices_, cloud_cam.getCloudOriginal());
  }

  // camera poses for 2-camera Baxter setup
  Eigen::Matrix4d base_tf, sqrt_tf;

  base_tf << 0, 0.445417, 0.895323, 0.215,
             1, 0, 0, -0.015,
             0, 0.895323, -0.445417, 0.23,
             0, 0, 0, 1;

  sqrt_tf << 0.9366, -0.0162, 0.3500, -0.2863,
             0.0151, 0.9999, 0.0058, 0.0058,
             -0.3501, -0.0002, 0.9367, 0.0554,
             0, 0, 0, 1;

  Eigen::Matrix4d cam_tf_left, cam_tf_right; // camera poses
  cam_tf_left = base_tf * sqrt_tf.inverse();
  cam_tf_right = base_tf * sqrt_tf;


  // 1. Generate grasp hypotheses.
  std::vector<GraspHypothesis> hands = hand_search_.generateHypotheses(cloud_cam, 0);
  ROS_INFO_STREAM("# grasp hypotheses: " << hands.size());
  if (!only_plot_output_ && plot_mode_ == PCL)
  {
//    plotter.plotHands(hands, cloud_cam.getCloudOriginal(), "Grasp Hypotheses");
    plotter.plotHands(hands, cloud_cam.getCloudProcessed(), "Grasp Hypotheses Within Workspace");
  }
//  plotter.drawCloud(cloud_cam.getCloudProcessed(), "Workspace Filtered Cloud");

  // 2. Prune on aperture and fingers below table surface.
  std::vector<GraspHypothesis> hands_filtered;
  if (indices_.size() == 0)
  {
    Eigen::VectorXd ws(6);
    ws << workspace_[0] + 0.04, workspace_[1] - 0.02, workspace_[2] + 0.03, workspace_[3] - 0.03, workspace_[4],
      workspace_[5];

    for (int i = 0; i < hands.size(); ++i)
    {
      double width = hands[i].getGraspWidth();
      Eigen::Vector3d left_bottom = hands[i].getGraspBottom() + 0.5 * width * hands[i].getBinormal();
      Eigen::Vector3d right_bottom = hands[i].getGraspBottom() - 0.5 * width * hands[i].getBinormal();
      Eigen::Vector3d left_top = hands[i].getGraspTop() + 0.5 * width * hands[i].getBinormal();
      Eigen::Vector3d right_top = hands[i].getGraspTop() - 0.5 * width * hands[i].getBinormal();
      Eigen::Vector4d z;
      z << left_bottom(2), right_bottom(2), left_top(2), right_top(2), hands[i].getGraspBottom()(2);

      if (hands[i].getGraspWidth() >= min_aperture_ && hands[i].getGraspWidth() <= max_aperture_
        && z.minCoeff() >= ws(4) && z.maxCoeff() <= ws(5)
        && left_bottom(0) >= ws(0) && left_bottom(0) <= ws(1)
        && left_bottom(1) >= ws(2) && left_bottom(1) <= ws(3))
      {
        hands_filtered.push_back(hands[i]);
      }
    }
    ROS_INFO_STREAM("# grasps within gripper width range and workspace: " << hands_filtered.size());
  }
  else
  {
    hands_filtered = hands;
  }

  if (!only_plot_output_ && plot_mode_ == PCL)
  {
    plotter.plotHands(hands_filtered, cloud_cam.getCloudProcessed(),
      "Grasp Hypotheses Within Gripper Width Limits And After Finger Pruning");
  }


  // 3. Predict or extract antipodal grasps.
  std::vector<GraspHypothesis> antipodal_hands;
  if (hand_search_.getAntipodalMethod() == hand_search_.NO_ANTIPODAL)
  {
    std::cout << "Creating grasp images ...\n";
    Eigen::Matrix<double, 3, 2> cams_mat;
    cams_mat.col(0) = cam_tf_left.block<3, 1>(0, 3);
    cams_mat.col(1) = cam_tf_right.block<3, 1>(0, 3);
  //  std::vector<cv::Mat> image_list = learn.storeGraspImages(hands, cams_mat, grasp_image_dir_, false);
    std::vector<cv::Mat> image_list = learning_->createGraspImages(hands_filtered, cams_mat, false);
    for (int i = 0; i < image_list.size(); i++)
    {
      std::vector<Prediction> predictions = classifier_->Classify(image_list[i]);
  //    std::cout << i << ": ";
  //    for (int j = 0; j < predictions.size(); j++)
  //      std::cout << std::fixed << std::setprecision(4) << predictions[j].second << " " << predictions[j].first << " ";

      if (predictions[1].second - predictions[0].second >= min_score_diff_)
      {
        antipodal_hands.push_back(hands_filtered[i]);
        antipodal_hands[antipodal_hands.size()-1].setFullAntipodal(true);
        antipodal_hands[antipodal_hands.size()-1].setScore(predictions[1].second - predictions[0].second);
  //      std::cout << " OK";
      }

  //    std::cout << std::endl;
    }
  }
  else
  {
    for (int i = 0; i < hands_filtered.size(); i++)
    {
      if (hands_filtered[i].isFullAntipodal())
        antipodal_hands.push_back(hands_filtered[i]);
    }
  }
  ROS_INFO_STREAM("# antipodal grasps: " << antipodal_hands.size());
  if (!only_plot_output_ && plot_mode_ == PCL)
  {
//    plotter.plotHands(hands, antipodal_hands, cloud_cam.getCloudOriginal(), "Antipodal Grasps");
//    plotter.plotFingers(antipodal_hands, cloud_cam.getCloudOriginal(), "Antipodal Grasps");
    plotter.plotHands(antipodal_hands, cloud_cam.getCloudOriginal(), "Antipodal Grasps");
  }


  // 4. Select the <num_selected_> highest ranking grasps.
  std::vector<GraspHypothesis> hands_selected;
  if (antipodal_hands.size() > num_selected_)
  {
    std::cout << "Partial Sorting the grasps based on their score ... \n";
    std::partial_sort(antipodal_hands.begin(), antipodal_hands.begin() + num_selected_, antipodal_hands.end(),
      isScoreGreater);
    hands_selected.assign(antipodal_hands.begin(), antipodal_hands.begin() + num_selected_);
//  std::vector<Handle> handles_selected;
//  std::partial_sort_copy(handles.begin(), handles.end(), handles_selected)
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
    //~ plotter.plotHands(hands_selected, cloud_cam.getCloudOriginal(), "Highest-Ranking Antipodal Grasps");
    plotter.plotFingers(hands_selected, cloud_cam.getCloudOriginal(), "Highest-Ranking Antipodal Grasps");
  }
//  for (int i = 0; i < hands_selected.size(); ++i)
//    std::cout << "hand " << i << ", score: " << hands_selected[i].getScore() << std::endl;

  return hands_selected;
}


void GraspDetectionNode::preprocessPointCloud(const std::vector<double>& workspace, double voxel_size, int num_samples,
  CloudCamera& cloud_cam)
{
  std::cout << "Processing cloud with: " << cloud_cam.getCloudOriginal()->size() << " points.\n";

  if (indices_.size() == 0)
  {
    // 1. Workspace filtering
    Eigen::VectorXd ws(6);
    ws << workspace[0], workspace[1], workspace[2], workspace[3], workspace[4], workspace[5];
    cloud_cam.filterWorkspace(ws);
    std::cout << "After workspace filtering: " << cloud_cam.getCloudProcessed()->size() << " points left.\n";

    // 2. Voxelization
    cloud_cam.voxelizeCloud(voxel_size);
    std::cout << "After voxelization: " << cloud_cam.getCloudProcessed()->size() << " points left.\n";

    // 3. Subsampling
    cloud_cam.subsampleUniformly(num_samples);
  }
  else
  {
    if (num_samples < cloud_cam.getCloudOriginal()->size())
    {
      std::vector<int> indices_rand(num_samples);
      for (int i=0; i < num_samples; i++)
        indices_rand[i] = indices_[rand() % indices_.size()];
      cloud_cam.setSampleIndices(indices_rand);
    }
    else
    {
      cloud_cam.setSampleIndices(indices_);
    }  
  }
}


bool GraspDetectionNode::graspsCallback(agile_grasp2::FindGrasps::Request& req, agile_grasp2::FindGrasps::Response& resp)
{
  ROS_INFO("Received grasp pose detection request ...");

  if (!has_cloud_)
  {
    ROS_INFO("No point cloud available!");
    return false;
  }

  CloudCamera* cloud_cam;

  // cloud with surface normals
  if (has_normals_)
    cloud_cam = new CloudCamera(cloud_normals_, size_left_cloud_);
  // cloud without surface normals
  else
    cloud_cam = new CloudCamera(cloud_, size_left_cloud_);

  // use all points in the point cloud
  if (req.grasps_signal == ALL_POINTS)
  {
    if (req.num_samples == 0)
      preprocessPointCloud(workspace_, voxel_size_, num_samples_, *cloud_cam);
    else
      preprocessPointCloud(workspace_, voxel_size_, req.num_samples, *cloud_cam);
  }
  // use points within a given radius from a given center point in the point cloud
  else if (req.grasps_signal == RADIUS)
  {
    pcl::PointXYZRGBA centroid;
    centroid.x = req.centroid.x;
    centroid.y = req.centroid.y;
    centroid.z = req.centroid.z;
    std::vector<int> indices_ball = getSamplesInBall(cloud_cam->getCloudOriginal(), centroid, req.radius);
    cloud_cam->setSampleIndices(indices_ball);
  }
  // use points given by a list of indices into the point cloud
  else if (req.grasps_signal == INDICES)
  {
    std::vector<int> indices(req.indices.size());
    for(int i=0; i < req.indices.size(); i++)
      indices[i] = req.indices[i];
    cloud_cam->setSampleIndices(indices);
  }

  std::vector<GraspHypothesis> hands = detectGraspPoses(*cloud_cam);
  // TODO: fill response

  delete cloud_cam;

  return true;
}


std::vector<int> GraspDetectionNode::getSamplesInBall(const PointCloudRGBA::Ptr& cloud,
  const pcl::PointXYZRGBA& centroid, float radius)
{
  std::vector<int> indices;
  std::vector<float> dists;
  pcl::KdTreeFLANN<pcl::PointXYZRGBA> kdtree;
  kdtree.setInputCloud(cloud);
  kdtree.radiusSearch(centroid, radius, indices, dists);
  return indices;
}


void GraspDetectionNode::cloud_callback(const sensor_msgs::PointCloud2ConstPtr& msg)
{
  if (!has_cloud_)
  {
    pcl::fromROSMsg(*msg, *cloud_);
    size_left_cloud_ = cloud_->size();
    has_cloud_ = true;
    ROS_INFO_STREAM("Received cloud with " << cloud_->size() << " points");
  }
}


void GraspDetectionNode::cloud_sized_callback(const agile_grasp2::CloudSized& msg)
{
  if (!has_cloud_)
  {
    if (msg.cloud.fields.size() == 6 && msg.cloud.fields[3].name == "normal_x" && msg.cloud.fields[4].name == "normal_y"
      && msg.cloud.fields[3].name == "normal_x")
    {
      pcl::fromROSMsg(msg.cloud, *cloud_normals_);
      has_normals_ = true;
    }
    else
    {
      pcl::fromROSMsg(msg.cloud, *cloud_);
    }

    size_left_cloud_ = msg.size_left.data;
    has_cloud_ = true;
    ROS_INFO_STREAM("Received cloud with " << cloud_->size() << " points (left: ) " << size_left_cloud_ << ", right: "
      << cloud_->points.size() - size_left_cloud_);
  }
}


void GraspDetectionNode::cloud_indexed_callback(const agile_grasp2::CloudIndexed& msg)
{
  if (!has_cloud_)
  {
    if (msg.cloud.fields.size() == 6 && msg.cloud.fields[3].name == "normal_x" && msg.cloud.fields[4].name == "normal_y"
      && msg.cloud.fields[3].name == "normal_x")
    {
      pcl::fromROSMsg(msg.cloud, *cloud_normals_);
      has_normals_ = true;
    }
    else
    {
      pcl::fromROSMsg(msg.cloud, *cloud_);
    }

    size_left_cloud_ = cloud_->size();
    indices_.resize(msg.indices.size());
    for (int i=0; i < indices_.size(); i++)
      indices_[i] = msg.indices[i].data;
    has_cloud_ = true;
    ROS_INFO_STREAM("Received cloud with " << cloud_->size() << " points and " << indices_.size() << " indices\n");
  }
}


agile_grasp2::GraspListMsg GraspDetectionNode::createGraspListMsg(const std::vector<Handle>& handles)
{
  agile_grasp2::GraspListMsg msg;
  for (int i = 0; i < handles.size(); i++)
    msg.grasps.push_back(handles[i].convertToGraspMsg());
  msg.header.stamp = ros::Time::now();
  return msg;
}


agile_grasp2::GraspListMsg GraspDetectionNode::createGraspListMsg(const std::vector<GraspHypothesis>& hands)
{
  agile_grasp2::GraspListMsg msg;
  for (int i = 0; i < hands.size(); i++)
    msg.grasps.push_back(hands[i].convertToGraspMsg());
  msg.header.stamp = ros::Time::now();
  return msg;
}


bool GraspDetectionNode::isScoreGreater(const GraspHypothesis& hypothesis1, const GraspHypothesis& hypothesis2)
{
  return hypothesis1.getScore() > hypothesis2.getScore();
}

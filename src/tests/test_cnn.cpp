#include <iostream>
#include <fstream>
#include <string>

#include <ros/ros.h>

#include <agile_grasp2/caffe_classifier.h>
#include <agile_grasp2/cloud_camera.h>
#include <agile_grasp2/hand_search.h>
#include <agile_grasp2/handle_search.h>
#include <agile_grasp2/learning.h>
#include <agile_grasp2/plot.h>


int main(int argc, char** argv)
{
  // initialize ROS
  ros::init(argc, argv, "test_svm");
  ros::NodeHandle node("~");
  
  // get parameters from ROS launch file
  std::string cloud_file_name, svm_file_name, root_dir, model_file, trained_file, label_file;
  std::vector<double> workspace, camera_pose;
  double min_score_diff, min_handle_length;
  int num_samples, num_threads, min_inliers, plot, antipodal_mode;
  bool forces_PSD, uses_NARF, plot_normals, save_hypotheses, reuse_inliers;
  node.getParam("cloud_file_name", cloud_file_name);
  node.getParam("svm_file_name", svm_file_name);
  node.getParam("workspace", workspace);
  node.getParam("camera_pose", camera_pose);
  node.getParam("num_samples", num_samples);
  node.getParam("num_threads", num_threads);
  node.getParam("forces_PSD", forces_PSD);
  node.getParam("plotting", plot);
  node.getParam("uses_NARF", uses_NARF);
  node.param("antipodal_mode", antipodal_mode, 0);
  node.param("plot_normals", plot_normals, false);
  node.param("save_hypotheses", save_hypotheses, false);
  node.param("model_file", model_file, std::string("/home/baxter/data/grasp_images/"));
  node.param("trained_file", trained_file, std::string(""));
  node.param("label_file", label_file, std::string(""));
  node.param("images_directory", root_dir, std::string(""));
  node.param("min_score_diff", min_score_diff, 500.0);
  node.param("min_handle_length", min_handle_length, 0.005);
  node.param("reuse_inliers", reuse_inliers, true);
  node.getParam("min_inliers", min_inliers);
  
  std::string file_name_left;
  std::string file_name_right;
      
  if (cloud_file_name.find(".pcd") == std::string::npos)
  {
    file_name_left = cloud_file_name + "l_reg.pcd";
    file_name_right = cloud_file_name + "r_reg.pcd";
  }
  else
  {
    file_name_left = cloud_file_name;
    file_name_right = "";
  }
  
  double taubin_radius = 0.03;
  double hand_radius = 0.08;
  
  Eigen::VectorXd ws(6);
  ws << workspace[0], workspace[1], workspace[2], workspace[3], workspace[4], workspace[5];

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

  std::cout << model_file << " " << trained_file << " " << label_file << std::endl;

  // 1. Generate grasp hypotheses.
  CloudCamera cloud_cam(file_name_left, file_name_right);
  std::cout << "Loaded cloud with: " << cloud_cam.getCloudProcessed()->size() << " points.\n";
  cloud_cam.filterWorkspace(ws);
  std::cout << "After workspace filtering: " << cloud_cam.getCloudProcessed()->size() << " points left.\n";
  cloud_cam.voxelizeCloud(0.003);
  std::cout << "After voxelization: " << cloud_cam.getCloudProcessed()->size() << " points left.\n";

  Plot plotter;
//  plotter.drawCloud(cloud_cam.getCloudOriginal(), "Original Cloud");
//  plotter.drawCloud(cloud_cam.getCloudProcessed(), "Processed Cloud");

//  cloud_cam.subsampleUniformly(num_samples);

  std::vector<int> sample_indices;
  sample_indices.push_back(122);
  cloud_cam.setSampleIndices(sample_indices);

  std::cout << "Subsampled " << cloud_cam.getSampleIndices().size() << " points.\n";
  HandSearch::Parameters params;
  params.finger_width_ = 0.01;
  params.hand_outer_diameter_ = 0.09;
  params.hand_depth_ = 0.06;
  params.hand_height_ = 0.02;
  params.init_bite_ = 0.015;
  params.nn_radius_taubin_ = 0.01;
  params.nn_radius_hands_ = 0.1;
  params.num_threads_ = num_threads;
  params.antipodal_mode_ = antipodal_mode;
  params.num_orientations_ = 8;
  HandSearch hand_search(params);
  std::vector<GraspHypothesis> hands = hand_search.generateHypotheses(cloud_cam, 1, forces_PSD, plot_normals);
  ROS_INFO_STREAM("# grasp hypotheses: " << hands.size());
  for (int i = 0; i < hands.size(); i++)
   {
 //    ROS_INFO_STREAM(i << ": " << hands[i].getGraspBottom());
     hands[i].print();
     std::cout << "\n";
   }
  plotter.plotHands(hands, cloud_cam.getCloudOriginal(), "Grasp Hypotheses");

  cloud_cam.writeNormalsToFile("/home/andreas/data/grasp_images/normals.csv", hands[0].getNormalsForLearning());


  // 2. Predict antipodal grasps.
  std::cout << "Storing grasp images ...\n";
  Eigen::Matrix<double, 3, 2> cams_mat;
  cams_mat.col(0) = cam_tf_left.block<3, 1>(0, 3);
  cams_mat.col(1) = cam_tf_right.block<3, 1>(0, 3);
  Learning learn(60, 1);
  std::vector<cv::Mat> image_list = learn.storeGraspImages(hands, cams_mat, root_dir, true);
  std::vector<GraspHypothesis> antipodal_hands;
  Classifier classifier(model_file, trained_file, label_file);
  for (int i = 0; i < image_list.size(); i++)
  {
    std::vector<Prediction> predictions = classifier.Classify(image_list[i]);
    std::cout << i << " (memory): ";
    for (int j = 0; j < predictions.size(); j++)
      std::cout << std::fixed << std::setprecision(4) << predictions[j].second << " " << predictions[j].first << " ";
    std::cout << std::endl;

    if (predictions[1].second - predictions[0].second >= min_score_diff)
    {
      antipodal_hands.push_back(hands[i]);
      antipodal_hands[antipodal_hands.size()-1].setFullAntipodal(true);
    }
  }
  ROS_INFO_STREAM("# antipodal grasps: " << antipodal_hands.size());
  plotter.plotHands(antipodal_hands, cloud_cam.getCloudOriginal(), "Antipodal Grasps");
//  plotter.plotHands(hands, antipodal_hands, cloud_cam.getCloudOriginal(), "Antipodal Grasps");
//  plotter.plotFingers(antipodal_hands, cloud_cam.getCloudOriginal(), "Antipodal Grasps");


  // 3. Cluster the grasps.
  HandleSearch handle_search;
  handle_search.setMinInliers(min_inliers);
  handle_search.setMinLength(min_handle_length);
  handle_search.setReuseInliers(reuse_inliers);
  std::vector<Handle> handles;
  std::vector<Eigen::Vector3d> pos_deltas;
  handle_search.findClusters(antipodal_hands, handles, pos_deltas);
  ROS_INFO_STREAM("# handles: " << handles.size());
  plotter.plotHandles(handles, cloud_cam.getCloudOriginal(), "Clusters");

//    cv::Mat img = cv::imread(root_dir + "jpgs/img_" + boost::lexical_cast<std::string>(i) + ".jpeg", -1);
//    predictions = classifier.Classify(img);
//    std::cout << i << " (file): ";
//    for (int j = 0; j < predictions.size(); j++)
//      std::cout << std::fixed << std::setprecision(4) << predictions[j].second << " ";
//    std::cout << std::endl;
//
//    std::cout << "memory: " << image_list[i].type() << ", file: " << img.type() << std::endl;
//    std::cout << "memory: " << image_list[i].dims << ", file: " << img.dims << std::endl;
//    std::cout << "memory: " << image_list[i].rows << ", file: " << img.rows << std::endl;
//    std::cout << "memory: " << image_list[i].cols << ", file: " << img.cols << std::endl;

//    for (int r = 0; r < img.rows; r++)
//    {
//      for (int c = 0; c < img.cols; c++)
//      {
//        std::cout << "file: " << img.at<cv::Vec3i>(r,c) << "; memory: " << image_list[i].at<cv::Vec3i>(r,c) << std::endl;
//      }
//    }
//
//    cv::namedWindow("Grasp Image from Memory", cv::WINDOW_NORMAL);
//    cv::imshow("Grasp Image from Memory", image_list[i]);
//    cv::waitKey(0);
//  }

  return 0;
}

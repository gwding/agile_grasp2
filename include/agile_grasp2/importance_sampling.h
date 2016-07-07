#ifndef IMPORTANCE_SAMPLING_H
#define IMPORTANCE_SAMPLING_H


#include <ros/ros.h>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

#include <agile_grasp2/grasp_hypothesis.h>
#include <agile_grasp2/grasp_detector.h>
#include <agile_grasp2/handle_search.h>
#include <agile_grasp2/plot.h>


typedef pcl::PointCloud<pcl::PointXYZRGBA> PointCloudRGB;

typedef boost::variate_generator<boost::mt19937, boost::normal_distribution<> > Gaussian;


class ImportanceSampling : public GraspDetector
{
public:

  ImportanceSampling(ros::NodeHandle& node);

  std::vector<GraspHypothesis> detectGraspPoses(const CloudCamera& cloud_cam_in);


private:

  void drawSamplesFromSumOfGaussians(const std::vector<GraspHypothesis>& hands, Gaussian& generator,
    double sigma, int num_gauss_samples, Eigen::Matrix3Xd& samples_out);

  void drawSamplesFromMaxOfGaussians(const std::vector<GraspHypothesis>& hands, Gaussian& generator, double sigma,
    int num_gauss_samples, Eigen::Matrix3Xd& samples_out, double term);

  int num_iterations_;
  int num_samples_;
  int num_init_samples_;
  double prob_rand_samples_;
  double radius_;
  bool visualizes_;
  int sampling_method_;

  // standard parameters
  static const int NUM_ITERATIONS;
  static const int NUM_SAMPLES;
  static const int NUM_INIT_SAMPLES;
  static const double PROB_RAND_SAMPLES;
  static const double RADIUS;
  static const bool VISUALIZE_STEPS;
  static const int METHOD;
};

#endif

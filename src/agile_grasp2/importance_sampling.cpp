#include <agile_grasp2/importance_sampling.h>


// sampling methods
const int SUM = 1; // sum of Gaussians
const int MAX = 2; // max of Gaussians

// standard parameters
const int ImportanceSampling::NUM_ITERATIONS = 5;
const int ImportanceSampling::NUM_SAMPLES = 100;
const int ImportanceSampling::NUM_INIT_SAMPLES = 100;
const double ImportanceSampling::PROB_RAND_SAMPLES = 0.1;
const double ImportanceSampling::RADIUS = 0.02;
const bool ImportanceSampling::VISUALIZE_STEPS = false;
const int ImportanceSampling::METHOD = MAX;


ImportanceSampling::ImportanceSampling(ros::NodeHandle& node) : GraspDetector(node)
{
  num_init_samples_ = NUM_INIT_SAMPLES;
  num_iterations_ = NUM_ITERATIONS;
  num_samples_ = NUM_SAMPLES;
  prob_rand_samples_ = PROB_RAND_SAMPLES;
  radius_ = RADIUS;
  visualizes_ = VISUALIZE_STEPS;
  sampling_method_ = METHOD;
}


std::vector<GraspHypothesis> ImportanceSampling::detectGraspPoses(const CloudCamera& cloud_cam_in)
{
  double t0 = omp_get_wtime();
  CloudCamera cloud_cam = cloud_cam_in;

  // 1. Find initial grasp hypotheses.
  std::vector<GraspHypothesis> hands = GraspDetector::detectGraspPoses(cloud_cam, false);
  std::cout << "Initially detected " << hands.size() << " grasp hypotheses" << std::endl;
  if (hands.size() == 0)
  {
    return hands;
  }

  // 2. Create random generator for normal distribution.
  int num_rand_samples = prob_rand_samples_ * num_samples_;
  int num_gauss_samples = num_samples_ - num_rand_samples;
  double sigma = 2.0 * radius_;
  Eigen::Matrix3d diag_sigma = Eigen::Matrix3d::Zero();
  diag_sigma.diagonal() << sigma, sigma, sigma;
  Eigen::Matrix3d inv_sigma = diag_sigma.inverse();
  double term = 1.0 / sqrt(pow(2.0 * M_PI, 3.0) * pow(sigma, 3.0));
  boost::mt19937 *rng = new boost::mt19937();
  rng->seed(time(NULL));
  boost::normal_distribution<> distribution(0.0, 1.0);
  boost::variate_generator<boost::mt19937, boost::normal_distribution<> > generator(*rng, distribution);
  Eigen::Matrix3Xd samples(3, num_samples_);

  // 3. Find grasp hypotheses using importance sampling.
  for (int i = 0; i < num_iterations_; i++)
  {
    std::cout << i << " " << num_gauss_samples << std::endl;

    // draw samples close to affordances (importance sampling)
    if (this->sampling_method_ == SUM)
    {
      for (std::size_t j = 0; j < num_gauss_samples; j++)
      {
        int idx = rand() % hands.size();
        samples(0, j) = hands[idx].getGraspSurface()(0) + generator() * sigma;
        samples(1, j) = hands[idx].getGraspSurface()(1) + generator() * sigma;
        samples(2, j) = hands[idx].getGraspSurface()(2) + generator() * sigma;
      }
    }
    else // max of Gaussians
    {
      int j = 0;
      while (j < num_gauss_samples) // draw samples using rejection sampling
      {
        // draw from sum of Gaussians
        int idx = rand() % hands.size();
        Eigen::Vector3d x;
        x(0) = hands[idx].getGraspSurface()(0) + generator() * sigma;
        x(1) = hands[idx].getGraspSurface()(1) + generator() * sigma;
        x(2) = hands[idx].getGraspSurface()(2) + generator() * sigma;

        double maxp = 0;
        for (std::size_t k = 0; k < hands.size(); k++)
        {
          double p = (x - hands[k].getGraspSurface()).transpose() * (x - hands[k].getGraspSurface());
          p = term * exp((-1.0 / (2.0 * sigma)) * p);
          if (p > maxp)
            maxp = p;
        }

        double p = (x - hands[idx].getGraspSurface()).transpose() * (x - hands[idx].getGraspSurface());
        p = term * exp((-1.0 / (2.0 * sigma)) * p);
        if (p >= maxp)
        {
          samples.col(j) = x;
          j++;
        }
      }
    }

    // draw random samples
    const PointCloudRGB::Ptr& cloud = cloud_cam.getCloudProcessed();
    for (int j = num_samples_ - num_rand_samples; j < num_samples_; j++)
    {
      int r = std::rand() % cloud->points.size();
//      while (!pcl::isFinite((*cloud)[r])
//          || !this->affordances.isPointInWorkspace(cloud->points[r].x, cloud->points[r].y, cloud->points[r].z))
//        r = std::rand() % cloud->points.size();
      samples.col(j) = cloud->points[r].getVector3fMap().cast<double>();
    }

    // evaluate grasp hypotheses at <samples>
    cloud_cam.setSamples(samples);
    std::vector<GraspHypothesis> hands_new = GraspDetector::detectGraspPoses(cloud_cam, false);
    hands.insert(hands.end(), hands_new.begin(), hands_new.end());
    std::cout << "Added/total: " << hands_new.size() << "/" << hands.size() << " grasp hypotheses in round " << i
      << std::endl;
  }

  std::cout << "Found " << hands.size() << " grasp hypotheses in " << omp_get_wtime() - t0 << " sec.\n";

  Plot plotter;
  plotter.plotFingers(hands, cloud_cam.getCloudOriginal(), "All Grasps");

  if (getHandleSearch().getMinInliers() > 0)
  {
    hands = getHandleSearch().findClusters(hands);
    plotter.plotFingers(hands, cloud_cam.getCloudOriginal(), "Clusters");
  }

  return hands;
}

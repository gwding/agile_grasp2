#include <agile_grasp2/handle_search.h>


std::vector<GraspHypothesis> HandleSearch::findClusters(const std::vector<GraspHypothesis>& hand_list,
  bool remove_inliers)
{
  const double AXIS_ALIGN_ANGLE_THRESH = 15.0 * M_PI/180.0;
  const double AXIS_ALIGN_DIST_THRESH = 0.005;
  const double MAX_DIST_THRESH = 0.05;
//  const int max_inliers = 25;

  std::vector<GraspHypothesis> hands_out;
  std::vector<bool> has_used;
  if (remove_inliers)
  {
    has_used.resize(hand_list.size());
    for (int i = 0; i < hand_list.size(); i++)
    {
      has_used[i] = false;
    }
  }

  for (int i = 0; i < hand_list.size(); i++)
  {
    int num_inliers = 0;
    Eigen::Vector3d grasp_pos_delta = Eigen::Vector3d::Zero();
    Eigen::Matrix3d axis_outer_prod = hand_list[i].getAxis() * hand_list[i].getAxis().transpose();
    double avg_score = 0.0;

    for (int j = 0; j < hand_list.size(); j++)
    {
      if (i == j || (remove_inliers && has_used[j]))
        continue;

      // Which hands have an axis within <AXIS_ALIGN_ANGLE_THRESH> of this one?
      double axis_aligned = hand_list[i].getAxis().transpose() * hand_list[j].getAxis();
      bool axis_aligned_binary = fabs(axis_aligned) > cos(AXIS_ALIGN_ANGLE_THRESH);

      // Which hands are within <MAX_DIST_THRESH> of this one?
      Eigen::Vector3d delta_pos = hand_list[i].getGraspBottom() - hand_list[j].getGraspBottom();
      double delta_pos_mag = delta_pos.norm();
      bool delta_pos_mag_binary = delta_pos_mag <= MAX_DIST_THRESH;

      // Which hands are within <AXIS_ALIGN_DIST_THRESH> of this one when projected onto the plane orthognal to this
      // one's axis?
      Eigen::Matrix3d axis_orth_proj = Eigen::Matrix3d::Identity() - axis_outer_prod;
      Eigen::Vector3d delta_pos_proj = axis_orth_proj * delta_pos;
      double delta_pos_proj_mag = delta_pos_proj.norm();
      bool delta_pos_proj_mag_binary = delta_pos_proj_mag <= AXIS_ALIGN_DIST_THRESH;

      bool inlier_binary = axis_aligned_binary && delta_pos_mag_binary && delta_pos_proj_mag_binary;
      if (inlier_binary)
      {
        num_inliers++;
        grasp_pos_delta += hand_list[j].getGraspBottom();
        avg_score += hand_list[j].getScore();
        if (remove_inliers)
          has_used[j] = true;
//        if (num_inliers >= max_inliers)
//          break;
      }
    }

    if (num_inliers >= min_inliers_)
    {
      grasp_pos_delta = grasp_pos_delta / (double) num_inliers - hand_list[i].getGraspBottom();
      avg_score = avg_score / (double) num_inliers;
      std::cout << "grasp " << i << ", num_inliers: " << num_inliers << ", pos_delta: " << grasp_pos_delta.transpose()
        << ", score: " << avg_score << "\n";
      GraspHypothesis hand = hand_list[i];
      hand.setGraspSurface(hand.getGraspSurface() + grasp_pos_delta);
      hand.setGraspBottom(hand.getGraspBottom() + grasp_pos_delta);
      hand.setGraspTop(hand.getGraspTop() + grasp_pos_delta);
      hand.setScore(avg_score);
      hands_out.push_back(hand);
    }
  }

  return hands_out;
}


std::vector<Handle> HandleSearch::findHandles(const std::vector<GraspHypothesis>& hand_list)
{
  double t0 = omp_get_wtime();
  std::vector<Handle> handle_list;
  std::vector<GraspHypothesis> reduced_hand_list(hand_list);

	for (int i = 0; i < hand_list.size(); i++)
	{
    if (reduced_hand_list[i].getGraspWidth() == -1)
      continue;
    
    Eigen::Vector3d iaxis = reduced_hand_list[i].getAxis();
		Eigen::Vector3d ipt = reduced_hand_list[i].getGraspBottom();
		Eigen::Vector3d inormal = reduced_hand_list[i].getApproach();
    std::vector<Eigen::Vector2d> inliers;
    
    for (int j = 0; j < hand_list.size(); j++)
    {
      if (reduced_hand_list[j].getGraspWidth() == -1)
        continue;
      
      Eigen::Vector3d jaxis = reduced_hand_list[j].getAxis();
			Eigen::Vector3d jpt = reduced_hand_list[j].getGraspBottom();
			Eigen::Vector3d jnormal = reduced_hand_list[j].getApproach();
      
      // how good is this match?
			double dist_from_line = ((Eigen::Matrix3d::Identity(3, 3) - iaxis * iaxis.transpose()) * (jpt - ipt)).norm();
			double dist_along_line = iaxis.transpose() * (jpt - ipt);
			Eigen::Vector2d angle_axis;
			angle_axis << safeAcos(iaxis.transpose() * jaxis), M_PI - safeAcos(iaxis.transpose() * jaxis);
			double dist_angle_axis = angle_axis.minCoeff();
			double dist_from_normal = safeAcos(inormal.transpose() * jnormal);
			if (dist_from_line < 0.01 && dist_angle_axis < 0.34 && dist_from_normal < 0.34)
			{
        Eigen::Vector2d inlier;
				inlier << j, dist_along_line;
				inliers.push_back(inlier);
			}
    }
    
    if (inliers.size() < min_inliers_)
			continue;

		// shorten handle to a continuous piece
		double handle_gap_threshold = 0.02;
		std::vector<Eigen::Vector2d> inliers_list(inliers.begin(), inliers.end());
		while (!shortenHandle(inliers_list, handle_gap_threshold))
		{	};
    
    if (inliers_list.size() < min_inliers_)
      continue;
    
		// find grasps farthest away from i-th grasp 
    double min_dist = 10000000;
		double max_dist = -10000000;
    std::vector<int> in(inliers_list.size());
		for (int k = 0; k < inliers_list.size(); k++)
		{
      in[k] = inliers_list[k](0);
      
			if (inliers_list[k](1) < min_dist)
				min_dist = inliers_list[k](1);
			if (inliers_list[k](1) > max_dist)
				max_dist = inliers_list[k](1);
		}
    
    // handle found
    if (max_dist - min_dist > min_length_)
    {
      handle_list.push_back(Handle(hand_list, in));
      std::cout << "handle found with " << in.size() << " inliers\n";
      
      // eliminate hands in this handle from future search
      if (!reuse_inliers_)
      {
        for (int k = 0; k < in.size(); k++)
        {
          reduced_hand_list[in[k]].setGraspWidth(-1);
        }
      }
    }
  }
  
  std::cout << "Handle Search\n";
  std::cout << " runtime: " << omp_get_wtime() - t0 << " sec\n";
	std::cout << " " << handle_list.size() << " handles found\n"; 
  return handle_list; 
}


bool HandleSearch::shortenHandle(std::vector<Eigen::Vector2d> &inliers, double gap_threshold)
{
	std::sort(inliers.begin(), inliers.end(), HandleSearch::LastElementComparator());
	bool is_done = true;

	for (int i = 0; i < inliers.size() - 1; i++)
	{
		double diff = inliers[i + 1](1) - inliers[i](1);
		if (diff > gap_threshold)
		{
			std::vector<Eigen::Vector2d> out;
			if (inliers[i](2) < 0)
			{
				out.assign(inliers.begin() + i + 1, inliers.end());
				is_done = false;
			}
			else
			{
				out.assign(inliers.begin(), inliers.begin() + i);
			}
			inliers = out;
			break;
		}
	}

	return is_done;
}


double HandleSearch::safeAcos(double x)
{
	if (x < -1.0)
		x = -1.0;
	else if (x > 1.0)
		x = 1.0;
	return acos(x);
}

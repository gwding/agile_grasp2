#include <agile_grasp2/finger_hand.h>


FingerHand::FingerHand(double finger_width, double hand_outer_diameter,	double hand_depth)
  :	finger_width_(finger_width), hand_outer_diameter_(hand_outer_diameter), hand_depth_(hand_depth)
{
	int n = 10; // number of finger placements to consider over a single hand diameter

	Eigen::VectorXd fs_half;
	fs_half.setLinSpaced(n, 0.0, hand_outer_diameter - finger_width);
	finger_spacing_.resize(2 * fs_half.size());
	finger_spacing_	<< (fs_half.array() - hand_outer_diameter_ + finger_width_).matrix(), fs_half;
	fingers_ = Eigen::Array<bool, 1, Eigen::Dynamic>::Constant(1, 2 * n, false);
}


void FingerHand::evaluateFingers(const Eigen::Matrix3Xd& points, double bite, int idx)
{
  // calculate fingertip distance (top) and hand base distance (bottom)
  top_ = bite;
  bottom_ = bite - hand_depth_;

  fingers_.setConstant(false);

  // crop points at bite
  std::vector<int> cropped_indices;
  for (int i = 0; i < points.cols(); i++)
  {
    if (points(1, i) < bite)
    {
      cropped_indices.push_back(i);

      // Check that the hand would be able to extend by <bite> onto the object without causing the back of the hand to
      // collide with <points>.
      if (points(1, i) < bottom_)
        return;
    }
  }

  // check that there is at least one point in between the fingers
  if (cropped_indices.size() == 0)
    return;

  Eigen::MatrixXd cropped_points(points.rows(), cropped_indices.size());
  for (int i = 0; i < cropped_indices.size(); i++)
  {
    cropped_points.col(i) = points.col(cropped_indices[i]);
  }

  // identify free gaps
  int m = finger_spacing_.size();
  if (idx == -1)
  {
    for (int i = 0; i < m; i++)
    {
      // int num_in_gap = (cropped_points.array() > finger_spacing_(i)
      // && cropped_points.array() < finger_spacing_(i) + finger_width_).count();
      int num_in_gap = 0;
      for (int j = 0; j < cropped_indices.size(); j++)
      {
        if (cropped_points(0, j) > finger_spacing_(i) && cropped_points(0, j) < finger_spacing_(i) + finger_width_)
          num_in_gap++;
      }

      if (num_in_gap == 0)
      {
        int sum;

        if (i <= m / 2)
          sum = (cropped_points.row(0).array() > finger_spacing_(i) + finger_width_).count();
        else
          sum = (cropped_points.row(0).array() < finger_spacing_(i)).count();

        if (sum > 0)
          fingers_(i) = true;
      }
    }
  }
  else
  {
    for (int i = 0; i < m; i++)
    {
      if (i == idx || i == m/2 + idx)
      {
        // int num_in_gap = (cropped_points.array() > finger_spacing_(i)
        // && cropped_points.array() < finger_spacing_(i) + finger_width_).count();
        int num_in_gap = 0;
        for (int j = 0; j < cropped_indices.size(); j++)
        {
          if (cropped_points(0, j) > finger_spacing_(i) && cropped_points(0, j) < finger_spacing_(i) + finger_width_)
            num_in_gap++;
        }

        if (num_in_gap == 0)
        {
          int sum;

          if (i <= m / 2)
            sum = (cropped_points.row(0).array() > finger_spacing_(i) + finger_width_).count();
          else
            sum = (cropped_points.row(0).array() < finger_spacing_(i)).count();

          if (sum > 0)
            fingers_(i) = true;
        }
      }
    }
  }
}


void FingerHand::deepenHand(const Eigen::Matrix3Xd& points, double min_depth, double max_depth)
{
  points_ = points;

  std::vector<int> hand_idx;

  for (int i = 0; i < hand_.cols(); i++)
  {
    if (hand_(i) == true)
      hand_idx.push_back(i);
  }

  if (hand_idx.size() == 0)
    return;

  // choose middle hand
  int hand_eroded_idx = hand_idx[ceil(hand_idx.size() / 2.0) - 1]; // middle index
  Eigen::Array<bool, 1, Eigen::Dynamic> hand_eroded;
  hand_eroded = Eigen::Array<bool, 1,Eigen::Dynamic>::Constant(hand_.cols(), false);
  hand_eroded(hand_eroded_idx) = true;

  // attempt to deepen hand
  double deepen_step_size = 0.005;
  FingerHand new_hand = *this;
  FingerHand last_new_hand = new_hand;

  for (double depth = min_depth + deepen_step_size; depth <= max_depth; depth +=  deepen_step_size)
  {
    new_hand.evaluateFingers(points, depth, hand_eroded_idx);
    if (new_hand.getFingers().cast<int>().sum() < 2)
      break;

    new_hand.evaluateHand();
    last_new_hand = new_hand;
  }

  *this = last_new_hand; // recover the deepest hand
  hand_ = hand_eroded;
}


std::vector<int> FingerHand::computePointsInClosingRegion(const Eigen::Matrix3Xd& points)
{
  // calculate gripper bounding box
  int idx = -1;
  for (int i = 0; i < hand_.cols(); i++)
  {
    if (hand_(i) == true)
    {
      idx = i;
      break;
    }
  }
  if (idx == -1)
  {
    std::cout << "ERROR: Something went wrong!\n";
  }
//  std::cout << "  idx: " << idx << ", |hand|: " << hand_.cols() << "\n";
//  std::cout << "  finger_spacing: " << finger_spacing_.transpose() << "\n";
  left_ = finger_spacing_(idx) + finger_width_;
  right_ = finger_spacing_(hand_.cols() + idx);
  center_ = 0.5 * (left_ + right_);
  surface_ = points.row(1).minCoeff();
//  std::cout << "  left_: " << left_ << "\n";
//  std::cout << "  right_: " << right_ << "\n";
//  std::cout << "  center_: " << center_ << "\n";
//  std::cout << "  surface_: " << surface_ << "\n";

  // find points inside bounding box
  std::vector<int> indices;
  for (int i = 0; i < points.cols(); i++)
  {
//    std::cout << i << ": " << points.col(i).transpose() << "\n";

    if (points(1, i) < top_ && points(0, i) > left_ && points(0, i) < right_)
      indices.push_back(i);
  }
//  if (indices.size() == 0)
//  {
//    for (int i = 0; i < points.cols(); i++)
//      std::cout << i << ": " << points.col(i).transpose() << "\n";
//  }

  return indices;
}


Eigen::MatrixXd FingerHand::calculateGraspParameters(const Eigen::Matrix3d& frame, const Eigen::Vector3d& sample)
{
  Eigen::MatrixXd params(3,7);

  Eigen::Vector3d pos_surface, pos_bottom, pos_top, left_bottom, right_bottom, left_top, right_top;

  // calculate position of hand middle on object surface
  pos_surface << center_, surface_, 0.0;
  params.col(0) = frame * pos_surface + sample;

  // calculate position of hand middle closest to robot hand base
  pos_bottom << center_, bottom_, 0.0;
  params.col(1) = frame * pos_bottom + sample;

  // calculate position of hand middle between fingertips
  pos_top << center_, top_, 0.0;
  params.col(2) = frame * pos_top + sample;

  // calculate bottom and top positions for left and right finger
  left_bottom << left_, bottom_, 0.0;
  right_bottom << right_, bottom_, 0.0;
  left_top << left_, top_, 0.0;
  right_top << right_, top_, 0.0;
  params.col(3) = frame * left_bottom + sample;
  params.col(4) = frame * right_bottom + sample;
  params.col(5) = frame * left_top + sample;
  params.col(6) = frame * right_top + sample;

//  std::cout << "center_: " << center_ << ", surface_: " << surface_ << ", bottom_: " << bottom_ << ", top_: " << top_ << std::endl;

  return params;
}

double FingerHand::calculateGraspWidth(double bite)
{
  // 1. Find eroded hand.
  std::vector<int> hand_idx;
  for (int i = 0; i < hand_.cols(); i++)
  {
    if (hand_(i) == true)
      hand_idx.push_back(i);
  }

  //  2. Take points contained within two fingers of that hand.
  int hand_eroded_idx = hand_idx[hand_idx.size() / 2];
  double left_finger_pos = finger_spacing_(hand_eroded_idx);
  double right_finger_pos = finger_spacing_(hand_.cols() + hand_eroded_idx);
  double max = -100000.0;
  double min = 100000.0;

  for (int i = 0; i < points_.cols(); i++)
  {
    if (points_(1, i) < bite && points_(0, i) > left_finger_pos && points_(0, i) < right_finger_pos)
    {
      if (points_(0, i) < min)
        min = points_(0, i);
      if (points_(0, i) > max)
        max = points_(0, i);
    }
  }

  grasp_width_ = max - min;
  return grasp_width_;
}

void FingerHand::evaluateFingers(double bite)
{
	back_of_hand_ = -1.0 * (hand_depth_ - bite);

	fingers_.setConstant(false);

	// crop points at bite
	std::vector<int> cropped_indices;
	for (int i = 0; i < points_.cols(); i++)
	{
		if (points_(1, i) < bite)
		{
			cropped_indices.push_back(i);

			// Check that the hand would be able to extend by <bite> onto the object without causing back of hand
			// to collide with points.
			if (points_(1, i) < back_of_hand_)
			{
				return;
			}
		}
	}

	Eigen::MatrixXd cropped_points(points_.rows(), cropped_indices.size());
	for (int i = 0; i < cropped_indices.size(); i++)
	{
		cropped_points.col(i) = points_.col(cropped_indices[i]);
	}


	// identify free gaps
	int m = finger_spacing_.size();
	for (int i = 0; i < m; i++)
	{
		// int num_in_gap = (cropped_points.array() > finger_spacing_(i)
		// && cropped_points.array() < finger_spacing_(i) + finger_width_).count();
		int num_in_gap = 0;
		for (int j = 0; j < cropped_indices.size(); j++)
		{
			if (cropped_points(0, j) > finger_spacing_(i)
					&& cropped_points(0, j) < finger_spacing_(i) + finger_width_)
				num_in_gap++;
		}

		if (num_in_gap == 0)
		{
			int sum;

			if (i <= m / 2)
			{
				sum = (cropped_points.row(0).array() > finger_spacing_(i) + finger_width_).count();
			}
			else
			{
				sum = (cropped_points.row(0).array() < finger_spacing_(i)).count();
			}

			if (sum > 0)
			{
				fingers_(i) = true;
			}
		}
	}
}

void FingerHand::evaluateHand()
{
	int n = fingers_.size() / 2;
	hand_.resize(1, n);

	for (int i = 0; i < n; i++)
	{
		if (fingers_(i) == true && fingers_(n + i) == true)
			hand_(i) = 1;
		else
			hand_(i) = 0;
	}
}

void FingerHand::evaluateGraspParameters(double bite)
{
	double fs_sum = 0.0;
	for (int i = 0; i < hand_.size(); i++)
	{
		fs_sum += finger_spacing_(i) * hand_(i);
	}
	double hor_pos = (hand_outer_diameter_ / 2.0) + (fs_sum / hand_.sum());

	grasp_bottom << hor_pos, points_.row(1).maxCoeff();
	grasp_surface << hor_pos, points_.row(1).minCoeff();

	// calculate hand width. First find eroded hand. Then take points contained within two fingers of that hand.
	std::vector<int> hand_idx;
	for (int i = 0; i < hand_.cols(); i++)
	{
		if (hand_(i) == true)
			hand_idx.push_back(i);
	}

	int hand_eroded_idx = hand_idx[hand_idx.size() / 2];
	double left_finger_pos = finger_spacing_(hand_eroded_idx);
	double right_finger_pos = finger_spacing_(hand_.cols() + hand_eroded_idx);
	double max = -100000.0;
	double min = 100000.0;

	for (int i = 0; i < points_.cols(); i++)
	{
		if (points_(1, i) < bite && points_(0, i) > left_finger_pos	&& points_(0, i) < right_finger_pos)
		{
			if (points_(0, i) < min)
				min = points_(0, i);
			if (points_(0, i) > max)
				max = points_(0, i);
		}
	}

	grasp_width_ = max - min;
}


void FingerHand::deepenHand(double init_deepness, double max_deepness)
{
  std::vector<int> hand_idx;

  for (int i = 0; i < hand_.cols(); i++)
  {
    if (hand_(i) == true)
      hand_idx.push_back(i);
  }

  if (hand_idx.size() == 0)
    return;

  // choose middle hand
  int hand_eroded_idx = hand_idx[ceil(hand_idx.size() / 2.0) - 1]; // middle index
  Eigen::Array<bool, 1, Eigen::Dynamic> hand_eroded = Eigen::Array<bool, 1,
    Eigen::Dynamic>::Constant(hand_.cols(), false);
  hand_eroded(hand_eroded_idx) = true;

  // attempt to deepen hand
  double deepen_step_size = 0.005;
  FingerHand new_hand = *this;
  FingerHand last_new_hand = new_hand;

  for (double d = init_deepness + deepen_step_size; d <= max_deepness; d += deepen_step_size)
  {
    new_hand.evaluateFingers(d);

    if (new_hand.getFingers().cast<int>().sum() < 2)
      break;

    new_hand.evaluateHand();
    last_new_hand = new_hand;
  }

  *this = last_new_hand; // recover most deep hand
  hand_ = hand_eroded;
}

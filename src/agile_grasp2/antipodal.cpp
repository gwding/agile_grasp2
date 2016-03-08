#include <agile_grasp2/antipodal.h>

const int Antipodal::NO_GRASP = 0; // normals point not toward any finger
const int Antipodal::HALF_GRASP = 1; // normals point towards one finger
const int Antipodal::FULL_GRASP = 2; // normals point towards both fingers


int Antipodal::evaluateGrasp(const Eigen::Matrix3Xd& pts, const Eigen::Matrix3Xd& normals, double extremal_thresh)
{
  double friction_coeff = 20.0; // angle of friction cone in degrees
  
  // calculate extremal points
  Eigen::Array<bool,1,Eigen::Dynamic> left_extremal = pts.row(0).array() < (pts.row(0).minCoeff() + extremal_thresh);
  Eigen::Array<bool,1,Eigen::Dynamic> right_extremal = pts.row(0).array() > (pts.row(0).maxCoeff() - extremal_thresh);
  
  // calculate points with surface normal within <friction_coeff> of closing direction
  Eigen::Vector3d l, r;
  l << -1.0, 0.0, 0.0;
  r << 1.0, 0.0, 0.0;
  double cos_friction_coeff = cos(friction_coeff*M_PI/180.0);
  Eigen::Array<bool,1,Eigen::Dynamic> left_close_direction = (l.transpose() * normals).array() > cos_friction_coeff;
  Eigen::Array<bool,1,Eigen::Dynamic> right_close_direction = (r.transpose() * normals).array() > cos_friction_coeff;
  
  // std::cout << left_extremal.count() << " " << right_extremal.count() << "\n";
  // std::cout << left_close_direction.count() << " " << right_close_direction.count() << "\n";
  
  // select points that are extremal and have their surface normal within the friction cone of the closing direction  
  std::vector<int> left_idx_viable, right_idx_viable;
  for (int i=0; i < pts.cols(); i++)    
  {
    // if (left_extremal(i))
      // left_idx_viable.push_back(i);    
    // if (right_extremal(i))
      // right_idx_viable.push_back(i);
    
    if (left_close_direction(i) && left_extremal(i))
      left_idx_viable.push_back(i);    
    if (right_close_direction(i) && right_extremal(i))
      right_idx_viable.push_back(i);
  }
  // std::cout << "left_idx_viable.size: " << left_idx_viable.size() << ", right_idx_viable.size(): " << right_idx_viable.size() << "\n";
  
  // if there are viable points on both sides
  if (left_idx_viable.size() > 0 && right_idx_viable.size() > 0)
  {
    Eigen::Matrix3Xd left_pts_viable(3, left_idx_viable.size()), right_pts_viable(3, right_idx_viable.size());
    for (int i=0; i < left_idx_viable.size(); i++)
    {
      left_pts_viable.col(i) = pts.col(left_idx_viable[i]);
    }
    for (int i=0; i < right_idx_viable.size(); i++)
    {
      right_pts_viable.col(i) = pts.col(right_idx_viable[i]);
    }
    
    double top_viable_y = std::min(left_pts_viable.row(1).maxCoeff(), right_pts_viable.row(1).maxCoeff());
    double bottom_viable_y = std::max(left_pts_viable.row(1).minCoeff(), right_pts_viable.row(1).minCoeff());
    
    double top_viable_z = std::min(left_pts_viable.row(2).maxCoeff(), right_pts_viable.row(2).maxCoeff());
    double bottom_viable_z = std::max(left_pts_viable.row(2).minCoeff(), right_pts_viable.row(2).minCoeff());
    
    // std::cout << "top_viable_y: " << top_viable_y << "\n";
    // std::cout << "bottom_viable_y: " << bottom_viable_y << "\n";
    // std::cout << "top_viable_z: " << top_viable_z << "\n";
    // std::cout << "bottom_viable_z: " << bottom_viable_z << "\n";
    
    if (top_viable_y > bottom_viable_y && top_viable_z > bottom_viable_z)
      return FULL_GRASP;    
  }
    
  return NO_GRASP;
}


int Antipodal::evaluateGrasp(const Eigen::Matrix3Xd& normals, double thresh_half, double thresh_full)
{
  int num_thresh = 6;
  int grasp = 0;
  double cos_thresh = cos(thresh_half * M_PI / 180.0);
  int numl = 0;
  int numr = 0;
  Eigen::Vector3d l, r;
  l << -1, 0, 0;
  r << 1, 0, 0;
  bool is_half_grasp = false;
  bool is_full_grasp = false;

  // check whether this is a half grasp
  for (int i = 0; i < normals.cols(); i++)
  {
    if (l.dot(normals.col(i)) > cos_thresh)
    {
      numl++;
      if (numl > num_thresh)
      {
        is_half_grasp = true;
        break;
      }
    }

    if (r.dot(normals.col(i)) > cos_thresh)
    {
      numr++;
      if (numr > num_thresh)
      {
        is_half_grasp = true;
        break;
      }
    }
  }

  // check whether this is a full grasp
  cos_thresh = cos(thresh_full * M_PI / 180.0);
  numl = 0;
  numr = 0;

  for (int i = 0; i < normals.cols(); i++)
  {
    if (l.dot(normals.col(i)) > cos_thresh)
    {
      numl++;
      if (numl > num_thresh && numr > num_thresh)
      {
        is_full_grasp = true;
        break;
      }
    }

    if (r.dot(normals.col(i)) > cos_thresh)
    {
      numr++;
      if (numl > num_thresh && numr > num_thresh)
      {
        is_full_grasp = true;
        break;
      }
    }
  }

  if (is_full_grasp)
    return FULL_GRASP;
  else if (is_half_grasp)
    return HALF_GRASP;

  return NO_GRASP;
}

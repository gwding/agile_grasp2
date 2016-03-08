#include <agile_grasp2/grasp_hypothesis.h>


void GraspHypothesis::writeHandsToFile(const std::string& filename, const std::vector<GraspHypothesis>& hands)
{
  std::ofstream myfile;
  myfile.open (filename.c_str());

  for (int i = 0; i < hands.size(); i++)
  {
    std::cout << "Hand " << i << std::endl;
    std::cout << " bottom: " << hands[i].getGraspBottom().transpose() << std::endl;
    std::cout << " surface: " << hands[i].getGraspSurface().transpose() << std::endl;
    std::cout << " axis: " << hands[i].getAxis().transpose() << std::endl;
    std::cout << " approach: " << hands[i].getApproach().transpose() <<std::endl;
    std::cout << " binormal: " << hands[i].getBinormal().transpose() << std::endl;
    std::cout << " width: " << hands[i].getGraspWidth() << std::endl;

    myfile << vectorToString(hands[i].getGraspBottom()) << vectorToString(hands[i].getGraspSurface())
        << vectorToString(hands[i].getAxis()) << vectorToString(hands[i].getApproach())
        << vectorToString(hands[i].getBinormal()) << boost::lexical_cast<double>(hands[i].getGraspWidth()) << "\n";
  }

  myfile.close();
}


void GraspHypothesis::print()
{
	std::cout << "axis: " << axis_.transpose() << std::endl;
	std::cout << "approach: " << approach_.transpose() << std::endl;
	std::cout << "binormal: " << binormal_.transpose() << std::endl;
	std::cout << "grasp width: " << grasp_width_ << std::endl;
	std::cout << "grasp surface: " << grasp_surface_.transpose() << std::endl;
	std::cout << "grasp bottom: " << grasp_bottom_.transpose() << std::endl;
	std::cout << "grasp top: " << grasp_top_.transpose() << std::endl;
}


agile_grasp2::GraspMsg GraspHypothesis::convertToGraspMsg() const
{
  agile_grasp2::GraspMsg msg;
  tf::pointEigenToMsg(grasp_surface_, msg.surface);
  tf::pointEigenToMsg(grasp_bottom_, msg.bottom);
  tf::pointEigenToMsg(grasp_top_, msg.top);
  tf::vectorEigenToMsg(axis_, msg.axis);
  tf::vectorEigenToMsg(approach_, msg.approach);
  tf::vectorEigenToMsg(binormal_, msg.binormal);
  msg.width.data = grasp_width_;
  msg.score.data = score_;
  return msg;
}


std::string GraspHypothesis::vectorToString(const Eigen::VectorXd& v)
{
  std::string s = "";
  for (int i = 0; i < v.rows(); i++)
  {
    s += boost::lexical_cast<std::string>(v(i)) + ",";
  }
  return s;
}

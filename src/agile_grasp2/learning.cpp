#include <agile_grasp2/learning.h>


std::vector<cv::Mat> Learning::createGraspImages(const std::vector<GraspHypothesis>& hands_list,
  const Eigen::Matrix3Xd& cam_pos, bool is_plotting, bool is_storing)
{
  std::vector<cv::Mat> image_list(hands_list.size());

  #ifdef _OPENMP // parallelization using OpenMP
  #pragma omp parallel for num_threads(num_threads_)
  #endif
  for (int i = 0; i < hands_list.size(); i++)
  {
    Instance ins = createInstance(hands_list[i], cam_pos);
    cv::Mat image = convertToImageRGB(ins);
    image.convertTo(image_list[i], CV_8UC3, 255.0);

    if (is_plotting)
    {
      cv::namedWindow("Grasp Image " + boost::lexical_cast<std::string>(i), cv::WINDOW_NORMAL);
      cv::imshow("Grasp Image " + boost::lexical_cast<std::string>(i), image_list[i]);
      cv::waitKey(0);
    }

    if (is_storing)
    {
      cv::imwrite("/home/baxter/data/cpp_grasp_images/img_" + boost::lexical_cast<std::string>(i) + ".jpeg", image_list[i]);
      std::cout << "Wrote: /home/baxter/data/cpp_grasp_images/img_" + boost::lexical_cast<std::string>(i) + ".jpeg\n";
    }
  }

  return image_list;
}


std::vector<cv::Mat> Learning::storeGraspImages(const std::vector<GraspHypothesis>& hands_list,
  const Eigen::Matrix3Xd& cam_pos, const std::string& root_dir, bool is_plotting)
{
  std::vector<cv::Mat> image_list(hands_list.size());
  std::string txt_filename = root_dir + "test.txt";
  std::ofstream txt_file;
  txt_file.open(txt_filename.c_str());

  for (int i = 0; i < hands_list.size(); i++)
  {
    Instance ins = createInstance(hands_list[i], cam_pos);
    cv::Mat image = convertToImageRGB(ins);
    std::string filename = root_dir + "jpgs/img_" + boost::lexical_cast<std::string>(i) + ".jpeg";
    image.convertTo(image, CV_8UC3, 255.0);
    cv::imwrite(filename, image);
    txt_file << "img_" + boost::lexical_cast<std::string>(i) + ".jpeg" + "\n";
//    std::cout << "Wrote image: " << filename << std::endl;
    image_list[i] = image;

    if (is_plotting)
    {
      cv::namedWindow("Grasp Image", cv::WINDOW_NORMAL);
      cv::imshow("Grasp Image", image);
      cv::waitKey(0);
    }
  }

//  std::cout << "Wrote text file: " << txt_filename << std::endl;
  txt_file.close();
  return image_list;
}


cv::Mat Learning::createGraspImage(const Eigen::Matrix3Xd& points, const Eigen::Matrix3Xd& normals,
  const agile_grasp2::GraspMsg& grasp, int idx, bool is_plotting)
{
  Instance ins = createInstanceFromPointsNormals(points, normals, grasp);
  cv::Mat image = convertToImageRGB(ins, false, is_plotting);
  image.convertTo(image, CV_8UC3, 255.0);
  
  if (idx > -1)
  {
    std::string filename = "/home/andreas/data/grasp_images/jpgs/img_" + boost::lexical_cast<std::string>(idx) + ".jpeg";
    cv::imwrite(filename, image);
  }
  
  if (is_plotting)
  {
    cv::namedWindow("Grasp Image", cv::WINDOW_NORMAL);
    cv::imshow("Grasp Image", image);
    cv::waitKey(0);
  }

  return image;
}


cv::Mat Learning::convertToImage(const Instance& ins)
{
	const double HORIZONTAL_LIMITS[2] = { -0.05, 0.05 };
	const double VERTICAL_LIMITS[2] = { 0.0, 0.08 };
	double cell_size = (HORIZONTAL_LIMITS[1] - HORIZONTAL_LIMITS[0]) / (double) num_horizontal_cells_;

	Eigen::VectorXi horizontal_cells(ins.pts.cols());
	Eigen::VectorXi vertical_cells(ins.pts.cols());
  
  // reverse x-direction to keep orientation consistent
	if (ins.binormal.dot(ins.source_to_center) > 0)
		horizontal_cells = floorVector((ins.pts.row(0).array() - HORIZONTAL_LIMITS[0]) / cell_size);
	else
  {
    Eigen::MatrixXd vec = (-ins.pts.row(0).array() - HORIZONTAL_LIMITS[0]) / cell_size;
    horizontal_cells = floorVector(vec);
  }
  
	vertical_cells = floorVector((ins.pts.row(1).array() - VERTICAL_LIMITS[0]) / cell_size);
  
	std::set<Eigen::Vector2i, UniqueVectorComparator> cells;
	for (int i = 0; i < ins.pts.cols(); i++)
	{
		Eigen::Vector2i c;
		c << horizontal_cells(i), vertical_cells(i);
		cells.insert(c);
	}
  
	Eigen::Matrix2Xi cells_mat(2, cells.size());
	int i = 0;
	cv::Mat image(num_vertical_cells_, num_horizontal_cells_, CV_8UC1);
	image.setTo(0);
  
	for (std::set<Eigen::Vector2i, UniqueVectorComparator>::iterator it = cells.begin(); it != cells.end(); it++)
	{
		Eigen::Vector2i c = *it;
		cells_mat(0, i) = std::max(0, c(0));
		cells_mat(1, i) = std::max(0, c(1));

		c = cells_mat.col(i);
		cells_mat(0, i) = std::min(num_horizontal_cells_ - 1, c(0));
		cells_mat(1, i) = std::min(num_vertical_cells_ - 1, c(1));
		image.at<uchar>(image.rows - 1 - cells_mat(1, i), cells_mat(0, i)) = 255;
		i++;
	}

	return image;
}


cv::Mat Learning::convertToImageRGB(const Instance& ins, bool aligns, bool plots)
{
  Eigen::VectorXd y = ins.pts.row(1);

  // align top of object to bottom of grasp window
  if (aligns)
    y = y.array() - ins.pts.row(1).minCoeff();

  // for each point, find the cell it falls into
  double cellsize = 1.0 / (double) num_horizontal_cells_;
  Eigen::VectorXi horizontal_cells = floorVector(ins.pts.row(0) / cellsize);
  Eigen::VectorXi vertical_cells = floorVector(y / cellsize);
  Eigen::VectorXi cell_indices;
  cell_indices = horizontal_cells + vertical_cells * num_horizontal_cells_;

  // for each cell, find the points that fall into it, and calculate their average surface normal
  Eigen::Matrix3Xd avg_normals(3, num_horizontal_cells_*num_horizontal_cells_);
  cv::Mat image(num_vertical_cells_, num_horizontal_cells_, CV_32FC3);
  image.setTo(0);
  cv::Size size = image.size();
  int rows = size.height;
  int cols = size.width;

  for (int i = 0; i < avg_normals.cols(); i++)
  {
    Eigen::Vector3d avg_normal;
    avg_normal << 0.0, 0.0, 0.0;
    int num_in_cell = 0;

    for (int j = 0; j < cell_indices.rows(); j++)
    {
      if (cell_indices(j) == i)
      {
        avg_normal += ins.normals.col(j);
        num_in_cell++;
      }
    }

    if (num_in_cell > 0)
    {
      double s = 1.0 / avg_normal.norm();
      Eigen::Vector3d scale;
      scale << s, s, s;
      avg_normal = (scale.array() * avg_normal.array()).abs();
      avg_normals.col(i) = avg_normal;
      int row = i / cols;
      int col = i % cols;
      image.at<cv::Vec3f>(image.rows - 1 - row, col) = cv::Vec3f(avg_normal(0), avg_normal(1), avg_normal(2));
    }
  }

  if (plots)
  {
    cv::namedWindow("Grasp Image Before Dilation", cv::WINDOW_NORMAL);
    cv::imshow("Grasp Image Before Dilation", image);
    cv::waitKey(0);
  }

  // dilate the image to fill in holes
  cv::Mat dilation_element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3,3));
  dilate(image, image, dilation_element);

  // convert image from BGR to RGB
  cv::cvtColor(image, image, CV_BGR2RGB);

  return image;
}


Eigen::VectorXi Learning::floorVector(const Eigen::VectorXd& a)
{
  Eigen::VectorXi b(a.size());
	for (int i = 0; i < b.size(); i++)
		b(i) = floor(a(i));
	return b;
}


Learning::Instance Learning::createInstance(const GraspHypothesis& h, const Eigen::Matrix3Xd& cam_pos, int cam)
{
  Instance ins;
  ins.binormal = h.getBinormal();
  ins.label = h.isFullAntipodal();
  
  // calculate camera position to center vector
//  const Eigen::Vector3d& source = cam_pos.col(h.getCamSource());
//  ins.source_to_center = h.getGraspSurface() - source;
  
  if (cam == -1)
  {
    ins.pts = h.getPointsForLearning();
    ins.normals = h.getNormalsForLearning();
  }
  else
  {
    const std::vector<int>& indices_cam = (cam == 0) ? h.getIndicesPointsForLearningCam1() : h.getIndicesPointsForLearningCam2();
    ins.pts.resize(3, indices_cam.size());
    ins.normals.resize(3, indices_cam.size());
    for (int i = 0; i < indices_cam.size(); i++)
    {
      ins.pts.col(i) = h.getPointsForLearning().col(indices_cam[i]);
      ins.normals.col(i) = h.getNormalsForLearning().col(indices_cam[i]);
    } 
  }

  return ins;
}


Learning::Instance Learning::createInstanceFromPointsNormals(const Eigen::Matrix3Xd& points,
  const Eigen::Matrix3Xd& normals, const agile_grasp2::GraspMsg& grasp)
{
  Instance ins;
  tf::vectorMsgToEigen(grasp.binormal, ins.binormal);
  ins.label = 0;
  ins.pts = points;
  ins.normals = normals;
  return ins;
}

//  Eigen::Matrix3Xd points_instance(3, points.size());
//  Eigen::Matrix3Xd normals_instance(3, points.size());
//  Eigen::Vector3d centroid;
//  centroid << 0.0, 0.0, 0.0;
//  for (int i = 0; i < points.size(); i++)
//  {
//    points_instance.col(i) = points[i];
//    normals_instance.col(i) = normals[i];
//    centroid += points[i];
//  }
//  centroid = centroid / (double) points.size();
//
//  // center the points
//  for (int i = 0; i < points.size(); i++)
//  {
//    points_instance.col(i) -= centroid;
//  }
//
//  Eigen::Vector3d binormal, approach, axis;
//  tf::vectorMsgToEigen(grasp.binormal, binormal);
//  tf::vectorMsgToEigen(grasp.approach, approach);
//  tf::vectorMsgToEigen(grasp.axis, axis);
//  Eigen::Matrix3d frame;
//  frame << binormal, approach, axis;
////  frame.col(0) = binormal;
////  frame.col(1) = approach;
////  frame.col(2) = axis;
//  Eigen::Matrix3Xd points_rot = frame.transpose() * points_instance;
//  Eigen::Matrix3Xd normals_rot = frame.transpose() * normals_instance;
//
//  // crop points on hand height
//  std::vector<int> indices(points_rot.cols());
//  int num_valid = 0;
//  for (int i = 0; i < points_rot.cols(); i++)
//  {
//    std::cout << points_rot(2, i) << "\n";
//    if (points_rot(2, i) > -1.0 * hand_height && points_rot(2, i) < hand_height)
//    {
//      indices[num_valid] = i;
//      num_valid++;
//    }
//  }
//  std::cout << "#points_rot: " << points_rot.cols() << "\n";
//
//  Eigen::Matrix3Xd points_cropped(3, num_valid);
//  Eigen::Matrix3Xd normals_cropped(3, num_valid);
//  for (int i = 0; i < num_valid; i++)
//  {
//    points_cropped.col(i) = points_rot.col(indices[i]);
//    normals_cropped.col(i) = normals_rot.col(indices[i]);
//  }
//
//  std::cout << "#points_cropped: " << points_cropped.cols() << "\n";

  // scale <pts_rot> to fit into unit square
//  double left = -0.5*grasp.width.data;
//  double right = 0.5*grasp.width.data;
//  double top = bite;
//  double bottom = top - hand_depth;
//  double baseline_const = 0.1;
//  double left_const = left - 0.5 * (baseline_const - (right - left));
//  Eigen::Vector3d lower, scales;
//  lower << left_const, bottom, -1.0 * hand_height;
//  scales << 1.0 / baseline_const, 1.0 / ((double) top - bottom), 1.0 / (2.0 * hand_height);
//  Eigen::Matrix3Xd points_in_box(3, points_cropped.cols());
//  points_in_box = scales.replicate(1, points_in_box.cols()).array() * (points_cropped - lower.replicate(1, points_in_box.cols())).array();

//  ins.binormal = h.getBinormal();
//  ins.label = h.isFullAntipodal();
//
//  // calculate camera position to center vector
////  const Eigen::Vector3d& source = cam_pos.col(h.getCamSource());
////  ins.source_to_center = h.getGraspSurface() - source;
//
//  if (cam == -1)
//  {
//    ins.pts = h.getPointsForLearning();
//    ins.normals = h.getNormalsForLearning();
//  }
//  else
//  {
//    const std::vector<int>& indices_cam = (cam == 0) ? h.getIndicesPointsForLearningCam1() : h.getIndicesPointsForLearningCam2();
//    ins.pts.resize(3, indices_cam.size());
//    ins.normals.resize(3, indices_cam.size());
//    for (int i = 0; i < indices_cam.size(); i++)
//    {
//      ins.pts.col(i) = h.getPointsForLearning().col(indices_cam[i]);
//      ins.normals.col(i) = h.getNormalsForLearning().col(indices_cam[i]);
//    }
//  }
//
//  return ins;
//}


Learning::Instance Learning::createInstanceFromHandle(const Handle& h, const Eigen::Matrix3Xd& cam_pos, int cam)
{
  Instance ins;
  ins.binormal = h.getBinormal();
  ins.label = h.isFullAntipodal();

  // calculate camera position to center vector
  const Eigen::Vector3d& source = cam_pos.col(h.getCamSource());
  ins.source_to_center = h.getGraspSurface() - source;

  if (cam == -1)
  {
    ins.pts = h.getPointsForLearning();
    ins.normals = h.getNormalsForLearning();
  }
  else
  {
    const std::vector<int>& indices_cam = (cam == 0) ? h.getIndicesPointsForLearningCam1() : h.getIndicesPointsForLearningCam2();
    ins.pts.resize(3, indices_cam.size());
    ins.normals.resize(3, indices_cam.size());
    for (int i = 0; i < indices_cam.size(); i++)
    {
      ins.pts.col(i) = h.getPointsForLearning().col(indices_cam[i]);
      ins.normals.col(i) = h.getNormalsForLearning().col(indices_cam[i]);
    }
  }

  return ins;
}

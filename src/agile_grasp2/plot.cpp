#include <agile_grasp2/plot.h>


void Plot::plotFingers(const std::vector<GraspHypothesis>& hand_list, const PointCloudRGBA::Ptr& cloud, std::string str)
{
  const int WIDTH = pcl::visualization::PCL_VISUALIZER_LINE_WIDTH;

  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer = createViewer(str);

  pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBA> rgb(cloud);
  viewer->addPointCloud<pcl::PointXYZRGBA>(cloud, rgb, "cloud");
  viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "cloud");
  
  //~ PointCloudRGBA::Ptr cloud_fingers(new PointCloudRGBA);
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_fingers(new pcl::PointCloud<pcl::PointXYZ>);

  for (int i = 0; i < hand_list.size(); i++)
  {
    //~ pcl::PointXYZRGBA pc;
    pcl::PointXYZ pc;
    pc.getVector3fMap() = hand_list[i].getGraspBottom().cast<float>();
    double width = hand_list[i].getGraspWidth();    
    double hw = 0.5 * width;
    double step = hw / 30.0;
    Eigen::Vector3d left_bottom = hand_list[i].getGraspBottom() + hw * hand_list[i].getBinormal();
    Eigen::Vector3d right_bottom = hand_list[i].getGraspBottom() - hw * hand_list[i].getBinormal();
    //~ pcl::PointXYZRGBA p1, p2;
    pcl::PointXYZ p1, p2;
    p1.getVector3fMap() = left_bottom.cast<float>();
    p2.getVector3fMap() = right_bottom.cast<float>();
    cloud_fingers->points.push_back(pc);
    cloud_fingers->points.push_back(p1);
    cloud_fingers->points.push_back(p2);
        
    for (double j=step; j < hw; j+=step)
    {
      Eigen::Vector3d lb, rb, a;
      lb = hand_list[i].getGraspBottom() + j * hand_list[i].getBinormal();
      rb = hand_list[i].getGraspBottom() - j * hand_list[i].getBinormal();
      a = hand_list[i].getGraspBottom() - j * hand_list[i].getApproach();
      //~ pcl::PointXYZRGBA plb, prb, pa;
      pcl::PointXYZ plb, prb, pa;
      plb.getVector3fMap() = lb.cast<float>();
      prb.getVector3fMap() = rb.cast<float>();
      pa.getVector3fMap() = a.cast<float>();
      cloud_fingers->points.push_back(plb);
      cloud_fingers->points.push_back(prb);      
      cloud_fingers->points.push_back(pa);
	  }
    
    double dist = (hand_list[i].getGraspTop() - hand_list[i].getGraspBottom()).norm();
    step = dist / 40.0;
    for (double j=step; j < dist; j+=step)
    {
      Eigen::Vector3d lt, rt;
      lt = left_bottom + j * hand_list[i].getApproach();
      rt = right_bottom + j * hand_list[i].getApproach();
      //~ pcl::PointXYZRGBA plt, prt;
      pcl::PointXYZ plt, prt;
      plt.getVector3fMap() = lt.cast<float>();
      prt.getVector3fMap() = rt.cast<float>();
      cloud_fingers->points.push_back(plt);
      cloud_fingers->points.push_back(prt);
    }

    //~ std::string istr = boost::lexical_cast<std::string>(i);
    //~ viewer->addLine<pcl::PointXYZ>(center, left_bottom, 0.0, 0.0, 1.0, "left_bottom_" + istr);
    //~ viewer->addLine<pcl::PointXYZ>(center, right_bottom, 0.0, 0.0, 1.0, "right_bottom_" + istr);
    //~ viewer->addLine<pcl::PointXYZ>(left_bottom, left_top, 0.0, 0.0, 1.0, "left_top_" + istr);
    //~ viewer->addLine<pcl::PointXYZ>(right_bottom, right_top, 0.0, 0.0, 1.0, "right_top_" + istr);
    //~ viewer->addLine<pcl::PointXYZ>(center, approach_center, 0.0, 0.0, 1.0, "approach_" + istr);
//~ 
    //~ viewer->setShapeRenderingProperties(WIDTH, 5, "left_bottom_" + boost::lexical_cast<std::string>(i));
    //~ viewer->setShapeRenderingProperties(WIDTH, 5, "right_bottom_" + boost::lexical_cast<std::string>(i));
    //~ viewer->setShapeRenderingProperties(WIDTH, 5, "left_top_" + boost::lexical_cast<std::string>(i));
    //~ viewer->setShapeRenderingProperties(WIDTH, 5, "right_top_" + boost::lexical_cast<std::string>(i));
    //~ viewer->setShapeRenderingProperties(WIDTH, 5, "approach_" + boost::lexical_cast<std::string>(i));
    //~ viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_REPRESENTATION,
      //~ pcl::visualization::PCL_VISUALIZER_REPRESENTATION_WIREFRAME, "approach_" + boost::lexical_cast<std::string>(i));
  }
  
  viewer->addPointCloud<pcl::PointXYZ>(cloud_fingers, "fingers");
  viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0.0, 0.0, 0.8, "fingers");
  viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "fingers");
  viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY, 0.4, "fingers");

  runViewer(viewer);
}


void Plot::plotHands(const std::vector<GraspHypothesis>& hand_list,
	const std::vector<GraspHypothesis>& antipodal_hand_list, const PointCloudRGBA::Ptr& cloud, std::string str,
	bool use_grasp_bottom)
{
	PointCloudNormal::Ptr hands_cloud = createNormalsCloud(hand_list, false, false);
	PointCloudNormal::Ptr antipodal_hands_cloud = createNormalsCloud(antipodal_hand_list, true,
		false);
	plotHandsHelper(hands_cloud, antipodal_hands_cloud, cloud, str, use_grasp_bottom);
}


void Plot::plotHands(const std::vector<GraspHypothesis>& hand_list, const PointCloudRGBA::Ptr& cloud,
	std::string str, bool use_grasp_bottom)
{
	PointCloudNormal::Ptr hands_cloud = createNormalsCloud(hand_list, false, false);
	PointCloudNormal::Ptr antipodal_hands_cloud = createNormalsCloud(hand_list, true, false);
	plotHandsHelper(hands_cloud, antipodal_hands_cloud, cloud, str, use_grasp_bottom);
}


void Plot::plotSamples(const std::vector<int>& index_list, const PointCloudRGBA::Ptr& cloud)
{
	PointCloudRGBA::Ptr samples_cloud(new PointCloudRGBA);
	for (int i = 0; i < index_list.size(); i++)
		samples_cloud->points.push_back(cloud->points[index_list[i]]);

	plotSamples(samples_cloud, cloud);
}


void Plot::plotSamples(const PointCloudRGBA::Ptr& samples_cloud, const PointCloudRGBA::Ptr& cloud)
{
  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer = createViewer("Samples");

  // draw the point cloud
  pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBA> rgb(cloud);
	viewer->addPointCloud<pcl::PointXYZRGBA>(cloud, rgb, "registered point cloud");
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "registered point cloud");

	// draw the samples
	viewer->addPointCloud<pcl::PointXYZRGBA>(samples_cloud, "samples cloud");
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "samples cloud");
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 1.0, 0.0, 0.0, "samples cloud");

	runViewer(viewer);
}


void Plot::plotNormals(const PointCloudRGBA::Ptr& cloud, const Eigen::Matrix3Xd& normals)
{		
  PointCloudNormal::Ptr normals_cloud(new PointCloudNormal);
  for (int i=0; i < normals.cols(); i++)
  {
    pcl::PointNormal p;
    p.x = cloud->points[i].x;
    p.y = cloud->points[i].y;
    p.z = cloud->points[i].z;
    p.normal_x = normals(0,i);
    p.normal_y = normals(1,i);
    p.normal_z = normals(2,i);
    normals_cloud->points.push_back(p);
  }
  std::cout << "Drawing " << normals_cloud->size() << " normals\n";
    
  double red[3] = {1.0, 0.0, 0.0};
  double blue[3] = {0.0, 0.0, 1.0};
  
  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer = createViewer("Normals");  
	addCloudNormalsToViewer(viewer, normals_cloud, 2, blue, red, std::string("cloud"), std::string("normals"));
  runViewer(viewer);
}


void Plot::plotLocalAxes(const std::vector<Quadric>& quadric_list, const PointCloudRGBA::Ptr& cloud)
{
	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer = createViewer("Local Axes");
	viewer->addPointCloud<pcl::PointXYZRGBA>(cloud, "registered point cloud");
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1,
		"registered point cloud");

	for (int i = 0; i < quadric_list.size(); i++)
		quadric_list[i].plotAxes((void*) &viewer, i);

	runViewer(viewer);
}


void Plot::plotCameraSource(const Eigen::VectorXi& pts_cam_source_in, const PointCloudRGBA::Ptr& cloud)
{
	PointCloudRGBA::Ptr left_cloud(new PointCloudRGBA);
	PointCloudRGBA::Ptr right_cloud(new PointCloudRGBA);

	for (int i = 0; i < pts_cam_source_in.size(); i++)
	{
		if (pts_cam_source_in(i) == 0)
			left_cloud->points.push_back(cloud->points[i]);
		else if (pts_cam_source_in(i) == 1)
			right_cloud->points.push_back(cloud->points[i]);
	}
		
	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer = createViewer("Camera Sources");  
	viewer->addPointCloud<pcl::PointXYZRGBA>(left_cloud, "left point cloud");
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1,
		"left point cloud");
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 1.0, 0.0, 0.0,
		"left point cloud");
	viewer->addPointCloud<pcl::PointXYZRGBA>(right_cloud, "right point cloud");
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1,
		"right point cloud");
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0.0, 1.0, 0.0,
		"right point cloud");
	runViewer(viewer);
}


PointCloudNormal::Ptr Plot::createNormalsCloud(const std::vector<GraspHypothesis>& hand_list,
  bool plots_only_antipodal, bool plots_grasp_bottom)
{
	PointCloudNormal::Ptr cloud(new PointCloudNormal);

	for (int i = 0; i < hand_list.size(); i++)
	{
		Eigen::Matrix3Xd grasp_surface = hand_list[i].getGraspSurface();
		Eigen::Matrix3Xd grasp_bottom = hand_list[i].getGraspBottom();
		Eigen::Matrix3Xd hand_approach = hand_list[i].getApproach();

    if (!plots_only_antipodal || (plots_only_antipodal && hand_list[i].isFullAntipodal()))
    {
      pcl::PointNormal p;
      if (!plots_grasp_bottom)
      {
        p.x = grasp_surface(0);
        p.y = grasp_surface(1);
        p.z = grasp_surface(2);
      }
      else
      {
        p.x = grasp_bottom(0);
        p.y = grasp_bottom(1);
        p.z = grasp_bottom(2);
      }
      p.normal[0] = -hand_approach(0);
      p.normal[1] = -hand_approach(1);
      p.normal[2] = -hand_approach(2);
      cloud->points.push_back(p);
    }
	}

	return cloud;
}


void Plot::addCloudNormalsToViewer(boost::shared_ptr<pcl::visualization::PCLVisualizer>& viewer,
	const PointCloudNormal::Ptr& cloud, double line_width, double* color_cloud,
	double* color_normals, const std::string& cloud_name, const std::string& normals_name)
{
	viewer->addPointCloud<pcl::PointNormal>(cloud, cloud_name);
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, color_cloud[0],
		color_cloud[1], color_cloud[2], cloud_name);
  viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 6,
		cloud_name);
	viewer->addPointCloudNormals<pcl::PointNormal>(cloud, 1, 0.01, normals_name);
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, line_width,
		normals_name);
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, color_normals[0],
		color_normals[1], color_normals[2], normals_name);
}


void Plot::plotHandsHelper(const PointCloudNormal::Ptr& hands_cloud,
	const PointCloudNormal::Ptr& antipodal_hands_cloud, const PointCloudRGBA::Ptr& cloud,
	std::string str, bool use_grasp_bottom)
{
	std::cout << "Drawing " << hands_cloud->size() << " grasps of which " << antipodal_hands_cloud->size()
			<< " are antipodal grasps.\n";

	std::string title = "Hands: " + str;
	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer = createViewer(title);

	pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBA> rgb(cloud);
  viewer->addPointCloud<pcl::PointXYZRGBA>(cloud, rgb, "cloud");
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "cloud");

	double green[3] = { 0.0, 1.0, 0.0 };
	double cyan[3] = { 0.0, 0.4, 0.8 };
	addCloudNormalsToViewer(viewer, hands_cloud, 2, green, cyan, std::string("hands"), std::string("approaches"));

	if (antipodal_hands_cloud->size() > 0)
	{
		double red[3] = { 1.0, 0.0, 0.0 };
		addCloudNormalsToViewer(viewer, antipodal_hands_cloud, 2, green, red, std::string("antipodal_hands"),
			std::string("antipodal_approaches"));
	}

	runViewer(viewer);
}


void Plot::runViewer(boost::shared_ptr<pcl::visualization::PCLVisualizer>& viewer)
{
	while (!viewer->wasStopped())
	{
		viewer->spinOnce(100);
		boost::this_thread::sleep(boost::posix_time::microseconds(100000));
	}
  
  viewer->close();
}


boost::shared_ptr<pcl::visualization::PCLVisualizer> Plot::createViewer(std::string title)
{
  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer(title));  
  viewer->initCameraParameters();
  viewer->setPosition(0, 0);
  viewer->setSize(640, 480);
  viewer->setBackgroundColor(1.0, 1.0, 1.0);
  // viewer->addCoordinateSystem(1.0, "", 0);
  return viewer;
}


void Plot::createVisualPublishers(ros::NodeHandle& node, double marker_lifetime)
{
  hypotheses_pub_ = node.advertise<visualization_msgs::MarkerArray>("grasp_hypotheses_visual", 10);
  antipodal_pub_ = node.advertise<visualization_msgs::MarkerArray>("antipodal_grasps_visual", 10);
  handles_pub_ = node.advertise<visualization_msgs::MarkerArray>("handles_visual", 10);
  marker_lifetime_ = marker_lifetime;
}


void Plot::plotGraspsRviz(const std::vector<GraspHypothesis>& hand_list, const std::string& frame, bool is_antipodal)
{  
  double red[3] = {1, 0, 0};
  double cyan[3] = {0, 1, 1};
  double* color;
  if (is_antipodal)
  {
		color = red;
		std::cout << "Visualizing antipodal grasps in Rviz ...\n";
	}
	else
	{
		color = cyan;
		std::cout << "Visualizing grasp hypotheses in Rviz ...\n";
	}
  
  visualization_msgs::MarkerArray marker_array;
  marker_array.markers.resize(hand_list.size());  
  
  for (int i=0; i < hand_list.size(); i++)
  {
    geometry_msgs::Point position;
    position.x = hand_list[i].getGraspSurface()(0);
    position.y = hand_list[i].getGraspSurface()(1);
    position.z = hand_list[i].getGraspSurface()(2);
    visualization_msgs::Marker marker = createApproachMarker(frame, position, hand_list[i].getApproach(), i, color, 0.4, 
			0.004);
		marker.ns = "grasp_hypotheses";
		marker.id = i;
    marker_array.markers[i] = marker;
  }
  
  if (is_antipodal)
		antipodal_pub_.publish(marker_array);
  else
		hypotheses_pub_.publish(marker_array);
  
  ros::Duration(1.0).sleep();
}


void Plot::plotHandlesRviz(const std::vector<Handle>& handle_list, const std::string& frame)
{
	std::cout << "Visualizing handles in Rviz ...\n";
  double green[3] = {0, 1, 0};
  visualization_msgs::MarkerArray marker_array;
  marker_array.markers.resize(handle_list.size());  
  
  for (int i=0; i < handle_list.size(); i++)
  {
    geometry_msgs::Point position;
    position.x = handle_list[i].getGraspSurface()(0);
    position.y = handle_list[i].getGraspSurface()(1);
    position.z = handle_list[i].getGraspSurface()(2);
    visualization_msgs::Marker marker = createApproachMarker(frame, position, handle_list[i].getApproach(), i, green, 0.6,
			0.008);		
    marker.ns = "handles";
		marker_array.markers[i] = marker;
  }
  
  handles_pub_.publish(marker_array);
  ros::Duration(1.0).sleep();
}


void Plot::plotHandlesComplete(const std::vector<Handle>& handle_list, const PointCloudRGBA::Ptr& cloud, std::string str)
{
  const int WIDTH = pcl::visualization::PCL_VISUALIZER_LINE_WIDTH;
  const double OFFSET = 0.06;

  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer = createViewer(str);
  pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBA> rgb(cloud);
  viewer->addPointCloud<pcl::PointXYZRGBA>(cloud, rgb, "cloud");
  viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "cloud");

  for (int i = 0; i < handle_list.size(); i++)
  {
    std::cout << "axis.norm: " << handle_list[i].getAxis().norm() << ", approach.norm: "
      << handle_list[i].getApproach().norm() << ", binormal.norm: " << handle_list[i].getBinormal().norm() << std::endl;
    std::cout << "axis*approach: " << handle_list[i].getAxis().transpose() * handle_list[i].getApproach()
      << ", axis*binormal: " << handle_list[i].getAxis().transpose() * handle_list[i].getBinormal()
      << ", binormal*approach: " << handle_list[i].getApproach().transpose() * handle_list[i].getBinormal() << "\n";

    pcl::PointXYZ p, q, r, s;
    p.getVector3fMap() = handle_list[i].getGraspSurface().cast<float>();
    q.getVector3fMap() = (handle_list[i].getGraspSurface() + OFFSET*handle_list[i].getAxis()).cast<float>();
    r.getVector3fMap() = (handle_list[i].getGraspSurface() + OFFSET*handle_list[i].getApproach()).cast<float>();
    s.getVector3fMap() = (handle_list[i].getGraspSurface() + OFFSET*handle_list[i].getBinormal()).cast<float>();
    std::string istr = boost::lexical_cast<std::string>(i);
    viewer->addLine<pcl::PointXYZ> (p, q, 1.0, 0.0, 0.0, "axis_" + istr);
    viewer->addLine<pcl::PointXYZ> (p, r, 0.0, 1.0, 0.0, "approach_" + istr);
    viewer->addLine<pcl::PointXYZ> (p, s, 0.0, 0.0, 1.0, "binormal_" + istr);

    viewer->setShapeRenderingProperties(WIDTH, 5, "axis_" + boost::lexical_cast<std::string>(i));
    viewer->setShapeRenderingProperties(WIDTH, 5, "approach_" + boost::lexical_cast<std::string>(i));
    viewer->setShapeRenderingProperties(WIDTH, 5, "binormal_" + boost::lexical_cast<std::string>(i));
  }

  runViewer(viewer);
}


void Plot::plotHandles(const std::vector<Handle>& handle_list, const PointCloudRGBA::Ptr& cloud, std::string str)
{
	double colors[6][3] = { { 1.0, 0.0, 0.0 }, { 0.0, 1.0, 0.0 }, { 0.0, 0.0, 1.0 },
	  { 1.0, 1.0, 0.0 }, { 1.0, 0.0, 1.0 }, { 0.0, 1.0, 1.0 } };
//  { 0, 0.4470, 0.7410 },
//	    { 0, 0.4470, 0.7410 }, { 0, 0.4470, 0.7410 }, { 0.8500, 0.3250, 0.0980 }, { 0.9290, 0.6940, 0.1250 },
//	    { 0.4940, 0.1840, 0.5560 }, { 0.4660, 0.6740, 0.1880 }, { 0.3010, 0.7450, 0.9330 }, { 0.6350, 0.0780,	0.1840 },
//	    { 0, 0.4470, 0.7410 }, { 0.8500, 0.3250, 0.0980 }, { 0.9290, 0.6940, 0.1250} };

	//	    0.4940    0.1840    0.5560
	//	    0.4660    0.6740    0.1880
	//	    0.3010    0.7450    0.9330
	//	    0.6350    0.0780    0.1840
	//	         0    0.4470    0.7410
	//	    0.8500    0.3250    0.0980
	//	    0.9290    0.6940    0.1250
	//	    0.4940    0.1840    0.5560
	//	    0.4660    0.6740    0.1880
	//	    0.3010    0.7450    0.9330
	//	    0.6350    0.0780    0.1840

	std::vector<pcl::PointCloud<pcl::PointNormal>::Ptr> clouds;
	pcl::PointCloud<pcl::PointNormal>::Ptr handle_cloud(new pcl::PointCloud<pcl::PointNormal>());

	for (int i = 0; i < handle_list.size(); i++)
	{
		pcl::PointNormal p;
		p.x = handle_list[i].getGraspSurface()(0);
		p.y = handle_list[i].getGraspSurface()(1);
		p.z = handle_list[i].getGraspSurface()(2);
		p.normal[0] = -handle_list[i].getApproach()(0);
		p.normal[1] = -handle_list[i].getApproach()(1);
		p.normal[2] = -handle_list[i].getApproach()(2);
		handle_cloud->points.push_back(p);

		const std::vector<int>& inliers = handle_list[i].getInliers();
		const std::vector<GraspHypothesis>& hand_list = handle_list[i].getHandList();
		pcl::PointCloud<pcl::PointNormal>::Ptr axis_cloud(new pcl::PointCloud<pcl::PointNormal>);

		for (int j = 0; j < inliers.size(); j++)
		{
			pcl::PointNormal p;
			p.x = hand_list[inliers[j]].getGraspSurface()(0);
			p.y = hand_list[inliers[j]].getGraspSurface()(1);
			p.z = hand_list[inliers[j]].getGraspSurface()(2);
			p.normal[0] = -hand_list[inliers[j]].getApproach()(0);
			p.normal[1] = -hand_list[inliers[j]].getApproach()(1);
			p.normal[2] = -hand_list[inliers[j]].getApproach()(2);
			axis_cloud->points.push_back(p);
		}
		clouds.push_back(axis_cloud);
	}

	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer(str));
	viewer->setBackgroundColor(1, 1, 1);
  viewer->setPosition(0, 0);
  viewer->setSize(640, 480);
	pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBA> rgb(cloud);
	viewer->addPointCloud<pcl::PointXYZRGBA>(cloud, rgb, "registered point cloud");
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1,
			"registered point cloud");
//	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0.0, 1.0, 0.0,
//			"registered point cloud");

	for (int i = 0; i < clouds.size(); i++)
	{
		std::string name = "handle_" + boost::lexical_cast<std::string>(i);
		int ci = i % 6;
		viewer->addPointCloud<pcl::PointNormal>(clouds[i], name);
		viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, colors[ci][0],
				colors[ci][1], colors[ci][2], name);
		std::string name2 = "approach_" + boost::lexical_cast<std::string>(i);
		viewer->addPointCloudNormals<pcl::PointNormal>(clouds[i], 1, 0.04, name2);
		viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 2, name2);
		viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, colors[ci][0],
				colors[ci][1], colors[ci][2], name2);
	}

	viewer->addPointCloud<pcl::PointNormal>(handle_cloud, "handles");
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0, 0, 0, "handles");
	viewer->addPointCloudNormals<pcl::PointNormal>(handle_cloud, 1, 0.08, "approach");
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 4, "approach");
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0, 0, 0, "approach");

	// viewer->addCoordinateSystem(1.0, "", 0);
	viewer->initCameraParameters();
  viewer->setPosition(0, 0);
	viewer->setSize(640, 480);

	while (!viewer->wasStopped())
	{
		viewer->spinOnce(100);
		boost::this_thread::sleep(boost::posix_time::microseconds(100000));
	}
  
  viewer->close();
}


void Plot::drawCloud(const PointCloudRGBA::Ptr& cloud_rgb, const std::string& title)
{
  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer = createViewer(title);
  pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBA> rgb(cloud_rgb);
  viewer->addPointCloud<pcl::PointXYZRGBA>(cloud_rgb, rgb, "registered point cloud");
  viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "registered point cloud");
  runViewer(viewer);
}


visualization_msgs::Marker Plot::createApproachMarker(const std::string& frame, const geometry_msgs::Point& center, 
	const Eigen::Vector3d& approach, int id, const double* color, double alpha, double diam)
{
  visualization_msgs::Marker marker = createMarker(frame);
  marker.type = visualization_msgs::Marker::ARROW;
  marker.id = id;
  marker.scale.x = diam; // shaft diameter
  marker.scale.y = diam; // head diameter
  marker.scale.z = 0.01; // head length
  marker.color.r = color[0];
  marker.color.g = color[1];
  marker.color.b = color[2];
  marker.color.a = alpha;
  geometry_msgs::Point p, q;
  p.x = center.x;
  p.y = center.y;
  p.z = center.z;
  q.x = p.x - 0.03 * approach(0);
  q.y = p.y - 0.03 * approach(1);
  q.z = p.z - 0.03 * approach(2);
  marker.points.push_back(p);
  marker.points.push_back(q);
  return marker;
}


visualization_msgs::Marker Plot::createMarker(const std::string& frame)
{
  visualization_msgs::Marker marker;
  marker.header.frame_id = frame;
  marker.header.stamp = ros::Time::now();
  marker.lifetime = ros::Duration(marker_lifetime_);
  marker.action = visualization_msgs::Marker::ADD;
  return marker;
}

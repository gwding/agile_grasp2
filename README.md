# agile_grasp 2.0

* **Author:** Andreas ten Pas (atp@ccs.neu.edu)
* **Version:** 1.0.0
* **ROS Wiki page:** not available yet
* **Author's website:** [http://www.ccs.neu.edu/home/atp/](http://www.ccs.neu.edu/home/atp/)


## 1) Overview

This package localizes antipodal grasps in 3D point clouds. AGILE stands for **A**ntipodal **G**rasp **I**dentification and 
**LE**arning. The reference for this package is: 
[High precision grasp pose detection in dense clutter](http://arxiv.org/abs/1603.01564). *agile_grasp 2.0* is an improved 
version of our previous package, [agile_grasp](http://wiki.ros.org/agile_grasp).

The package already comes with a pre-trained machine learning classifier and can be used (almost) out-of-the-box with 
RGBD cameras such as the Microsoft Kinect and the Asus Xtion Pro as well as point clouds stored as *.pcd files.


## 2) Requirements

1. [ROS Indigo](http://wiki.ros.org/indigo) (installation instructions for [Ubuntu](http://wiki.ros.org/indigo/Installation/Ubuntu))
2. [Lapack](http://www.netlib.org/lapack/) (install in Ubuntu: `$ sudo apt-get install liblapack-dev`) 
3. [OpenNI](http://wiki.ros.org/openni_launch) or a similar range sensor driver
4. [Caffe](http://caffe.berkeleyvision.org/) 


## 3) Installation

1. Install Caffe. [Instructions](https://github.com/BVLC/caffe/wiki/Install-Caffe-on-EC2-from-scratch-%28Ubuntu,-CUDA-7,-cuDNN%29) for Ubuntu 14.04. 
2. Compile Caffe as a cmake library ([https://github.com/BVLC/caffe/pull/1667](instructions)):

   ```
   $ cd caffe && mkdir cmake_build && cd cmake_build
   $ cmake .. -DBUILD_SHARED_LIB=ON
   $ make -j 12
   $ ln -s ../build .
   ```
   
   If the second line gives the error *Manually-specified variables were not used by the project: BUILD_SHARED_LIB*, 
   just run the second line again.
3. Clone the agile_grasp 2.0 repository into your ros workspace: 

   ```
   $ cd <location_of_your_ros_workspace>/src
   $ git clone https://github.com/atenpas/agile_grasp2
   ```
4. Recompile your ROS workspace: 

   ```
   $ cd ..
   $ catkin_make
   ```


## 4) Detect Grasp Poses With a Robot

1. Connect a range sensor, such as a Microsoft Kinect or Asus Xtion Pro, to your robot.
2. Adjust the file *robot_detect_grasps.launch* for your sensor and robot hand.
3. Run the grasp pose detection: 
   
   ```
   $ roslaunch agile_grasp2 robot_detect_grasps.launch
   ```

![Image Alt](readme/robot1.png)
![Image Alt](readme/robot2.png)


## 5) Detect Grasp Poses in a PCD File

1. Have a *.pcd file available. Let's say it's located at */home/user/cloud.pcd*. 
2. Change the parameter *cloud_file_name* in the file *file_detect_grasps.launch* to */home/user/cloud.pcd*.
3. Detect grasp poses: 
  
   ```
   roslaunch agile_grasp2 file_detect_grasps.launch
   ```
![Image Alt](readme/file1.png)


## 6) Using Precomputed Normals

The ROS node, *grasp_detection*, can take point clouds with normals as input. The normals need to be stored in the 
normal_x, normal_y, and normal_z fields. **Attention**: the only preprocessing operation that works with this is 
workspace filtering, so avoid voxelization and manual indexing.


## 7) Parameters

The most important parameters are cloud_type and cloud_topic to define the input point cloud, and workspace and 
num_samples to define the workspace dimensions and the number of grasp hypotheses to be sampled. The other parameters 
only need to be modified in special cases.

#### Input
* cloud_type: the type of the input point cloud. 
  * 0: *.pcd file
  * 1: sensor_msgs/PointCloud2
  * 2: agile_grasp/CloudSized
  * 3: agile_grasp/CloudIndexed.
* cloud_topic: the ROS topic from which the input point cloud is received.

#### Visualization
* plot_mode: what type of visualization is used (0: no visualization, 1: PCL, 2: rviz (not supported yet))
* only_plot_output: set this to *false* to see additional visualizations.

#### Grasp Hypothesis Search
* workspace: the limits of the robot's workspace, given as a cube centered at the origin: [x_min, x_max, y_min, y_max, z_min, z_max] (in meters).
* camera_pose: the camera pose as a 4x4 homogeneous transformation matrix.
* num_samples: the number of grasp hypotheses to be sampled.
* num_threads: the number of CPU threads to be used.
* nn_radius_taubin: the neighborhood search radius for normal estimation (in meters).
* nn_radius_hands: the neighborhood search radius for the grasp hypothesis search (in meters).
* num_orientations: the number of hand orientations to be considered.
* antipodal_mode: the output of the algorithm. 0: grasp hypotheses, 1: antipodal graps (prediction), 2: antipodal grasps (geometric).
* normal_estimation_method: the method used to estimate normals. 0: Taubin quadric fitting, 1: PCL normal estimation.
* voxelize: if the point cloud gets voxelized.
* filter_half_grasps: if half-grasps are filtered out.
* gripper_width_range: the aperture range of the robot hand: [aperture_min, aperture_max].

#### Robot Hand Geometry
* finger_width: the finger width.
* hand_outer_diameter: the hand's outer diameter (aperture + finger_width).
* hand_depth: the hand's depth (length of fingers).
* hand_height: the hand's height.
* init_bite: the initial amount that the hand extends into the object to be grasped.

#### Classifier
* images_directory: where images are stored (not used).
* model_file: the Caffe prototxt file that specifies the network.
* trained_file: the Caffe model file that contains the weights for the network.
* label_file: a txt file that contains the label for each class.
* min_score_diff: the minimum difference between the positive and the negative score for a grasp to be classfied as positive.

#### Grasp Selection
* num_selected: the number of selected grasps. If antipodal grasps are predicted/calculated, then the selected grasps will be 
the highest-scoring grasps.

#### Other
* use_service: uses a ROS service instead of topics (untested).


## 8) Citation

If you like this package and use it in your own work, please cite our [arXiv paper](http://arxiv.org/abs/1603.01564):

```
@misc{1603.01564,
Author = {Marcus Gualtieri and Andreas ten Pas and Kate Saenko and Robert Platt},
Title = {High precision grasp pose detection in dense clutter},
Year = {2016},
Eprint = {arXiv:1603.01564},
} 
```

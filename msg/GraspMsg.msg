# This message describes a grasp by its 6-DOF pose, consisting of a 3-DOF position and 3-DOF orientation, and the 
# required aperture (opening width) of the robot hand.

# Grasp position in the middle of the hand ...
geometry_msgs/Point surface # On the object surface
geometry_msgs/Point bottom # At the robot hand base
geometry_msgs/Point top # At the robot hand fingertips

# Grasp orientation represented as a frame
geometry_msgs/Vector3 axis # The hand axis 
geometry_msgs/Vector3 approach # The grasp approach direction
geometry_msgs/Vector3 binormal # The binormal

std_msgs/Float32 width # The required aperture (opening width) of the robot hand 

std_msgs/Float32 score # The score assigned to the grasp by the classifier

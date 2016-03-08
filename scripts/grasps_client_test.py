# ROS
import rospy

# custom messages/services
from agile_grasp.srv import *


# connect to agile_grasp ROS service
rospy.wait_for_service('/grasps_server/find_grasps')
findGraspsClient = rospy.ServiceProxy('/grasps_server/find_grasps', FindGrasps)

try:
  req = FindGraspsRequest()
  req.grasps_signal = 0
  req.calculate_antipodal = True
  # if signal == 1:
    # req.centroid = geometry_msgs.msg.Vector3(0, 0, centroid[2])
  resp = findGraspsClient(req)
  print "Received", len(resp.grasps_msg.grasps), "grasp poses"
except rospy.ServiceException, e:
  print "Service call failed: %s"%e  

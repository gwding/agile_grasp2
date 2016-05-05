import numpy as np
import matplotlib.pyplot as plt
from pylab import *
import sys
import matplotlib.pyplot as plt


# Check command line arguments
if len(sys.argv) < 3:
  print "Not enough commandline arguments!"
  print "Usage: python deploy_classifier.py caffe_directory grasp_images_directory"
  exit(-1)

# Make sure that caffe is on the python path:
caffe_root = sys.argv[1]
sys.path.insert(0, caffe_root + 'python')
import caffe

caffe.set_device(0)
caffe.set_mode_gpu()

net = caffe.Net('../caffe/test_1batch2.prototxt', '../caffe/bottles_boxes_cans_5xNeg.caffemodel', caffe.TEST)
net.forward()

rootdir = sys.argv[2]
testFilename = sys.argv[3]

predictionList = []
predictionProbsList = []

# Classify the images specified in rootdir/test.txt
with open(rootdir + testFilename, "r") as fh:
	lines = fh.readlines()
	for line in lines:
		filename = line[:-1]
		img = plt.imread(rootdir + 'jpgs/' + filename)
		imgbgr = array([img.transpose(2,0,1)[2], img.transpose(2,0,1)[1], img.transpose(2,0,1)[0]])
		net.blobs['data'].data[0] = imgbgr
		net.forward(start='conv1')
		
		predictionProbs = net.blobs['ip2'].data[0].copy()
		prediction = net.blobs['ip2'].data[0].argmax(0)
		
		predictionList.append(prediction)
		predictionProbsList.append(predictionProbs)

# Write the prediction scores to a file
predictionFilename = rootdir + 'predictionList_' + testFilename
with open(predictionFilename, "w") as fh:
	for predictionProbs in predictionProbsList:
		#~ fh.write(str(predictionProbs) + '\n')
		fh.write(str(predictionProbs[0]) + ', ' + str(predictionProbs[1]) + '\n')
    
print "Wrote:", predictionFilename

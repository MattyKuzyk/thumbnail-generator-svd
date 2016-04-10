# import the necessary packages
from matplotlib import pyplot as plt
import numpy as np
import argparse
import cv2
import os
 
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", required = False, help = "Path to the video")
ap.add_argument("-f", "--frames", required = False, help = "Path to the video")
args = vars(ap.parse_args())
frames = []
A = [] 
# load the video and show it

if args["video"]:
	vidcap = cv2.VideoCapture(args["video"])
	count = 0;
	success = True
	while success:
	    success,frame = vidcap.read()
	    print "success"
	    if success:
	        frames.append(frame)

elif args["frames"]: # Hack for windows
	for i in os.listdir(args["frames"]):
		frames.append(cv2.imread(args["frames"] + "\\" + i))


print "Finished converting video!"

for frame in frames:
	y = len(frame)
	x = len(frame[0])
	for i in range(0, 3):
		for j in range(0, 3):
			window = frame[i:(y/3)*(i+1)][j:(x/3)*(j+1)]
			hist = cv2.calcHist([frame], [0, 1, 2], 
				None, [8, 8, 8], [0, 256, 0, 256, 0, 256]).flatten()
			A.append(hist)

A = np.array(A)
A.transpose()
rank = np.linalg.matrix_rank(A)
svd = np.linalg.svd(A,compute_uv=True)
V = svd[2]
norms = []
for eigenvec in V:
	norm = 0
	for j in range(0,rank):
		norm += np.power(eigenvec[j],2)
	norms.append(norm)

static_cluster = np.sum(norms)
print static_cluster
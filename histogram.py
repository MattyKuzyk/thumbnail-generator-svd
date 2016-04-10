# import the necessary packages
from matplotlib import pyplot as plt
import numpy as np
import argparse
import cv2
 
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", required = True, help = "Path to the video")
args = vars(ap.parse_args())
 
# load the video and show it
vidcap = cv2.VideoCapture(args["video"])

frames = []
A = []
count = 0;
success = True
while success:
    success,frame = vidcap.read()
    if success:
        frames.append(frame)

print "Finished converting video!"
print frames

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
svd = np.linalg.svd(A)
print svd
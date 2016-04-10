# import the necessary packages
from matplotlib import pyplot as plt
import numpy as np
import argparse
import cv2
 
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Path to the image")
args = vars(ap.parse_args())
 
# load the image and show it
image = cv2.imread(args["image"])
y = len(image)
x = len(image[0])

for i in range(0, 3):
	for j in range(0, 3):
		window = image[i:(y/3)*(i+1)][j:(x/3)*(j+1)]
		hist = cv2.calcHist([image], [0, 1, 2], 
			None, [8, 8, 8], [0, 256, 0, 256, 0, 256]).flatten()
		A.append(hist)
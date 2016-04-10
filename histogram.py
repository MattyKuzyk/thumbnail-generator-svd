# import the necessary packages
from matplotlib import pyplot as plt
import numpy as np
import argparse
import cv2
import os
import itertools as it
import more_itertools as mit

# Returns a generator of numpy matrices for each frame of a video from the given
# file name.
def frames_from_file(filename):
    vidcap = cv2.VideoCapture(filename)
    success = True
    while success:
        success,frame = vidcap.read()
        if success:
            yield frame

# Returns a generator of numpy matrices for each frame of a video from the given
# folder
def frames_from_folder(folder_path):
    for i in os.listdir(folder_path):
        yield cv2.imread(folder_path + "\\" + i)

# Returns a generator which splits up a frame into nxm blocks from the given
# frame.
def partion_frame(frame, m, n):
    y = frame.shape[0]
    x = frame.shape[1]
    for i in range(0, m):
        for j in range(0, n):
            yield frame[i:(y/m)*(i+1)][j:(x/n)*(j+1)]

# Returns a generator which calculates a 3D histogram of a block
def histogram(block, bins):
    hist = cv2.calcHist([block], [0, 1, 2],
        None, [bins, bins, bins], [0, 256, 0, 256, 0, 256]).flatten()
    return hist

def frame_vector(frame, bins, m, n):
    hs = [histogram(block, bins) for block in partion_frame(frame, m, n)]
    return np.array(hs).flatten()

def video_to_feature_matrix(filename, bins, m, n):
    frames = frames_from_file(filename)
    vecs = ([v for v in frame_vector(f, bins, m, n)] for f in frames)
    return np.stack(vecs, axis=1)

def folder_to_feature_matrix(path, bins, m, n):
    frames = frames_from_folder(path)
    vecs = ([v for v in frame_vector(f, bins, m, n)] for f in frames)
    return np.stack(vecs, axis=1)

# A = np.array(A)
# A = A.transpose()
# rank = np.linalg.matrix_rank(A)
# svd = np.linalg.svd(A,compute_uv=True)
# V = svd[2] # Returns eigenvectors as rows
# norms = []
# for eigenvec_columns in V.transpose(): # The metric is the sum of the the jth position in every eigenvector for all i eigenvectors
#     norm = 0
#     for j in range(0,rank):
#         norm += np.power(eigenvec_columns[j],2)
#     norms.append(norm)
#
# static_cluster = np.sum(norms)
# print static_cluster

if __name__ == "__main__":
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", required = False, help = "Path to the video")
    ap.add_argument("-f", "--frames", required = False, help = "Path to the ideo")
    args = parser.parse_args()

    A = None;
    if args.video:
        A = video_to_feature_matrix(args.video)
    elif args.frames:
        A = folder_to_feature_matrix(args.frames)

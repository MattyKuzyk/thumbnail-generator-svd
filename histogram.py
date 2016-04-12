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

def frames_to_matrix(frames, bins, m, n):
    vecs = ([v for v in frame_vector(f, bins, m, n)] for f in frames)
    return np.stack(vecs, axis=1)

def video_to_feature_matrix(filename, bins, m, n):
    frames = frames_from_file(filename)
    return frames_to_matrix(frames, bins, m, n)

def folder_to_feature_matrix(path, bins, m, n):
    frames = frames_from_folder(path)
    return frames_to_matrix(frames, bins, m, n)

def norm(Vi, rank):
    return np.sqrt(np.sum(np.power(Vi[0:rank], 2)))

# Euclidean distance weighted by the singular values from S. t specifies how
# many singular values to use.
def frame_distance(Vi, Vj, V, S, t):
    s_values = S[0:t-1]
    diff = (Vi - Vj)[0:t-1]
    return np.sqrt(np.power(np.dot(s_values, diff), 2))

def cluster_content(vecs, rank):
    return sum([np.power(norm(Vi, rank), 2) for Vi in vecs])

# A singular value decomposition which takes into account large numbers of frame
def frame_decomp(A):
    if A.shape[0] < A.shape[1]:
        At = A.transpose()
        U, S, V = np.linalg.svd(At)
        rank = np.linalg.matrix_rank(At)
        return (U, S, rank)
    else:
        U, S, V = np.linalg.svd(A)
        rank = np.linalg.matrix_rank(A)
        return (V, S, rank)

def cluster(V, S, rank, t):
    norms = np.apply_along_axis(lambda Vi: norm(Vi, rank), 1, V)
    # Keep track of original order of V to relate back to frame
    V_idx = np.array([range(0,V.shape[0])]).T
    V_with_idx = np.append(V_idx, V, axis=1)
    sorted_V = V_with_idx[np.argsort(norms),:]
    unclustered_V = sorted_V
    clusters = []

    def min_dist(Vi, cluster, V, S, t):
        return min([frame_distance(Vi, Vj[1:], V, S, t) for Vj in cluster['children']])

    def new_internal_dist(c, Vi, V, S, t):
        c_len = len(c['children'])
        return ((c_len - 1)*c['internal_distance'] + min_dist(Vi, c, V, S, t))/c_len

    while len(unclustered_V) > 0:
        clusters.append({
            'internal_distance': 0,
            'children': [unclustered_V[0,:]]
        })
        unclustered_V = unclustered_V[1:,:]
        to_delete_idx = []

        for i, Vi in zip(it.count(), unclustered_V):
            if len(clusters) == 1:
                c = clusters[0]
                if (c['internal_distance'] == 0 or
                    (min_dist(Vi[1:], c, V, S, t)/c['internal_distance'] < 5.0)):
                    c['internal_distance'] = new_internal_dist(c, Vi[1:], V, S, t)
                    c['children'].append(Vi)
                    to_delete_idx.append(i)
            else:
                for c in clusters:
                    if (c['internal_distance'] == 0 or
                        cluster_content(c['children'], rank) < cluster_content([Vi[1:]], rank) or
                        (min_dist(Vi[1:], c, V, S, t)/c['internal_distance'] < 2.0)):
                        c['internal_distance'] = new_internal_dist(c, Vi[1:], V, S, t)
                        c['children'].append(Vi)
                        to_delete_idx.append(i)

        unclustered_V = np.delete(unclustered_V, to_delete_idx, axis=0)
        print 'Cluster count: {}'.format(len(clusters))
        print 'Unclustered V: {}'.format(len(unclustered_V))

    return clusters

def play_cluster(frames, cluster, fps):
    for v in sorted(cluster['children'], key=lambda v: v[0]):
        cv2.imshow('cluster', frames[int(v[0])])
        cv2.waitKey(1000/fps)

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

    V, S, rank = frame_decomp(A)

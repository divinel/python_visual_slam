#! /usr/bin/env python

import numpy as np
import cv2 as cv
from landmark import Landmark
import pose

def reconstruct(K, frame1, frame2, matches, inliers, landmark_map):
    '''
        given two frames and matches, reconstruct 3d landmarks in world frame.
        landmark_map, which is a map of index : Landmark,  will be updated
        newly reconstucted landmarks in world frame will be returned
    '''
    new_matched_pts1 = []
    new_matched_kps_idx1 = []
    new_matched_pts2 = []
    new_matched_kps_idx2 = []
    for i, match in enumerate(matches):
        if inliers[i]:
            if frame1.landmark_idx[match.queryIdx] < 0: # newly reconstructed landmark
                new_matched_pts1.append(frame1.kps[match.queryIdx].pt)
                new_matched_kps_idx1.append(match.queryIdx)
                new_matched_pts2.append(frame2.kps[match.trainIdx].pt)
                new_matched_kps_idx2.append(match.trainIdx)
            else:   # previousely constructed landmark, already in landmark_map
                landmark_idx = frame1.landmark_idx[match.queryIdx]
                assert(landmark_idx in landmark_map)
                landmark_map[landmark_idx].add_observation(frame2.pose, frame2.kps[match.trainIdx].pt)
                frame2.landmark_idx[match.trainIdx] = landmark_idx
    matched_pts1 = np.array(new_matched_pts1)
    matched_pts2 = np.array(new_matched_pts2)
    P1 = K.dot(pose.get_3x4_pose_mat(pose.get_inverse_pose(frame1.pose)))
    P2 = K.dot(pose.get_3x4_pose_mat(pose.get_inverse_pose(frame2.pose)))
    point3d_homo = cv.triangulatePoints(P1, P2, matched_pts1.T, matched_pts2.T)
    point3d_homo /= point3d_homo[3, :]
    pts_3d = point3d_homo[:3, :].T.tolist()

    new_landmark_idx = 0
    if len(landmark_map):
        new_landmark_idx = max(landmark_map.keys()) + 1
    for i, pt_3d in enumerate(pts_3d):
        assert(new_landmark_idx not in landmark_map)
        landmark_map[new_landmark_idx] = Landmark(pt_3d)
        landmark_map[new_landmark_idx].add_observation(frame1.pose, 
                                                       frame1.kps[new_matched_kps_idx1[i]].pt)
        frame1.landmark_idx[new_matched_kps_idx1[i]] = new_landmark_idx
        landmark_map[new_landmark_idx].add_observation(frame2.pose,
                                                       frame2.kps[new_matched_kps_idx2[i]].pt)
        frame2.landmark_idx[new_matched_kps_idx2[i]] = new_landmark_idx
        new_landmark_idx += 1
    return pts_3d



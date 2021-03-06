#!/usr/bin/env python3
import frame
import cv2 as cv
import numpy as np
import pose

class RelativeEstimator:
    def __init__(self):
        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm = 6,
                        table_number = 6,
                        key_size = 12,
                        multi_probe_level = 1)
        search_params = dict(checks = 50)
        self.matcher = cv.FlannBasedMatcher(index_params, search_params)

    def match_frames(self, frame1, frame2):
        matches =self.matcher.knnMatch(frame1.desc, frame2.desc, k = 2)
        good_matches = []
        matched_uvs = []
        for match in matches:
            if len(match) < 2:
                continue
            first, second = match
            if first.distance < (0.75 * second.distance):
                good_matches.append(first)
        good_matches.sort(key = lambda m : m.distance)
        
        for match in good_matches:
            matched_pt1 = frame1.kps[match.queryIdx].pt
            matched_pt2 = frame2.kps[match.trainIdx].pt
            uv1 = tuple(int(round(coord)) for coord in matched_pt1)
            uv2 = tuple(int(round(coord)) for coord in matched_pt2)
            matched_uvs.append((uv1, uv2))

        return good_matches, matched_uvs

def decompose_essential(E):
    U, D, Vt = np.linalg.svd(E)
    if np.linalg.det(U) < 0:
        U *= -1
    if np.linalg.det(Vt) < 0:
        Vt *= -1
    W = np.array([[0, -1, 0],
                  [1,  0, 0],
                  [0,  0, 1]])
    Wt = W.T
    R1 = U.dot(W).dot(Vt)
    R2 = U.dot(Wt).dot(Vt)
    t1 = U[:,2].reshape(3,1)
    t2 = -t1
    return ((R1, t1), (R1, t2), (R2, t1), (R2, t2))

def get_R_t(E, K, pts_1, pts_2, matches, inliers):
    '''
        Output parameter:
        R is cur_R_prev
        t is cur_t_prev
        x_cur = R * x_prev + t
    '''

    R_ts = decompose_essential(E)
    pts_inliers_1 = np.array([pts_1[match.queryIdx].pt for i, match in enumerate(matches) if inliers[i] > 0])
    pts_inliers_2 = np.array([pts_2[match.trainIdx].pt for i, match in enumerate(matches) if inliers[i] > 0])
    
    max_cnt = 0
    result = None
    for R, t in R_ts:
        P1 = K.dot(np.hstack((np.identity(3), np.zeros((3,1)))))
        P2 = K.dot(np.hstack((R, t.reshape(3,1))))
        x_1 = cv.triangulatePoints(P1, P2, pts_inliers_1.T, pts_inliers_2.T)
        x_1 /= x_1[3, :]
        x_2 = P2.dot(x_1)
        assert x_1.shape[1] == x_2.shape[1]
        count = 0
        for i in range(x_1.shape[1]):
            if x_1[2, i] > 0 and x_2[2, i] > 0:
                count += 1
        if count > max_cnt:
            result = (pose.orthonomalize_rot_mat(R), t)
            max_cnt = count

    return result

def get_pose(relative_Rt, prev_Rt):
    '''
        pose is representated as (world_R_cam, world_T_cam)
        relative_Rt is prev to cur
    '''
    relative_R, relative_T = relative_Rt
    prev_R_cur = relative_R.T
    prev_T_cur = prev_R_cur.dot(-relative_T)

    prev_R, prev_T = prev_Rt    
    new_R = prev_R.dot(prev_R_cur)
    new_T = prev_R.dot(prev_T_cur) + prev_T
    return (pose.orthonomalize_rot_mat(new_R), new_T)


def get_essential(F, K):
    K_trans = K.T
    return K_trans.dot(F).dot(K)


def estimate_fundamental(matches, frame1, frame2):
    # from matches build np.ndarray for matched points
    matched_kps1 = np.float32([frame1.kps[match.queryIdx].pt for match in matches])
    matched_kps2 = np.float32([frame2.kps[match.trainIdx].pt for match in matches])
    # 8 points RANSAC is built in opencv's findFundamentalMat function
    # return F and inliers mask
    return cv.findFundamentalMat(matched_kps1, matched_kps2, method = cv.FM_RANSAC, 
                                 ransacReprojThreshold = 1., confidence = 0.9)
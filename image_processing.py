#!/usr/bin/env python3
import cv2 as cv
import numpy as np

def color2gray(img):
    return cv.cvtColor(img, cv.COLOR_BGR2GRAY)

def offset_keypoint(kp, offset_x, offset_y):
    u,v = kp.pt
    kp.pt = (u + offset_x, v + offset_y)
    return kp
class ORBFeatureExtractor:
    def __init__(self, max_kps):
        self.max_kps = max_kps
        self.orb = cv.ORB_create(nfeatures = self.max_kps)
    def extract(self, img):
        '''
            img: expected a gray image
        ''' 
        return self.orb.detectAndCompute(img, None)

class FeatureExtractor:
    def __init__(self, max_kps, quality_level, min_dist):
        self.max_kps = max_kps
        self.quality_level = quality_level
        self.min_dist = min_dist
    def extract(self, img):
        '''
            img: expected a gray image
        ''' 
        corners = cv.goodFeaturesToTrack(img, maxCorners = self.max_kps, 
                                         qualityLevel = self.quality_level, 
                                         minDistance = self.min_dist)
        kps = np.split(corners, corners.shape[0], axis = 0)
        kps = [kp.ravel().astype(int) for kp in kps]
        return kps 
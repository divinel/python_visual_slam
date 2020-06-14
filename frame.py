#!/usr/bin/env python

import numpy as np
class Frame:
    def __init__(self, pose, keypoints, descriptors):
        self.pose = pose
        self.kps = keypoints
        self.desc = descriptors
        self.landmark_idx = [-1] * len(self.kps)
#!/usr/bin/env python
class Frame:
    def __init__(self, pose, keypoints, descriptors):
        self.pose = pose
        self.kps = keypoints
        self.desc = descriptors
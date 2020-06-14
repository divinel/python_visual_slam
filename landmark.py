#!/usr/bin/env python
class Landmark:
    def __init__(self, xyz):
        self.xyz = xyz
        self.views = []
        self.uvs = []
    def add_observation(self, cam, uv):
        self.views.append(cam)
        self.uvs.append(uv)
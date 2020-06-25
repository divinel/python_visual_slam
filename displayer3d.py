#! /usr/bin/env python

import OpenGL.GL as gl
import pangolin

import numpy as np
from scipy.spatial.transform import Rotation

import threading
import time

class Renderer(threading.Thread):
    def __init__(self, win_name, cam_color = [1.0, 0.0, 0.0], map_pts_color = [0.0, 1.0, 0.0]):
        threading.Thread.__init__(self)
        self.win_name = win_name
        self.cam_color = cam_color
        self.map_pts_color = map_pts_color
        self.map_pts = None
        self.new_pts_obs = None 
        self.poses = []
        self.data_lock = threading.Lock()
        self.tree = None

    def add_map_pts(self, new_map_pts):
        self.data_lock.acquire()
        if self.map_pts is None:
            self.map_pts = new_map_pts
        else:
            self.map_pts = np.append(self.map_pts, new_map_pts, 0)
        self.new_pts_obs = new_map_pts
        self.data_lock.release()

    def add_pose(self, new_pose):
        self.data_lock.acquire()
        self.poses.append(new_pose)
        self.data_lock.release()
    def run(self):
        pangolin.CreateWindowAndBind(self.win_name, 640, 480)
        gl.glEnable(gl.GL_DEPTH_TEST)
        self.scam = pangolin.OpenGlRenderState(
            pangolin.ProjectionMatrix(640, 480, 420, 420, 320, 240, 0.2, 1000),
            pangolin.ModelViewLookAt(2, -10, -30, 0, 0, 0, pangolin.AxisNegY))

        self.tree = pangolin.Renderable()
        self.tree.Add(pangolin.Axis())
        self.handler = pangolin.SceneHandler(self.tree, self.scam)
        
        self.dcam = pangolin.CreateDisplay()
        self.dcam.SetBounds(0.0, 1.0, 0.0, 1.0, -640.0/480.0)
        self.dcam.SetHandler(self.handler)
        while not pangolin.ShouldQuit():
            gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
            gl.glClearColor(1.0, 1.0, 1.0, 1.0)
            self.dcam.Activate(self.scam)
            self.tree.Render()
            self.data_lock.acquire()

            if self.poses:
                gl.glPointSize(5)
                gl.glColor3f(*self.cam_color)
                for pose in self.poses:
                    pangolin.DrawPoints(pose[1].T)

            if self.map_pts is not None:
                gl.glPointSize(2)
                gl.glColor3f(*self.map_pts_color)
                pangolin.DrawPoints(self.map_pts)

            if self.new_pts_obs is not None:
                cur_pos = self.poses[-1]
                cam_centers = np.repeat(cur_pos[1].T, self.new_pts_obs.shape[0], axis = 0)
                gl.glLineWidth(1)
                gl.glColor3f(0.0, 0.0, 1.0)
                pangolin.DrawLines(cam_centers, self.new_pts_obs)

            self.data_lock.release()

            time.sleep(0.02)

            pangolin.FinishFrame()
    
class Displayer3D:
    def __init__(self, win_name):
        self.win_name = win_name
        self.scam = None
        self.handler = None
        self.dcam = None
        self.renderer = None
        self.gl_R_global = Rotation.from_euler('x', -180, degrees = True).as_matrix()

    def add_pose(self, pose):
        if self.renderer:
            self.renderer.add_pose(pose)
    def add_map_pts(self, map_pts):
        if self.renderer:
            self.renderer.add_map_pts(map_pts)
    def display(self):
        self.renderer = Renderer(self.win_name)
        self.renderer.start()



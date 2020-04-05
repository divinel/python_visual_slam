#!/usr/bin/env python3

import cv2 as cv

class Displayer:
    def __init__(self, win_name):
        self.win_name = win_name
        cv.namedWindow(self.win_name, cv.WINDOW_AUTOSIZE)
    def __del__(self):
        cv.destroyWindow(self.win_name)
    def display(self, image, delay):
        cv.imshow(self.win_name, image)
        cv.waitKey(delay)
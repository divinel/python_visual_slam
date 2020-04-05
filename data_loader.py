#!/usr/bin/env python3

import os
import cv2 as cv

def get_images_filenames(dir):
    image_dir = os.path.abspath(dir)
    images = os.listdir(image_dir)
    images.sort()
    images_filenames = [os.path.join(image_dir, image) for image in images]
    return images_filenames

class ImageLoader:
    def __init__(self, image_names):
        self.image_names = image_names
        self.cur_idx = 0
    def empty(self):
        return self.cur_idx >= len(self.image_names)
    def get_next_image(self):
        if self.empty():
            return None
        else:
            img = cv.imread(self.image_names[self.cur_idx])
            self.cur_idx += 1
            return img
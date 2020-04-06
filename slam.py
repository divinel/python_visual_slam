import data_loader, displayer, image_processing
import sys, copy
from collections import namedtuple
import cv2 as cv
import numpy as np

Frame = namedtuple('Frame', 'img kps desc')
def estimate_relative_pose(frame1, frame2):
    FLANN_INDEX_LSH = 6
    index_params = dict(algorithm = 6,
                       table_number = 6,
                       key_size = 12,
                       multi_probe_level = 1)
    search_params = dict(checks = 50)
    matcher = cv.FlannBasedMatcher(index_params, search_params)
    matches = matcher.knnMatch(frame1.desc, frame2.desc, k = 2)
    good_matches = []
    for match in matches:
        if len(match) < 2:
            continue
        first, second = match
        if first.distance < (0.75 * second.distance):
            good_matches.append(first)

    good_matches.sort(key = lambda m : m.distance)

    return good_matches


def main():
    data_folder = sys.argv[1]
    print("data folder is {path}".format(path = data_folder))
    img_files = data_loader.get_images_filenames(data_folder)
    print("{image_size} images are loaded".format(image_size = len(img_files)))
    image_loader = data_loader.ImageLoader(img_files)
    image_displayer = displayer.Displayer("data")
    # feature_extractor_shi = image_processing.FeatureExtractor(max_kps = 1000, quality_level = 0.01, min_dist = 5)
    feature_extractor = image_processing.ORBFeatureExtractor(max_kps = 1000)
    
    prev_frame = None
    cur_frame = None
    while not image_loader.empty():
        img = image_loader.get_next_image()
        disp_img = img.copy()
        gray_img = image_processing.color2gray(img)

        kps, desc = feature_extractor.extract(gray_img)

        matches = None
        cur_frame = Frame(img, kps, desc)
        if prev_frame:
            matches = estimate_relative_pose(prev_frame, cur_frame)

        for kp in kps:
            u, v = int(round(kp.pt[0])), int(round(kp.pt[1]))
            cv.circle(disp_img, (u,v), radius=2, color=(0, 255, 0), thickness=1)

        if matches is not None:
            for match in matches:
                prev_kp = prev_frame.kps[match.queryIdx].pt
                cur_kp = cur_frame.kps[match.trainIdx].pt
                cv.line(disp_img, 
                        tuple(int(round(coord)) for coord in prev_kp), 
                        tuple(int(round(coord)) for coord in cur_kp), 
                        (255, 0, 0), thickness = 1)
        if disp_img is not None:
            image_displayer.display(disp_img, 10)

        prev_frame = cur_frame

if __name__ == "__main__":
    main()
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

    match_img = cv.drawMatches(frame1.img, frame1.kps, frame2.img, frame2.kps, good_matches, 
                               None, flags = cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return match_img


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

        match_img = None
        cur_frame = Frame(img, kps, desc)
        if prev_frame:
            match_img = estimate_relative_pose(prev_frame, cur_frame)
        prev_frame = cur_frame

        for kp in kps:
            u, v = int(round(kp.pt[0])), int(round(kp.pt[1]))
            cv.circle(disp_img, (u,v), radius=2, color=(0, 255, 0), thickness=1)

        if (match_img is not None) and (disp_img is not None):
            orig_size = match_img.shape
            new_size = (orig_size[1] // 2, orig_size[0] // 2)
            match_img = cv.resize(match_img, new_size, interpolation = cv.INTER_AREA)
            disp_img = np.vstack((disp_img, match_img))
            image_displayer.display(disp_img, 10)
        elif disp_img is not None:
            image_displayer.display(disp_img, 10)

if __name__ == "__main__":
    main()
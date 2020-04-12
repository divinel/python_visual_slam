import data_loader, displayer
import image_processing, frame, relative_estimation
import sys
import cv2 as cv

def main():
    data_folder = sys.argv[1]
    print("data folder is {path}".format(path = data_folder))
    img_files = data_loader.get_images_filenames(data_folder)
    print("{image_size} images are loaded".format(image_size = len(img_files)))
    image_loader = data_loader.ImageLoader(img_files)
    image_displayer = displayer.Displayer("data")
    # feature_extractor_shi = image_processing.FeatureExtractor(max_kps = 1000, quality_level = 0.01, min_dist = 5)
    feature_extractor = image_processing.ORBFeatureExtractor(max_kps = 1000)
    relative_estimator = relative_estimation.RelativeEstimator()
    
    prev_frame = None
    cur_frame = None
    while not image_loader.empty():
        img = image_loader.get_next_image()
        disp_img = img.copy()
        gray_img = image_processing.color2gray(img)

        kps, desc = feature_extractor.extract(gray_img)
        displayer.draw_keypoints(disp_img, kps)

        # Display tracked movement between frame
        F = None
        inliers = []
        cur_frame = frame.Frame(img, kps, desc)
        if prev_frame:
            matches, matched_uvs = relative_estimator.match_frames(prev_frame, cur_frame)
            F, inliers = relative_estimation.estimate_fundamental(matches, prev_frame, cur_frame)
            print("num of matches = {}, num inliers for 8 Pts RANSAC = {}".format(len(matches), sum(inliers.ravel())))
            uv_inliers = [matched_uv for i, matched_uv in enumerate(matched_uvs) if inliers[i, 0] > 0]
            displayer.draw_relative_movements(disp_img, uv_inliers)
        prev_frame = cur_frame       
        
        image_displayer.display(disp_img, 10)


if __name__ == "__main__":
    main()
import data_loader, calib_loader, displayer, displayer3d
import image_processing, frame, relative_estimation, reconstruction
import sys
import cv2 as cv
import numpy as np

def main():
    data_folder = sys.argv[1]
    calib_file = sys.argv[2]
    print("data folder is {path}".format(path = data_folder))
    print("calibration parameters file is {}".format(calib_file))
    K, K_inv, Cam_Rts = calib_loader.load_calib_params(calib_file)
    print("camera intrinsic matrix:\n{}".format(K))
    img_files = data_loader.get_images_filenames(data_folder)
    print("{image_size} images are loaded".format(image_size = len(img_files)))
    image_loader = data_loader.ImageLoader(img_files)
    image_displayer = displayer.Displayer("cam")
    displayer_slam = displayer3d.Displayer3D("slam")
    displayer_slam.display()
    # feature_extractor_shi = image_processing.FeatureExtractor(max_kps = 1000, quality_level = 0.01, min_dist = 5)
    feature_extractor = image_processing.ORBFeatureExtractor(max_kps = 1000)
    relative_estimator = relative_estimation.RelativeEstimator()
    
    prev_frame = None
    cur_frame = None
    frames = []
    landmark_map = {}
    while not image_loader.empty():
        frame_idx, img = image_loader.get_next_image()
        disp_img = img.copy()
        gray_img = image_processing.color2gray(img)

        kps, desc = feature_extractor.extract(gray_img)
        displayer.draw_keypoints(disp_img, kps)

        # Display tracked movement between frame
        F = None
        inliers = []
        cur_frame = frame.Frame((np.identity(3), np.zeros((3,1))), kps, desc)
        if frames:
            prev_frame = frames[-1]
            matches, matched_uvs = relative_estimator.match_frames(prev_frame, cur_frame)
            F, inliers = relative_estimation.estimate_fundamental(matches, prev_frame, cur_frame)
            E = relative_estimation.get_essential(F, K)
            if len(matches) > 0:
                print("frame {} : num of matches = {}, num inliers for 8 Pts RANSAC = {}".format(frame_idx, len(matches), sum(inliers.ravel())))
                relative_R_t = relative_estimation.get_R_t(E, K, prev_frame.kps, cur_frame.kps, matches, inliers)
                cur_frame.pose = relative_estimation.get_pose(relative_R_t, prev_frame.pose)
                print(f"cur_frame R:\n {cur_frame.pose[0]}\n T:\n {cur_frame.pose[1]}")
                new_landmarks = reconstruction.reconstruct(K, prev_frame, cur_frame, matches, inliers, landmark_map)
                print(f"{len(new_landmarks)} new landmarks are reconstructed")
                uv_inliers = [matched_uv for i, matched_uv in enumerate(matched_uvs) if inliers[i]]
                displayer.draw_relative_movements(disp_img, uv_inliers)
                displayer_slam.add_pose(cur_frame.pose)
                displayer_slam.add_map_pts(np.array(new_landmarks))
        frames.append(cur_frame)
        image_displayer.display(disp_img, 50)



if __name__ == "__main__":
    main()
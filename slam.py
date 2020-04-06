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
    
    prev_frame = None
    cur_frame = None
    while not image_loader.empty():
        img = image_loader.get_next_image()
        disp_img = img.copy()
        gray_img = image_processing.color2gray(img)

        kps, desc = feature_extractor.extract(gray_img)

        # Display detected key points
        for kp in kps:
            u, v = int(round(kp.pt[0])), int(round(kp.pt[1]))
            cv.circle(disp_img, (u,v), radius=2, color=(0, 255, 0), thickness=1)

        # Dsiplay tracked movement between frame
        cur_frame = frame.Frame(img, kps, desc)
        if prev_frame:
            matches, matched_uvs = relative_estimation.estimate_relative_pose(prev_frame, cur_frame)
            for match in matched_uvs:
                prev_kp = match[0]
                cur_kp = match[1]
                cv.line(disp_img, prev_kp, cur_kp, (255, 0, 0), thickness = 1)
        
        if disp_img is not None:
            image_displayer.display(disp_img, 10)

        prev_frame = cur_frame

if __name__ == "__main__":
    main()
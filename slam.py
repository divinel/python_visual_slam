import data_loader, displayer, image_processing
import sys, copy
import cv2 as cv

def main():
    data_folder = sys.argv[1]
    print("data folder is {path}".format(path = data_folder))
    img_files = data_loader.get_images_filenames(data_folder)
    print("{image_size} images are loaded".format(image_size = len(img_files)))
    image_loader = data_loader.ImageLoader(img_files)
    image_displayer = displayer.Displayer("data")
    feature_extractor = image_processing.FeatureExtractor(quality_level = 0.01, min_dist = 5)

    while not image_loader.empty():
        img = image_loader.get_next_image()
        disp_img = img.copy()
        gray_img = image_processing.color2gray(img)
        kps = feature_extractor.extract(gray_img, 1000)
        for kp in kps:
            u,v = kp
            cv.circle(disp_img, (u,v), radius=2, color=(0, 255, 0), thickness=1)

        if disp_img is not None:
            image_displayer.display(disp_img, 10)


if __name__ == "__main__":
    main()
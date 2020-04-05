import data_loader, displayer
import sys

def main():
    data_folder = sys.argv[1]
    print("data folder is {path}".format(path = data_folder))
    img_files = data_loader.get_images_filenames(data_folder)
    print("{image_size} images are loaded".format(image_size = len(img_files)))
    image_loader = data_loader.ImageLoader(img_files)
    image_displayer = displayer.Displayer("data")
    while not image_loader.empty():
        img = image_loader.get_next_image()
        if img is not None:
            image_displayer.display(img, 100)


if __name__ == "__main__":
    main()
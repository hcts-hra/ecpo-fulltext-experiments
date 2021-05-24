import cv2
import sys, os
import numpy as np

def show(img):
    cv2.imshow('img',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def pepper(img):
    row,col = img.shape
    amount = 0.002
    num_pepper = np.ceil(amount * img.size)
    coords = [np.random.randint(0, i-1, int(num_pepper)) for i in img.shape]
    img[coords] = 0

def noisify(filename, noise_level):
    # different settings for different levels of noise intensity
    if noise_level == "light":
        open_close_kernel = np.ones((2,2),np.uint8)
        open_its = 8
        close_its = 8
    elif noise_level == "medium":
        open_close_kernel = np.ones((3,4), np.uint8)
        open_its = 6
        close_its = 5
    elif noise_level == "strong":
        open_close_kernel = np.ones((5,5),np.uint8)
        open_its = 4
        close_its = 3

    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img,(299,299))
    _,img = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    pepper(img)

    kernel = np.ones((5,5),np.uint8)
    img = cv2.erode(img,kernel)
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN,  open_close_kernel, iterations=open_its,  borderType=cv2.BORDER_REPLICATE)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, open_close_kernel, iterations=close_its, borderType=cv2.BORDER_REPLICATE)

    return img

if __name__ == "__main__":

    path_to_clean_data = sys.argv[1]
    iterations = 5

    filenames = [f for _,_,fi in os.walk(path_to_clean_data) for f in fi]
    noise_levels = ["light", "medium", "strong"]

    for i in range(iterations):
        for noise_level in noise_levels:
            for filename in filenames:
                img = noisify(os.path.join(path_to_clean_data,filename), noise_level)
                new_filename = f"{filename.split('.')[0]}-{noise_level}-{i+1}.png"
                cv2.imwrite(new_filename, img)
                print("generated", new_filename, "done")

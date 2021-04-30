import sys
import cv2
import numpy as np
# from rlsa import rlsa
# import math

def load_gray(path):
    """
    loads image as grayscale image
    """
    original_img = cv2.imread(path)
    return original_img, cv2.cvtColor(original_img, cv2.COLOR_RGB2GRAY)

def binarize(img, kernel_size=305, c=0):
    """
    uses adaptive thresholding to get a binary image
    https://docs.opencv.org/master/d7/d1b/group__imgproc__misc.html#ga72b913f352e4a1b1b397736707afcde3
    """
    return cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
           cv2.THRESH_BINARY,kernel_size,c)

def remove_big_objects(img, min_width, min_height):
    """
    draws contours wider than min_width OR taller than min_height over in white
    (which is equal to removing them as long as the image is binary with black objects on white ground)
    https://docs.opencv.org/master/d6/d6e/group__imgproc__draw.html#ga746c0625f1781f1ffc9056259103edbc
    """
    img_inv = cv2.bitwise_not(img)
    contours, hierarchy = cv2.findContours(img_inv, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for idx, contour in enumerate(contours):
        [x,y,w,h] = cv2.boundingRect(contour)
        if w > min_width or h > min_height:
            # -1 instead of idx to draw all contours
            # -1 instead of thickness to fill bounded area instead of drawing line contours
            cv2.drawContours(img, contours, idx, (255,255,255), -1)

def open_then_close(img, kernel_size, open_its, close_its):
    # morphological closing (to make text blocks connect)
    # (since black on white instead of white on black use OPEN instead of CLOSE)
    kernel = np.ones((kernel_size,kernel_size),np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel,iterations=open_its)
    cv2.imwrite('3-1.png',img)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel,iterations=close_its)
    cv2.imwrite('3-2.png',img)
    # binary_img = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel,iterations=5)
    # binary_img = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel,iterations=7)

    # binary_img = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel,iterations=8)
    # binary_img = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel,iterations=4)
    return img

def draw_boxes(contour_src, target_img, min_width, min_height):
    """
    draws boxes around contours wider than min_width OR taller than min_height
    """
    contours, hierarchy = cv2.findContours(contour_src, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        [x,y,w,h] = cv2.boundingRect(contour)
        if w > min_width or h > min_height:
            cv2.rectangle(target_img, (x,y), (x+w,y+h), (200, 0, 0), 3)


if __name__ == "__main__":
    path_to_image = sys.argv[1]
    original_img, grayscale_img = load_gray(path_to_image)
    binary_img = binarize(grayscale_img,305,10)

    # close (to connect potentially broken lines that otherwise end up
    # too short to be whitened out later)
    # kernel = np.ones((3,3),np.uint8)
    # binary_img = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel)

    # write to file
    cv2.imwrite('1.png',binary_img)

    # invert so contours have correct normals
    remove_big_objects(binary_img,100,100)

    cv2.imwrite('2.png',binary_img)

    morph_img = open_then_close(binary_img,kernel_size=3,open_its=5,close_its=3)

    # invert so contours have correct normals
    morph_img_inv = cv2.bitwise_not(morph_img)
    draw_boxes(morph_img_inv,original_img,100,100)

    cv2.imwrite('4.png',original_img)

    # x, y = img.shape
    #
    # # rlsa
    # value = max(math.ceil(x/100),math.ceil(y/100))+1 #heuristic
    # img = rlsa(img, False, True, value) #rlsa application
    # cv2.imwrite('3.png',img)

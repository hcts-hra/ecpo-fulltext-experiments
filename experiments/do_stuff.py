import sys
import cv2
from morphology import *
# from rlsa import rlsa
from pythonRLSA import rlsa
from pythonRLSA.rlsa_fast import rlsa_fast
import math


def remove_separators(img, min_size, min_aspect_ratio):
    """
    draws over in white:
     - contours wider than min_width OR taller than min_height
     - after that, remaining contours exceeding min_aspect_ratio (= very slim contours = separators)
    https://docs.opencv.org/master/d6/d6e/group__imgproc__draw.html#ga746c0625f1781f1ffc9056259103edbc
    """
    img_inv = cv2.bitwise_not(img)
    contours, hierarchy = cv2.findContours(img_inv, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for idx, contour in enumerate(contours):
        [x,y,w,h] = cv2.boundingRect(contour)
        if w > min_size or h > min_size:
            # -1 instead of idx to draw all contours
            # -1 instead of thickness to fill bounded area instead of drawing line contours
            cv2.drawContours(img, contours, idx, (255,255,255), -1)
    cv2.imwrite("1-1.png", img)
    for idx, contour in enumerate(contours):
        [x,y,w,h] = cv2.boundingRect(contour)
        if w/h > min_aspect_ratio or h/w > min_aspect_ratio:
            # -1 instead of idx to draw all contours
            # -1 instead of thickness to fill bounded area instead of drawing line contours
            cv2.drawContours(img, contours, idx, (255,255,255), -1)
    cv2.imwrite("1-2.png", img)

def extend_vertical_lines(img, min_size, min_aspect_ratio):
    """
    finds vertical contours with height/width > min_aspect_ratio
    draws pixel bridges at the top and/or bottom of these contours
        if another object is within reach of min_size
    """
    img_inv = cv2.bitwise_not(img)
    contours, hierarchy = cv2.findContours(img_inv, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        [x,y,w,h] = cv2.boundingRect(contour)

        if h/w > min_aspect_ratio:

            # top
            seed_x = round((x+(x+w))/2)-1 # horizontal middle of bounding box
            seed_y = y # top of bounding box
            y_min = max(0, seed_y-min_size) # just in case y-min_size < 0
            if 0 in img[y_min:seed_y,seed_x]: # if there's another object close enough
                # draw a pixel bridge to that object
                while img[seed_y-1,seed_x] == 255:
                    img[seed_y-1,seed_x] = 0
                    seed_y -= 1

            # bottom (repeat as above, seed_x stays the same)
            seed_y = y+h # bottom of bounding box
            y_max = min(img.shape[0]-1, seed_y+min_size)
            if 0 in img[seed_y:y_max,seed_x]:
                while img[seed_y+1,seed_x] == 255:
                    img[seed_y+1,seed_x] = 0
                    seed_y += 1

            # make sure contour itself is connected to either pixel bridge
            img[y:y+h+1,seed_x] = 0

def draw_white_box_contours(contour_src, target_img):
    """
    draws contours wider than min_width OR taller than min_height over in white
    (which is equal to removing them as long as the image is binary with black objects on white ground)
    https://docs.opencv.org/master/d6/d6e/group__imgproc__draw.html#ga746c0625f1781f1ffc9056259103edbc
    """
    # make sure findContours gets uint8
    # contour_src = cv2.bitwise_not(contour_src)
    contour_src = np.uint8(contour_src)
    contours, hierarchy = cv2.findContours(contour_src, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    unsmoothed_contours = target_img.copy()
    cv2.drawContours(unsmoothed_contours, contours, -1, (200,0,0), 3)
    cv2.imwrite('5.png',unsmoothed_contours)

    print("computing and drawing smoothed box contours ...")
    for idx, contour in enumerate(contours):
        mask = np.zeros(contour_src.shape, np.uint8)
        cv2.drawContours(mask, contours, idx, (255), -1)
        cv2.imwrite(f'temp/{idx}.png',mask)
        # smallest register = 4 characters = 90-100 px so take 80 as kernel size
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((80,80),np.uint8))
        mask = np.uint8(mask)
        cv2.imwrite(f'temp/{idx}_opened.png',mask)
        mask_contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # contours[idx] = mask_contours[0]

        if len(mask_contours) > 0: # make sure the contour didn't get eroded away
            # draw them right away
            x,y,w,h = cv2.boundingRect(mask_contours[0])
            # width of one page (= half a fold): ~1800 px
            # height of one page: ~2600 px
            # one character: ~25x25 px
            # if box size is ≤ page size and bigger than either one column of characters or one line of characters
            if w < 2000 and h < 3000 and ((w > 20 and h > 100) or (w > 100 and h > 20)):
                cv2.drawContours(target_img, mask_contours, 0, (200,0,0), 3)

    # cv2.drawContours(target_img, contours, -1, (200,0,0), 3)
    cv2.imwrite('6.png',target_img)








if __name__ == "__main__":
    path_to_image = sys.argv[1]
    original_img, grayscale_img = load_gray(path_to_image)
    print("binarizing image ...")
    binary_img = binarize(grayscale_img,35,0)
    cv2.imwrite('0.png',binary_img)

    binary_img_inv = cv2.bitwise_not(binary_img)

    print("removing all but separators ...")
    # connected 4-side rectangular separator > 300 > height of potentially connected heading
    # aspect ration of potentially connected character patches > 5 > aspect ration of separators
    remove_separators(binary_img,300,5)
    only_separators_img = cv2.bitwise_not(binary_img_inv - cv2.bitwise_not(binary_img))
    cv2.imwrite('2.png',only_separators_img)

    print("morphological closing ...")
    only_separators_img = cv2.morphologyEx(only_separators_img, cv2.MORPH_CLOSE, np.ones([3,3]))
    cv2.imwrite('2-1.png',only_separators_img)

    print("extending vertical lines ...")
    extend_vertical_lines(only_separators_img,300,10)
    cv2.imwrite('3.png',only_separators_img)

    # rlsa
    print("applying RLSA ...")
    x, y = only_separators_img.shape

    # for one-column boxes there is usually 50 px space, we need to go below that
    # so as not to close them up
    only_separators_img = rlsa_fast(only_separators_img, True, True, 40)
    cv2.imwrite('4.png',only_separators_img)

    # draw contours on original img
    print("drawing box contours ...")
    draw_white_box_contours(only_separators_img,original_img)
    # cv2.imwrite('5.png',original_img)

    print("done")

    # to crop
    # x,y,w,h = cv2.boundingRect(contour)
    # cropped_section = image[y:y+h, x:x+w]

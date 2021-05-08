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

def extend_vertical_lines(img, max_distance, min_height, min_aspect_ratio):
    """
    finds vertical contours with height/width > min_aspect_ratio
    draws pixel bridges at the top and/or bottom of these contours
        if another object is within reach of max_distance
    """
    img_inv = cv2.bitwise_not(img)
    contours, hierarchy = cv2.findContours(img_inv, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        [x,y,w,h] = cv2.boundingRect(contour)

        if h/w > min_aspect_ratio and h > min_height:

            # # to test which separators are found by this approach:
            # cv2.rectangle(img, (x,y), (x+w,y+h), (180), 2)

            # top
            seed_y = y # top of bounding box
            # find where on the top line of the contour there are actually black pixels
            black_pixels_at_top = np.where(img[seed_y,x:x+w]==0)[0]
            y_min = max(0, seed_y-max_distance) # just in case y-max_distance is off the image
            for idx in black_pixels_at_top:
                seed_x = x+idx
                # -2 below is just to make sure we're safely outside the contour
                black_pixels_over_contour = np.where(img[y_min:seed_y-2,seed_x]==0)[0]
                if len(black_pixels_over_contour) > 0: # if there's another object within max_distance
                    # draw a connecting line to that object
                    black_pixel_closest_to_contour = np.max(black_pixels_over_contour)
                    line_from = (seed_x,seed_y)
                    line_to = (seed_x,y_min+black_pixel_closest_to_contour)
                    cv2.line(img,line_from,line_to,(0),1)

            # bottom (repeat analogous to top)
            seed_y = y+h-1 # bottom of bounding box
            black_pixels_at_btm = np.where(img[seed_y,x:x+w]==0)[0]
            print(black_pixels_at_btm)
            y_max = min(img.shape[0]-1, seed_y+max_distance)
            for idx in black_pixels_at_btm:
                seed_x = x+idx
                black_pixels_under_contour = np.where(img[seed_y+2:y_max,seed_x]==0)[0]
                if len(black_pixels_under_contour) > 0:
                    black_pixel_closest_to_contour = np.min(black_pixels_under_contour)
                    line_from = (seed_x,seed_y)
                    line_to = (seed_x,seed_y+black_pixel_closest_to_contour)
                    cv2.line(img,line_from,line_to,(0),1)


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


def crop(img, contour):
    """
    crops a contour from an image
    """
    x,y,w,h = cv2.boundingRect(contour)
    return img[y:y+h, x:x+w]


def find_angle(img):
    """
    finds the angle an image containing text is rotated by
    """
    # use bigger kernel for average thresholding and then invert
    binary_img_inv = cv2.bitwise_not(binarize(grayscale_img,555,50))
    cv2.imwrite('0.png',binary_img_inv)

    # erode to remove noise and get text areas only
    # eroded = cv2.erode(binary_img_inv,np.ones((15,15),np.uint8),iterations=1)
    img = binary_img_inv
    cv2.imwrite('1.png',img)
    img = cv2.morphologyEx(img,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8),iterations=3)
    cv2.imwrite('2.png',img)
    # img = cv2.morphologyEx(img,cv2.MORPH_OPEN,np.ones((3,3),np.uint8),iterations=3)
    # cv2.imwrite('3.png',img)
    # dilated = cv2.dilate(opened,np.ones((2,2),np.uint8),iterations=1)

    contours,_ = cv2.findContours(img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    # iniatialising the empty angles list to collet the angles of each contour
    angles = []

    # obtaining the angles of each contour using a for loop
    for idx, cnt in enumerate(contours):
        # the last output of the cv2.minAreaRect() is the orientation of the contour
        [x,y,w,h] = cv2.boundingRect(cnt)
        if (w > 50 or h > 50) and (w < 50):

            rect = cv2.minAreaRect(cnt)
            box = np.int0(cv2.boxPoints(rect))
            cv2.drawContours(img, [box], 0, (100), 3)
            # cv2.rectangle(img, (x,y), (x+w,y+h), (100), 3)
            angles.append(rect[-1])

    # finding the median of the collected angles
    angles.sort()
    median_angle = np.median(angles)
    cv2.imwrite('4.png',img)

    # returning the median angle
    # for i, a in enumerate(angles):
    #     print(i,a)
    return median_angle

def deskew(no_separators_img):
    """
    finds angle using rlsa
    """
    pass

def skeletize(img):
    """
    returns skeleton of black elements in img
    """
    size = np.size(img)
    skel = np.zeros(img.shape,np.uint8)

    element = cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5))
    done = False

    while( not done):
        eroded = cv2.erode(img,element)
        temp = cv2.dilate(eroded,element)
        temp = cv2.subtract(img,temp)
        skel = cv2.bitwise_or(skel,temp)
        img = eroded.copy()

        zeros = size - cv2.countNonZero(img)
        if zeros==size:
            done = True

    cv2.imwrite('4-2.png',skel)
    return skel

def free_vertical_sepators(img):
    """
    cuts vertical separators free from horizontal ones connecting from the side
    works for ├, ┼, ┤ as well as └, ┴, ┬, ┌, etc.
    """
    # kernels to draw 4~5px-wide frame left and right from all contours
    kernel_right = np.array([[-1,-1,-1, 1, 1, 1, 0, 0, 0, 0, 0]])
    kernel_left  = np.array([[ 0, 0, 0, 0, 0, 1, 1, 1,-1,-1,-1]])
    mask = cv2.bitwise_or(cv2.filter2D(img, -1, kernel_right), cv2.filter2D(img, -1, kernel_left))
    # cv2.imshow('image',mask)
    # cv2.waitKey(0)

    # only keep those parts of the mask that are long vertical lines
    opening_kernel = np.tile(np.array([[1]]),(50,1))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, opening_kernel)
    # cv2.imshow('image',mask)
    # cv2.waitKey(0)

    # make every white pixel in mask turn into 20 vertically stacked white pixels
    extension_kernel = np.tile(np.array([[1]]),(20,1))
    mask = cv2.filter2D(mask, -1, extension_kernel)
    # cv2.imshow('image',mask)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # cv2.imwrite('2-10.png',mask)
    # cv2.imwrite('2-13.png',cv2.bitwise_or(img,mask))
    return cv2.bitwise_or(img,mask)


if __name__ == "__main__":
    path_to_image = sys.argv[1]
    original_img, grayscale_img = load_gray(path_to_image)

    print("binarizing image ...")
    binary_img = binarize(grayscale_img,21,20)
    cv2.imwrite('0.png',binary_img)
    print("cutting separators free")
    binary_img = free_vertical_sepators(binary_img)
    cv2.imwrite('0-1.png', binary_img)

    binary_img_inv = cv2.bitwise_not(binary_img)

    print("removing all but separators ...")
    # connected 4-side rectangular separator > 300 > height of potentially connected heading
    # aspect ration of potentially connected character patches < 5 < aspect ration of separators
    remove_separators(binary_img,300,5)
    # no_separators_img = binary_img
    # cv2.imwrite("2.png",no_separators_img)
    only_separators_img = cv2.bitwise_not(binary_img_inv - cv2.bitwise_not(binary_img))
    cv2.imwrite('2.png',only_separators_img)

    # print("removing some noise ...")
    # only_separators_img = cv2.morphologyEx(only_separators_img, cv2.MORPH_CLOSE, np.ones([3,3]))
    # cv2.imwrite('2-1.png',only_separators_img)


    print("cutting vertical separators free ...")
    only_separators_img = free_vertical_sepators(only_separators_img)

    print("extending vertical lines ...")
    # the shortest free-standing wave-line separator I saw was 98 px > 80 = min_height
    extend_vertical_lines(only_separators_img,max_distance=300,min_height=80,min_aspect_ratio=6)
    cv2.imwrite('3.png',only_separators_img)

    # rlsa
    print("applying RLSA ...")
    x, y = only_separators_img.shape

    # for one-column boxes there is usually 50 px space, we need to go below that
    # so as not to close them up
    only_separators_img = rlsa_fast(only_separators_img, True, True, 40)
    cv2.imwrite('4.png',only_separators_img)
    only_separators_img = np.uint8(only_separators_img)

    # print("skeletizing ...")
    # skel = skeletize(cv2.bitwise_not(only_separators_img))

    print("connecting line ends and corners ...")
    corners = cv2.goodFeaturesToTrack(only_separators_img,1000,0.2,10)
    corners = np.int0(corners)
    copy = only_separators_img.copy() # to draw points without modifying img
    for i in corners:
        xi,yi = i.ravel()
        for j in corners:
            if cv2.norm(i,j) < 60:
                xj,yj = j.ravel()
                cv2.line(only_separators_img,(xi,yi),(xj,yj),(0),3)
        cv2.circle(copy,(xi,yi),7,150,-1)


    cv2.imwrite('4-2.png', copy)
    cv2.imwrite('4-3.png', only_separators_img)

    # draw contours on original img
    print("drawing box contours ...")
    cv2.imwrite('5.png',original_img)
    draw_white_box_contours(only_separators_img,original_img)

    print("done")

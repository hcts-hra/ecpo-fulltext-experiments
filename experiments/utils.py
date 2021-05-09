import cv2
import numpy as np
import time

#################################
############ GENERAL ############

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

####################################
############ MORPHOLOGY ############

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
    cv2.imwrite('morphology_3-1.png',img)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel,iterations=close_its)
    cv2.imwrite('morphology_3-2.png',img)
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

#########################################################
############ CONNTECTING SEPARATORS APPROACH ############

def remove_separators(img, min_size, min_aspect_ratio):
    """
    draws over in white:
     - contours wider than min_width OR taller than min_height
       (min_width should be rather high since small contours will be found by next step)
     - after that, remaining contours exceeding min_aspect_ratio (= very slim contours = probably separators)
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
    cv2.imwrite("1_1.png", img)
    for idx, contour in enumerate(contours):
        [x,y,w,h] = cv2.boundingRect(contour)
        if w/h > min_aspect_ratio or h/w > min_aspect_ratio:
            # -1 instead of idx to draw all contours
            # -1 instead of thickness to fill bounded area instead of drawing line contours
            cv2.drawContours(img, contours, idx, (255,255,255), -1)
    cv2.imwrite("1_2.png", img)

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

def extend_separators(img, mode, max_distance, min_length, min_aspect_ratio):
    """
    mode="vertical":
        finds vertical contours with height/width > min_aspect_ratio and height > min_length
    mode="horizontal":
        and horizontal contours with width/height > min_aspect_ratio and width > min_length
    draws pixel bridges at the top and/or bottom [mode="vertical"] /
    left and/or right side [mode="horizontal"] of these contours if another object is within reach of max_distance
    """
    img_inv = cv2.bitwise_not(img)
    contours, hierarchy = cv2.findContours(img_inv, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    img_copy = img.copy()


    for contour in contours:
        [x,y,w,h] = cv2.boundingRect(contour)

        # leave a 100 px margin because contours at the edge of the image behave weirdly
        if mode=="vertical" and y > 10 and y+h < img.shape[0]-10:

            if h/w > min_aspect_ratio and h > min_length:

                # # to test which contours are found:
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
                y_max = min(img.shape[0]-1, seed_y+max_distance)
                for idx in black_pixels_at_btm:
                    seed_x = x+idx
                    black_pixels_under_contour = np.where(img[seed_y+2:y_max,seed_x]==0)[0]
                    if len(black_pixels_under_contour) > 0:
                        black_pixel_closest_to_contour = np.min(black_pixels_under_contour)
                        line_from = (seed_x,seed_y)
                        line_to = (seed_x,seed_y+black_pixel_closest_to_contour)
                        cv2.line(img,line_from,line_to,(0),1)

        elif mode=="horizontal" and x > 10 and x+w < img.shape[1]-10: # analogous to the vertical case

            if w/h > min_aspect_ratio and w > min_length:

                # to test which contours are found:
                cv2.rectangle(img_copy, (x,y), (x+w,y+h), (180), 2)

                # left
                seed_x = x # left end of bounding box
                black_pixels_at_left = np.where(img[y:y+h,seed_x]==0)[0]
                x_min = max(0, seed_x-max_distance)
                for idx in black_pixels_at_left:
                    seed_y = y+idx
                    black_pixels_left_of_contour = np.where(img[seed_y,x_min:seed_x-2]==0)[0]
                    if len(black_pixels_left_of_contour) > 0:
                        black_pixel_closest_to_contour = np.max(black_pixels_left_of_contour)
                        line_from = (seed_x,seed_y)
                        line_to = (x_min+black_pixel_closest_to_contour,seed_y)
                        cv2.line(img,line_from,line_to,(0),1)

                # right
                seed_x = x+w-1 # right end of bounding box
                black_pixels_at_right = np.where(img[y:y+h,seed_x]==0)[0]
                x_max = min(img.shape[1]-1, seed_x+max_distance)
                for idx in black_pixels_at_right:
                    seed_y = y+idx
                    black_pixels_right_of_contour = np.where(img[seed_y,seed_x+2:x_max]==0)[0]
                    if len(black_pixels_right_of_contour) > 0:
                        black_pixel_closest_to_contour = np.min(black_pixels_right_of_contour)
                        line_from = (seed_x,seed_y)
                        line_to = (seed_x+black_pixel_closest_to_contour,seed_y)
                        cv2.line(img,line_from,line_to,(0),1)


    cv2.imwrite('3-01.png', img_copy)

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
    cv2.imwrite('8.png',unsmoothed_contours)

    print("computing and drawing smoothed box contours ...")
    for idx, contour in enumerate(contours):
        mask = np.zeros(contour_src.shape, np.uint8)
        cv2.drawContours(mask, contours, idx, (255), -1)
        # cv2.imwrite(f'temp/{idx}.png',mask)
        # smallest register = 4 characters = 90-100 px so take 80 as kernel size
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((80,80),np.uint8))
        mask = np.uint8(mask)
        # cv2.imwrite(f'temp/{idx}_opened.png',mask)
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

################################
############ UNUSED ############

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
    cv2.imwrite('angle_0.png',binary_img_inv)

    # erode to remove noise and get text areas only
    # eroded = cv2.erode(binary_img_inv,np.ones((15,15),np.uint8),iterations=1)
    img = binary_img_inv
    cv2.imwrite('angle_1.png',img)
    img = cv2.morphologyEx(img,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8),iterations=3)
    cv2.imwrite('angle_2.png',img)
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
    cv2.imwrite('angle_4.png',img)

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

    cv2.imwrite('skeletized.png',skel)
    return skel

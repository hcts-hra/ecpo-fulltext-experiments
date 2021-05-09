import sys
import cv2
from morphology import *
from utils import *
from pythonRLSA import rlsa
from pythonRLSA.rlsa_fast import rlsa_fast

if __name__ == "__main__":
    path_to_image = sys.argv[1]
    original_img, grayscale_img = load_gray(path_to_image)

    print("binarizing image ...")
    img = binarize(grayscale_img,21,20)
    cv2.imwrite('0.png',img)

    print("generating mask to deal with horizontal line noise on scan ...")
    mask = cv2.bitwise_not(binarize(grayscale_img,355,0))
    cv2.imwrite('mask0.png',mask)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((6,3)))
    cv2.imwrite('mask1.png',mask)
    mask = cv2.erode(mask,np.ones((45,1)))
    cv2.imwrite('mask2.png',mask)

    print("morphological opening to connect separator fragments ...")
    # during erosion (black contours grow) extend contours 2 px
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, np.ones((2,2)),iterations=2)
    cv2.imwrite('0_1.png',img)

    print("extracting separators ...")
    img_inv = cv2.bitwise_not(img)
    # connected 4-side rectangular separator > 300 > height of potentially connected heading
    # aspect ration of potentially connected character patches < 5 < aspect ration of separators
    remove_separators(img,300,5) # writes 1-1.png and 1-2.png
    # no_separators_img = binary_img # needed for RLSA-based de-skewing
    img = cv2.bitwise_not(img_inv - cv2.bitwise_not(img))
    cv2.imwrite('2.png',img)

    print("cutting vertical separators free  ...")
    img = free_vertical_sepators(img) # doesn't work too well for wavy separators

    vertical = img.copy()
    cv2.imwrite('vertical.png',vertical)
    horizontal = img.copy()
    print("vertical separator extension ...")
    # the shortest free-standing wave-line separator I saw was 98 px > 80 = min_length
    extend_separators(vertical, mode="vertical", max_distance=300, min_length=80, min_aspect_ratio=6)
    cv2.imwrite('3.png',vertical)

    print("horizontal separator extension ...")
    # shortest gap caused by a heading between a horizontal separator to the left
    # and a vertical one to the right is ~85 px < 70 = max_distance
    # this is to connect remainders of very thin separators => use lower min_length and min_aspect_ratio
    extend_separators(horizontal, mode="horizontal", max_distance=70, min_length=15, min_aspect_ratio=4)
    img = cv2.bitwise_and(vertical,horizontal)
    cv2.imwrite('3_1.png', img)


    line_width = 5
    line_end_length = 50
    line_extension = 100
    margin = 20
    bottom_line_end_kernel = np.append(np.tile(np.array([-10]*margin+[1]*line_width+[-10]*margin),(10,1)), np.tile(np.array([-10]*margin+[-10]*line_width+[-10]*margin),(margin,1)), 0)
    line_ends = cv2.dilate(cv2.filter2D(cv2.erode(cv2.bitwise_not(img),np.ones((line_end_length,1))), -1, bottom_line_end_kernel),np.ones((line_extension,1)))
    img = cv2.bitwise_and(cv2.bitwise_not(line_ends),img)
    cv2.imwrite('3_2.png',img)


    # rlsa
    print("applying RLSA ...")
    x, y = img.shape

    # for one-column boxes there is usually 50 px space, we need to go below that
    # so as not to close them up
    img = rlsa_fast(img, True, True, 40)
    cv2.imwrite('4.png',img)
    img = np.uint8(img)

    # print("skeletizing ...")
    # skel = skeletize(cv2.bitwise_not(img))


    # create kernel like this:


    print("connecting line ends and corners ...")
    corners = cv2.goodFeaturesToTrack(img,1000,0.2,10)
    corners = np.int0(corners)
    copy = img.copy() # to draw points without modifying img
    for i in corners:
        xi,yi = i.ravel()
        for j in corners:
            if cv2.norm(i,j) < 60:
                xj,yj = j.ravel()
                cv2.line(img,(xi,yi),(xj,yj),(0),3)
        cv2.circle(copy,(xi,yi),7,150,-1)

    cv2.imwrite('4_1.png', copy)
    cv2.imwrite('4_2.png', img)

    # draw contours on original img
    print("drawing box contours ...")
    draw_white_box_contours(img,original_img)
    cv2.imwrite('9.png',original_img)


    print("done")

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
    binary_img = binarize(grayscale_img,21,20)
    cv2.imwrite('0.png',binary_img)

    print("cutting separators free")
    binary_img = free_vertical_sepators(binary_img)
    cv2.imwrite('0-1.png', binary_img)

    binary_img_inv = cv2.bitwise_not(binary_img)

    print("extracting separators ...")
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

    print("extending horizontal separators ...")
    # shortest gap caused by a heading between a horizontal separator to the left
    # and a vertical one to the right is ~85 px < 65 = max_distance
    # this is to connect remainders of very thin separators => use lower min_length and min_aspect_ratio
    extend_lines(binary_img, mode="horizontal", max_distance=65, min_length=15, min_aspect_ratio=4)
    cv2.imwrite('2-1.png', binary_img)

    print("cutting vertical separators free ...")
    only_separators_img = free_vertical_sepators(only_separators_img)

    print("extending vertical separators ...")
    # the shortest free-standing wave-line separator I saw was 98 px > 80 = min_length
    extend_lines(only_separators_img, mode="vertical", max_distance=300, min_length=80, min_aspect_ratio=6)
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

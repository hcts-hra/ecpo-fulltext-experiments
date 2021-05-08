import sys
from utils import *

if __name__ == "__main__":
    path_to_image = sys.argv[1]
    original_img, grayscale_img = load_gray(path_to_image)
    binary_img = binarize(grayscale_img,305,10)
    cv2.imwrite('morphology_1_binarized.png',binary_img)

    remove_big_objects(binary_img,100,100)
    cv2.imwrite('morphology_2.png_without_big_objects',binary_img)

    morph_img = open_then_close(binary_img,kernel_size=3,open_its=5,close_its=0)
    # the open_then_close function will imwrite 3-1.png and 3-2.png

    morph_img_inv = cv2.bitwise_not(morph_img) # invert so contours have correct normals
    draw_boxes(morph_img_inv,original_img,200,200)
    cv2.imwrite('morphology_4_result.png',original_img)

import os, sys
import cv2
import numpy as np
from pythonRLSA import rlsa
from pythonRLSA.rlsa_fast import rlsa_fast
from scipy.ndimage import interpolation as inter
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import imutils

def correct_skew(img, delta=0.5, limit=2):
    def determine_score(arr, angle):
        data = inter.rotate(arr, angle, reshape=False, order=0)
        proj_hor = np.sum(data, axis=0)
        proj_ver = np.sum(data, axis=1)
        score_hor = np.sum((proj_hor[1:] - proj_hor[:-1]) ** 2)
        score_ver = np.sum((proj_ver[1:] - proj_ver[:-1]) ** 2)
        return score_ver+score_hor

    binary,_ = better_binarize(img,c=10)

    scores = []
    angles = np.arange(-limit, limit + delta, delta)
    for angle in angles:
        score = determine_score(binary, angle)
        scores.append(score)

    best_angle = angles[scores.index(max(scores))]

    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, best_angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, \
              borderMode=cv2.BORDER_REPLICATE)

    # print(best_angle)
    return rotated

# rectangle methods
def union(a,b):
    x = min(a[0], b[0])
    y = min(a[1], b[1])
    w = max(a[0]+a[2], b[0]+b[2]) - x
    h = max(a[1]+a[3], b[1]+b[3]) - y
    return (x, y, w, h)

def check_intersection(a,b):
    x = max(a[0], b[0])
    y = max(a[1], b[1])
    w = min(a[0]+a[2], b[0]+b[2]) - x
    h = min(a[1]+a[3], b[1]+b[3]) - y
    if w<0 or h<0:
        return False
    else:
        return True

def better_binarize(original_img,c=0,leave_partly_gray=False):
    """
    performs adaptive thresholding but leaves the background white even for
    """
    img = original_img.copy()
    kernel_size = 7 # to compute averages of local surrounding area
    # supposing that we have more background pixels than content pixels,
    # all pixels above threshold will become white later
    thresh = np.median(img) # TODO: don't use global img median but array of local median thresholds
    averages = cv2.blur(img,(kernel_size,kernel_size))

    minimum_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25,25))
    minima = cv2.erode(img,minimum_kernel)

    averages[averages>thresh] = 0 # for the next two lines
    img[img>averages-c] = 255 # since averages > thresh (background pixels) are already 0 they all become white
    if not leave_partly_gray:
        img[img<=averages-c] = 0

    # U+4E00 bis U+9FA5

    return img,thresh

def draw_projection_profiles(proj_ver,proj_hor,img_idx):
    """
    draws and saves projection profiles
    """

    # horizontal projections
    m = np.max(proj_ver)
    w = img.shape[0] // 2
    result = np.zeros((w,len(proj_ver)))
    # Draw a line for each row
    for row in range(img.shape[1]):
        cv2.line(result, (row,0), (row,proj_ver[row]*w // m), (255), 1)
    cv2.imwrite(f'horizontal_projection-{img_idx}.jpg', result)

    # vertical projection
    m = np.max(proj_ver)
    w = img.shape[0] // 2
    result = np.zeros((w,len(proj_ver)))
    # Draw a line for each row
    for row in range(img.shape[1]):
        cv2.line(result, (row,0), (row,proj_ver[row]*w // m), (255), 1)
    cv2.imwrite(f'horizontal_projection-{img_idx}.jpg', result)



def crop_chars(image,img_idx,write_processed_images=False):
    """
    uses horizontal and vertical projections to crop characters
    """
    # derotate original
    image = correct_skew(image)
    # perform binarization and projection operations on copy
    img = image.copy()
    # two different binarization methods
    # img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
    #        cv2.THRESH_BINARY,125,0)
    bin_img,_ = better_binarize(img,c=0)
    if write_processed_images:
        cv2.imwrite(f"binary-{img_idx}.jpg",bin_img)

    # black rectangle at the outside to find peak outside of outmost character
    cv2.rectangle(bin_img,(0,0),(bin_img.shape[1],bin_img.shape[0]),(0),2)
    proj_ver = np.sum(bin_img, axis=0)
    proj_hor = np.sum(bin_img, axis=1)

    peaks_ver, _ = find_peaks(proj_ver, distance=22) # average hor. char dist.: 31
    peaks_hor, _ = find_peaks(proj_hor, distance=20) # average vert. char dist.: 24

    # open annotations
    with open(f"{img_idx}.txt") as f:
        columns = [line.strip().strip("<lb/>").replace("&gaiji;","e") for line in f.readlines()]
        assert len(columns) == len(peaks_ver)-1, f"at img_idx {img_idx} there are {len(columns)} lines in the annotation but {len(peaks_ver)-1} columns in the image"

    img_annotation_dict = {char:[] for column in columns for char in column}
    for i in range(len(peaks_ver)-1): # iterate over columns
        # crop a column and draw its left border
        column = bin_img[0:bin_img.shape[0],peaks_ver[i]:peaks_ver[i+1]]

        # draw column border for wiki image

        column = rlsa_fast(column, True, False, 10)

        # do the same as above vertically per column
        proj_hor_local = np.sum(column, axis=1)
        proj_hor_local = proj_hor_local / np.max(proj_hor_local)
        peaks_hor_local, _ = find_peaks(proj_hor_local, prominence=0.7, distance=15) # average vert. char dist.: 24

        peaks_hor_adjusted = peaks_hor.copy()
        thresh = 6
        for peak in peaks_hor_local:
            # adjust peaks_hor_adjusted to peak wherever peak is within thresh of any element
            peaks_hor_adjusted = np.where(np.abs(peaks_hor_adjusted-peak)<thresh,peak,peaks_hor_adjusted)

        if write_processed_images:
            cv2.line(image,(peaks_ver[i],0),(peaks_ver[i],image.shape[0]),(50),2)
            for j in range(len(peaks_hor)-1): # iterate over characters within one column
                cv2.line(image,(peaks_ver[i],peaks_hor_adjusted[j]),(peaks_ver[i+1],peaks_hor_adjusted[j]),(50),2)

        # preprocess the image before cropping because single char imgs are hard to preprocess without img context
        gray_img,bin_thresh = better_binarize(image,c=-50,leave_partly_gray=True)
        gray_col = image[0:image.shape[0],peaks_ver[i]:peaks_ver[i+1]]

        for j in range(len(peaks_hor)-1):
            # crop actual character
            char_image = gray_col[peaks_hor_adjusted[j]:peaks_hor_adjusted[j+1],0:gray_col.shape[1]]
            # to re-scale array from a range between min and max to (0,1): (array-min)/(max-min)
            # we want to stretch the darkest pixel to 0 and the brightest (= bin_thresh) to 255
            darkest_pixel = char_image.min()
            char_image = np.where(
                char_image<bin_thresh, # all non-background pixels
                np.uint8((char_image-darkest_pixel)/(bin_thresh-darkest_pixel)*255),
                255
            )
            # add white (255) padding around char_image to make it squared
            img_size = max(char_image.shape[0],char_image.shape[1])
            vertical_padding = img_size - char_image.shape[0]
            if vertical_padding%2: # odd number
                top_padding = vertical_padding // 2 + 1
                btm_padding = vertical_padding // 2
            else: # even number
                top_padding = btm_padding = vertical_padding // 2
            horizontal_padding = img_size - char_image.shape[1]
            if horizontal_padding%2:
                left_padding = horizontal_padding // 2 + 1
                right_padding = horizontal_padding // 2
            else:
                left_padding = right_padding = horizontal_padding // 2
            char_image = cv2.copyMakeBorder(
                char_image,
                top_padding,
                btm_padding,
                left_padding,
                right_padding,
                cv2.BORDER_CONSTANT,
                value=255
            )
            # # compute centroid (mean of non-zero indices in the inverted binary)
            # _,b = cv2.threshold(char_image,bin_thresh,255,cv2.THRESH_BINARY)
            # b = cv2.bitwise_not(b)
            # centroid = tuple(np.uint8(nz.mean()) for nz in b.nonzero())
            # cv2.circle(b, centroid, 3,(100),1)
            char_image = cv2.resize(char_image,(50,50),interpolation=cv2.INTER_CUBIC)

            # append  idx [-i-1] to read columns from right to left
            if len(columns[-i-1]) > j:
                char_annotation = columns[-i-1][j]
                img_annotation_dict[char_annotation].append((char_image,len(columns)-i,j+1))

    # "e" has been inserted as a placeholder where a character is missing or illegible (&gaiji;)
    img_annotation_dict.pop("e", None)
    for char,img_list in img_annotation_dict.items():
        for char_image,col_idx,row_idx in img_list:
            if np.isnan(char_image).any():
                print(f"char img {char}-{img_idx}-{col_idx}-{row_idx} contains nans!")
            cv2.imwrite(os.path.join("char_images",f"{char}-{img_idx}-{col_idx}-{row_idx}.png"),char_image)

    if write_processed_images:
        cv2.imwrite(f"hybrid-{img_idx}.jpg",image)


if __name__ == "__main__":

    if len(sys.argv) == 2: # e.g. python3 extract_character_images.py 215
        img = cv2.imread(f"{sys.argv[1]}.png", cv2.IMREAD_GRAYSCALE)
        crop_chars(img,sys.argv[1],write_processed_images=True)
    else: # all png files
        for file in os.listdir("."):
            if file.endswith(".png"):
                print("processing", file, end="\r")
                img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
                crop_chars(img,file[:3],write_processed_images=False)
        print(" "*30, end="\r")
        print("done")

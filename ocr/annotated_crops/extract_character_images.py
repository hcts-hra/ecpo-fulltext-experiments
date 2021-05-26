import os
import cv2
import numpy as np
from pythonRLSA import rlsa
from pythonRLSA.rlsa_fast import rlsa_fast
from scipy.ndimage import interpolation as inter
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

def correct_skew(img, delta=0.5, limit=2):
    def determine_score(arr, angle):
        data = inter.rotate(arr, angle, reshape=False, order=0)
        proj_hor = np.sum(data, axis=0)
        proj_ver = np.sum(data, axis=1)
        score_hor = np.sum((proj_hor[1:] - proj_hor[:-1]) ** 2)
        score_ver = np.sum((proj_ver[1:] - proj_ver[:-1]) ** 2)
        return score_ver+score_hor

    binary = better_binarize(img,c=-10)

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

def better_binarize(original_img,c=0):
    """
    performs adaptive thresholding but leaves the background white
    """
    img = original_img.copy()
    kernel_size = 7 # to compute averages of local surrounding area
    # supposing that we have more background pixels than content pixels,
    # all pixels above threshold will become white later
    thresh = np.median(img)
    averages = cv2.blur(img,(kernel_size,kernel_size))

    averages[averages>thresh] = 0 # for the line after the next
    img[img>averages+c] = 255
    img[img<=averages+c] = 0 # since averages > thresh (background pixels) are 0 they are not set to 0 here

    # U+4E00 bis U+9FA5

    return img

def draw_projection_profiles(proj_ver,proj_hor,idx):
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
    cv2.imwrite(f'horizontal_projection-{idx}.jpg', result)

    # vertical projection
    m = np.max(proj_ver)
    w = img.shape[0] // 2
    result = np.zeros((w,len(proj_ver)))
    # Draw a line for each row
    for row in range(img.shape[1]):
        cv2.line(result, (row,0), (row,proj_ver[row]*w // m), (255), 1)
    cv2.imwrite(f'horizontal_projection-{idx}.jpg', result)



def crop_chars(image,idx):
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
    img = better_binarize(img,c=0)
    # cv2.imwrite(f"binary-{idx}.jpg",img)

    # black rectangle at the outside to find peak outside of outmost character
    cv2.rectangle(img,(0,0),(img.shape[1],img.shape[0]),(0),2)
    proj_ver = np.sum(img, axis=0)
    proj_hor = np.sum(img, axis=1)

    peaks_ver, _ = find_peaks(proj_ver, distance=22) # average hor. char dist.: 31
    peaks_hor, _ = find_peaks(proj_hor, distance=21) # average vert. char dist.: 24

    # open annotations
    with open(f"{idx}.txt") as f:
        columns = [line.strip().strip("<lb/>").replace("&gaiji;","e") for line in f.readlines()]
        assert len(columns) == len(peaks_ver)-1, f"wrong number of columns at idx {idx}"

    img_annotation_dict = {char:[] for column in columns for char in column}
    for i in range(len(peaks_ver)-1): # iterate over columns
        # crop a column and draw its left border
        column = img[0:img.shape[0],peaks_ver[i]:peaks_ver[i+1]]

        # draw column border for wiki image
        # cv2.line(image,(peaks_ver[i],0),(peaks_ver[i],img.shape[0]),(50),2)

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

        # for wiki image
        # for j in range(len(peaks_hor)-1): # iterate over characters within one column
            # cv2.line(image,(peaks_ver[i],peaks_hor_adjusted[j]),(peaks_ver[i+1],peaks_hor_adjusted[j]),(50),2)

        # this time don't use binary but original image for cropping
        # NOTE: in the future, preprocess the image before cropping because single char imgs are hard to preprocess
        orig_column = image[0:image.shape[0],peaks_ver[i]:peaks_ver[i+1]]
        # cv2.imwrite(f"column-{i}.png",orig_column)
        for j in range(len(peaks_hor)-1):
            # crop actual character
            char_image = orig_column[peaks_hor_adjusted[j]:peaks_hor_adjusted[j+1],0:orig_column.shape[1]]
            # print(char_image.shape)

            # add padding around char_image
            img_size = 50
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
            # print(char_image)

            # not good: resizing leads to distortion (duh), better work with center of mass?
            # char_image = cv2.resize(char_image,(30,30),interpolation = cv2.INTER_CUBIC)
            if len(columns[-i-1]) > j:
                char_annotation = columns[-i-1][j]
            img_annotation_dict[char_annotation].append((char_image,len(columns)-i,j+1))

    img_annotation_dict.pop("e", None) # "e" is inserted as a placeholder where a new paragraph is intented
    for char,img_list in img_annotation_dict.items():
        for char_image,i,j in img_list:
            cv2.imwrite(os.path.join("char_images",f"{char}-{idx}-{i}-{j}.png"),char_image)

    # save image for wiki
    # cv2.imwrite(f"hybrid-{idx}.jpg",image)


if __name__ == "__main__":

    for file in os.listdir("."):
        if file.endswith(".png"):

            img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
            crop_chars(img,file[:2])

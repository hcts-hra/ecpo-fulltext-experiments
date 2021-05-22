import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('../../HRCenterNet/images/crop_4.jpg',0)

kernel_size = 7 # to compute averages
threshold = np.median(img) # all pixels above this will become white later
print(threshold)

averages = cv2.blur(img,(kernel_size,kernel_size))
# plt.imshow(averages, cmap='gray')
# plt.show()

averages[averages>threshold] = 0
img[img>averages] = 255
img[img<=averages] = 0
# img[img>180] = 255

# plt.imshow(img, cmap='gray')
# plt.show()
#
# img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,5,2)
# _,img = cv2.threshold(img,0,255,cv2.THRESH_OTSU)

# plt.imshow(img, cmap='gray')
# plt.show()

cv2.imwrite('binarized.png', img)

# U+4E00 bis U+9FA5

import cv2
import os, sys
import numpy as np

def ceil_divide(n, d):
    return -(n // -d) # ceiling equivalent of // operator (== np.floor_divide)

# manually enter here, then do "python show_train_samples.py" and open samples_*.png
path = "train_data"
char = sys.argv[1]
img_size = 40
chars_per_row = 1

files = [x for x in os.listdir(path) if x.startswith(sys.argv[1])]
final_img = np.zeros((ceil_divide(len(files),chars_per_row)*img_size,chars_per_row*img_size))

for idx,file in enumerate(files):
    img = cv2.imread(os.path.join(path,file), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img,((32,32)))
    cv2.rectangle(img,(0,0),(img.shape[1]-1,img.shape[0]-1),(180),1)
    vertical_padding = img_size - img.shape[0]
    if vertical_padding%2: # odd number
        top_padding = vertical_padding // 2 + 1
        btm_padding = vertical_padding // 2
    else: # even number
        top_padding = btm_padding = vertical_padding // 2
    horizontal_padding = img_size - img.shape[1]
    if horizontal_padding%2:
        left_padding = horizontal_padding // 2 + 1
        right_padding = horizontal_padding // 2
    else:
        left_padding = right_padding = horizontal_padding // 2
    try:
        img = cv2.copyMakeBorder(
            img,
            top_padding,
            btm_padding,
            left_padding,
            right_padding,
            cv2.BORDER_CONSTANT,
            value=255
        )
    except cv2.error:
        print(f"leaving out {file} since its shape {img.shape} is bigger than {img_size}")
        continue
    i = idx//chars_per_row
    j = idx%chars_per_row
    final_img[i*img_size:(i+1)*img_size,j*img_size:(j+1)*img_size] = img

# cv2.imwrite(f"samples_{char}.png",final_img)
cv2.imwrite("xx.png",final_img)

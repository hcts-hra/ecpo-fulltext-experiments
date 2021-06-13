import cv2
import os
import numpy as np

# manually enter here, then do "python show_train_samples.py" and open samples_*.png
char = "當"
img_size = 40
chars_per_row = 13

files = [x for x in os.listdir("train_data") if x.startswith(char)]
final_img = np.zeros(((len(files)//chars_per_row)*img_size,chars_per_row*img_size))

for idx,file in enumerate(files):
    img = cv2.imread(os.path.join("train_data",file), cv2.IMREAD_GRAYSCALE)
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

cv2.imwrite(f"samples_{char}.png",final_img)

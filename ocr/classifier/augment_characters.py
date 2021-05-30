import numpy as np
import cv2
import sys

def show(name,img):
    cv2.imshow(name,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def pepper(img,color,amount,padding):
    num_pepper = np.ceil(amount * img.size)
    coords = [np.random.randint(padding, i-1, int(num_pepper)) for i in np.subtract(img.shape,(padding,padding))]
    img[coords] = color

# def increase_brightness(img,increment):
#     return np.where(img<=255-increment,img+increment,255)

def add_patches(img,number,size,brightness,padding):
    coords = [np.random.randint(padding, i-1, int(number)) for i in np.subtract(img.shape,(padding,padding))]
    patches = np.zeros(img.shape,dtype="uint8")
    for x,y in zip(*coords):
        patches = cv2.circle(patches,(x,y),size,brightness,-1) # -1 to fill instead of drawing outline
    patches = cv2.blur(patches,(10,10))
    # show("img",patches)
    # add patches to img wherever this wouldn't cause integer overflow
    return np.where(img<=255-patches,img+patches,255)

def augment(char_img, noise_level, brightness=0, open_its=2):

    img = cv2.resize(char_img, (40,40))
    padding = 10

    # padding but random.
    # later morphology will cause pixels to shift to btm-right
    # so create more padding at btm and right (using padding//2 instead of padding)
    top_padding = np.random.randint(0,padding//2)
    btm_padding = padding-top_padding
    left_padding = np.random.randint(0,padding//2)
    right_padding = padding-left_padding

    img = cv2.copyMakeBorder(
        img,
        top_padding,
        btm_padding,
        left_padding,
        right_padding,
        cv2.BORDER_CONSTANT,
        value=255
    )

    # pepper
    if noise_level == "light":
        pepper(img, 30,0.005,padding=padding)
        pepper(img, 90,0.008,padding=padding)
        pepper(img,200,0.020,padding=padding)
    elif noise_level == "medium":
        pepper(img, 30,0.01,padding=padding)
        pepper(img, 90,0.02,padding=padding)
        pepper(img,180,0.03,padding=padding) # less padding for more noise
    elif noise_level == "strong":
        pepper(img, 0,0.01,padding=padding)
        pepper(img, 70,0.03,padding=padding)
        pepper(img,150,0.05,padding=padding) # less padding for more noise

    # salt
    pepper(img,255,0.05,padding=10)
    img = cv2.erode(img,np.ones((2,2)))

    # morphology
    open_close_kernel = np.ones((2,2),np.uint8)
    close_its = 1
    if open_its != 0:
        img = cv2.morphologyEx(img, cv2.MORPH_OPEN,  open_close_kernel, iterations=open_its,  borderType=cv2.BORDER_REPLICATE)
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, open_close_kernel, iterations=close_its, borderType=cv2.BORDER_REPLICATE)

    # add patches
    img = add_patches(img,6,7,brightness,padding)
    # blur
    img = cv2.blur(img,(3,3))
    show(f"{noise_level}-{brightness}-{open_its}", img)

if __name__ == "__main__":
    img = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)

    # 3 x 3 x 2 = 24 imgs
    for noise_level in ["light","medium","strong"]: # 3
        for brightness in [0,50,100]: # 3
            for open_its in [0,1]: # 2
                for _ in range(2):
                    augment(img,noise_level,brightness,open_its)

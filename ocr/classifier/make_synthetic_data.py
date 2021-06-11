import numpy as np
import cv2
import sys, os
import argparse

from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter

def elastic_transform(image, alpha, sigma, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
       source:
       https://gist.github.com/chsasank/4d8f68caf01f041a6453e67fb30f8f5a
    """
    assert len(image.shape)==2

    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    indices = np.reshape(x+dx, (-1, 1)), np.reshape(y+dy, (-1, 1))

    return map_coordinates(image, indices, order=1, mode="constant", cval=255).reshape(shape)

def show(name,img):
    cv2.imshow(name,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def pepper(img,color,amount,padding=0):
    num_pepper = np.ceil(amount * img.size)
    coords = [np.random.randint(padding, i-1, int(num_pepper)) for i in np.subtract(img.shape,(padding,padding))]
    img[tuple(coords)] = color

def add_patches(img,number,size,brightness,padding):
    coords = [np.random.randint(padding, i-1, int(number)) for i in np.subtract(img.shape,(padding,padding))]
    patches = np.zeros(img.shape,dtype="uint8")
    for x,y in zip(*coords):
        patches = cv2.rectangle(patches,(x,y),(x+size,y+size),brightness,-1) # -1 to fill instead of drawing outline
    patches = cv2.blur(patches,(15,15))
    # show("img",patches)
    # add patches to img wherever this wouldn't cause integer overflow
    return np.where(img<=255-patches,img+patches,255)

def augment(char_img):

    # resizing
    content_dim = (100,100)
    padding = np.random.randint(12,18)
    img = char_img.copy()
    img = cv2.resize(img, content_dim)

    # padding but random = random translation
    # total_padding = padding*2
    # top_padding = np.random.randint(0,total_padding-padding//2)
    # btm_padding = total_padding-top_padding
    # left_padding = np.random.randint(0,total_padding-padding//2)
    # right_padding = total_padding-left_padding
    img = cv2.copyMakeBorder(
        img,
        padding,
        padding,
        padding,
        padding,
        cv2.BORDER_CONSTANT,
        value=255
    )

    # pepper + morphology = nice
    morph_its = np.random.randint(1,4)
    pepper(img,0,0.01*morph_its)
    open_close_kernel = np.ones((2,2),np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN,  open_close_kernel, iterations=morph_its, borderType=cv2.BORDER_REPLICATE)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, open_close_kernel, iterations=morph_its, borderType=cv2.BORDER_REPLICATE)

    img = cv2.erode(img,np.ones((3,3)))
    # show("img",img)
    vertical_lines = cv2.dilate(img.copy(),np.ones((7,1)))
    img = cv2.erode(img,np.ones((3,3)))
    img = cv2.blur(img,(10,10))
    # b = brightness
    # img = np.where(img<=255-b,img+b,255)
    img = add_patches(img,10,12,150,padding)
    img = cv2.bitwise_and(img,vertical_lines)

    # add patches of increased brightness
    # img = add_patches(img,12,15,brightness,padding)

    # blur
    blur_kernel_size = np.random.randint(8,15)
    img = cv2.blur(img,(blur_kernel_size,blur_kernel_size))
    # to re-scale array from a range between min and max to (0,1): (array-min)/(max-min)
    img = np.uint8((img-img.min()) / (img.max()-img.min()) * 255)

    # distort
    alpha = 30 # bigger alpha => distort more
    sigma = 8 # bigger sigma => bigger radius of gausian distortions
    img = elastic_transform(img,alpha,sigma)

    # show("img",img)
    return img

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_dir", required=True)
    parser.add_argument("-o", "--output_dir")
    args = parser.parse_args()

    count = 0
    file_list = os.listdir(args.input_dir)
    for file_name in file_list:
        for i in range(2):
            img = cv2.imread(os.path.join(args.input_dir,file_name), cv2.IMREAD_GRAYSCALE)
            brightness = 50
            morph_its = np.random.randint(1,3) # 1 o. 2
            aug_img = augment(img,brightness,morph_its)
            if args.output_dir:
                cv2.imwrite(os.path.join(args.output_dir,f"{file_name[:4]}-{i}.png"),aug_img)
                count += 1
                print(f"wrote {count} images to {args.output_dir}", end="\r")
            else:
                show(f"{brightness}-{morph_its}",aug_img)
                break

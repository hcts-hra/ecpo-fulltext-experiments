import argparse
import os
import random

def extract_indices(char_img_filename):
    """
    transforms e.g. '同-038-2-1.png' into ('540C', '038', 2, 1)
    """
    l = char_img_filename.split(".")[0].split("-")
    unicodestring = hex(ord(l[0]))[2:].upper()
    crop_idx = l[1]
    col = int(l[2])
    row = int(l[3])
    return unicodestring, crop_idx, col, row

def build_dataset(path,train_size,exclude):
    """
    splits character imgs of the shape "<char>-<crop_idx>-<col>-<row>.png"
    into train and val set
    """

    # create dictionary like this: {'8B39': [(120, 10, 4), (119, 8, 4)], '542B': ...}
    char_imgs = os.listdir(path)
    char_img_indices = {extract_indices(f) for f in char_imgs}
    crop_indices = list({
        crop_idx
        for _,crop_idx,_,_ in char_img_indices
        if crop_idx not in exclude
    })
    random.shuffle(crop_indices)
    cut_off = len(crop_indices) * train_size // 100
    train_crop_indices = crop_indices[:cut_off]
    val_crop_indices = crop_indices[cut_off:]

    train_imgs = {char_img for char_img in char_imgs if extract_indices(char_img)[1] in train_crop_indices}
    val_imgs   = {char_img for char_img in char_imgs if extract_indices(char_img)[1] in val_crop_indices}

    for img in train_imgs: # move using os.replace
        os.replace(os.path.join(path,img),os.path.join("train_data",img))
    for img in val_imgs: # move using os.replace
        os.replace(os.path.join(path,img),os.path.join("val_data",img))

    print(f"moved {len(train_imgs)} imgs from {path} to train_data")
    print(f"moved {len(val_imgs)} imgs from {path} to val_data")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("path", help="path to directory containing char imgs")
    parser.add_argument("-t", "--train_size", type=int, help="percentage of char imgs to be used for training, e.g. 80", required=True)
    parser.add_argument("-e", "--exclude", help="file containing idx of crops to be excluded when creating the dataset")
    args = parser.parse_args()

    abort = False
    if not os.path.exists("train_data"):
        os.makedirs("train_data")
    else:
        print("directory 'train_data' already exists, remove or rename it first")
        abort = True

    if not os.path.exists("val_data"):
        os.makedirs("val_data")
    else:
        print("directory 'val_data' already exists, remove or rename it first")
        abort = True

    path = args.path
    train_size = args.train_size
    exclude = {x.strip() for x in open(args.exclude).readlines()} if args.exclude else set()

    if not abort:
        print("building dataset ...")
        build_dataset(path,train_size,exclude)



    # print(extract_indices())

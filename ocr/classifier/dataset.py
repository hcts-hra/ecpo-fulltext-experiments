import os
import cv2
import numpy as np
import json

from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor, Compose
from torchvision import transforms

from make_synthetic_data import augment

class CharacterDataset(Dataset):
    def __init__(self, root_dir, file_naming, resize_dim, transform=None):
        self.root_dir = root_dir
        self.file_list = os.listdir(root_dir)
        self.transform = transform
        self.resize_dim = resize_dim
        # file_naming == "uni" for files of format '9F13.png'
        # file_naming == "char" for files of format 鼓-178-14-2.png
        self.file_naming = file_naming

        with open("glyph_dict.json") as f:
            self.unicode2label = json.load(f)
            self.label2unicode = {v:k for k,v in self.unicode2label.items()}

    def __len__(self):
        return len(self.file_list)

    def load_image(self,path):
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        return img

    def __getitem__(self,idx):
        file_name = self.file_list[idx]
        img = self.load_image(os.path.join(self.root_dir,file_name))
        img = cv2.resize(img, self.resize_dim, interpolation=cv2.INTER_CUBIC)
        img = cv2.bitwise_not(img)
        if self.transform:
            img = self.transform(img)

        # class index for CrossEntropyLoss:
        if self.file_naming == "uni":
            # file_name == e.g. '9F13-medium-50-1-2.png'
            label = self.unicode2label[file_name[:4]]
        elif self.file_naming == "char":
            # convert '鼓-178-14-2.png' to '9F13'
            unicodestring = file_name[0] # '鼓'
            label = self.unicode2label[hex(ord(unicodestring))[2:].upper()]

        return img, label

class RandomizedCharacterDataset(CharacterDataset):
    def load_image(self,path):
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)

        # randomize here and load original glyph imgs as dataset
        # instead of creating a whole new dataset and training on it
        brightness = np.random.randint(0,255)
        morph_its = np.random.randint(1,4)
        aug_img = augment(img,brightness,morph_its)

        # proceed as in parent class
        return aug_img

if __name__ == "__main__":

    import matplotlib.pyplot as plt

    transform = transforms.Compose([transforms.ToTensor()])
    train_data = CharacterDataset("synthetic_train_data/", "uni", (299,299), transform=transform)
    trainloader = DataLoader(train_data, shuffle=True, batch_size=1)

    print("showing 10 random samples from the dataset")

    data_iter = iter(trainloader)

    for i in range(10):
        img, label = next(data_iter)
        print(label)
        unicode = train_data.label2unicode[int(label)]
        print(unicode)
        print(i, img.shape)
        print(chr(int(unicode,16)))
        plt.imshow(img[0][0])
        plt.show()

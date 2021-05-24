import os
import cv2
import numpy as np

from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor, Compose
from torchvision import transforms

class SongTiDataset(Dataset):
    def __init__(self, root_dir, resize_dim, transform=None):
        self.file_list = [file for _,_,files in os.walk(root_dir) for file in files]
        self.root_dir = root_dir
        self.transform = transform
        self.resize_dim = resize_dim
        self.unicodestrings = sorted({file_name.split("-")[0].split(".")[0][3:] for file_name in self.file_list})

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self,idx):
        file_name = self.file_list[idx]
        img = cv2.imread(os.path.join(self.root_dir,file_name),cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, self.resize_dim, interpolation=cv2.INTER_LINEAR)
        img = cv2.bitwise_not(img)
        unicodestring = file_name.split("-")[0].split(".")[0][3:]
        # class index for CrossEntropyLoss:
        label = self.unicodestrings.index(unicodestring)
        # to get unicodestring from label do: unicodestring = self.unicodestrings[label]
        if self.transform:
            img = self.transform(img)
        return img, label


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    transform = transforms.Compose([transforms.ToTensor()])
    train_data = SongTiDataset("train_data", (299,299), transform=transform)
    trainloader = DataLoader(train_data, shuffle=True, batch_size=1)

    print("showing 10 random samples from the dataset")

    data_iter = iter(trainloader)

    for i in range(10):
        img, label = next(data_iter)
        label = train_data.unicodestrings[label]
        # img.shape == (batch_size, channels, height, width) == (1, 1, 101, 101)

        print(i, img.shape)
        print(chr(int(label,16)))
        plt.imshow(img[0][0])
        plt.show()





# class CustomDataset(torch.utils.data.Dataset):
#     def __init__(self, csv_path, images_folder, transform = None):
#         self.df = pd.read_csv(csv_path)
#         self.images_folder = images_folder
#         self.transform = transform
#         self.class2index = {"cat":0, "dog":1}
#
#     def __len__(self):
#         return len(self.df)
#
#     def __getitem__(self, index):
#         filename = self.df[index, "FILENAME"]
#         label = self.class2index[self.df[index, "LABEL"]]
#         image = PIL.Image.open(os.path.join(self.images_folder, filename))
#         if self.transform is not None:
#             image = self.transform(image)
#         return image, label

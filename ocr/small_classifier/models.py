import os
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchsummary import summary

from dataset import SongTiDataset

class BasicConv2d(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        **kwargs
    ) -> None:
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


class InceptionModel(nn.Module):
    def __init__(self, num_classes):
        super(InceptionModel, self).__init__()

        # layers
        self.inception = models.inception_v3(
            num_classes=num_classes,
            aux_logits=False,
            transform_input=False,
            pretrained=False,
            progress=True
        )
        self.inception.Conv2d_1a_3x3 = BasicConv2d(1, 32, kernel_size=3, stride=2)


        # test input shape
        self.print_shape = False

    def forward(self,img):
        if self.print_shape:
            print("shape before inception net:", img.shape)

        x = self.inception(img)

        if self.print_shape:
            print("shape after inception net:", x.shape)

        # return F.log_softmax(x, dim=1) # for nll_loss
        return x # for CrossEntropyLoss
        # expl. see here: https://stackoverflow.com/a/65193236/9920677


if __name__ == "__main__":

    MODEL = "inception"
    NUM_CLASSES = 257

    # initialize model
    model = InceptionModel(num_classes=NUM_CLASSES)
    summary(model, (1, 299, 299)) # but we have (1, 101, 101)!

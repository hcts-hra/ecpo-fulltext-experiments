import os
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models


# copy-pasta from https://pytorch.org/vision/stable/_modules/torchvision/models/inception.html#inception_v3
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
        self.inception = models.inception_v3(
            num_classes=num_classes,
            aux_logits=False,
            transform_input=False,
            pretrained=False,
            progress=False
        )
        # modify to deal with 1-channel inputs as we have grayscale images
        self.inception.Conv2d_1a_3x3 = BasicConv2d(1, 32, kernel_size=3, stride=1)

    def forward(self,img):
        x = self.inception(img)
        # return F.log_softmax(x, dim=1) # for nll_loss
        return x # for CrossEntropyLoss
        # expl. see here: https://stackoverflow.com/a/65193236/9920677

class GoogleNetModel(nn.Module):

    def __init__(self, num_classes):
        super(GoogleNetModel, self).__init__()
        self.googlenet = models.googlenet(
            num_classes=num_classes,
            aux_logits=False,
            transform_input=False,
            pretrained=False,
            progress=False
        )
        # modify to deal with 1-channel inputs as we have grayscale images
        self.googlenet.conv1 = BasicConv2d(1, 64, kernel_size=7, stride=2, padding=3)

    def forward(self,img):
        return self.googlenet(img)

if __name__ == "__main__":

    from torchsummary import summary
    import json

    NUM_CLASSES = len(json.load(open("glyph_dict.json")))

    # initialize model
    model = GoogleNetModel(num_classes=NUM_CLASSES)
    summary(model, (1, 224,224)) # but we have (1, 101, 101)!

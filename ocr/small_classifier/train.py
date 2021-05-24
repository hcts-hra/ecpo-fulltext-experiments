import argparse

import models, dataset

import numpy as np
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from eval import evaluate

import time

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--input_dim", type=int, required=True)
    parser.add_argument("-b", "--batch_size", type=int, required=True)
    parser.add_argument("-e", "--epochs", type=int, required=True)
    args = parser.parse_args()

    # hyperparameters
    INPUT_DIM = args.input_dim
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    LEARNING_RATE = 0.001
    MOMENTUM = 0.5

    # other constants
    SAVE_PATH = "./models"
    SLUG = f"d{INPUT_DIM}-b{BATCH_SIZE}"

    # load dataset, prepare data loader, find number of classes
    transform = transforms.Compose([transforms.ToTensor()])

    train_data = dataset.SongTiDataset("train_data", (INPUT_DIM,INPUT_DIM), transform=transform)
    val_data = dataset.SongTiDataset("val_data", (INPUT_DIM,INPUT_DIM), transform=transform)
    assert len(train_data.unicodestrings) == len(val_data.unicodestrings), \
        f"unequal number of classes in train data and val data ({len(train_data.unicodestrings)} =/= {len(val_data.unicodestrings)})"
    NUM_CLASSES = len(train_data.unicodestrings)

    train_loader = DataLoader(train_data, shuffle=True, batch_size=BATCH_SIZE)
    val_loader   = DataLoader(val_data, shuffle=False)

    # set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'device: {device}')

    # load model
    model = models.InceptionModel(num_classes=NUM_CLASSES)
    model.to(device)

    # loss function and optimizer
    criterion = nn.CrossEntropyLoss() # does not expect one-hot encoded vector as target, but class indices
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)

    # train dis shit
    print("===> started training")
    start_time = time.time()
    best_val_acc = 0

    for epoch in range(1, EPOCHS+1):

        model.train()
        train_losses = []

        for idx, (imgs, labels) in enumerate(train_loader):

            if device == 'cuda':
                # move to GPU
                imgs, labels = imgs.cuda(), labels.cuda()

            # get output and loss
            outputs = model(imgs)
            loss = criterion(outputs, labels)

            print(f'epoch: {epoch}, batch: {idx+1}/{len(train_loader)}, time since start: {time.time()-start_time}', end="\r")
            train_losses.append(loss.item())

            # backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # evaluate after every epoch
        val_acc, mean_val_loss = evaluate(model,val_loader,device,criterion=criterion)
        if val_acc > best_val_acc:
            torch.save(model.state_dict(),f"{SAVE_PATH}/{SLUG}-{epoch}.pth.tar")
            best_val_acc = val_acc

        with open(f"{SLUG}.logs", "a") as f:
            f.write(
                f'epoch: {epoch}, '
                f'train loss: {np.round(np.mean(train_losses),3)}, '
                f'val los: {np.round(mean_val_loss,3)}, '
                f'val acc: {val_acc}, '
                f'time since start: {np.round(time.time()-start_time)}'
            )
            f.write('\n')

        print(end="\r") # delete last print

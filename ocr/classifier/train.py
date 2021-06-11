import argparse
import os
import json
import numpy as np
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import time

from eval import evaluate
import models, dataset


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", required=True)
    parser.add_argument("-c", "--checkpoint")
    parser.add_argument("-t", "--training_data", required=True)
    parser.add_argument("-f", "--file_naming", required=True,
        help="'char' if train imgs start with character (like '鼓'), 'uni' if with 4-char unicode (like '9F13')")
    parser.add_argument("-v", "--validation_data", required=True)
    parser.add_argument("-d", "--input_dim", type=int, required=True)
    parser.add_argument("-b", "--batch_size", type=int, required=True)
    parser.add_argument("-e", "--epochs", type=int, required=True)
    parser.add_argument("-l", "--learning_rate", type=float, required=True)
    parser.add_argument("-s", "--slug")
    args = parser.parse_args()

    # hyperparameters
    MODEL = args.model
    CHECKPOINT = args.checkpoint
    TRAIN_PATH = args.training_data.rstrip("/")
    TRAIN_FILE_NAMING = args.file_naming
    VAL_PATH = args.validation_data
    INPUT_DIM = args.input_dim
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    LEARNING_RATE = args.learning_rate
    MOMENTUM = 0.5

    # other constants
    SAVE_PATH = "./models"
    SLUG = f"{MODEL}-b{BATCH_SIZE}-l{LEARNING_RATE}-on-{TRAIN_PATH}-{args.slug if args.slug else ''}"

    # load dataset, prepare data loader, find number of classes
    transform = transforms.Compose([transforms.ToTensor()])

    train_data = dataset.RandomizedCharacterDataset(TRAIN_PATH, TRAIN_FILE_NAMING, (INPUT_DIM,INPUT_DIM), transform=transform)
    val_data = dataset.CharacterDataset(VAL_PATH, "char", (INPUT_DIM,INPUT_DIM), transform=transform)
    NUM_CLASSES = len(json.load(open("glyph_dict.json")))

    train_loader = DataLoader(train_data, shuffle=True, batch_size=BATCH_SIZE)
    val_loader   = DataLoader(val_data, shuffle=False)

    # set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'device: {device}')

    # load model
    if MODEL == "inception":
        model = models.InceptionModel(num_classes=NUM_CLASSES)
    elif MODEL == "googlenet":
        model = models.GoogleNetModel(num_classes=NUM_CLASSES)

    # load checkpoint if provided
    if CHECKPOINT:
        model.load_state_dict(torch.load(CHECKPOINT,map_location=torch.device(device)))

    # check for	GPUs
    print(f"will be using {torch.cuda.device_count()} GPUs")
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
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

        # adapt learning_rate to 0.001 after 50 epochs if started with 0.01
        if LEARNING_RATE == 0.01 and epoch == 50:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 0.001

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
            torch.save(model.state_dict(),os.path.join(SAVE_PATH,f"{SLUG}-{epoch}.pth.tar"))
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

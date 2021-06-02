import sys, os
import models, dataset
import json
import cv2
import numpy as np
import torch
import argparse
# from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

def evaluate(model,eval_loader,device,criterion=None):
    # eval_loader might be val loader or test loader

    eval_losses = []
    model.to(device)
    model.eval()

    with torch.no_grad():
        correct_count = 0

        for imgs, labels in eval_loader:
            if device == "cuda":
                imgs, labels = imgs.cuda(), labels.cuda()

            preds = model(imgs)

            if criterion:
                loss = criterion(preds, labels)
                eval_losses.append(loss.item())

            classidx = torch.argmax(preds)
            # print(classidx, labels)
            correct_count += 1 if classidx==labels else 0

    rounded_acc_in_percent = np.round(correct_count/len(eval_loader)*100,2)

    if criterion:
        return rounded_acc_in_percent, np.mean(eval_losses)
    return rounded_acc_in_percent

def get_char_from_unicode(unicodestring):
    return chr(int(unicodestring,16))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", required=True)
    parser.add_argument("-v", "--validation_data", required=True)
    parser.add_argument("-d", "--input_dim", type=int, required=True)
    args = parser.parse_args()

    MODEL_PATH = args.model
    VAL_DATA = args.validation_data.rstrip("/")
    INPUT_DIM = args.input_dim

    # load dataset and prepare data loader
    transform = transforms.Compose([transforms.ToTensor()])
    NUM_CLASSES = len(json.load(open("glyph_dict.json")))

    # set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'device: {device}')

    # load val data
    eval_data = dataset.CharacterDataset(VAL_DATA, "char", (INPUT_DIM,INPUT_DIM), transform=transform)
    eval_loader = DataLoader(eval_data, shuffle=False)

    # load model
    model = models.GoogleNetModel(num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH,map_location=torch.device(device)))

    # evaluate
    model.to(device)
    model.eval()

    with torch.no_grad():
        count = 0
        correct_count = 0

        for imgs, labels in eval_loader:
            if device == "cuda":
                imgs, labels = imgs.cuda(), labels.cuda()

            preds = model(imgs)
            classidx = torch.argmax(preds)
            predicted_char = chr(int(eval_data.label2unicode[int(classidx)],16))
            goldlabel_char = chr(int(eval_data.label2unicode[int(labels)],16))
            if predicted_char == goldlabel_char:
                correct_count += 1
            # else:
            #     print(predicted_char,goldlabel_char)

            count += 1
            print(f"{count}/{len(eval_loader)}", end="\r")

        print("accuracy:", np.round(correct_count/len(eval_loader)*100,2))
    # rounded_acc_in_percent = np.round(correct_count/len(eval_loader)*100,2)
    #
    # if len(sys.argv) == 2: # evaluate manually on argv[1]
    #     for file_name in os.listdir(sys.argv[1]):
    #
    #         # load image as in Dataset.__getitem__
    #         img = cv2.imread(os.path.join(sys.argv[1],file_name),cv2.IMREAD_GRAYSCALE)
    #         print(img.shape)
    #         img = cv2.resize(img, (INPUT_DIM,INPUT_DIM), interpolation=cv2.INTER_LINEAR)
    #         img = cv2.bitwise_not(img)
    #         unicodestring = file_name.split("-")[0].split(".")[0][3:]
    #         # class index for CrossEntropyLoss:
    #         label = train_data.unicodestrings.index(unicodestring)
    #         img = transform(img)
    #
    #         # predict using model
    #         model.eval()
    #         with torch.no_grad():
    #             # if device == "cuda":
    #             #     imgs, labels = imgs.cuda(), labels.cuda()
    #             img = np.expand_dims(img, axis=0) # reshape to get batch size == 1
    #             img = torch.tensor(img)
    #             preds = model(img)
    #
    #             classidx = torch.argmax(preds)
    #
    #             pred_unicodestring = train_data.unicodestrings[classidx]
    #             true_unicodestring = train_data.unicodestrings[label]
    #
    #             pred_char = get_char_from_unicode(pred_unicodestring)
    #             true_char = get_char_from_unicode(true_unicodestring)
    #
    #             print(f"predicted image of {true_char} as {pred_char}")
    #
    # elif len(sys.argv) == 1: # val_data/ as input to 'evaluate' function
    #
    #     val_data = dataset.SongTiDataset("val_data", (299,299), transform=transform)
    #     assert len(train_data.unicodestrings) == len(val_data.unicodestrings), \
    #         f"unequal number of classes in train data and val data ({len(train_data.unicodestrings)} =/= {len(val_data.unicodestrings)})"
    #
    #     val_loader   = DataLoader(val_data, shuffle=False)
    #     print(evaluate(model,val_loader,device))
    #
    # else:
    #     raise Exception("too many arguments")

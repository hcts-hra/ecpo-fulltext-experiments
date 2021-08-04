import sys, os
sys.path.append("../classifier")
import numpy as np
import cv2
import json
import models
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from transformers import (
  BertTokenizerFast,
  BertForMaskedLM,
  pipeline
)

def print_top_k_results(seq,pipe):
    for i,char in enumerate(seq):
        masked = seq[:i] + tokenizer.mask_token + seq[i+1:]
        # print(masked.replace(tokenizer.mask_token,"＿"))
        preds = pipe(masked)
        out = [(pred["token_str"],round(pred["score"],2)) for pred in preds]
        try:
            predicted_index = [pred["token_str"] for pred in preds].index(char)
        except ValueError:
            predicted_index = None
        print(f" {char}: {predicted_index} – ", end="")
        for x in out:
            print(x[0], x[1], end=" ")
        print()

def load_image_for_ocr_model(path): # doing the same thing as in classifier/dataset.py
    img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (224,224), interpolation=cv2.INTER_CUBIC)
    img = cv2.bitwise_not(img)
    img = transforms.functional.to_tensor(img).view(1,1,224,224) # to get torch.Size([1, 1, 224, 224]) needed for the model
    # show("bla",img[0,0].numpy())
    return img

def show(name,img):
    cv2.imshow(name,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":

    ocr_model        = "../classifier/googlenet-b4-l0.001-on-train_data-afterPretrainingOnAllFonts-26.pth.tar"
    path_to_crops    = "../classifier/annotated_crops/"
    path_to_val_data = "../classifier/val_data/"
    val_crop_indices = {file.split("-")[1] for file in os.listdir(path_to_val_data)} # {'001', '003', '005', '011', '012', '018', ...}
    val_txt_paths    = {f"{path_to_crops}{val_crop_idx}.txt" for val_crop_idx in val_crop_indices}
    glyph_dict       = json.load(open("../classifier/glyph_dict.json"))

    sort_by_column_and_idx = lambda x:(int(x[6:].split("-")[0]),int(x[6:].split("-")[1].strip(".png")))
    img_lists_per_crop = [sorted(
                            filter(
                              lambda x:x[2:5]==idx,
                              os.listdir(path_to_val_data)
                            ),
                            key=sort_by_column_and_idx
                          ) for idx in val_crop_indices]
    # [['本-469-1-2.png', '月-469-1-3.png', ...], ['沿-346-1-1.png', '東-346-1-2.png', '江-346-1-3.png'], ...]

    k = 10
    NUM_CLASSES = len(glyph_dict)
    label2unicode = {v:k for k,v in glyph_dict.items()}

    # set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'device: {device}')

    # load model
    model = models.GoogleNetModel(num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(ocr_model,map_location=torch.device(device)))

    # evaluate
    model.to(device)
    model.eval()

    scores_of_correct_preds = []
    scores_of_wrong_preds = []
    bins = np.arange(20,40,0.1)

    with torch.no_grad():

        count = 0
        for img_list in img_lists_per_crop:

            # one img_list looks like this: ['本-469-1-2.png', '月-469-1-3.png', ...]

            # first OCR all of it
            pred_seq = []
            gold_seq = []
            for file_name in img_list:
                img_tensor = load_image_for_ocr_model(os.path.join(path_to_val_data,file_name))
                img_tensor = img_tensor.cuda() if device == "cuda" else img_tensor

                preds = model(img_tensor)
                print(torch.max(preds), torch.min(preds), torch.mean(preds))
                # preds = F.log_softmax(preds, dim=1) # not needed for CrossEntropyLoss but here
                print(torch.max(preds), torch.min(preds), torch.mean(preds))
                top_k = torch.topk(preds,k)

                goldlabel_char = file_name[0]
                top_k_chars = [chr(int(label2unicode[int(idx)],16)) for idx in top_k.indices[0]]
                top_k_scores = [val.item() for val in top_k.values[0]]

                top_k_scores -= np.mean(top_k_scores)

                if goldlabel_char == top_k_chars[0]: # correct prediction
                    scores_of_correct_preds.append(top_k_scores[0]-top_k_scores[1])
                    print("y", top_k_scores[0]-top_k_scores[1])
                else:
                    scores_of_wrong_preds.append(top_k_scores[0]-top_k_scores[1])
                    print("-", top_k_scores[0]-top_k_scores[1])

                pred_seq.append(top_k_chars[0]) # use top-1 predictions for BERT later
                gold_seq.append(goldlabel_char) # to check if accuracy improves with LM

            pred_seq = "".join(pred_seq)

            count += 1
            print(f"processed crops: {count}/{len(img_lists_per_crop)}", end="\r")

        # plot histograms
        fig, (ax1, ax2) = plt.subplots(2,1, sharex=True)
        ax1.hist(scores_of_correct_preds,color="royalblue",bins=bins)
        ax2.hist(scores_of_wrong_preds,color="cornflowerblue",bins=bins)
        ax2.set_xlabel('logit difference between top 1 and top 2 candidate')
        plt.savefig("histograms.png",bbox_inches='tight')

    # bert_model = BertForMaskedLM.from_pretrained('ckiplab/bert-base-chinese')
    # tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')
    # pipe = pipeline("fill-mask", model=bert_model, tokenizer=tokenizer, top_k=k)
    #
    # bert_model.eval()

    # # checking LMs performance on GT:
    # for txt in val_txt_paths:
    #     print("="*20)
    #     print(txt)
    #     with open(txt) as f:
    #         lines = [line.strip()
    #                      .rstrip("<lb/>")
    #                      .lstrip("e")
    #                      .replace("cc","，")
    #                      .replace("&gaiji;",tokenizer.unk_token)
    #                      .replace("　","")
    #                 for line in f.readlines()]
    #         seq = "".join(lines)
    #         print_top_k_results(seq,pipe)
    # print_top_k_results(, pipe)

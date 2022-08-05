# ECPO full text

This repository provides two-fold contribution: The directory `ocr/classifier/` contains the training environment to obtain the OCR classification model presented in [Henke (2021)](https://doi.org/10.11588/heidok.00030845) which is the basis for the [ecpo-ocr-pipeline](https://github.com/exc-asia-and-europe/ecpo-ocr-pipeline) (see directory `GoogLeNet/`). The rest of the repository contains the code for the experiments documented in the [wiki](https://github.com/exc-asia-and-europe/ecpo-fulltext-experiments/wiki). 

## Installation

Clone the repo:
```
git clone https://github.com/exc-asia-and-europe/ecpo-fulltext-experiments.git
```
Install requirements:
```
cd ecpo-fulltext-experiments
python -m pip install -r requirements.txt
```

## Directory Structure

`experiments/` contains mainly undocumented code for page segmentation experiments that has largely lead into developmental dead-ends documented in [Morphological Opening to Connect Text Blocks](https://github.com/exc-asia-and-europe/ecpo-fulltext-experiments/wiki/Morphological-Opening-to-Connect-Text-Blocks) and [Finding and Connecting Separators](https://github.com/exc-asia-and-europe/ecpo-fulltext-experiments/wiki/Finding-and-Connecting-Separators). We consider these approaches not robust enough and conclude that page segmentation should be approached by ML-driven methods such as has been demonstrated in the proof-of-concept documented in [ecpo-segment](https://github.com/exc-asia-and-europe/ecpo-segment).

`examples/` and `wiki_images/` are for storing image material only and can be ignored.

`ocr/small_classifier` has been documented in [First Experiments](https://github.com/exc-asia-and-europe/ecpo-fulltext-experiments/wiki/First-Experiments) and has only been left included here as a proof of concept for reference purposes. The actual classifier development is documented in `ocr/classifier/`.

`ocr/lm_improvement/` is the original code used in [Henke (2021)](https://doi.org/10.11588/heidok.00030845) for post-OCR correction using a BERT language model. Further development has taken place in the [ecpo-ocr-pipeline](https://github.com/exc-asia-and-europe/ecpo-ocr-pipeline) (see `GoogLeNet/bert.py`), please refer to that repository instead.

`ocr/classifier/` contains the code used to cut text block images into single character images in order to obtain a training set as explained [here](https://github.com/exc-asia-and-europe/ecpo-fulltext-experiments/wiki/Extracting-Character-Images) and in more detail in [Henke (2021)](https://doi.org/10.11588/heidok.00030845).

## Usage of `ocr/classifier/`

```
python3.8 train.py -m "googlenet" -t train_data -f "char" -v val_data -d 224 -b 4 -l 0.001 -e 5000 -s "slug" -c models/my_checkpoint.pth.tar
```

---


This is part of the [Early Chinese Periodicals Online project](https://uni-heidelberg.de/ecpo) and its related GitHub repositories:
- [ecpo](https://github.com/exc-asia-and-europe/ecpo) full text ground truth
- [ecpo-backend](https://github.com/exc-asia-and-europe/ECPO-backend) backend of the sql system
- [ecpoweb](https://github.com/exc-asia-and-europe/ecpoweb) web frontend (private)
- [ecpo-segment](https://github.com/exc-asia-and-europe/ecpo-segment) document segmentation based on dh-segment
- [ecpo-annotator](https://github.com/exc-asia-and-europe/ecpo-annotator) annotation tool
- [iipsrv](https://github.com/exc-asia-and-europe/iipsrv) iiif image service

Our progress is documented on the [Wiki pages](https://github.com/exc-asia-and-europe/ecpo-full-text/wiki).

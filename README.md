# SFDDNet: A Spatial-Frequency Dual Decoder Network for Breast Tumor Segmentation in Ultrasound Images

This repository contains the official PyTorch implementation for the paper "SFDDNet: A Spatial-Frequency Dual Decoder Network for Breast Tumor Segmentation in Ultrasound Images".

## Introduction

Automatic segmentation of breast ultrasound (BUS) images is a critical yet challenging task due to issues like low contrast, speckle noise, and ambiguous boundaries. This work introduces SFDDNet, a novel dual-decoder architecture that simultaneously processes information in both spatial and frequency domains. By synergistically fusing features from these two complementary perspectives, SFDDNet generates more comprehensive and robust representations, leading to highly accurate and reliable tumor segmentation.

## Key Features

-   **Spatial-Frequency Dual Decoder Network (SFDDNet):** A novel architecture with two parallel decoders to process spatial and frequency information simultaneously, generating robust, multi-domain feature representations.
-   **Enhanced Frequency-Aware Block (EFAB):** Utilizes wavelet decomposition to decouple and differentially process low-frequency global context and high-frequency boundary details, significantly improving the model's robustness to noise and artifacts.
-   **Dual-Selective Transformer (DST):** An efficient module integrated into the decoder that enhances computational efficiency and performance by adaptively pruning redundant tokens and channels.
-   **Spatial-Frequency Fusion Block (SFFB):** A multi-scale fusion mechanism that adaptively integrates the outputs of the dual decoders, ensuring effective synergy between spatial and frequency features.


## Usage

### Dataset Preparation

This project does not require a specific folder structure for the image data. Instead, it relies on a `dataset.json` file located in the root directory of the project. This file specifies the paths for the training, validation, and test sets.

The structure of `dataset.json` should be as follows:

```json
{
    "train": [
        {"image": "path/to/train/image1.png", "mask": "path/to/train/mask1.png"},
        {"image": "path/to/train/image2.png", "mask": "path/to/train/mask2.png"}
    ],
    "val": [
        {"image": "path/to/val/image1.png", "mask": "path/to/val/mask1.png"}
    ],
    "test": [
        {"image": "path/to/test/image1.png", "mask": "path/to/test/mask1.png"}
    ]
}
```
### Training

To train the SFDDNet model, run the following command:

```Bash
python train.py --root_path "/path/to/your/data/directory" --dataset "SFDDNet_BUSI_Experiment" --num_classes 2
```
--root_path: The root directory where your image and mask folders (e.g., DATA, MASK) are located. The paths in dataset.json are relative to this root path.

--dataset: A name for your experiment, which will be used for saving logs and model checkpoints.

--num_classes: The number of output channels for the network. For binary segmentation (lesion vs. background), this should be 2.

### Testing / Inference
To evaluate a trained model on the test set, use the following command. Make sure you have the trained model weights saved in the appropriate directory from the training process.

```Bash
python test.py --root_path "/path/to/your/data/directory" --dataset "SFDDNet_BUSI_Experiment"
```

# Acknowledgements
This project is developed based on the excellent CSwin-UNet repository. We express our sincere gratitude to the original authors for making their code publicly available, which has provided a valuable foundation for our research. The original repository can be found at: https://github.com/eatbeanss/CSWin-UNet.

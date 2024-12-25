# VideoMAE-v2 SL Training Pipeline

This repository contains scripts and instructions for training VideoMAE-v2 models on custom video datasets.

## Prerequisites

Before starting, ensure you have Python installed on your system. Then install the required packages:

```bash
pip install -r requirements.txt
```

## Environment Setup

1. Create an Ananconda environment
```
conda create -n signlanguage-env python=3.9
conda activate signlanguage-env
```
2. Install the required packages
```
pip install -r requirements.txt
```
3. Create an .env file in the root directory.
4. Add your Weights & Biases API key:
```
WANDB_API_KEY=YOUR_WANDB_API_KEY
```

## Model Checkpoint

Download the VideoMAE-v2-base checkpoint from MMAction2:

```bash
bash getting_started.sh
```

This script will download and rename the checkpoint parameters according to the [official MMAction2 repository](https://github.com/open-mmlab/mmaction2/blob/main/configs/recognition/videomaev2/README.md).

## Data Preparation

### Frame Extraction

Edit `scripts/extract_frames.sh`:
   - Replace `INPUT_FOLDER` with your video directory
   - Replace `OUTPUT_BASE_FOLDER` with your desired output frame directory

```bash
bash extract_frames.sh
```

### Dataset Splitting

For each frame folder, run the dataset splitting script:

```bash
bash split_datasets.sh
```

Before running, modify the script parameters:
- `SOURCE_DIR`: Directory containing the extracted frames
- `OUTPUT_DIR`: Directory where the split datasets will be saved

## Training

1. Edit `scripts/videomae_v2.sh` with the following parameters:
   - `MODEL_PATH`: Path to the VideoMAE-v2 checkpoint
   - `TRAIN_DATA_PATH`: Path to training data folder
   - `VAL_DATA_PATH`: Path to validation data folder

2. Start training:
```bash
bash videomae_v2.sh
```

## Notes

- Ensure all paths in scripts are absolute or correctly referenced relative to the script location
- Monitor training progress through Weights & Biases dashboard
- Check script permissions if you encounter execution issues (`chmod +x script.sh`)

## Acknowledgments

Based on the [MMAction2](https://github.com/open-mmlab/mmaction2) implementation of VideoMAE-v2.

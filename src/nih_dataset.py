# src/nih_dataset.py
import os
import glob
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

import config


class NIHChestXrayDataset(Dataset):
    """
    NIH ChestXray14 dataset, single-label multi-class.

    Uses config.CSV_PATH ("Data_Entry_2017.csv") for labels and
    config.TRAIN_VAL_LIST / config.TEST_LIST for the official split.
    Only rows whose "Finding Labels" is a single label (no '|') are kept,
    matching config.CLASS_TO_IDX built in config.py.

    Images are expected somewhere under config.DATA_PATH in the standard
    NIH layout: images_001/images/*.png ... images_012/images/*.png
    (falls back to images_XXX/*.png if there's no nested "images" folder).
    """

    def __init__(self, split="train", transform=None):
        assert split in ("train", "test"), "split must be 'train' or 'test'"
        self.transform = transform

        df = pd.read_csv(config.CSV_PATH)

        list_path = config.TRAIN_VAL_LIST if split == "train" else config.TEST_LIST
        with open(list_path, "r") as f:
            split_images = {line.strip() for line in f if line.strip()}

        df = df[df["Image Index"].isin(split_images)]
        df = df[~df["Finding Labels"].str.contains(r"\|")]
        df = df[df["Finding Labels"].isin(config.CLASS_TO_IDX)]

        self.samples = list(zip(
            df["Image Index"].tolist(),
            df["Finding Labels"].tolist()
        ))

        self._image_index = self._build_image_index()

        # Drop any listed samples whose image file wasn't found on disk,
        # rather than crashing mid-training on a missing file.
        self.samples = [
            (fname, label) for fname, label in self.samples
            if fname in self._image_index
        ]

    def _build_image_index(self):
        """Builds filename -> full path lookup once (avoids per-item glob)."""
        pattern = os.path.join(config.DATA_PATH, "images_*", "images", "*.png")
        paths = glob.glob(pattern)

        if not paths:
            pattern = os.path.join(config.DATA_PATH, "images_*", "*.png")
            paths = glob.glob(pattern)

        return {os.path.basename(p): p for p in paths}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filename, label_name = self.samples[idx]
        path = self._image_index[filename]

        # NIH images are grayscale; convert to RGB to match the
        # ImageNet-style 3-channel normalization used everywhere else.
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        label = config.CLASS_TO_IDX[label_name]
        return image, label
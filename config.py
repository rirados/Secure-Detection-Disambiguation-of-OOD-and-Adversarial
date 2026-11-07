# config.py
import os
import torch
import pandas as pd

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 64
IMAGE_SIZE = 224

# --------------------------------------------------
# Dataset paths (NIH ChestXray14)
# --------------------------------------------------
DATA_PATH = "./data/NIH_ChestXray14"
CSV_PATH = os.path.join(DATA_PATH, "Data_Entry_2017.csv")
TRAIN_VAL_LIST = os.path.join(DATA_PATH, "train_val_list.txt")
TEST_LIST = os.path.join(DATA_PATH, "test_list.txt")

MODEL_PATH = "artifacts/resnet18_nih_chestxray.pth"
ARTIFACT_PATH = "artifacts/detector_artifacts.pkl"
SECRET_KEY_PATH = "artifacts/secret.key"
LOG_FILE_PATH = "artifacts/system.log"


def _build_class_list(csv_path):
    """
    Builds the sorted list of classes from the NIH "Finding Labels" column.

    Only single-label rows (labels that do NOT contain '|') are treated as
    classes. This keeps the dataset single-label multi-class, which is what
    the rest of the pipeline assumes (per-class Mahalanobis means/precision,
    a single softmax head, CrossEntropyLoss). Multi-label rows (e.g.
    "Cardiomegaly|Emphysema") are simply excluded by the dataset loader.
    """
    if not os.path.exists(csv_path):
        # Lets `import config` succeed even before the CSV is in place
        # (e.g. first checkout on a machine without the dataset yet).
        return []

    df = pd.read_csv(csv_path)
    single_label_mask = ~df["Finding Labels"].str.contains(r"\|")
    labels = sorted(df.loc[single_label_mask, "Finding Labels"].unique().tolist())
    return labels


CLASSES = _build_class_list(CSV_PATH)
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASSES)}
NUM_CLASSES = len(CLASSES) if CLASSES else 15  # fallback matching checkpoint size
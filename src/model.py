import os
import torch
import torch.nn as nn
from torchvision import models
import config

def load_model(weights_path=None):
    if weights_path is None:
        weights_path = getattr(config, "MODEL_PATH", "artifacts/resnet18_nih_chestxray.pth")

    # If fine-tuned checkpoint exists, load without ImageNet weights
    has_checkpoint = os.path.exists(weights_path)
    base_weights = None if has_checkpoint else models.ResNet18_Weights.IMAGENET1K_V1

    model = models.resnet18(weights=base_weights)
    model.fc = nn.Linear(model.fc.in_features, config.NUM_CLASSES)

    if has_checkpoint:
        state_dict = torch.load(weights_path, map_location=config.DEVICE)
        model.load_state_dict(state_dict)
        print(f"Loaded fine-tuned model checkpoint from {weights_path}")
    else:
        print("Using ResNet-18 backbone with newly initialized 10-class head.")

    model.to(config.DEVICE)
    model.eval()
    return model
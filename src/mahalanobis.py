import torch
import numpy as np
import config
from src.hooks import captured_features

def get_mahalanobis_score(inputs, model, stats, magnitude=0.002):
    inputs = inputs.to(config.DEVICE).requires_grad_(True)

    captured_features.clear()
    logits = model(inputs)[:, :config.NUM_CLASSES]
    best_class = logits.argmax(1)

    features = captured_features[-1]
    means = stats[-1]["means"].to(config.DEVICE)
    precision = stats[-1]["precision"].to(config.DEVICE)

    target_means = means[best_class]

    diff = features - target_means
    intermediate = torch.mm(diff, precision)
    distance = torch.sum(intermediate * diff, dim=1)

    loss = torch.mean(-0.5 * distance)
    model.zero_grad()
    loss.backward()

    gradient = inputs.grad.data.sign()
    perturbed = inputs.data - magnitude * gradient

    return perturbed


def calculate_layer_scores(inputs, model, stats):
    model.eval()
    captured_features.clear()

    with torch.no_grad():
        _ = model(inputs)

    layer_scores = []

    for i, feat in enumerate(captured_features):
        means = stats[i]["means"].to(config.DEVICE)
        precision = stats[i]["precision"].to(config.DEVICE)

        # Vectorized Mahalanobis distance across batch:
        # feat: (B, D), means: (C, D), precision: (D, D)
        diff = feat.unsqueeze(1) - means.unsqueeze(0)  # (B, C, D)
        intermediate = torch.matmul(diff, precision)   # (B, C, D)
        dist = torch.sum(intermediate * diff, dim=-1)  # (B, C)
        dist = torch.clamp(dist, min=0.0)              # prevent numerical negatives
        min_dist, _ = torch.min(dist, dim=1)           # (B,)

        scores = -min_dist.detach().cpu().numpy()
        layer_scores.append(scores)

    return np.column_stack(layer_scores)
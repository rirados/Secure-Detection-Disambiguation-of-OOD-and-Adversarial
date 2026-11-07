import torch
from tqdm import tqdm
import config
from src.hooks import captured_features


def get_class_stats(dataloader, model, layers):
    """
    Computes class-wise feature means and precision matrices for each
    hooked layer using the training dataset.

    Returns:
        stats_per_layer: List containing:
            {
                "means": Tensor[num_classes, feature_dim],
                "precision": Tensor[feature_dim, feature_dim]
            }
    """

    all_features = {i: [] for i in range(len(layers))}
    all_labels = []

    model.eval()

    print("\n[Step 3.1] Extracting Deep Features...\n")

    with torch.no_grad():
        for inputs, labels in tqdm(
                dataloader,
                desc="Extracting Features",
                unit="batch",
                colour="cyan",
                dynamic_ncols=True):

            inputs = inputs.to(config.DEVICE)

            captured_features.clear()
            _ = model(inputs)

            for i, feat in enumerate(captured_features):
                all_features[i].append(feat.cpu())

            all_labels.append(labels)

    labels = torch.cat(all_labels)

    stats_per_layer = []

    print("\n[Step 3.2] Computing Mahalanobis Statistics...\n")

    for i in tqdm(
            range(len(layers)),
            desc="Processing Layers",
            unit="layer",
            colour="green",
            dynamic_ncols=True):

        layer_feats = torch.cat(all_features[i])

        means = []

        # Compute class-wise mean vectors
        for c in range(config.NUM_CLASSES):
            mask = (labels == c)
            means.append(layer_feats[mask].mean(dim=0))

        means = torch.stack(means)

        # Zero-center features
        zero_mean = layer_feats.clone()

        for c in range(config.NUM_CLASSES):
            zero_mean[labels == c] -= means[c]

        # Covariance Matrix
        covariance = (
            torch.mm(zero_mean.t(), zero_mean)
            / len(layer_feats)
        )

        # Precision Matrix (using pseudo-inverse with regularization for guaranteed stability)
        reg_eye = 1e-4 * torch.eye(covariance.shape[0], device=covariance.device)
        precision = torch.linalg.pinv(covariance + reg_eye)

        stats_per_layer.append({
            "means": means,
            "precision": precision
        })

    print("\n✓ Mahalanobis Statistics Successfully Computed!\n")

    return stats_per_layer
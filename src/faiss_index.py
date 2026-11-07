
import faiss
import numpy as np
import torch
import config

from tqdm import tqdm
from src.hooks import captured_features


def extract_top_layer_features(model, inputs):
    """
    Runs the model and returns the captured top-layer features.

    Expected output shape:
        [batch_size, feature_dimension]
    """
    captured_features.clear()
    _ = model(inputs)

    if not captured_features:
        raise RuntimeError(
            "No features were captured. Check your hooks "
            "and ensure the model's top layer is registered."
        )

    features = captured_features[-1]

    if not isinstance(features, torch.Tensor):
        raise TypeError(
            "Captured features must be a torch.Tensor."
        )

    # Convert spatial feature maps to vectors if necessary.
    if features.ndim > 2:
        features = torch.flatten(features, start_dim=1)

    if features.ndim != 2:
        raise ValueError(
            f"Expected 2D features [batch, dimension], "
            f"got shape {tuple(features.shape)}"
        )

    return features


def build_faiss_index(model, dataloader, num_samples=1000):
    """
    Extracts features from clean in-distribution training data
    and builds a FAISS L2 index.

    The threshold is calibrated using leave-one-out distances:
    each training embedding is compared with its 5 nearest
    OTHER embeddings, excluding itself.

    Returns:
        index: FAISS IndexFlatL2
        threshold: Mean + 2 * standard deviations of
                   leave-one-out neighbor distances
    """
    model.eval()
    embeddings = []
    samples_collected = 0

    if num_samples < 6:
        raise ValueError(
            "num_samples must be at least 6 for leave-one-out "
            "calibration with 5 neighbors."
        )

    with torch.no_grad():
        for batch in tqdm(
            dataloader,
            desc="Extracting FAISS Embeddings",
            unit="batch",
            colour="cyan"
        ):
            # Support dataloaders that return (inputs, labels)
            # or (inputs, labels, ...) tuples.
            inputs = batch[0] if isinstance(batch, (tuple, list)) else batch
            inputs = inputs.to(config.DEVICE)

            features = extract_top_layer_features(model, inputs)

            features = (
                features.detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )

            if not np.isfinite(features).all():
                raise ValueError(
                    "Non-finite values found in extracted features."
                )

            remaining = num_samples - samples_collected
            features = features[:remaining]

            embeddings.append(features)
            samples_collected += len(features)

            if samples_collected >= num_samples:
                break

    if not embeddings:
        raise ValueError(
            "No embeddings were extracted. Check the dataloader."
        )

    embeddings = np.vstack(embeddings).astype(np.float32)

    if len(embeddings) < 6:
        raise ValueError(
            f"Only {len(embeddings)} embeddings were extracted. "
            "At least 6 are required."
        )

    print(f"Extracted embeddings: {embeddings.shape}")

    # Build exact L2 index.
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    # Leave-one-out calibration.
    # Request 6 neighbors: self + 5 other nearest neighbors.
    distances, indices = index.search(embeddings, k=6)

    loo_distances = []

    for i in range(len(embeddings)):
        # Exclude this embedding's own index.
        mask = indices[i] != i
        other_distances = distances[i][mask]

        # Keep the 5 nearest non-self neighbors.
        other_distances = other_distances[:5]

        if len(other_distances) != 5:
            raise RuntimeError(
                f"Could not find 5 non-self neighbors for "
                f"embedding {i}."
            )

        loo_distances.extend(other_distances.tolist())

    loo_distances = np.asarray(
        loo_distances,
        dtype=np.float32
    )

    mean_dist = float(np.mean(loo_distances))
    std_dist = float(np.std(loo_distances))
    
    # Use 99.5th percentile for heavy-tailed distance distributions
    threshold = float(np.percentile(loo_distances, 99.5))

    print("\nFAISS calibration complete")
    print(f"Index size: {index.ntotal}")
    print(f"Feature dimension: {dimension}")
    print(f"LOO neighbor distances: {len(loo_distances)}")
    print(f"Mean distance: {mean_dist:.4f}")
    print(f"Std distance: {std_dist:.4f}")
    print(f"Distance threshold (99.5th %ile): {threshold:.4f}")

    return index, threshold


def disambiguate_sample(feature_vector, index, threshold, k=5):
    if index is None or index.ntotal == 0:
        raise ValueError("FAISS index is empty or missing.")

    if isinstance(feature_vector, torch.Tensor):
        feature_vector = feature_vector.detach().cpu().numpy()

    feature_vector = np.asarray(feature_vector, dtype=np.float32)

    if feature_vector.ndim == 1:
        feature_vector = feature_vector.reshape(1, -1)

    if feature_vector.shape[1] != index.d:
        raise ValueError(
            f"Feature dimension mismatch: {feature_vector.shape[1]} "
            f"vs {index.d}"
        )

    distances, indices = index.search(feature_vector, k)

    neighbor_distances = distances[0]
    neighbor_indices = indices[0]
    avg_distance = float(np.mean(neighbor_distances))

    print("\n========== FAISS INFERENCE LOG ==========")
    print(f"Index vectors: {index.ntotal}")
    print(f"Feature dimension: {index.d}")
    print(f"Query feature shape: {feature_vector.shape}")
    print(f"Number of neighbors (k): {k}")
    print(f"Neighbor indices: {neighbor_indices.tolist()}")
    print(f"Neighbor distances: {neighbor_distances.tolist()}")
    print(f"Average squared L2 distance: {avg_distance:.6f}")
    print(f"Threshold: {threshold:.6f}")
    print(f"Distance / threshold: {avg_distance / threshold:.6f}")

    if avg_distance <= threshold:
        category = "Adversarial Attack"
        print("Decision: BELOW threshold")
    else:
        category = "Out-of-Distribution (OOD)"
        print("Decision: ABOVE threshold")

    print(f"FAISS category: {category}")
    print("=========================================\n")

    return category, avg_distance
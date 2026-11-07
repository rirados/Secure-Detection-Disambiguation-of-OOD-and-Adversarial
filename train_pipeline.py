# train_pipeline.py
import os
import pickle
import numpy as np
import config
from src.model import load_model
from src.hooks import register_hooks
from src.utils import get_dataloader
from src.stats import get_class_stats
from src.mahalanobis import get_mahalanobis_score, calculate_layer_scores
from src.adversarial import generate_adversarial_image
from src.ensemble import train_judge
from src.faiss_index import build_faiss_index
from src.security import get_or_create_key
from tqdm import tqdm

# Ensure artifacts directory exists
os.makedirs("artifacts", exist_ok=True)

print("1. Loading Model and Hooks...")
model = load_model()
layers = register_hooks(model)
loader = get_dataloader()

print("2. Generating AES Encryption Key...")
_ = get_or_create_key()

print("3. Extracting Layer Statistics...")
stats = get_class_stats(loader, model, layers)

print("4. Building FAISS Index & Threshold for Disambiguation...")
faiss_idx, faiss_threshold = build_faiss_index(model, loader)

print("5. Collecting Mahalanobis Scores (Clean vs FGSM Adversarial)...")
X_clean, X_adv = [], []

for inputs, labels in tqdm(
    loader,
    desc="Collecting Mahalanobis Scores",
    unit="batch",
    colour="green",
):
    inputs, labels = inputs.to(config.DEVICE), labels.to(config.DEVICE)

    clean = get_mahalanobis_score(inputs, model, stats)
    clean_scores = calculate_layer_scores(clean, model, stats)

    adv = generate_adversarial_image(model, inputs, labels)
    adv = get_mahalanobis_score(adv, model, stats)
    adv_scores = calculate_layer_scores(adv, model, stats)

    X_clean.append(clean_scores)
    X_adv.append(adv_scores)

    if len(X_clean) * config.BATCH_SIZE > 500:
        break

X_clean = np.vstack(X_clean)
X_adv = np.vstack(X_adv)

print("6. Training Logistic Regression Classifier...")
judge = train_judge(X_clean, X_adv)

print("7. Saving Artifacts...")
with open(config.ARTIFACT_PATH, "wb") as f:
    pickle.dump({
        "stats": stats, 
        "judge": judge,
        "faiss_index": faiss_idx,
        "faiss_threshold": faiss_threshold
    }, f)

print(" Pipeline trained and ready for deployment!")
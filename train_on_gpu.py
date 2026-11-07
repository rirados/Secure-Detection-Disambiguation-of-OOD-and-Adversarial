import os
import argparse
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm
try:
    import faiss
except ImportError:
    raise ImportError(
        "FAISS is not installed. Please run: pip install faiss-cpu (or pip install -r requirements.txt)"
    )
import config
from src.hooks import register_hooks, captured_features
from src.stats import get_class_stats
from src.mahalanobis import get_mahalanobis_score, calculate_layer_scores
from src.adversarial import generate_adversarial_image
from src.ensemble import train_judge
from src.security import get_or_create_key
from src.nih_dataset import NIHChestXrayDataset
from src.faiss_index import build_faiss_index


def parse_args():
    parser = argparse.ArgumentParser(description="Train Model and Detector Pipeline on GPU")
    parser.add_argument("--epochs", type=int, default=5, help="Number of fine-tuning epochs (default: 5)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for training and feature extraction (default: 64)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for fine-tuning (default: 1e-4)")
    parser.add_argument("--adv-samples", type=int, default=3000, help="Number of clean/adv samples for judge training (default: 3000)")
    parser.add_argument("--faiss-samples", type=int, default=5000, help="Number of clean embeddings for FAISS index (default: 5000)")
    return parser.parse_args()


def check_environment():
    print("\n" + "=" * 60)
    print(" 🚀 HARDWARE ACCELERATION CHECK")
    print("=" * 60)

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f" [✓] NVIDIA CUDA is AVAILABLE!")
        print(f" [✓] Detected GPU : {gpu_name}")
        print(f" [✓] Total VRAM   : {vram_gb:.2f} GB")
        torch.backends.cudnn.benchmark = True
    else:
        print(" [!] CUDA NOT DETECTED. Running on CPU fallback.")
        print("     Tip: If you have an NVIDIA GPU, ensure NVIDIA drivers and CUDA-enabled PyTorch are installed.")

    print(f" [✓] Selected Device: {config.DEVICE}")
    print(f" [✓] Classes ({config.NUM_CLASSES}): {config.CLASSES}")
    print("=" * 60 + "\n")


def get_nih_dataloaders(batch_size):
    """Prepares augmented training loader and standard test loader for NIH ChestXray14."""
    train_transform = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = NIHChestXrayDataset(split="train", transform=train_transform)
    test_dataset = NIHChestXrayDataset(split="test", transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=torch.cuda.is_available())

    return train_loader, test_loader


def finetune_resnet(train_loader, test_loader, epochs, lr):
    """Fine-tunes ResNet-18 on NIH ChestXray14 with an adapted classification head."""
    print(f"\n[Step 1/5] Fine-tuning ResNet-18 on NIH ChestXray14 ({epochs} epochs)...")

    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, config.NUM_CLASSES)
    model.to(config.DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_acc = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]", unit="batch", colour="blue")
        for inputs, targets in pbar:
            inputs, targets = inputs.to(config.DEVICE), targets.to(config.DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            pbar.set_postfix(loss=f"{total_loss/total:.4f}", acc=f"{100.*correct/total:.2f}%")

        # Evaluate on test set
        model.eval()
        test_correct, test_total = 0, 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(config.DEVICE), targets.to(config.DEVICE)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                test_total += targets.size(0)
                test_correct += predicted.eq(targets).sum().item()

        test_acc = 100.0 * test_correct / test_total
        print(f" -> Epoch {epoch} Evaluation Accuracy: {test_acc:.2f}%")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), config.MODEL_PATH)
            print(f"    [*] Checkpoint saved to {config.MODEL_PATH} (Acc: {best_acc:.2f}%)")

    print(f"\n✓ Fine-tuning completed! Best Test Accuracy: {best_acc:.2f}%\n")

    # Reload best weights
    model.load_state_dict(torch.load(config.MODEL_PATH, map_location=config.DEVICE))
    model.eval()
    return model





def train_detector_judge(model, loader, stats, target_samples=3000):
    """Collects multi-layer Mahalanobis scores for clean and FGSM adversarial inputs."""
    print(f"\n[Step 4/5] Collecting Mahalanobis Features (Target: {target_samples} samples)...")
    X_clean, X_adv = [], []

    for inputs, labels in tqdm(loader, desc="Generating Clean vs Adv Scores", colour="green"):
        inputs, labels = inputs.to(config.DEVICE), labels.to(config.DEVICE)

        # 1. Clean perturbed Mahalanobis scores
        clean_perturbed = get_mahalanobis_score(inputs, model, stats)
        clean_scores = calculate_layer_scores(clean_perturbed, model, stats)

        # 2. FGSM Adversarial Mahalanobis scores
        adv_inputs = generate_adversarial_image(model, inputs, labels, epsilon=0.01)
        adv_perturbed = get_mahalanobis_score(adv_inputs, model, stats)
        adv_scores = calculate_layer_scores(adv_perturbed, model, stats)

        X_clean.append(clean_scores)
        X_adv.append(adv_scores)

        if len(X_clean) * config.BATCH_SIZE >= target_samples:
            break

    X_clean = np.vstack(X_clean)
    X_adv = np.vstack(X_adv)

    print(f" [✓] Collected {len(X_clean)} clean and {len(X_adv)} adversarial score vectors.")
    judge = train_judge(X_clean, X_adv)

    # Quick verification on training data
    train_preds = judge.predict(np.vstack([X_clean, X_adv]))
    train_labels = np.hstack([np.ones(len(X_clean)), np.zeros(len(X_adv))])
    acc = 100.0 * np.mean(train_preds == train_labels)
    print(f" [✓] Logistic Regression Judge Training Accuracy: {acc:.2f}%")

    return judge


def main():
    args = parse_args()
    os.makedirs("artifacts", exist_ok=True)
    os.makedirs(config.DATA_PATH, exist_ok=True)

    check_environment()
    train_loader, test_loader = get_nih_dataloaders(args.batch_size)

    # 1. Fine-tune ResNet-18 on NIH ChestXray14
    model = finetune_resnet(train_loader, test_loader, epochs=args.epochs, lr=args.lr)

    # 2. Extract multi-layer stats
    print("\n[Step 2/5] Extracting Class-wise Mahalanobis Statistics...")
    layer_handles = register_hooks(model)
    stats = get_class_stats(train_loader, model, layer_handles)

    # 3. FAISS index & calibrated threshold
    print("\n[Step 3/5] Building FAISS Index & Calibrating Disambiguation Threshold...")
    faiss_index, faiss_threshold = build_faiss_index(model, train_loader, num_samples=args.faiss_samples)

    # 4. Train Judge
    judge = train_detector_judge(model, train_loader, stats, target_samples=args.adv_samples)

    # 5. Security key & Artifact packaging
    print("\n[Step 5/5] Packaging System Artifacts...")
    _ = get_or_create_key()

    with open(config.ARTIFACT_PATH, "wb") as f:
        pickle.dump({
            "stats": stats,
            "judge": judge,
            "faiss_index": faiss_index,
            "faiss_threshold": faiss_threshold
        }, f)

    print("\n" + "=" * 60)
    print(" TRAINING & ARTIFACT GENERATION COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(" Generated Files in ./artifacts/:")
    print(f"  1. {config.MODEL_PATH}          (Trained ResNet-18 weights)")
    print(f"  2. {config.ARTIFACT_PATH}       (Detector stats, Judge, FAISS index)")
    print(f"  3. {config.SECRET_KEY_PATH}             (AES Fernet Encryption Key)")
    print("=" * 60)
    print("\nNext Step:")
    print(" Copy the 'artifacts/' folder from that laptop to this project folder.")
    print(" Then run 'streamlit run app.py' on any machine!\n")


if __name__ == "__main__":
    main()
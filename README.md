# Secure Multi-Stage Detection and Disambiguation of OOD and Adversarial Samples in Neural Networks

A application and research-based software framework that detects and differentiates **Out-of-Distribution (OOD) inputs** and **adversarial attacks** in deep learning models without retraining the original model.

### 1.1 Background
Machine learning and deep learning models power critical applications: image classification, medical diagnosis, autonomous systems, cybersecurity. But they break when faced with inputs that differ from training data.

**2 major threats:**
1. **Out-of-Distribution (OOD) inputs**: Data not seen during training → model gives wrong predictions with high confidence.
2. **Adversarial attacks**: Tiny, intentional perturbations added to input → misleads neural network into wrong classification.

This project builds a multi-stage detection system that analyzes internal feature representations and filters unsafe inputs before they reach the model. Validated inputs can then be processed securely with encryption.

### 1.2 Problem Statement
Modern DL systems in healthcare, autonomous driving, surveillance, and finance are vulnerable to:
- **OOD inputs** that differ from training distribution
- **Adversarial attacks** with small crafted perturbations

Existing solutions usually handle them separately or require model retraining.  
**Goal**: A unified, efficient, model-agnostic framework that detects both OOD + adversarial inputs without modifying the original model.

### 1.3 Objectives
1. Identify abnormal inputs in DL models: OOD samples + adversarial attacks
2. Improve robustness and trustworthiness by preventing unsafe decisions  
3. Disambiguate between OOD vs adversarial inputs for better interpretability
4. Validate performance on standard benchmark datasets + metrics

**Applications**: Cybersecurity, medical image analysis, autonomous systems, intelligent monitoring

### 5. Implementation Overview

#### 5.1 Feature Extraction
Uses a pretrained **ResNet-18** model without retraining.  
- Captures feature maps from multiple intermediate layers using forward hooks
- Applies adaptive average pooling + flattening for uniform vectors
- Computes class-wise statistical distributions: mean vectors + covariance matrices
- Multi-layer features form the base for anomaly detection

#### 5.2 OOD and Adversarial Detection
Unified approach combining feature-based statistical analysis + model sensitivity evaluation.

**5.2.1 Statistical Distance-Based Detection**
Uses **Mahalanobis distance** to measure deviation from known class distributions:
- Computed per layer using shared covariance matrix
- Small input perturbation added in gradient direction to increase separation
- Layer-wise scores aggregated into feature vector

**5.2.2 Adversarial Sample Generation and Detection**
- Adversarial samples generated using gradient-based perturbations
- Logistic regression classifier trained on layer-wise Mahalanobis scores
- Clean samples = consistent statistical patterns  
- Adversarial/OOD samples = deviations in feature distribution
- Classifier outputs probability + category: clean vs abnormal

#### 5.3 Disambiguation Mechanism
Differentiates OOD samples from adversarial inputs by comparing:
- Feature similarity  
- Model response patterns
Improves interpretability and enables appropriate handling per anomaly type.

#### 5.4 Secure Data Handling
Validated in-distribution data processed through encryption module to protect sensitive info during storage + transmission.

### Tech Stack
- **Language**: Python
- **Framework**: PyTorch/TensorFlow
- **Model**: Pretrained ResNet-18 
- **Detection**: Mahalanobis Distance, Logistic Regression
- **Security**: Encryption module for validated data

### Key Features
- **Model-agnostic**: Works without retraining your base model
- **Unified detection**: Handles both OOD + adversarial attacks
- **Multi-layer analysis**: Uses hierarchical features from multiple ResNet layers
- **Disambiguation**: Classifies anomaly type, not just "abnormal"
- **Secure pipeline**: Encrypts validated inputs post-detection

### Usage

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train / Calibrate Pipeline on GPU (Fine-tunes ResNet-18, builds FAISS index & Mahalanobis stats)
python train_on_gpu.py

# 3. Launch the Streamlit Web Application
streamlit run app.py
```

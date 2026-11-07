# src/security.py
import os
import logging
from datetime import datetime
from cryptography.fernet import Fernet
import config

# Setup Audit Logger
logging.basicConfig(
    filename=config.LOG_FILE_PATH,
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s"
)

def get_or_create_key():
    """Generates or retrieves the secret AES encryption key."""
    os.makedirs(os.path.dirname(config.SECRET_KEY_PATH), exist_ok=True)
    if not os.path.exists(config.SECRET_KEY_PATH):
        key = Fernet.generate_key()
        with open(config.SECRET_KEY_PATH, "wb") as f:
            f.write(key)
    else:
        with open(config.SECRET_KEY_PATH, "rb") as f:
            key = f.read()
    return Fernet(key)

def encrypt_image_bytes(raw_bytes: bytes) -> bytes:
    """Encrypts clean image bytes."""
    fernet = get_or_create_key()
    return fernet.encrypt(raw_bytes)

def log_event(status: str, confidence: float, details: str = ""):
    """Logs the verification and classification results."""
    message = f"Status: {status} | Confidence: {confidence:.4f} | Details: {details}"
    logging.info(message)
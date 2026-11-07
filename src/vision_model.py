import os
import io
import json
import base64
from typing import Dict, Any, Tuple
from PIL import Image
from groq import Groq


DEFAULT_MODEL = "qwen/qwen3.8-27b"


def get_vision_client(api_key: str = None) -> Groq:
    """Creates a vision client instance using the provided key or environment variable."""
    key = api_key or os.environ.get("GROQ_API_KEY")
    if not key:
        raise ValueError("Vision API Key is not set. Please provide a key or set it in your environment.")
    return Groq(api_key=key)


def encode_image_to_base64(image: Image.Image) -> str:
    """Converts a PIL Image to a base64-encoded JPEG string."""
    buffered = io.BytesIO()
    # Convert RGBA or grayscale to RGB for JPEG encoding
    if image.mode != "RGB":
        image = image.convert("RGB")
    image.save(buffered, format="JPEG", quality=95)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def analyze_image_with_vision_model(
    image: Image.Image,
    api_key: str = None,
    model_name: str = DEFAULT_MODEL
) -> Dict[str, Any]:
    """
    Analyzes an uploaded image using an advanced vision model to determine:
    1. Whether it is an In-Distribution Chest X-ray or Out-of-Distribution (OOD).
    2. Whether it is a Clean image, an Adversarial Attack, or OOD.
    3. Detailed radiographic & perturbation analysis.

    Returns:
        dict containing:
            - classification: 'CLEAN' | 'ADVERSARIAL_ATTACK' | 'OUT_OF_DISTRIBUTION'
            - confidence: float (0.0 to 1.0)
            - is_chest_xray: bool
            - summary: str
            - details: str
            - findings: str
    """
    client = get_vision_client(api_key)
    base64_img = encode_image_to_base64(image)

    system_prompt = (
        "You are a world-class AI Security and Medical Imaging Vision Analyst. "
        "Your task is to inspect an uploaded image and classify it into EXACTLY ONE of three categories:\n"
        "1. 'CLEAN': Authentic, unaltered in-distribution Chest X-ray (radiograph). Can be normal or show real pathological findings.\n"
        "2. 'OUT_OF_DISTRIBUTION': Any image that is NOT a human chest X-ray. This includes photos of people, animals, landscapes, "
        "objects, cartoons, memes, natural images, abstract patterns, solid colors, noise, or other non-chest-radiology scans.\n"
        "3. 'ADVERSARIAL_ATTACK': A chest X-ray that has clearly been perturbed or manipulated by an adversarial attack (e.g. FGSM, PGD, "
        "high-frequency grid/checkerboard noise, unnatural digital salt-and-pepper patterns, gradient banding, or artificial pixel distortion).\n\n"
        "You MUST respond ONLY with a valid JSON object matching this schema:\n"
        "{\n"
        '  "classification": "CLEAN" | "OUT_OF_DISTRIBUTION" | "ADVERSARIAL_ATTACK",\n'
        '  "confidence": 0.95,\n'
        '  "is_chest_xray": true,\n'
        '  "summary": "Brief 1-line verdict",\n'
        '  "details": "Detailed explanation of image characteristics, why it belongs to this category, and any visual or digital artifacts observed",\n'
        '  "findings": "If a chest X-ray, describe visible anatomy or pathology (e.g., clear lungs, cardiomegaly, opacity). If not an X-ray, describe what the image actually depicts."\n'
        "}\n"
        "Do not include markdown code ticks (```json) or commentary outside the JSON."
    )

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Analyze this image thoroughly for AI security validation, OOD detection, and adversarial perturbation."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_img}"
                            }
                        }
                    ]
                }
            ],
            temperature=0.1,
            max_tokens=600,
            response_format={"type": "json_object"}
        )

        content = response.choices[0].message.content.strip()
        data = json.loads(content)

        # Standardize classification values
        cat = data.get("classification", "").upper()
        if "CLEAN" in cat:
            data["classification"] = "CLEAN"
        elif "ADVERSARIAL" in cat:
            data["classification"] = "ADVERSARIAL_ATTACK"
        else:
            data["classification"] = "OUT_OF_DISTRIBUTION"

        # Ensure float confidence
        data["confidence"] = float(data.get("confidence", 0.90))
        return data

    except Exception as e:
        raise RuntimeError(f"Vision API Error: {str(e)}")

# app/services/pretrained_antispoof_service.py
"""
Inference wrapper for MiniFASNetV2 anti-spoofing model.
Uses pretrained weights to classify images as live vs spoof.

Model output classes (based on 2.7_80x80_MiniFASNetV2.pth):
    0 → spoof (printed/photo attack)
    1 → spoof (other attack type)
    2 → live / real

Exported function: check_spoof(image: np.ndarray) → (real_score: float, is_real: bool)
"""

import logging
from typing import Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from app.services.MiniFASNet import MiniFASNetV2
from config import Config

logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────────
# Global model (loaded once at import time)
# ────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    model = MiniFASNetV2(conv6_kernel=(5, 5)).to(DEVICE)
    state_dict = torch.load(Config.MINIFASNET_MODEL_PATH, map_location=DEVICE)
    
    # Remove 'module.' prefix if model was saved with DataParallel
    cleaned_state = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(cleaned_state, strict=True)
    
    model.eval()
    logger.info(f"MiniFASNetV2 loaded successfully on {DEVICE}")
except Exception as e:
    logger.critical(f"Failed to load MiniFASNetV2 model: {str(e)}")
    raise RuntimeError("MiniFASNetV2 model loading failed. Check model path and weights.")


INPUT_SIZE = (80, 80)
NORM_MEAN = 127.5
NORM_STD  = 128.0
REAL_CLASS_INDEX = 2


def _preprocess(image: np.ndarray) -> torch.Tensor:
    """
    Prepare image for MiniFASNetV2:
        - Resize to 80×80
        - Convert to float32
        - Normalize to [-1, 1]
        - Add batch dimension → (1, 3, 80, 80)
    """
    if image is None or image.size == 0:
        raise ValueError("Invalid input image (None or empty)")

    if len(image.shape) != 3 or image.shape[2] != 3:
        raise ValueError("Input image must be BGR 3-channel")

    # Resize
    image = cv2.resize(image, INPUT_SIZE, interpolation=cv2.INTER_LINEAR)

    # Convert and normalize
    image = image.astype(np.float32)
    image = (image - NORM_MEAN) / NORM_STD

    # HWC → CHW + batch
    image = np.transpose(image, (2, 0, 1))          # (3,80,80)
    image = np.expand_dims(image, axis=0)           # (1,3,80,80)

    return torch.from_numpy(image).float().to(DEVICE)


def check_spoof(image: np.ndarray) -> Tuple[float, bool]:
    """
    Run MiniFASNetV2 inference on a single face crop (BGR).

    Returns:
        (real_score: float [0–1], is_real: bool)
    """
    try:
        with torch.no_grad():
            input_tensor = _preprocess(image)
            logits = model(input_tensor)
            probs = F.softmax(logits, dim=1)[0]          # shape: (num_classes,)
            real_prob = float(probs[REAL_CLASS_INDEX].item())

        logger.debug(f"MiniFASNetV2 real probability: {real_prob:.4f}")

        return real_prob, real_prob > Config.ANTISPOOF_REAL_THRESHOLD

    except ValueError as ve:
        logger.warning(f"Preprocess error in anti-spoof check: {str(ve)}")
        return 0.0, False

    except RuntimeError as re:
        logger.error(f"Model inference error: {str(re)}")
        return 0.0, False

    except Exception as e:
        logger.exception("Unexpected error in pretrained anti-spoof inference")
        return 0.0, False
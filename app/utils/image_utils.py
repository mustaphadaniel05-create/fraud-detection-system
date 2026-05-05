# app/utils/image_utils.py
"""
Utility functions for image processing in the fraud detection system.

Mainly handles base64-encoded image decoding from frontend uploads.
"""

import base64
import binascii
import logging
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def decode_base64_image(base64_str: str, *, quality_check: bool = False) -> Optional[np.ndarray]:
    """
    Decode a base64-encoded image string (with or without data URI prefix)
    into an OpenCV BGR numpy array.

    Handles common formats sent from browsers:
    - Plain base64
    - Data URI: data:image/jpeg;base64,...
    - Missing padding (= characters)

    Args:
        base64_str: The base64 string from the client
        quality_check: If True, performs basic image quality checks (optional)

    Returns:
        np.ndarray: BGR image if successful, None otherwise
    """
    if not isinstance(base64_str, str) or not base64_str.strip():
        logger.debug("Empty or invalid base64 input")
        return None

    try:
        # Remove data URI prefix if present (data:image/...;base64,)
        if ',' in base64_str:
            _, base64_str = base64_str.split(',', 1)

        # Fix missing padding
        base64_str += '=' * (-len(base64_str) % 4)

        # Decode base64 to bytes
        image_bytes = base64.b64decode(base64_str)

    except (binascii.Error, ValueError) as e:
        logger.warning(f"Base64 decoding failed: {str(e)}")
        return None

    try:
        # Convert bytes to numpy array and decode as image
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            logger.warning("cv2.imdecode returned None - corrupt or unsupported image format")
            return None

        # Optional: basic quality check
        if quality_check:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            if lap_var < 50:  # very blurry image
                logger.debug(f"Low sharpness detected (Laplacian var={lap_var:.1f})")
                # You could return None here if you want to reject blurry images

        return img

    except Exception as e:
        logger.exception(f"OpenCV image decoding failed: {str(e)}")
        return None


def encode_image_to_base64(image: np.ndarray, format: str = '.jpg', quality: int = 85) -> Optional[str]:
    """
    Encode OpenCV image (BGR) back to base64 string (useful for debugging or responses).

    Args:
        image: BGR numpy array
        format: Output format ('.jpg', '.png', etc.)
        quality: JPEG quality (0–100)

    Returns:
        str: base64 string (without data URI prefix), or None on failure
    """
    try:
        success, buffer = cv2.imencode(format, image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if not success:
            return None

        return base64.b64encode(buffer).decode('utf-8')

    except Exception as e:
        logger.exception(f"Image encoding to base64 failed: {str(e)}")
        return None
"""
Face verification service using ArcFace.
STRICTER MODE – Accepts faces with similarity ≥ 0.60.
"""

import json
import logging
import tempfile
import os
from typing import Dict, Any, Optional

import cv2
import numpy as np
from deepface import DeepFace
import mediapipe as mp

# TensorFlow/Keras compatibility fix
import keras
import tf_keras
os.environ['TF_USE_LEGACY_KERAS'] = '1'

from app.db import get_connection
from config import Config

logger = logging.getLogger(__name__)

# MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
_FACE_DETECTION = mp_face_detection.FaceDetection(
    model_selection=0,
    min_detection_confidence=0.5
)

# Constants
MIN_CROP_SIZE = 120
TARGET_SIZE = (160, 160)

# STRICTER THRESHOLD – 0.60 (was 0.55)
SAME_FACE_THRESHOLD = 0.60
DIFFERENT_FACE_MAX = 0.45


def _crop_face(image: np.ndarray) -> Optional[np.ndarray]:
    """Detect and crop the largest face."""
    if image is None or image.size == 0:
        return None

    try:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = _FACE_DETECTION.process(rgb)

        if not results or not results.detections:
            return None

        detection = max(results.detections, key=lambda d: 
                       d.location_data.relative_bounding_box.width * 
                       d.location_data.relative_bounding_box.height)
        
        bbox = detection.location_data.relative_bounding_box
        h, w = image.shape[:2]
        x = max(0, int(bbox.xmin * w))
        y = max(0, int(bbox.ymin * h))
        width = min(w - x, int(bbox.width * w))
        height = min(h - y, int(bbox.height * h))

        pad_x = int(width * 0.3)
        pad_y = int(height * 0.3)

        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(w, x + width + pad_x)
        y2 = min(h, y + height + pad_y)

        cropped = image[y1:y2, x1:x2]

        if cropped.size == 0:
            return None
            
        if cropped.shape[0] < MIN_CROP_SIZE or cropped.shape[1] < MIN_CROP_SIZE:
            cropped = image[y:y+height, x:x+width]
            
        cropped = cv2.resize(cropped, TARGET_SIZE, interpolation=cv2.INTER_LINEAR)
        
        return cropped

    except Exception as e:
        logger.error(f"Face cropping error: {e}")
        return None


def _generate_embedding(image: np.ndarray) -> Optional[np.ndarray]:
    """Generate ArcFace embedding."""
    temp_path = None
    
    try:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            cv2.imwrite(tmp.name, image, [cv2.IMWRITE_JPEG_QUALITY, 95])
            temp_path = tmp.name

        representations = DeepFace.represent(
            img_path=temp_path,
            model_name="ArcFace",
            enforce_detection=False,
            detector_backend="skip",
            align=True,
            normalization="base"
        )

        if not representations:
            return None

        embedding = representations[0]["embedding"]
        
        if len(embedding) != 512:
            logger.error(f"Unexpected embedding size: {len(embedding)}")
            return None
        
        embedding_array = np.array(embedding, dtype=np.float32)
        norm = np.linalg.norm(embedding_array) + 1e-10
        embedding_array = embedding_array / norm
        
        return embedding_array

    except Exception as e:
        logger.error(f"ArcFace embedding error: {e}")
        return None
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)


def _cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """Calculate cosine similarity."""
    dot = np.dot(emb1, emb2)
    return float(dot)


def _scale_similarity(cosine_sim: float) -> float:
    """Scale from [-1, 1] to [0, 1]."""
    return (cosine_sim + 1) / 2


def verify_identity(email: str, frame: np.ndarray) -> Dict[str, Any]:
    """
    STRICTER face verification – accepts only matches with similarity ≥ 0.60.
    """
    result = {
        "success": False,
        "user_id": None,
        "similarity": 0.0,
        "raw_similarity": 0.0,
        "email_exists": False,
        "face_detected": False
    }
    
    # Validate inputs
    if not email or not email.strip():
        return result
    
    email = email.strip().lower()
    
    if frame is None or frame.size == 0:
        return result
    
    # Detect face
    cropped_face = _crop_face(frame)
    if cropped_face is None:
        logger.debug(f"No face detected for {email}")
        return result
    
    result["face_detected"] = True
    
    # Check if email exists
    try:
        with get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    "SELECT id, face_embedding FROM users WHERE email = %s",
                    (email,)
                )
                user = cursor.fetchone()
        
        if not user:
            logger.info(f"Email not registered: {email}")
            return result
        
        result["email_exists"] = True
        user_id = user["id"]
        
        stored_embedding = np.array(json.loads(user["face_embedding"]), dtype=np.float32)
        norm = np.linalg.norm(stored_embedding) + 1e-10
        stored_embedding = stored_embedding / norm
        
    except Exception as e:
        logger.error(f"Database error: {e}")
        return result
    
    # Generate live embedding
    live_embedding = _generate_embedding(cropped_face)
    if live_embedding is None:
        logger.warning(f"Failed to generate embedding for {email}")
        return result
    
    # Calculate similarity
    raw_similarity = _cosine_similarity(live_embedding, stored_embedding)
    scaled_similarity = _scale_similarity(raw_similarity)
    
    result["raw_similarity"] = round(raw_similarity, 4)
    result["similarity"] = round(scaled_similarity, 4)
    
    # Log with clear indicators
    logger.info(
        f"ArcFace | email={email} | raw={raw_similarity:.4f} | scaled={scaled_similarity:.4f} | "
        f"threshold={SAME_FACE_THRESHOLD}"
    )
    
    # STRICTER decision – accept only if similarity >= 0.60
    if scaled_similarity >= SAME_FACE_THRESHOLD:
        result["success"] = True
        result["user_id"] = user_id
        logger.info(f"✅ FACE MATCHED: {email} (sim={scaled_similarity:.3f}) - ACCEPTED")
    else:
        logger.warning(f"❌ FACE REJECTED: {email} (sim={scaled_similarity:.3f}) - below threshold {SAME_FACE_THRESHOLD}")
    
    return result
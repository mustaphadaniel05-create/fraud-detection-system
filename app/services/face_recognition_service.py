"""
Face verification service using ArcFace.
STRICT cross-user identity enforcement.

SECURITY RULES:
- enrolled face + unenrolled email = rejected
- enrolled email + unenrolled face = rejected
- enrolled face + wrong enrolled email = rejected
- claimed identity must be the strongest match
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

import keras
import tf_keras

os.environ['TF_USE_LEGACY_KERAS'] = '1'

from app.db import get_connection
from config import Config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------
# MediaPipe Face Detection
# ---------------------------------------------------------
mp_face_detection = mp.solutions.face_detection

_FACE_DETECTION = mp_face_detection.FaceDetection(
    model_selection=0,
    min_detection_confidence=0.5
)

# ---------------------------------------------------------
# Constants
# ---------------------------------------------------------
MIN_CROP_SIZE = 120
TARGET_SIZE = (160, 160)

# STRICT FACE MATCH THRESHOLD
SAME_FACE_THRESHOLD = 0.82

# Different-person threshold (kept for future use)
DIFFERENT_FACE_MAX = 0.45


# ---------------------------------------------------------
# Face Cropping
# ---------------------------------------------------------
def _crop_face(image: np.ndarray) -> Optional[np.ndarray]:

    if image is None or image.size == 0:
        return None

    try:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        results = _FACE_DETECTION.process(rgb)

        if not results or not results.detections:
            return None

        detection = max(
            results.detections,
            key=lambda d:
            d.location_data.relative_bounding_box.width *
            d.location_data.relative_bounding_box.height
        )

        bbox = detection.location_data.relative_bounding_box

        h, w = image.shape[:2]

        x = max(0, int(bbox.xmin * w))
        y = max(0, int(bbox.ymin * h))

        width = min(w - x, int(bbox.width * w))
        height = min(h - y, int(bbox.height * h))

        # Padding
        pad_x = int(width * 0.3)
        pad_y = int(height * 0.3)

        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)

        x2 = min(w, x + width + pad_x)
        y2 = min(h, y + height + pad_y)

        cropped = image[y1:y2, x1:x2]

        if cropped.size == 0:
            return None

        # fallback if too small
        if (
            cropped.shape[0] < MIN_CROP_SIZE or
            cropped.shape[1] < MIN_CROP_SIZE
        ):
            cropped = image[y:y+height, x:x+width]

        cropped = cv2.resize(
            cropped,
            TARGET_SIZE,
            interpolation=cv2.INTER_LINEAR
        )

        return cropped

    except Exception as e:
        logger.error(f"Face cropping error: {e}")
        return None


# ---------------------------------------------------------
# ArcFace Embedding
# ---------------------------------------------------------
def _generate_embedding(image: np.ndarray) -> Optional[np.ndarray]:

    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            suffix=".jpg",
            delete=False
        ) as tmp:

            cv2.imwrite(
                tmp.name,
                image,
                [cv2.IMWRITE_JPEG_QUALITY, 95]
            )

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
            logger.error(
                f"Unexpected embedding size: {len(embedding)}"
            )
            return None

        embedding_array = np.array(
            embedding,
            dtype=np.float32
        )

        embedding_array = embedding_array / (
            np.linalg.norm(embedding_array) + 1e-10
        )

        return embedding_array

    except Exception as e:
        logger.error(f"ArcFace embedding error: {e}")
        return None

    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)


# ---------------------------------------------------------
# Similarity Functions
# ---------------------------------------------------------
def _cosine_similarity(
    emb1: np.ndarray,
    emb2: np.ndarray
) -> float:

    return float(np.dot(emb1, emb2))


def _scale_similarity(cosine_sim: float) -> float:
    return (cosine_sim + 1) / 2


# ---------------------------------------------------------
# Main Identity Verification
# ---------------------------------------------------------
def verify_identity(
    email: str,
    frame: np.ndarray
) -> Dict[str, Any]:

    result = {
        "success": False,
        "user_id": None,
        "similarity": 0.0,
        "raw_similarity": 0.0,
        "email_exists": False,
        "face_detected": False,
        "matched_other_user": False,
        "best_other_similarity": 0.0,
        "best_other_email": None
    }

    # -----------------------------------------------------
    # Validate email
    # -----------------------------------------------------
    if not email or not email.strip():
        logger.warning("Empty email supplied")
        return result

    email = email.strip().lower()

    # -----------------------------------------------------
    # Validate frame
    # -----------------------------------------------------
    if frame is None or frame.size == 0:
        logger.warning(f"Invalid frame for {email}")
        return result

    # -----------------------------------------------------
    # Crop face
    # -----------------------------------------------------
    cropped_face = _crop_face(frame)

    if cropped_face is None:
        logger.warning(f"No face detected for {email}")
        return result

    result["face_detected"] = True

    # -----------------------------------------------------
    # Load claimed user + all other users
    # -----------------------------------------------------
    try:
        with get_connection() as conn:
            with conn.cursor() as cursor:

                # Claimed user
                cursor.execute(
                    """
                    SELECT id, email, face_embedding
                    FROM users
                    WHERE email = %s
                    """,
                    (email,)
                )

                claimed_user = cursor.fetchone()

                # Email not enrolled
                if not claimed_user:
                    logger.warning(
                        f"Email not registered: {email}"
                    )
                    return result

                result["email_exists"] = True

                claimed_user_id = claimed_user["id"]

                claimed_embedding = np.array(
                    json.loads(
                        claimed_user["face_embedding"]
                    ),
                    dtype=np.float32
                )

                claimed_embedding = claimed_embedding / (
                    np.linalg.norm(claimed_embedding) + 1e-10
                )

                # Load ALL other enrolled users
                cursor.execute(
                    """
                    SELECT id, email, face_embedding
                    FROM users
                    WHERE email != %s
                    """,
                    (email,)
                )

                other_users = cursor.fetchall()

    except Exception as e:
        logger.error(f"Database error: {e}")
        return result

    # -----------------------------------------------------
    # Generate live embedding
    # -----------------------------------------------------
    live_embedding = _generate_embedding(cropped_face)

    if live_embedding is None:
        logger.warning(
            f"Failed to generate embedding for {email}"
        )
        return result

    # -----------------------------------------------------
    # Compare with claimed user
    # -----------------------------------------------------
    try:
        raw_similarity = _cosine_similarity(
            live_embedding,
            claimed_embedding
        )

        scaled_similarity = _scale_similarity(
            raw_similarity
        )

    except Exception as e:
        logger.error(
            f"Claimed user comparison error: {e}"
        )
        return result

    result["raw_similarity"] = round(
        raw_similarity,
        4
    )

    result["similarity"] = round(
        scaled_similarity,
        4
    )

    logger.info(
        f"[CLAIMED USER] "
        f"email={email} | "
        f"raw={raw_similarity:.4f} | "
        f"scaled={scaled_similarity:.4f}"
    )

    # -----------------------------------------------------
    # Compare with ALL other enrolled users
    # -----------------------------------------------------
    best_other_similarity = 0.0
    best_other_email = None

    for other in other_users:

        try:
            other_embedding = np.array(
                json.loads(
                    other["face_embedding"]
                ),
                dtype=np.float32
            )

            other_embedding = other_embedding / (
                np.linalg.norm(other_embedding) + 1e-10
            )

            other_raw_similarity = _cosine_similarity(
                live_embedding,
                other_embedding
            )

            other_scaled_similarity = _scale_similarity(
                other_raw_similarity
            )

            logger.info(
                f"[OTHER USER CHECK] "
                f"candidate={other['email']} | "
                f"similarity={other_scaled_similarity:.4f}"
            )

            if other_scaled_similarity > best_other_similarity:

                best_other_similarity = (
                    other_scaled_similarity
                )

                best_other_email = other["email"]

        except Exception as e:
            logger.error(
                f"Other user comparison error "
                f"({other.get('email', 'unknown')}): {e}"
            )

    result["best_other_similarity"] = round(
        best_other_similarity,
        4
    )

    result["best_other_email"] = best_other_email

    logger.info(
        f"[BEST OTHER MATCH] "
        f"email={best_other_email} | "
        f"similarity={best_other_similarity:.4f}"
    )

    # -----------------------------------------------------
    # SECURITY RULE 1
    # Claimed user must pass threshold
    # -----------------------------------------------------
    if scaled_similarity < SAME_FACE_THRESHOLD:

        logger.warning(
            f"❌ REJECTED: "
            f"Claimed user similarity below threshold | "
            f"email={email} | "
            f"similarity={scaled_similarity:.4f} | "
            f"threshold={SAME_FACE_THRESHOLD}"
        )

        return result

    # -----------------------------------------------------
    # SECURITY RULE 2
    # Face matches another enrolled user
    # -----------------------------------------------------
    if best_other_similarity >= SAME_FACE_THRESHOLD:

        logger.warning(
            f"❌ SECURITY REJECT: "
            f"Live face matched another enrolled user | "
            f"claimed_email={email} | "
            f"matched_email={best_other_email} | "
            f"similarity={best_other_similarity:.4f}"
        )

        result["matched_other_user"] = True

        return result

    # -----------------------------------------------------
    # SECURITY RULE 3
    # Claimed user must be strongest match
    # -----------------------------------------------------
    if best_other_similarity >= scaled_similarity:

        logger.warning(
            f"❌ SECURITY REJECT: "
            f"Another user matched equal/better "
            f"than claimed user | "
            f"claimed_email={email} | "
            f"other_email={best_other_email} | "
            f"claimed_similarity={scaled_similarity:.4f} | "
            f"other_similarity={best_other_similarity:.4f}"
        )

        result["matched_other_user"] = True

        return result

    # -----------------------------------------------------
    # SUCCESS
    # -----------------------------------------------------
    result["success"] = True
    result["user_id"] = claimed_user_id

    logger.info(
        f"✅ VERIFIED SUCCESSFULLY | "
        f"email={email} | "
        f"similarity={scaled_similarity:.4f}"
    )

    return result
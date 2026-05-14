"""
Production-ready enrollment service.
Strict but realistic enrollment quality checks.

Features:
- Sharpness validation
- Brightness validation
- Contrast validation
- Strict frontal pose validation
- Stable ArcFace embedding generation
- Safer face cropping
- Better MediaPipe resource handling
"""

import json  # <-- ADDED
import logging
import tempfile
import os

from typing import Tuple, Dict, Any, Optional

import cv2
import numpy as np
import mediapipe as mp
from deepface import DeepFace

os.environ["TF_USE_LEGACY_KERAS"] = "1"

from app.db import get_connection
from app.utils.image_utils import decode_base64_image

logger = logging.getLogger(__name__)

# =========================================================
# MEDIAPIPE FACE DETECTOR
# =========================================================

mp_face_detection = mp.solutions.face_detection

face_detection = mp_face_detection.FaceDetection(
    model_selection=0,
    min_detection_confidence=0.85
)

TARGET_SIZE = (160, 160)

# =========================================================
# ENROLLMENT THRESHOLDS
# =========================================================

# Slightly relaxed from ultra-strict
# to reduce false rejection of real users
ENROLL_CLARITY_THRESHOLD = 70.0

ENROLL_BRIGHTNESS_MIN = 65
ENROLL_BRIGHTNESS_MAX = 205

ENROLL_CONTRAST_MIN = 35

ENROLL_FACE_SIZE_MIN = 20
ENROLL_FACE_SIZE_MAX = 48

# Strict frontal pose
MAX_YAW = 20
MAX_PITCH = 20
MAX_ROLL = 25

# =========================================================
# HEAD POSE ESTIMATION
# =========================================================

def _estimate_head_pose(image: np.ndarray) -> Dict[str, float]:

    try:

        mp_face_mesh = mp.solutions.face_mesh

        with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        ) as face_mesh:

            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            results = face_mesh.process(rgb)

            if not results.multi_face_landmarks:
                return {
                    "yaw": 0.0,
                    "pitch": 0.0,
                    "roll": 0.0
                }

            landmarks = results.multi_face_landmarks[0].landmark

            h, w = image.shape[:2]

            model_points = np.array([
                (0.0, 0.0, 0.0),
                (0.0, -330.0, -65.0),
                (-225.0, 170.0, -135.0),
                (225.0, 170.0, -135.0),
                (-150.0, -150.0, -125.0),
                (150.0, -150.0, -125.0)
            ], dtype=np.float64)

            idx = [1, 152, 33, 263, 61, 291]

            image_points = np.array([
                (landmarks[i].x * w, landmarks[i].y * h)
                for i in idx
            ], dtype=np.float64)

            focal_length = w

            center = (w / 2, h / 2)

            camera_matrix = np.array([
                [focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1]
            ], dtype="double")

            dist_coeffs = np.zeros((4, 1))

            success, rotation_vector, _ = cv2.solvePnP(
                model_points,
                image_points,
                camera_matrix,
                dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )

            if not success:
                return {
                    "yaw": 0.0,
                    "pitch": 0.0,
                    "roll": 0.0
                }

            rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

            sy = np.sqrt(
                rotation_matrix[0, 0] ** 2 +
                rotation_matrix[1, 0] ** 2
            )

            singular = sy < 1e-6

            if not singular:

                pitch = np.arctan2(
                    -rotation_matrix[2, 0],
                    sy
                )

                yaw = np.arctan2(
                    rotation_matrix[1, 0],
                    rotation_matrix[0, 0]
                )

                roll = np.arctan2(
                    rotation_matrix[2, 1],
                    rotation_matrix[2, 2]
                )

            else:

                pitch = np.arctan2(
                    -rotation_matrix[2, 0],
                    sy
                )

                yaw = np.arctan2(
                    -rotation_matrix[1, 2],
                    rotation_matrix[1, 1]
                )

                roll = 0

            yaw = np.degrees(yaw)
            pitch = np.degrees(pitch)
            roll = np.degrees(roll)

            # Normalize roll
            roll = (roll + 180) % 360 - 180

            if roll > 90:
                roll = 180 - roll
            elif roll < -90:
                roll = -180 - roll

            return {
                "yaw": round(float(yaw), 1),
                "pitch": round(float(pitch), 1),
                "roll": round(float(roll), 1)
            }

    except Exception as e:

        logger.error(f"Head pose estimation error: {e}")

        return {
            "yaw": 0.0,
            "pitch": 0.0,
            "roll": 0.0
        }

# =========================================================
# HEAD POSE CHECK
# =========================================================

def _check_head_pose(
    image: np.ndarray
) -> Tuple[bool, str, Dict]:

    pose = _estimate_head_pose(image)

    yaw = abs(pose["yaw"])
    pitch = abs(pose["pitch"])
    roll = abs(pose["roll"])

    ok = (
        yaw <= MAX_YAW and
        pitch <= MAX_PITCH and
        roll <= MAX_ROLL
    )

    msg = (
        f"Head pose: "
        f"yaw={pose['yaw']}°, "
        f"pitch={pose['pitch']}°, "
        f"roll={pose['roll']}°"
    )

    if not ok:
        msg = f"Please look straight at the camera. {msg}"

    return ok, msg, pose

# =========================================================
# IMAGE QUALITY CHECKS
# =========================================================

def _check_clarity(
    image: np.ndarray
) -> Tuple[bool, str, float]:

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    lap_var = cv2.Laplacian(
        gray,
        cv2.CV_64F
    ).var()

    is_clear = lap_var >= ENROLL_CLARITY_THRESHOLD

    msg = (
        f"Sharpness={lap_var:.1f} "
        f"(required ≥ {ENROLL_CLARITY_THRESHOLD})"
    )

    if not is_clear:
        msg += " — Hold camera still and improve focus."

    logger.info(
        f"Enrollment clarity: {lap_var:.1f} "
        f"-> {'PASS' if is_clear else 'FAIL'}"
    )

    return is_clear, msg, lap_var

def _check_brightness(
    image: np.ndarray
) -> Tuple[bool, str, float]:

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    brightness = float(np.mean(gray))

    if brightness < ENROLL_BRIGHTNESS_MIN:
        return (
            False,
            f"Image too dark ({brightness:.0f})",
            brightness
        )

    if brightness > ENROLL_BRIGHTNESS_MAX:
        return (
            False,
            f"Image too bright ({brightness:.0f})",
            brightness
        )

    return (
        True,
        f"Brightness OK ({brightness:.0f})",
        brightness
    )

def _check_contrast(
    image: np.ndarray
) -> Tuple[bool, str, float]:

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    contrast = float(gray.std())

    if contrast < ENROLL_CONTRAST_MIN:
        return (
            False,
            f"Low contrast ({contrast:.0f})",
            contrast
        )

    return (
        True,
        f"Contrast OK ({contrast:.0f})",
        contrast
    )

def _check_face_size(
    face_box: tuple,
    frame_shape: tuple
) -> Tuple[bool, str, float]:

    h, w = frame_shape[:2]

    _, _, face_w, _ = face_box

    face_width_pct = (face_w / w) * 100

    if face_width_pct < ENROLL_FACE_SIZE_MIN:

        return (
            False,
            f"Face too far ({face_width_pct:.0f}%)",
            face_width_pct
        )

    if face_width_pct > ENROLL_FACE_SIZE_MAX:

        return (
            False,
            f"Face too close ({face_width_pct:.0f}%)",
            face_width_pct
        )

    return (
        True,
        f"Face size OK ({face_width_pct:.0f}%)",
        face_width_pct
    )

# =========================================================
# IMAGE ENHANCEMENT
# =========================================================

def _enhance_image(image: np.ndarray) -> np.ndarray:

    if image is None:
        return image

    try:

        image = cv2.resize(image, (640, 480))

        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8)
        )

        l = clahe.apply(l)

        enhanced = cv2.cvtColor(
            cv2.merge((l, a, b)),
            cv2.COLOR_LAB2BGR
        )

        return enhanced

    except Exception as e:

        logger.error(f"Enhancement error: {e}")

        return image

# =========================================================
# FACE CROP
# =========================================================

def _crop_face(
    image: np.ndarray
) -> Tuple[Optional[np.ndarray], Optional[tuple]]:

    try:

        if image is None:
            return None, None

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        results = face_detection.process(rgb)

        if not results or not results.detections:
            return None, None

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

        width = int(bbox.width * w)
        height = int(bbox.height * h)

        width = min(width, w - x)
        height = min(height, h - y)

        if width <= 0 or height <= 0:
            return None, None

        pad = int(0.22 * max(width, height))

        x1 = max(0, x - pad)
        y1 = max(0, y - pad)

        x2 = min(w, x + width + pad)
        y2 = min(h, y + height + pad)

        cropped = image[y1:y2, x1:x2]

        if cropped.size == 0:
            return None, None

        cropped = cv2.resize(
            cropped,
            TARGET_SIZE,
            interpolation=cv2.INTER_AREA
        )

        return cropped, (x, y, width, height)

    except Exception as e:

        logger.error(f"Face crop error: {e}")

        return None, None

# =========================================================
# ARC FACE EMBEDDING
# =========================================================

def _generate_embedding(
    image: np.ndarray
) -> Optional[np.ndarray]:

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

        embedding = np.array(
            representations[0]["embedding"],
            dtype=np.float32
        )

        embedding = embedding / (
            np.linalg.norm(embedding) + 1e-10
        )

        return embedding

    except Exception as e:

        logger.error(f"Embedding generation error: {e}")

        return None

    finally:

        try:
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)
        except:
            pass

# =========================================================
# MAIN ENROLLMENT
# =========================================================

def enroll_user(
    full_name: str,
    email: str,
    image_base64: str
) -> Tuple[Dict[str, Any], int]:

    try:

        email = (email or "").strip().lower()
        full_name = (full_name or "").strip()

        if len(full_name) < 3:

            return {
                "status": "error",
                "message": "Full name too short"
            }, 400

        if not email or "@" not in email:

            return {
                "status": "error",
                "message": "Valid email required"
            }, 400

        image = decode_base64_image(image_base64)

        if image is None:

            return {
                "status": "error",
                "message": "Invalid image"
            }, 400

        enhanced = _enhance_image(image)

        cropped_face, face_box = _crop_face(enhanced)

        if cropped_face is None:

            return {
                "status": "error",
                "message": "No face detected"
            }, 400

        # =================================================
        # QUALITY CHECKS
        # =================================================

        clear, msg, _ = _check_clarity(cropped_face)

        if not clear:
            return {"status": "error", "message": msg}, 400

        ok, msg, _ = _check_face_size(
            face_box,
            enhanced.shape
        )

        if not ok:
            return {"status": "error", "message": msg}, 400

        ok, msg, _ = _check_brightness(cropped_face)

        if not ok:
            return {"status": "error", "message": msg}, 400

        ok, msg, _ = _check_contrast(cropped_face)

        if not ok:
            return {"status": "error", "message": msg}, 400

        ok, msg, _ = _check_head_pose(cropped_face)

        if not ok:
            return {"status": "error", "message": msg}, 400

        # =================================================
        # EMBEDDING
        # =================================================

        embedding = _generate_embedding(cropped_face)

        if embedding is None:

            return {
                "status": "error",
                "message": "Failed to process face"
            }, 400

        # =================================================
        # DATABASE
        # =================================================

        with get_connection() as conn:

            with conn.cursor() as cursor:

                cursor.execute(
                    "SELECT id FROM users WHERE email = %s",
                    (email,)
                )

                if cursor.fetchone():

                    return {
                        "status": "error",
                        "message": "Email already registered"
                    }, 409

                cursor.execute(
                    """
                    INSERT INTO users
                    (full_name, email, face_embedding)
                    VALUES (%s, %s, %s)
                    """,
                    (
                        full_name,
                        email,
                        json.dumps(embedding.tolist())
                    )
                )

            conn.commit()

        logger.info(f"Enrollment successful: {email}")

        return {
            "status": "success",
            "message": "Enrollment successful"
        }, 201

    except Exception as e:

        logger.exception(f"Enrollment error: {e}")

        return {
            "status": "error",
            "message": "Enrollment failed"
        }, 500
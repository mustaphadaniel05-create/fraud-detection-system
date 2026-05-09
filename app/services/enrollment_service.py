"""
Enrollment service – VERY STRICT CLARITY + VERY FORGIVING FRONTAL POSE.
Only accepts clear faces, but allows moderate head turns and tilts.
"""
import json
import logging
import tempfile
import os
from typing import Tuple, Dict, Any, Optional

import cv2
import numpy as np
from deepface import DeepFace
import mediapipe as mp

os.environ['TF_USE_LEGACY_KERAS'] = '1'

from app.db import get_connection
from app.utils.image_utils import decode_base64_image

logger = logging.getLogger(__name__)

# MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(
    model_selection=0,
    min_detection_confidence=0.80
)

TARGET_SIZE = (160, 160)

# STRICT ENROLLMENT THRESHOLDS
ENROLL_CLARITY_THRESHOLD = 60.0      # Laplacian variance – sharp image required
ENROLL_BRIGHTNESS_MIN = 60
ENROLL_BRIGHTNESS_MAX = 210
ENROLL_CONTRAST_MIN = 35
ENROLL_FACE_SIZE_MIN = 20
ENROLL_FACE_SIZE_MAX = 50

# VERY FORGIVING head pose limits (angles in degrees)
MAX_YAW = 40          # looking left/right
MAX_PITCH = 40        # looking up/down
MAX_ROLL = 70         # head tilt (after normalisation)


# ======================================================================
# HEAD POSE ESTIMATION WITH NORMALISED ROLL
# ======================================================================
def _estimate_head_pose(image: np.ndarray) -> Dict[str, float]:
    """
    Estimate yaw, pitch, roll using MediaPipe FaceMesh.
    Returns angles in degrees, with roll normalised to [-90, 90].
    """
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    )
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    if not results.multi_face_landmarks:
        return {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}

    landmarks = results.multi_face_landmarks[0].landmark
    h, w = image.shape[:2]

    # 3D model points (nose tip, chin, left eye corner, right eye corner, left mouth, right mouth)
    model_points = np.array([
        (0.0, 0.0, 0.0),          # Nose tip
        (0.0, -330.0, -65.0),     # Chin
        (-225.0, 170.0, -135.0),  # Left eye left corner
        (225.0, 170.0, -135.0),   # Right eye right corner
        (-150.0, -150.0, -125.0), # Left mouth corner
        (150.0, -150.0, -125.0)   # Right mouth corner
    ], dtype=np.float64)

    idx = [1, 152, 33, 263, 61, 291]
    image_points = np.array([
        (landmarks[i].x * w, landmarks[i].y * h) for i in idx
    ], dtype=np.float64)

    focal_length = w
    center = (w / 2, h / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype="double")
    dist_coeffs = np.zeros((4, 1))

    success, rot_vec, _ = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not success:
        return {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}

    rot_mat, _ = cv2.Rodrigues(rot_vec)
    # Extract Euler angles (pitch, yaw, roll)
    sy = np.sqrt(rot_mat[0,0]**2 + rot_mat[1,0]**2)
    singular = sy < 1e-6
    if not singular:
        pitch = np.arctan2(-rot_mat[2,0], sy)
        yaw = np.arctan2(rot_mat[1,0], rot_mat[0,0])
        roll = np.arctan2(rot_mat[2,1], rot_mat[2,2])
    else:
        pitch = np.arctan2(-rot_mat[2,0], sy)
        yaw = np.arctan2(-rot_mat[1,2], rot_mat[1,1])
        roll = 0

    # Convert to degrees
    yaw = np.degrees(yaw)
    pitch = np.degrees(pitch)
    roll = np.degrees(roll)

    # Normalise roll to the range [-90, 90]
    # If roll is near ±180°, it actually means upright (no tilt)
    roll = (roll + 180) % 360 - 180
    if roll > 90:
        roll = 180 - roll
    elif roll < -90:
        roll = -180 - roll

    return {
        "yaw": round(yaw, 1),
        "pitch": round(pitch, 1),
        "roll": round(roll, 1)
    }


def _check_head_pose(image: np.ndarray) -> Tuple[bool, str, Dict]:
    """Check if face is frontal enough (very forgiving)."""
    pose = _estimate_head_pose(image)
    yaw = abs(pose["yaw"])
    pitch = abs(pose["pitch"])
    roll = abs(pose["roll"])
    ok = (yaw <= MAX_YAW and pitch <= MAX_PITCH and roll <= MAX_ROLL)
    msg = (f"Head pose: yaw={pose['yaw']}°, pitch={pose['pitch']}°, roll={pose['roll']}°"
           f" (limits: {MAX_YAW}°, {MAX_PITCH}°, {MAX_ROLL}°)")
    if not ok:
        msg = f"Please look straight – {msg}"
    else:
        msg = f"Head position good – {msg}"
    return ok, msg, pose


# ======================================================================
# CLARITY AND BASIC QUALITY CHECKS (strict)
# ======================================================================
def _check_clarity_strict(image: np.ndarray) -> Tuple[bool, str, float]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    is_clear = lap_var >= ENROLL_CLARITY_THRESHOLD
    msg = f"Image sharpness: {lap_var:.1f} (need ≥ {ENROLL_CLARITY_THRESHOLD})"
    if not is_clear:
        msg += " – Please hold camera perfectly still and ensure good focus."
    logger.info(f"Clarity: {lap_var:.1f} -> {'PASS' if is_clear else 'FAIL'}")
    return is_clear, msg, lap_var

def _check_brightness(image: np.ndarray) -> Tuple[bool, str, float]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)
    if brightness < ENROLL_BRIGHTNESS_MIN:
        return False, f"Image too dark ({brightness:.0f}). Improve lighting.", brightness
    if brightness > ENROLL_BRIGHTNESS_MAX:
        return False, f"Image too bright ({brightness:.0f}). Reduce lighting.", brightness
    return True, f"Brightness OK ({brightness:.0f})", brightness

def _check_contrast(image: np.ndarray) -> Tuple[bool, str, float]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    contrast = gray.std()
    if contrast < ENROLL_CONTRAST_MIN:
        return False, f"Low contrast ({contrast:.0f}). Ensure even lighting.", contrast
    return True, f"Contrast OK ({contrast:.0f})", contrast

def _check_face_size(face_box: tuple, frame_shape: tuple) -> Tuple[bool, str, float]:
    h, w = frame_shape[:2]
    _, _, face_w, _ = face_box
    face_width_pct = (face_w / w) * 100
    if face_width_pct < ENROLL_FACE_SIZE_MIN:
        return False, f"Face too far ({face_width_pct:.0f}%). Move closer.", face_width_pct
    if face_width_pct > ENROLL_FACE_SIZE_MAX:
        return False, f"Face too close ({face_width_pct:.0f}%). Move back.", face_width_pct
    return True, f"Face size OK ({face_width_pct:.0f}%)", face_width_pct

def _enhance_image(image: np.ndarray) -> np.ndarray:
    if image is None: return image
    try:
        image = cv2.resize(image, (640, 480))
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        enhanced = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
        return enhanced
    except:
        return image

def _crop_face(image: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[tuple]]:
    if image is None: return None, None
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb)
    if not results or not results.detections:
        return None, None
    detection = max(results.detections, key=lambda d: 
                    d.location_data.relative_bounding_box.width * 
                    d.location_data.relative_bounding_box.height)
    bbox = detection.location_data.relative_bounding_box
    h, w = image.shape[:2]
    x = max(0, int(bbox.xmin * w))
    y = max(0, int(bbox.ymin * h))
    width = min(w - x, int(bbox.width * w))
    height = min(h - y, int(bbox.height * h))
    pad = int(0.2 * max(width, height))
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(w, x + width + pad)
    y2 = min(h, y + height + pad)
    cropped = image[y1:y2, x1:x2]
    cropped = cv2.resize(cropped, TARGET_SIZE)
    return cropped, (x, y, width, height)

def _generate_embedding(image: np.ndarray) -> Optional[np.ndarray]:
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            cv2.imwrite(tmp.name, image, [cv2.IMWRITE_JPEG_QUALITY, 95])
            temp_path = tmp.name
        reps = DeepFace.represent(img_path=temp_path, model_name="ArcFace",
                                  enforce_detection=False, detector_backend="skip",
                                  align=True, normalization="base")
        if not reps:
            return None
        emb = np.array(reps[0]["embedding"])
        emb = emb / (np.linalg.norm(emb) + 1e-10)
        return emb
    except Exception as e:
        logger.error(f"Embedding error: {e}")
        return None
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)


# ======================================================================
# MAIN ENROLLMENT FUNCTION
# ======================================================================
def enroll_user(full_name: str, email: str, image_base64: str) -> Tuple[Dict[str, Any], int]:
    email = (email or "").strip().lower()
    full_name = (full_name or "").strip()
    if len(full_name) < 3:
        return {"status": "error", "message": "Full name at least 3 characters"}, 400
    if not email or "@" not in email:
        return {"status": "error", "message": "Valid email required"}, 400

    image = decode_base64_image(image_base64)
    if image is None:
        return {"status": "error", "message": "Invalid image"}, 400

    enhanced = _enhance_image(image)
    cropped_face, face_box = _crop_face(enhanced)
    if cropped_face is None:
        return {"status": "error", "message": "No face detected. Center your face."}, 400

    # 1. Strict clarity (sharpness)
    clear, msg, _ = _check_clarity_strict(cropped_face)
    if not clear:
        return {"status": "error", "message": msg}, 400

    # 2. Face size check
    ok, msg, _ = _check_face_size(face_box, enhanced.shape)
    if not ok:
        return {"status": "error", "message": msg}, 400

    # 3. Brightness
    ok, msg, _ = _check_brightness(cropped_face)
    if not ok:
        return {"status": "error", "message": msg}, 400

    # 4. Contrast
    ok, msg, _ = _check_contrast(cropped_face)
    if not ok:
        return {"status": "error", "message": msg}, 400

    # 5. Head pose (very forgiving)
    ok, msg, pose = _check_head_pose(cropped_face)
    if not ok:
        return {"status": "error", "message": msg}, 400

    # 6. Final re-check clarity (extra safety)
    clear2, msg2, _ = _check_clarity_strict(cropped_face)
    if not clear2:
        return {"status": "error", "message": "Image became distorted – please retry"}, 400

    # Generate embedding
    embedding = _generate_embedding(cropped_face)
    if embedding is None:
        return {"status": "error", "message": "Face processing failed. Ensure good lighting."}, 400

    # Save to DB
    try:
        with get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT id FROM users WHERE email = %s", (email,))
                if cursor.fetchone():
                    return {"status": "error", "message": "Email already registered"}, 409
                cursor.execute("INSERT INTO users (full_name, email, face_embedding) VALUES (%s, %s, %s)",
                               (full_name, email, json.dumps(embedding.tolist())))
            conn.commit()
        logger.info(f"Enrollment success: {email}")
        return {"status": "success", "message": "Enrollment successful"}, 201
    except Exception as e:
        logger.exception(f"DB error: {e}")
        return {"status": "error", "message": "Database error"}, 500
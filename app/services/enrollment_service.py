"""
Enrollment service – EXTREMELY STRICT + FACE UNIQUENESS CHECK.
Rejects multi‑face images and faces already enrolled under another email.
"""

import json
import logging
import tempfile
import os
from typing import Tuple, Dict, Any, Optional, List

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
# ENROLLMENT THRESHOLDS (strict)
# =========================================================

ENROLL_CLARITY_THRESHOLD = 70.0
ENROLL_BRIGHTNESS_MIN = 65
ENROLL_BRIGHTNESS_MAX = 205
ENROLL_CONTRAST_MIN = 35
ENROLL_FACE_SIZE_MIN = 20
ENROLL_FACE_SIZE_MAX = 48

# Frontal pose limits
MAX_YAW = 20
MAX_PITCH = 20
MAX_ROLL = 25

# Face‑uniqueness threshold
FACE_UNIQUENESS_THRESHOLD = 0.75   # if similarity to any existing user > 0.75, reject

# =========================================================
# HEAD POSE ESTIMATION (same as before)
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
                return {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}
            landmarks = results.multi_face_landmarks[0].landmark
            h, w = image.shape[:2]
            model_points = np.array([
                (0.0, 0.0, 0.0), (0.0, -330.0, -65.0),
                (-225.0, 170.0, -135.0), (225.0, 170.0, -135.0),
                (-150.0, -150.0, -125.0), (150.0, -150.0, -125.0)
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
            yaw = np.degrees(yaw)
            pitch = np.degrees(pitch)
            roll = np.degrees(roll)
            roll = (roll + 180) % 360 - 180
            if roll > 90: roll = 180 - roll
            elif roll < -90: roll = -180 - roll
            return {"yaw": round(float(yaw), 1), "pitch": round(float(pitch), 1), "roll": round(float(roll), 1)}
    except Exception as e:
        logger.error(f"Head pose error: {e}")
        return {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}

def _check_head_pose(image: np.ndarray) -> Tuple[bool, str, Dict]:
    pose = _estimate_head_pose(image)
    yaw, pitch, roll = abs(pose["yaw"]), abs(pose["pitch"]), abs(pose["roll"])
    ok = (yaw <= MAX_YAW and pitch <= MAX_PITCH and roll <= MAX_ROLL)
    msg = f"Head pose: yaw={pose['yaw']}°, pitch={pose['pitch']}°, roll={pose['roll']}°"
    if not ok:
        msg = f"Please look straight at the camera. {msg}"
    return ok, msg, pose

# =========================================================
# QUALITY CHECKS
# =========================================================

def _check_clarity(image: np.ndarray) -> Tuple[bool, str, float]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    is_clear = lap_var >= ENROLL_CLARITY_THRESHOLD
    msg = f"Sharpness={lap_var:.1f} (need ≥ {ENROLL_CLARITY_THRESHOLD})"
    if not is_clear:
        msg += " — Hold camera still and improve focus."
    logger.info(f"Enrollment clarity: {lap_var:.1f} -> {'PASS' if is_clear else 'FAIL'}")
    return is_clear, msg, lap_var

def _check_brightness(image: np.ndarray) -> Tuple[bool, str, float]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    b = float(np.mean(gray))
    if b < ENROLL_BRIGHTNESS_MIN:
        return False, f"Image too dark ({b:.0f})", b
    if b > ENROLL_BRIGHTNESS_MAX:
        return False, f"Image too bright ({b:.0f})", b
    return True, f"Brightness OK ({b:.0f})", b

def _check_contrast(image: np.ndarray) -> Tuple[bool, str, float]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    c = float(gray.std())
    if c < ENROLL_CONTRAST_MIN:
        return False, f"Low contrast ({c:.0f})", c
    return True, f"Contrast OK ({c:.0f})", c

def _check_face_size(face_box: tuple, frame_shape: tuple) -> Tuple[bool, str, float]:
    h, w = frame_shape[:2]
    _, _, face_w, _ = face_box
    pct = (face_w / w) * 100
    if pct < ENROLL_FACE_SIZE_MIN:
        return False, f"Face too far ({pct:.0f}%)", pct
    if pct > ENROLL_FACE_SIZE_MAX:
        return False, f"Face too close ({pct:.0f}%)", pct
    return True, f"Face size OK ({pct:.0f}%)", pct

# =========================================================
# IMAGE ENHANCEMENT & FACE CROP
# =========================================================

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
    except Exception as e:
        logger.error(f"Enhancement error: {e}")
        return image

def _crop_face(image: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[tuple]]:
    try:
        if image is None: return None, None
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb)
        if not results or not results.detections:
            return None, None
        # Get largest face
        detection = max(results.detections, key=lambda d:
                        d.location_data.relative_bounding_box.width *
                        d.location_data.relative_bounding_box.height)
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
        cropped = cv2.resize(cropped, TARGET_SIZE, interpolation=cv2.INTER_AREA)
        return cropped, (x, y, width, height)
    except Exception as e:
        logger.error(f"Face crop error: {e}")
        return None, None

# =========================================================
# EMBEDDING & UNIQUENESS CHECK
# =========================================================

def _generate_embedding(image: np.ndarray) -> Optional[np.ndarray]:
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            cv2.imwrite(tmp.name, image, [cv2.IMWRITE_JPEG_QUALITY, 95])
            temp_path = tmp.name
        reps = DeepFace.represent(
            img_path=temp_path,
            model_name="ArcFace",
            enforce_detection=False,
            detector_backend="skip",
            align=True,
            normalization="base"
        )
        if not reps:
            return None
        emb = np.array(reps[0]["embedding"], dtype=np.float32)
        emb = emb / (np.linalg.norm(emb) + 1e-10)
        return emb
    except Exception as e:
        logger.error(f"Embedding error: {e}")
        return None
    finally:
        try:
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)
        except:
            pass

def _check_face_uniqueness(embedding: np.ndarray, exclude_email: str = None) -> Tuple[bool, Optional[str], float]:
    """
    Compare embedding against all existing users.
    Returns (is_unique, conflicting_email, highest_similarity)
    """
    try:
        with get_connection() as conn:
            with conn.cursor() as cursor:
                if exclude_email:
                    cursor.execute("SELECT email, face_embedding FROM users WHERE email != %s", (exclude_email,))
                else:
                    cursor.execute("SELECT email, face_embedding FROM users")
                users = cursor.fetchall()
        if not users:
            return True, None, 0.0
        best_sim = 0.0
        best_email = None
        for u in users:
            stored_emb = np.array(json.loads(u["face_embedding"]), dtype=np.float32)
            stored_emb = stored_emb / (np.linalg.norm(stored_emb) + 1e-10)
            sim = float(np.dot(embedding, stored_emb))  # cosine similarity (raw, -1..1)
            scaled_sim = (sim + 1) / 2  # convert to 0..1
            logger.info(f"Uniqueness check vs {u['email']}: raw={sim:.4f}, scaled={scaled_sim:.4f}")
            if scaled_sim > best_sim:
                best_sim = scaled_sim
                best_email = u["email"]
        is_unique = best_sim < FACE_UNIQUENESS_THRESHOLD
        return is_unique, best_email, best_sim
    except Exception as e:
        logger.error(f"Uniqueness check error: {e}")
        return True, None, 0.0

# =========================================================
# MULTIPLE FACE DETECTION
# =========================================================

def _has_multiple_faces(image: np.ndarray) -> bool:
    try:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb)
        if not results or not results.detections:
            return False
        # Count faces with confidence > 0.6
        count = sum(1 for d in results.detections if d.score[0] > 0.6)
        return count > 1
    except Exception as e:
        logger.error(f"Multiple face detection error: {e}")
        return False

# =========================================================
# MAIN ENROLLMENT
# =========================================================

def enroll_user(full_name: str, email: str, image_base64: str) -> Tuple[Dict[str, Any], int]:
    email = (email or "").strip().lower()
    full_name = (full_name or "").strip()
    if len(full_name) < 3:
        return {"status": "error", "message": "Full name too short"}, 400
    if not email or "@" not in email:
        return {"status": "error", "message": "Valid email required"}, 400

    image = decode_base64_image(image_base64)
    if image is None:
        return {"status": "error", "message": "Invalid image"}, 400

    # =========================================================
    # 1. MULTIPLE FACES CHECK
    # =========================================================
    if _has_multiple_faces(image):
        logger.warning(f"Multiple faces detected during enrollment for {email}")
        return {"status": "error", "message": "Multiple faces detected. Please provide a single clear face."}, 400

    enhanced = _enhance_image(image)
    cropped_face, face_box = _crop_face(enhanced)
    if cropped_face is None:
        return {"status": "error", "message": "No face detected"}, 400

    # Quality checks
    clear, msg, _ = _check_clarity(cropped_face)
    if not clear:
        return {"status": "error", "message": msg}, 400

    ok, msg, _ = _check_face_size(face_box, enhanced.shape)
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

    # Generate embedding
    embedding = _generate_embedding(cropped_face)
    if embedding is None:
        return {"status": "error", "message": "Failed to process face"}, 400

    # =========================================================
    # 2. UNIQUENESS CHECK (face not already enrolled)
    # =========================================================
    is_unique, conflicting_email, similarity = _check_face_uniqueness(embedding, exclude_email=email)
    if not is_unique:
        logger.warning(f"Face already registered with {conflicting_email} (sim={similarity:.3f})")
        return {"status": "error", "message": f"This face is already registered with email {conflicting_email}. Please use your own face."}, 409

    # Save to DB
    try:
        with get_connection() as conn:
            with conn.cursor() as cursor:
                # Check if email already exists
                cursor.execute("SELECT id FROM users WHERE email = %s", (email,))
                if cursor.fetchone():
                    return {"status": "error", "message": "Email already registered"}, 409
                cursor.execute(
                    "INSERT INTO users (full_name, email, face_embedding) VALUES (%s, %s, %s)",
                    (full_name, email, json.dumps(embedding.tolist()))
                )
            conn.commit()
        logger.info(f"Enrollment successful: {email}")
        return {"status": "success", "message": "Enrollment successful"}, 201
    except Exception as e:
        logger.exception(f"Enrollment error: {e}")
        return {"status": "error", "message": "Enrollment failed"}, 500
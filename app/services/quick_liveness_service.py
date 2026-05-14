"""
Quick liveness detection – FORGIVING FOR REAL FACES.
Requires at least 1 blink (eyes closed for 1 frame) to pass.
Uses a higher EAR threshold (0.25) for easier blink detection.
"""

import logging
from typing import List, Dict, Any

import cv2
import numpy as np
import mediapipe as mp

logger = logging.getLogger(__name__)

mp_face_mesh = mp.solutions.face_mesh
_face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# MORE FORGIVING THRESHOLDS
EAR_THRESHOLD = 0.25            # increased from 0.22
MIN_CONSECUTIVE_FRAMES = 1
REQUIRED_BLINKS = 1
STATIC_VARIATION_THRESHOLD = 0.018
MIN_FRAMES_FOR_LIVENESS = 6

def _eye_aspect_ratio(landmarks, eye_indices, frame_w, frame_h) -> float:
    points = []
    for idx in eye_indices:
        lm = landmarks[idx]
        points.append((lm.x * frame_w, lm.y * frame_h))
    A = np.linalg.norm(np.array(points[1]) - np.array(points[5]))
    B = np.linalg.norm(np.array(points[2]) - np.array(points[4]))
    C = np.linalg.norm(np.array(points[0]) - np.array(points[3]))
    return (A + B) / (2.0 * C) if C != 0 else 0

def check_liveness(frames: List[np.ndarray]) -> Dict[str, Any]:
    logger.info(f"Quick liveness: {len(frames)} frames")

    if len(frames) < MIN_FRAMES_FOR_LIVENESS:
        return {
            "is_live": False,
            "blinks_detected": 0,
            "ear_variation": 0.0,
            "reason": "Insufficient frames – please retry"
        }

    real_blinks = 0
    consecutive_closed = 0
    frames_with_face = 0
    max_consecutive = 0
    multiple_faces_detected = False
    ear_values = []

    for frame in frames:
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = _face_mesh.process(rgb)
            if not results or not results.multi_face_landmarks:
                continue
            if len(results.multi_face_landmarks) > 1:
                multiple_faces_detected = True
                logger.warning("Multiple faces detected")
            frames_with_face += 1
            landmarks = results.multi_face_landmarks[0].landmark
            h, w = frame.shape[:2]
            ear_left = _eye_aspect_ratio(landmarks, LEFT_EYE, w, h)
            ear_right = _eye_aspect_ratio(landmarks, RIGHT_EYE, w, h)
            ear = (ear_left + ear_right) / 2
            ear_values.append(ear)

            if ear < EAR_THRESHOLD:
                consecutive_closed += 1
                if consecutive_closed > max_consecutive:
                    max_consecutive = consecutive_closed
            else:
                if consecutive_closed >= MIN_CONSECUTIVE_FRAMES:
                    real_blinks += 1
                    logger.info(f"Blink detected! frames closed={consecutive_closed}")
                consecutive_closed = 0
        except Exception:
            continue

    if consecutive_closed >= MIN_CONSECUTIVE_FRAMES:
        real_blinks += 1
        logger.info(f"Blink detected at end! frames closed={consecutive_closed}")

    ear_variation = float(np.std(ear_values)) if len(ear_values) > 1 else 0.0

    logger.info(f"Liveness results: blinks={real_blinks}, ear_var={ear_variation:.4f}")

    if multiple_faces_detected:
        return {
            "is_live": False,
            "blinks_detected": real_blinks,
            "ear_variation": ear_variation,
            "reason": "Multiple faces detected"
        }

    if real_blinks >= REQUIRED_BLINKS:
        return {
            "is_live": True,
            "blinks_detected": real_blinks,
            "ear_variation": ear_variation,
            "reason": f"Live face: {real_blinks} blink(s)"
        }

    if frames_with_face > 0:
        if real_blinks == 0 and ear_variation < STATIC_VARIATION_THRESHOLD:
            return {
                "is_live": False,
                "blinks_detected": 0,
                "ear_variation": ear_variation,
                "reason": "Static photo – no eye movement"
            }
        return {
            "is_live": False,
            "blinks_detected": 0,
            "ear_variation": ear_variation,
            "reason": "No blink detected – please blink naturally"
        }

    return {
        "is_live": False,
        "blinks_detected": 0,
        "ear_variation": ear_variation,
        "reason": "No face detected"
    }
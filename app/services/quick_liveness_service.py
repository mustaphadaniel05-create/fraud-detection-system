"""
Quick liveness detection – PRODUCTION BALANCED.
Reliable for real users while still blocking:
- Static photos
- Printed images
- Frozen replay attempts

Optimized for:
- Low FPS webcams
- Mobile phones
- Poor lighting
- Real-world browser capture

Features:
- Requires at least 1 natural blink
- EAR spike filtering
- Static-photo rejection
- Natural eye-motion validation
"""

import logging
from typing import List, Dict, Any

import cv2
import numpy as np
import mediapipe as mp

logger = logging.getLogger(__name__)

# =========================================================
# MEDIAPIPE SETUP
# =========================================================

mp_face_mesh = mp.solutions.face_mesh

_face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# =========================================================
# EYE LANDMARKS
# =========================================================

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# =========================================================
# PRODUCTION THRESHOLDS
# =========================================================

# Eye closed threshold
EAR_THRESHOLD = 0.22

# One closed frame counts as blink
# Important for low FPS cameras
MIN_CONSECUTIVE_FRAMES = 1

# Require at least one blink
REQUIRED_BLINKS = 1

# Static photos usually stay below this
# Real faces naturally exceed this
STATIC_VARIATION_THRESHOLD = 0.018

# Minimum frames required
MIN_FRAMES_FOR_LIVENESS = 6

# EAR sanity filtering
# Prevents MediaPipe spikes/glitches
MIN_VALID_EAR = 0.08
MAX_VALID_EAR = 0.60

# =========================================================
# EYE ASPECT RATIO
# =========================================================

def _eye_aspect_ratio(
    landmarks,
    eye_indices,
    frame_w,
    frame_h
) -> float:

    points = []

    for idx in eye_indices:
        lm = landmarks[idx]
        points.append((
            lm.x * frame_w,
            lm.y * frame_h
        ))

    A = np.linalg.norm(
        np.array(points[1]) - np.array(points[5])
    )

    B = np.linalg.norm(
        np.array(points[2]) - np.array(points[4])
    )

    C = np.linalg.norm(
        np.array(points[0]) - np.array(points[3])
    )

    if C == 0:
        return 0.0

    return (A + B) / (2.0 * C)

# =========================================================
# MAIN LIVENESS CHECK
# =========================================================

def check_liveness(
    frames: List[np.ndarray]
) -> Dict[str, Any]:

    logger.info(
        f"Quick liveness started | frames={len(frames)}"
    )

    # =====================================================
    # MINIMUM FRAMES
    # =====================================================

    if not frames or len(frames) < MIN_FRAMES_FOR_LIVENESS:

        return {
            "is_live": False,
            "blinks_detected": 0,
            "ear_variation": 0.0,
            "reason": "Insufficient frames – please retry"
        }

    real_blinks = 0
    consecutive_closed = 0
    max_consecutive = 0

    frames_with_face = 0
    multiple_faces_detected = False

    ear_values = []

    # =====================================================
    # PROCESS FRAMES
    # =====================================================

    for i, frame in enumerate(frames):

        try:

            if frame is None or frame.size == 0:
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = _face_mesh.process(rgb)

            if not results or not results.multi_face_landmarks:
                continue

            # Extra safety
            if len(results.multi_face_landmarks) > 1:

                multiple_faces_detected = True

                logger.warning(
                    f"Multiple faces detected in frame {i}"
                )

            frames_with_face += 1

            landmarks = (
                results.multi_face_landmarks[0].landmark
            )

            h, w = frame.shape[:2]

            # =================================================
            # CALCULATE EAR
            # =================================================

            ear_left = _eye_aspect_ratio(
                landmarks,
                LEFT_EYE,
                w,
                h
            )

            ear_right = _eye_aspect_ratio(
                landmarks,
                RIGHT_EYE,
                w,
                h
            )

            ear = (ear_left + ear_right) / 2.0

            # =================================================
            # FILTER MEDIAPIPE SPIKES
            # =================================================

            if (
                ear < MIN_VALID_EAR
                or ear > MAX_VALID_EAR
            ):
                continue

            ear_values.append(ear)

            # =================================================
            # BLINK DETECTION
            # =================================================

            if ear < EAR_THRESHOLD:

                consecutive_closed += 1

                if consecutive_closed > max_consecutive:
                    max_consecutive = consecutive_closed

            else:

                if (
                    consecutive_closed >=
                    MIN_CONSECUTIVE_FRAMES
                ):

                    real_blinks += 1

                    logger.info(
                        f"Blink detected | "
                        f"closed_frames={consecutive_closed} | "
                        f"total={real_blinks}"
                    )

                consecutive_closed = 0

        except Exception as e:

            logger.error(
                f"Liveness frame error: {e}"
            )

            continue

    # =====================================================
    # HANDLE BLINK ON FINAL FRAME
    # =====================================================

    if consecutive_closed >= MIN_CONSECUTIVE_FRAMES:

        real_blinks += 1

        logger.info(
            f"Blink detected at final frame | "
            f"closed_frames={consecutive_closed}"
        )

    # =====================================================
    # EAR VARIATION
    # =====================================================

    ear_variation = (
        float(np.std(ear_values))
        if len(ear_values) > 1
        else 0.0
    )

    logger.info(
        f"Quick liveness results | "
        f"blinks={real_blinks} | "
        f"ear_variation={ear_variation:.4f} | "
        f"frames_with_face={frames_with_face}"
    )

    # =====================================================
    # MULTIPLE FACE REJECTION
    # =====================================================

    if multiple_faces_detected:

        return {
            "is_live": False,
            "blinks_detected": real_blinks,
            "ear_variation": round(
                ear_variation,
                4
            ),
            "reason": "Multiple faces detected"
        }

    # =====================================================
    # PASS — NATURAL BLINK DETECTED
    # =====================================================

    if real_blinks >= REQUIRED_BLINKS:

        return {
            "is_live": True,
            "blinks_detected": real_blinks,
            "ear_variation": round(
                ear_variation,
                4
            ),
            "reason": (
                f"Live face detected "
                f"({real_blinks} blink)"
            )
        }

    # =====================================================
    # STATIC PHOTO DETECTION
    # =====================================================

    if frames_with_face > 0:

        if (
            ear_variation <
            STATIC_VARIATION_THRESHOLD
            and real_blinks == 0
            and max_consecutive == 0
        ):

            logger.warning(
                f"Static photo suspected | "
                f"ear_variation={ear_variation:.4f}"
            )

            return {
                "is_live": False,
                "blinks_detected": 0,
                "ear_variation": round(
                    ear_variation,
                    4
                ),
                "reason": (
                    "Static photo detected "
                    "- no natural eye movement"
                )
            }

        # =================================================
        # REAL FACE BUT NO BLINK
        # =================================================

        return {
            "is_live": False,
            "blinks_detected": 0,
            "ear_variation": round(
                ear_variation,
                4
            ),
            "reason": (
                "No blink detected "
                "- please blink naturally"
            )
        }

    # =====================================================
    # NO FACE
    # =====================================================

    return {
        "is_live": False,
        "blinks_detected": 0,
        "ear_variation": round(
            ear_variation,
            4
        ),
        "reason": "No face detected"
    }
"""
Production-ready passive liveness detection.
Optimized for 6 frames, more forgiving for real faces.
"""

import logging
from typing import List, Dict, Any, Tuple

import cv2
import numpy as np
import mediapipe as mp
from scipy.signal import find_peaks

from config import Config

logger = logging.getLogger(__name__)

mp_face_mesh = mp.solutions.face_mesh

_face_mesh_instance = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

def get_face_mesh():
    return _face_mesh_instance

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# Forgiving thresholds
EAR_THRESHOLD = 0.22
EAR_MIN_CONSECUTIVE_FRAMES = 3

STANDARD_SIZE = (320, 240)
MIN_FRAMES_FOR_PASSIVE = 6

# Lower thresholds for motion and texture to make it easier for real faces
MIN_MOTION_MEAN = 0.5          # was 0.8
MIN_MOTION_STD = 0.2           # was 0.3
MIN_TEXTURE_FOR_LIVE = 15.0    # was 20.0
MIN_RPPG_SCORE = 0.02          # was 0.03

def _eye_aspect_ratio(landmarks, eye_indices):
    points = [landmarks[i] for i in eye_indices]
    A = np.linalg.norm(np.array(points[1]) - np.array(points[5]))
    B = np.linalg.norm(np.array(points[2]) - np.array(points[4]))
    C = np.linalg.norm(np.array(points[0]) - np.array(points[3]))
    return (A + B) / (2.0 * C) if C != 0 else 0

def detect_blinks(frames: List[np.ndarray]) -> Tuple[int, float]:
    face_mesh = get_face_mesh()
    blink_count = 0
    consecutive_closed = 0
    ear_values = []
    for frame in frames:
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = face_mesh.process(rgb)
            if not result.multi_face_landmarks:
                consecutive_closed = 0
                continue
            lm = result.multi_face_landmarks[0].landmark
            h, w = frame.shape[:2]
            coords = [(int(l.x * w), int(l.y * h)) for l in lm]
            ear_left = _eye_aspect_ratio(coords, LEFT_EYE)
            ear_right = _eye_aspect_ratio(coords, RIGHT_EYE)
            ear = (ear_left + ear_right) / 2
            ear_values.append(ear)
            if ear < EAR_THRESHOLD:
                consecutive_closed += 1
            else:
                if consecutive_closed >= EAR_MIN_CONSECUTIVE_FRAMES:
                    blink_count += 1
                consecutive_closed = 0
        except Exception:
            consecutive_closed = 0
    ear_variation = float(np.std(ear_values)) if len(ear_values) > 1 else 0.0
    return blink_count, ear_variation

def estimate_head_pose(frame):
    face_mesh = get_face_mesh()
    try:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb)
        if not result.multi_face_landmarks:
            return {"yaw": 0.0, "pitch": 0.0, "roll": 0.0, "magnitude": 0.0}
        landmarks = result.multi_face_landmarks[0].landmark
        h, w = frame.shape[:2]
        image_points = np.array([
            (landmarks[1].x * w, landmarks[1].y * h),
            (landmarks[33].x * w, landmarks[33].y * h),
            (landmarks[263].x * w, landmarks[263].y * h),
            (landmarks[61].x * w, landmarks[61].y * h),
            (landmarks[291].x * w, landmarks[291].y * h),
            (landmarks[199].x * w, landmarks[199].y * h)
        ], dtype="double")
        model_points = np.array([
            (0.0, 0.0, 0.0),
            (-30.0, -30.0, -30.0),
            (30.0, -30.0, -30.0),
            (-40.0, 30.0, -30.0),
            (40.0, 30.0, -30.0),
            (0.0, 60.0, -30.0)
        ])
        focal_length = w
        center = (w / 2, h / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype="double")
        success, rotation_vector, _ = cv2.solvePnP(
            model_points, image_points, camera_matrix, np.zeros((4, 1)),
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not success:
            return {"yaw": 0.0, "pitch": 0.0, "roll": 0.0, "magnitude": 0.0}
        magnitude = float(np.linalg.norm(rotation_vector))
        yaw = float(rotation_vector[1]) * 180
        pitch = float(rotation_vector[0]) * 180
        roll = float(rotation_vector[2]) * 180
        return {"yaw": yaw, "pitch": pitch, "roll": roll, "magnitude": magnitude}
    except Exception:
        return {"yaw": 0.0, "pitch": 0.0, "roll": 0.0, "magnitude": 0.0}

def _motion_energy(frames):
    if len(frames) < 2:
        return 0, 0
    gray = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    diffs = [np.mean(cv2.absdiff(gray[i-1], gray[i])) for i in range(1, len(gray))]
    if not diffs:
        return 0, 0
    motion_mean = float(np.mean(diffs))
    motion_std = float(np.std(diffs)) if len(diffs) > 1 else 0
    return motion_mean, motion_std

def _texture_score(frames):
    vals = [cv2.Laplacian(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var() for f in frames]
    return float(np.median(vals)) if vals else 0

def _simple_rppg(frames):
    try:
        greens = [np.mean(f[:, :, 1]) for f in frames if f is not None]
        if len(greens) < 6:
            return 0
        signal = np.array(greens)
        signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-8)
        peaks, _ = find_peaks(signal, distance=2, prominence=0.1)
        return min(len(peaks) / 3.0, 1.0)
    except Exception:
        return 0

def passive_liveness(frames: List[np.ndarray]) -> Tuple[Dict[str, Any], bool]:
    if not frames or len(frames) < MIN_FRAMES_FOR_PASSIVE:
        logger.warning(f"Not enough frames for passive liveness: {len(frames)}")
        return {"status": "SPOOF", "confidence": 0, "reason": "insufficient_frames"}, False

    valid_frames = []
    for f in frames:
        if f is None or f.size == 0:
            continue
        try:
            f = cv2.resize(f, STANDARD_SIZE)
            valid_frames.append(f)
        except:
            continue

    if len(valid_frames) < MIN_FRAMES_FOR_PASSIVE:
        logger.warning(f"Not enough valid frames after resize: {len(valid_frames)}")
        return {"status": "SPOOF", "confidence": 0, "reason": "invalid_frames"}, False

    blinks, ear_variation = detect_blinks(valid_frames)
    motion_mean, motion_std = _motion_energy(valid_frames)
    texture = _texture_score(valid_frames)
    rppg = _simple_rppg(valid_frames)

    logger.info(f"Passive liveness raw: blinks={blinks}, ear_var={ear_variation:.4f}, motion={motion_mean:.2f}±{motion_std:.2f}, texture={texture:.1f}, rppg={rppg:.2f}")

    live_score = 0.0
    if blinks >= 2:
        live_score += 0.3
    elif blinks >= 1:
        live_score += 0.2

    if motion_mean > MIN_MOTION_MEAN:
        live_score += 0.3
    elif motion_mean > MIN_MOTION_MEAN * 0.7:
        live_score += 0.2

    if texture > MIN_TEXTURE_FOR_LIVE:
        live_score += 0.2
    elif texture > MIN_TEXTURE_FOR_LIVE * 0.7:
        live_score += 0.1

    if rppg > MIN_RPPG_SCORE:
        live_score += 0.2
    elif rppg > MIN_RPPG_SCORE * 0.5:
        live_score += 0.1

    live_score = min(live_score, 1.0)
    status = "LIVE" if live_score >= Config.PASSIVE_LIVENESS_THRESHOLD else "SPOOF"

    result = {
        "status": status,
        "confidence": round(live_score, 3),
        "details": {
            "blinks": blinks,
            "ear_variation": round(ear_variation, 4),
            "motion_mean": round(motion_mean, 3),
            "motion_std": round(motion_std, 3),
            "texture": round(texture, 3),
            "rppg": round(rppg, 3),
        },
    }
    logger.info(f"Passive liveness result: {status} (score={live_score:.2f})")
    return result, status == "LIVE"
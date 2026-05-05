import cv2
import numpy as np
import mediapipe as mp
from typing import List, Tuple

mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# 3D model points of face
MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),          # Nose tip
    (0.0, -330.0, -65.0),     # Chin
    (-225.0, 170.0, -135.0),  # Left eye left corner
    (225.0, 170.0, -135.0),   # Right eye right corner
    (-150.0, -150.0, -125.0), # Left mouth corner
    (150.0, -150.0, -125.0)   # Right mouth corner
], dtype=np.float64)

LANDMARK_IDS = [1, 152, 33, 263, 61, 291]


def _estimate_head_pose(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    if not result.multi_face_landmarks:
        return None

    h, w = frame.shape[:2]
    landmarks = result.multi_face_landmarks[0].landmark

    image_points = []
    for idx in LANDMARK_IDS:
        lm = landmarks[idx]
        image_points.append((lm.x * w, lm.y * h))

    image_points = np.array(image_points, dtype=np.float64)

    focal_length = w
    center = (w/2, h/2)

    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype="double")

    dist_coeffs = np.zeros((4,1))

    success, rotation_vector, translation_vector = cv2.solvePnP(
        MODEL_POINTS,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        return None

    # Convert rotation vector to Euler angles
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    
    # Calculate Euler angles
    pitch = np.arctan2(-rotation_matrix[2,1], rotation_matrix[2,2]) * 180 / np.pi
    yaw = np.arctan2(rotation_matrix[2,0], np.sqrt(rotation_matrix[0,0]**2 + rotation_matrix[1,0]**2)) * 180 / np.pi
    roll = np.arctan2(rotation_matrix[1,0], rotation_matrix[0,0]) * 180 / np.pi

    return pitch, yaw, roll


def head_pose_liveness(frames: List[np.ndarray]) -> Tuple[bool, float]:
    """
    Detect natural head movement across frames.
    Photos have no head movement.
    """
    poses = []

    for frame in frames:
        pose = _estimate_head_pose(frame)
        if pose:
            poses.append(pose)

    if len(poses) < 5:  # Need enough frames for movement detection
        return False, 0.0

    pitches = [p[0] for p in poses]
    yaws = [p[1] for p in poses]
    rolls = [p[2] for p in poses]

    # Calculate movement ranges
    pitch_range = max(pitches) - min(pitches)
    yaw_range = max(yaws) - min(yaws)
    roll_range = max(rolls) - min(rolls)

    # Total movement score
    motion_score = (pitch_range + yaw_range + roll_range) / 3

    # Photos have almost zero movement
    is_live = motion_score > 4.0  # Back to original threshold

    # Confidence based on movement amount
    confidence = min(motion_score / 15.0, 1.0)

    return is_live, confidence
import cv2
import numpy as np
import mediapipe as mp
import logging

logger = logging.getLogger(__name__)

mp_face_mesh = mp.solutions.face_mesh


def micro_movement_liveness(frames):
    """
    Detect biological micro-movements that cannot be reproduced
    by phone screens or printed photos.
    """

    if len(frames) < 10:
        return False, 0.0

    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    eye_movements = []
    skin_signals = []

    prev_landmarks = None

    for frame in frames:

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb)

        if not result.multi_face_landmarks:
            continue

        landmarks = result.multi_face_landmarks[0]

        coords = np.array(
            [(lm.x, lm.y) for lm in landmarks.landmark],
            dtype=np.float32
        )

        # Eye movement detection
        left_eye = coords[33:42]
        eye_center = np.mean(left_eye)

        if prev_landmarks is not None:
            prev_eye = np.mean(prev_landmarks[33:42])
            movement = np.linalg.norm(eye_center - prev_eye)
            eye_movements.append(movement)

        prev_landmarks = coords

        # Skin color pulse (blood flow signal)
        face_roi = frame[
            int(coords[:,1].min()*frame.shape[0]):
            int(coords[:,1].max()*frame.shape[0]),
            int(coords[:,0].min()*frame.shape[1]):
            int(coords[:,0].max()*frame.shape[1])
        ]

        if face_roi.size > 0:
            mean_color = np.mean(face_roi[:, :, 1])  # green channel
            skin_signals.append(mean_color)

    if len(eye_movements) < 3:
        return False, 0.0

    eye_var = np.var(eye_movements)

    if len(skin_signals) > 5:
        pulse_var = np.var(skin_signals)
    else:
        pulse_var = 0

    score = eye_var + pulse_var

    if score > 0.002:
        return True, float(score)

    logger.warning("Micro movement liveness failed")

    return False, float(score)
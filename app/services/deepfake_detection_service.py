# app/services/deepfake_detection_service.py

import logging
from typing import Dict, Tuple, List

import cv2
import numpy as np
from scipy.fft import fft2, fftshift
from scipy.signal import find_peaks

# Try to import skimage, fall back if not available
try:
    from skimage.feature import local_binary_pattern
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False

logger = logging.getLogger(__name__)

if not SKIMAGE_AVAILABLE:
    logger.warning("scikit-image not installed. Texture-based deepfake detection will be limited.")

# ---------------------------------------------------------------------
# Detection thresholds
# ---------------------------------------------------------------------

FREQ_STD_FAKE_THRESHOLD: float = 8.0
COLOR_VAR_FAKE_THRESHOLD: float = 12.0
NOISE_STD_FAKE_THRESHOLD: float = 2.0
LBP_HIST_FAKE_THRESHOLD: float = 0.3
FOCUS_MEASURE_FAKE_THRESHOLD: float = 50.0
BLINK_RATE_FAKE_THRESHOLD: float = 0.1
MOUTH_SYNC_THRESHOLD: float = 0.3
TEMPORAL_CONSISTENCY_THRESHOLD: float = 0.25

STANDARD_SIZE = (320, 240)


# ---------------------------------------------------------------------
# Frequency anomaly detection
# ---------------------------------------------------------------------

def _detect_frequency_anomaly(image: np.ndarray) -> Tuple[bool, float]:
    """
    Detect unnatural frequency distribution.
    Deepfakes often have smoother/unnatural frequency spectra.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    f = fft2(gray)
    fshift = fftshift(f)
    magnitude = 20 * np.log(np.abs(fshift) + 1e-8)
    std_val = float(np.std(magnitude))
    suspicious = std_val < FREQ_STD_FAKE_THRESHOLD
    return suspicious, std_val


# ---------------------------------------------------------------------
# Color inconsistency detection
# ---------------------------------------------------------------------

def _detect_color_inconsistency(image: np.ndarray) -> Tuple[bool, float]:
    """
    Detect abnormal color distribution.
    Deepfakes often have unnatural color patterns.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    variance = float(np.var(hsv[:, :, 1]))
    suspicious = variance < COLOR_VAR_FAKE_THRESHOLD
    return suspicious, variance


# ---------------------------------------------------------------------
# Camera noise residual detection
# ---------------------------------------------------------------------

def _detect_noise_residual(image: np.ndarray) -> Tuple[bool, float]:
    """
    Real camera images contain sensor noise.
    Generated images often have unnaturally low residual noise.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    residual = gray - blurred
    noise_std = float(np.std(residual))
    suspicious = noise_std < NOISE_STD_FAKE_THRESHOLD
    return suspicious, noise_std


# ---------------------------------------------------------------------
# Texture analysis with LBP
# ---------------------------------------------------------------------

def _detect_texture_anomaly(image: np.ndarray) -> Tuple[bool, float]:
    """
    Detect unnatural texture patterns using Local Binary Patterns.
    Deepfakes often have repeating or unnatural texture patterns.
    """
    if not SKIMAGE_AVAILABLE:
        return False, 0.0
    
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply LBP
        radius = 1
        n_points = 8 * radius
        lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
        
        # Calculate histogram of LBP
        hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), density=True)
        
        # Check histogram distribution
        hist_entropy = -np.sum(hist * np.log2(hist + 1e-10))
        suspicious = hist_entropy < LBP_HIST_FAKE_THRESHOLD
        return suspicious, hist_entropy
    except Exception as e:
        logger.debug(f"Texture analysis error: {e}")
        return False, 0.0


# ---------------------------------------------------------------------
# Focus measure (blur detection)
# ---------------------------------------------------------------------

def _detect_focus_anomaly(image: np.ndarray) -> Tuple[bool, float]:
    """
    Detect unnatural focus patterns using Laplacian variance.
    Deepfakes often have unnatural blur patterns.
    """
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        focus_measure = float(lap.var())
        suspicious = focus_measure < FOCUS_MEASURE_FAKE_THRESHOLD
        return suspicious, focus_measure
    except Exception as e:
        logger.debug(f"Focus analysis error: {e}")
        return False, 0.0


# ---------------------------------------------------------------------
# Eye blink detection for video sequences
# ---------------------------------------------------------------------

def _detect_unnatural_blinks(frames: List[np.ndarray]) -> Tuple[bool, float]:
    """
    Detect unnatural blink patterns.
    Deepfake videos often have unnatural blink rates or missing blinks.
    """
    if len(frames) < 30:
        return False, 0.0
    
    try:
        import mediapipe as mp
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        LEFT_EYE = [33, 160, 158, 133, 153, 144]
        RIGHT_EYE = [362, 385, 387, 263, 373, 380]
        
        def eye_aspect_ratio(landmarks, eye_indices, w, h):
            points = [(landmarks[i].x * w, landmarks[i].y * h) for i in eye_indices]
            A = np.linalg.norm(np.array(points[1]) - np.array(points[5]))
            B = np.linalg.norm(np.array(points[2]) - np.array(points[4]))
            C = np.linalg.norm(np.array(points[0]) - np.array(points[3]))
            return (A + B) / (2.0 * C) if C != 0 else 0
        
        ear_values = []
        for frame in frames[::3]:  # Sample every 3rd frame
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)
            if results and results.multi_face_landmarks:
                h, w = frame.shape[:2]
                lm = results.multi_face_landmarks[0].landmark
                ear_left = eye_aspect_ratio(lm, LEFT_EYE, w, h)
                ear_right = eye_aspect_ratio(lm, RIGHT_EYE, w, h)
                ear_values.append((ear_left + ear_right) / 2)
        
        if len(ear_values) < 5:
            return False, 0.0
        
        # Detect blinks (EAR < 0.2)
        blinks = sum(1 for ear in ear_values if ear < 0.2)
        blink_rate = blinks / len(ear_values)
        
        suspicious = blink_rate < BLINK_RATE_FAKE_THRESHOLD or blink_rate > 0.5
        return suspicious, blink_rate
        
    except Exception as e:
        logger.debug(f"Blink detection error: {e}")
        return False, 0.0


# ---------------------------------------------------------------------
# Mouth movement analysis
# ---------------------------------------------------------------------

def _detect_mouth_sync_anomaly(frames: List[np.ndarray]) -> Tuple[bool, float]:
    """
    Detect unnatural mouth movement patterns.
    Deepfakes often have poor lip sync or unnatural mouth movements.
    """
    if len(frames) < 30:
        return False, 0.0
    
    try:
        import mediapipe as mp
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        mouth_aspect_ratios = []
        
        for frame in frames[::3]:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)
            if results and results.multi_face_landmarks:
                h, w = frame.shape[:2]
                lm = results.multi_face_landmarks[0].landmark
                
                # Upper and lower lip
                upper_lip = (lm[13].y * h, lm[13].x * w)
                lower_lip = (lm[14].y * h, lm[14].x * w)
                mouth_height = abs(upper_lip[0] - lower_lip[0])
                
                # Mouth width
                left_mouth = (lm[61].x * w, lm[61].y * h)
                right_mouth = (lm[291].x * w, lm[291].y * h)
                mouth_width = np.linalg.norm(np.array(left_mouth) - np.array(right_mouth))
                
                mar = mouth_height / mouth_width if mouth_width > 0 else 0
                mouth_aspect_ratios.append(mar)
        
        if len(mouth_aspect_ratios) < 5:
            return False, 0.0
        
        # Check for unnatural variation
        variation = np.std(mouth_aspect_ratios)
        suspicious = variation < MOUTH_SYNC_THRESHOLD
        
        return suspicious, variation
        
    except Exception as e:
        logger.debug(f"Mouth sync detection error: {e}")
        return False, 0.0


# ---------------------------------------------------------------------
# Temporal consistency
# ---------------------------------------------------------------------

def _detect_temporal_inconsistency(frames: List[np.ndarray]) -> Tuple[bool, float]:
    """
    Detect temporal inconsistencies between frames.
    Deepfakes often have frame-to-frame artifacts.
    """
    if len(frames) < 5:
        return False, 0.0
    
    try:
        differences = []
        prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        
        for i in range(1, min(len(frames), 30)):
            curr_gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
            diff = cv2.absdiff(curr_gray, prev_gray)
            differences.append(np.mean(diff))
            prev_gray = curr_gray
        
        if len(differences) < 4:
            return False, 0.0
        
        # Check for unnatural frame-to-frame variation
        variation = np.std(differences)
        suspicious = variation > TEMPORAL_CONSISTENCY_THRESHOLD * 100
        
        return suspicious, variation
        
    except Exception as e:
        logger.debug(f"Temporal consistency error: {e}")
        return False, 0.0


# ---------------------------------------------------------------------
# Single frame deepfake detection
# ---------------------------------------------------------------------

def detect_deepfake(image: np.ndarray) -> Dict[str, any]:
    """
    Run deepfake detection on a single frame.
    Returns dictionary with vote ratio and diagnostic scores.
    """
    if image is None or image.size == 0:
        logger.warning("Deepfake detection received empty image")
        return {
            "is_fake": False,
            "confidence": 0.0,
            "votes": 0,
            "total_votes": 0,
            "frequency_score": 0.0,
            "color_score": 0.0,
            "noise_score": 0.0,
            "texture_score": 0.0,
            "focus_score": 0.0,
            "frequency_suspicious": False,
            "color_suspicious": False,
            "noise_suspicious": False,
            "texture_suspicious": False,
            "focus_suspicious": False,
        }

    try:
        image = cv2.resize(image, STANDARD_SIZE)
        
        # Run all detection methods
        freq_fake, freq_val = _detect_frequency_anomaly(image)
        color_fake, color_val = _detect_color_inconsistency(image)
        noise_fake, noise_val = _detect_noise_residual(image)
        texture_fake, texture_val = _detect_texture_anomaly(image)
        focus_fake, focus_val = _detect_focus_anomaly(image)
        
        # Count votes (5 methods)
        votes = sum([freq_fake, color_fake, noise_fake, texture_fake, focus_fake])
        total_votes = 5
        
        # Weighted confidence
        confidence = votes / total_votes
        
        result = {
            "is_fake": bool(votes >= 3),  # 3+ votes = fake
            "confidence": round(confidence, 3),
            "votes": int(votes),
            "total_votes": total_votes,
            "frequency_score": round(freq_val, 3),
            "color_score": round(color_val, 3),
            "noise_score": round(noise_val, 3),
            "texture_score": round(texture_val, 3),
            "focus_score": round(focus_val, 3),
            "frequency_suspicious": bool(freq_fake),
            "color_suspicious": bool(color_fake),
            "noise_suspicious": bool(noise_fake),
            "texture_suspicious": bool(texture_fake),
            "focus_suspicious": bool(focus_fake),
        }

        logger.info(
            f"Deepfake detection | votes={votes}/{total_votes} "
            f"freq={freq_val:.2f} color={color_val:.2f} noise={noise_val:.2f} "
            f"texture={texture_val:.2f} focus={focus_val:.2f}"
        )

        return result

    except Exception as e:
        logger.exception(f"Deepfake detection failed: {str(e)}")
        return {
            "is_fake": False,
            "confidence": 0.0,
            "votes": 0,
            "total_votes": 5,
            "frequency_score": 0.0,
            "color_score": 0.0,
            "noise_score": 0.0,
            "texture_score": 0.0,
            "focus_score": 0.0,
            "frequency_suspicious": False,
            "color_suspicious": False,
            "noise_suspicious": False,
            "texture_suspicious": False,
            "focus_suspicious": False,
        }


# ---------------------------------------------------------------------
# Video / frame sequence deepfake analysis
# ---------------------------------------------------------------------

def analyze_frame_sequence(frames: List[np.ndarray]) -> Dict[str, any]:
    """
    Run deepfake detection across multiple frames.
    Includes temporal analysis for video deepfakes.
    """
    if not frames:
        return {
            "is_fake": False,
            "confidence": 0.0,
            "vote_ratio": 0.0,
            "frames_analyzed": 0,
            "detection_methods": []
        }
    
    fake_votes = 0
    total_votes = 0
    single_frame_results = []
    
    # Analyze individual frames
    for frame in frames[:20]:  # Analyze first 20 frames
        result = detect_deepfake(frame)
        single_frame_results.append(result)
        fake_votes += result["votes"]
        total_votes += result["total_votes"]
    
    # Additional video-specific checks
    temporal_fake, temporal_score = _detect_temporal_inconsistency(frames[:30])
    blink_fake, blink_rate = _detect_unnatural_blinks(frames[:60])
    mouth_fake, mouth_variation = _detect_mouth_sync_anomaly(frames[:60])
    
    # Add video-specific votes
    video_votes = 0
    if temporal_fake:
        video_votes += 1
    if blink_fake:
        video_votes += 1
    if mouth_fake:
        video_votes += 1
    
    total_votes += 3
    fake_votes += video_votes
    
    vote_ratio = fake_votes / total_votes if total_votes > 0 else 0.0
    
    # Calculate confidence with video checks weighted higher
    is_fake = vote_ratio >= 0.4  # Lower threshold for video detection
    
    # Determine detection method that caught it
    detection_methods = []
    if temporal_fake:
        detection_methods.append("temporal_inconsistency")
    if blink_fake:
        detection_methods.append("unnatural_blink_rate")
    if mouth_fake:
        detection_methods.append("mouth_sync_anomaly")
    
    # Add single-frame detection methods
    if any(r.get("frequency_suspicious", False) for r in single_frame_results[:5]):
        detection_methods.append("frequency_anomaly")
    if any(r.get("color_suspicious", False) for r in single_frame_results[:5]):
        detection_methods.append("color_inconsistency")
    if any(r.get("noise_suspicious", False) for r in single_frame_results[:5]):
        detection_methods.append("noise_anomaly")
    if any(r.get("texture_suspicious", False) for r in single_frame_results[:5]):
        detection_methods.append("texture_anomaly")
    if any(r.get("focus_suspicious", False) for r in single_frame_results[:5]):
        detection_methods.append("focus_anomaly")
    
    logger.info(
        f"Deepfake sequence analysis | frames={len(frames)} | "
        f"vote_ratio={vote_ratio:.3f} | is_fake={is_fake} | "
        f"detected_by={detection_methods[:3]}"
    )

    return {
        "is_fake": bool(is_fake),
        "confidence": round(vote_ratio, 3),
        "vote_ratio": round(vote_ratio, 3),
        "frames_analyzed": len(frames),
        "detection_methods": detection_methods[:3],
        "temporal_score": round(temporal_score, 3),
        "blink_rate": round(blink_rate, 3),
        "mouth_variation": round(mouth_variation, 3)
    }
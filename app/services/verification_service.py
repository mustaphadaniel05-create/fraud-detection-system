"""
Production-ready face verification service - FRIENDLY MESSAGES
Returns specific, helpful errors for real-face issues.
Deepfake messages are generic "AI-generated face detected."
"""

import logging
import json
from typing import Dict, Any, Tuple, List, Optional

import cv2
import numpy as np
from flask import request
import mediapipe as mp

from app.db import get_connection
from app.utils.image_utils import decode_base64_image

from app.services.liveness_service import passive_liveness
from app.services.face_recognition_service import verify_identity
from app.services.antispoof_service import AntiSpoofService
from app.services.quick_liveness_service import check_liveness as quick_liveness_check
from app.services.xception_deepfake_service import detect_deepfake_advanced
from app.services.face_swap_detection_service import detect_face_swap
from app.services.attempt_tracker_service import record_attempt, is_blocked
from app.services.fraud_engine import calculate_risk, decide
from app.services.email_alert_service import send_fraud_alert
from app.services.quality_service import QualityService
from app.services.jwt_service import get_jwt_service

from config import Config

logger = logging.getLogger(__name__)

_ANTISPOOF_SERVICE = AntiSpoofService()

mp_face_detection = mp.solutions.face_detection
_FACE_DETECTION = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

# ----------------------------------------------------------------------
# Utility functions (unchanged)
# ----------------------------------------------------------------------
def _detect_faces(frame: np.ndarray) -> List:
    if frame is None or frame.size == 0: return []
    try:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = _FACE_DETECTION.process(rgb)
        return results.detections if results.detections else []
    except Exception:
        return []

def _has_face(frames: List[np.ndarray]) -> bool:
    return any(len(_detect_faces(f)) > 0 for f in frames)

def _has_multiple_faces(frames: List[np.ndarray]) -> bool:
    for frame in frames:
        detections = _detect_faces(frame)
        high_conf = [d for d in detections if d.score[0] > 0.6]
        if len(high_conf) > 1: return True
    return False

def _frame_motion_score(frames: List[np.ndarray]) -> float:
    if len(frames) < 3: return 0.0
    try:
        gray = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
        diffs = [np.mean(cv2.absdiff(gray[i-1], gray[i])) for i in range(1, len(gray))]
        return float(np.mean(diffs)) if diffs else 0.0
    except Exception:
        return 0.0

def _get_best_frame(frames: List[np.ndarray]) -> Optional[np.ndarray]:
    best = None
    best_size = 0
    for f in frames:
        dets = _detect_faces(f)
        if dets:
            h, w = f.shape[:2]
            det = max(dets, key=lambda d: d.location_data.relative_bounding_box.width * d.location_data.relative_bounding_box.height)
            bbox = det.location_data.relative_bounding_box
            size = bbox.width * bbox.height
            if size > best_size:
                best_size = size
                best = f
    return best

def _get_face_size_percentage(frame: np.ndarray, detections: List) -> Optional[float]:
    if not detections: return None
    h, w = frame.shape[:2]
    det = max(detections, key=lambda d: d.location_data.relative_bounding_box.width * d.location_data.relative_bounding_box.height)
    bbox = det.location_data.relative_bounding_box
    face_width = int(bbox.width * w)
    return (face_width / w) * 100

# ========== UPDATED: Now includes risk_score and details ==========
def _log_verification(user_id: Optional[int], similarity: float, status: str, risk_score: int = 0, details: Dict = None) -> None:
    try:
        details_json = json.dumps(details) if details else None
        with get_connection() as conn:
            with conn.cursor() as cursor:
                if user_id is None or user_id == 0:
                    cursor.execute(
                        "INSERT INTO verification_logs (user_id, similarity_score, status, risk_score, details, created_at) VALUES (NULL, %s, %s, %s, %s, NOW())",
                        (similarity, status, risk_score, details_json)
                    )
                else:
                    cursor.execute(
                        "INSERT INTO verification_logs (user_id, similarity_score, status, risk_score, details, created_at) VALUES (%s, %s, %s, %s, %s, NOW())",
                        (user_id, similarity, status, risk_score, details_json)
                    )
                conn.commit()
    except Exception as e:
        logger.error(f"Logging error: {e}")

def _check_and_send_alerts(email, client_ip, status, reason, risk_score, attempts, similarity, liveness_conf, antispoof_conf, deepfake_score=0):
    if status == "VERIFIED": return
    alert_data = {
        'email': email, 'ip_address': client_ip, 'status': status, 'reason': reason,
        'risk_score': risk_score, 'attempt_number': attempts,
        'similarity': round(similarity, 3) if similarity else 0,
        'liveness_confidence': round(liveness_conf, 2) if liveness_conf else 0,
        'antispoof_confidence': round(antispoof_conf, 2) if antispoof_conf else 0,
        'deepfake_score': round(deepfake_score, 3) if deepfake_score else 0,
    }
    if risk_score and risk_score > Config.ALERT_RISK_THRESHOLD:
        alert_data['alert_type'] = 'high_risk'
        send_fraud_alert(alert_data)
    elif status in ["SPOOF", "DEEPFAKE"]:
        alert_data['alert_type'] = 'spoof_attack'
        send_fraud_alert(alert_data)
    elif attempts >= Config.ALERT_ATTEMPT_THRESHOLD:
        alert_data['alert_type'] = 'multiple_attempts'
        send_fraud_alert(alert_data)

def _check_email_exists(email: str) -> Tuple[bool, Optional[int], Optional[str]]:
    try:
        with get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT id, full_name FROM users WHERE email = %s", (email,))
                user = cursor.fetchone()
                return (True, user["id"], user["full_name"]) if user else (False, None, None)
    except Exception as e:
        logger.error(f"Email check error: {e}")
        return False, None, None

# ----------------------------------------------------------------------
# DEEPFAKE DETECTION METHODS (same as before)
# ----------------------------------------------------------------------
def _analyze_temporal_consistency(frames: List[np.ndarray]) -> Tuple[bool, float, str]:
    if len(frames) < 4: return False, 0.0, "insufficient_frames"
    sub = frames[:4][::2]
    if len(sub) < 2: return False, 0.0, "insufficient_data"
    luminance_vars = []
    for i in range(1, len(sub)):
        prev_gray = cv2.cvtColor(sub[i-1], cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(sub[i], cv2.COLOR_BGR2GRAY)
        luminance_vars.append(np.mean(cv2.absdiff(prev_gray, curr_gray)))
    if len(luminance_vars) < 1: return False, 0.0, "insufficient_data"
    motion_variance = float(np.std(luminance_vars)) if len(luminance_vars) > 1 else 0
    motion_mean = float(np.mean(luminance_vars))
    if motion_variance < 0.20 and motion_mean < 0.35:
        return True, 0.85, "AI-smoothed motion detected"
    return False, 0.0, "normal"

def _check_face_consistency(frames: List[np.ndarray]) -> Tuple[bool, float, str]:
    if len(frames) < 4: return False, 0.0, "insufficient_frames"
    sub = frames[:4][::2]
    if len(sub) < 2: return False, 0.0, "insufficient_landmarks"
    try:
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)
        nose = []
        eye = []
        for frame in sub:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = face_mesh.process(rgb)
            if res and res.multi_face_landmarks:
                lm = res.multi_face_landmarks[0].landmark
                nose.append((lm[1].x, lm[1].y))
                le = lm[33]; re = lm[263]
                eye.append(((le.x+re.x)/2, (le.y+re.y)/2))
        if len(nose) < 2: return False, 0.0, "insufficient_landmarks"
        nose_mov = [abs(nose[i][0]-nose[i-1][0])+abs(nose[i][1]-nose[i-1][1]) for i in range(1,len(nose))]
        eye_mov = [abs(eye[i][0]-eye[i-1][0])+abs(eye[i][1]-eye[i-1][1]) for i in range(1,len(eye))]
        if len(nose_mov) < 1: return False, 0.0, "insufficient_movement"
        corr = np.corrcoef(nose_mov, eye_mov)[0,1] if len(nose_mov)>1 else 0
        if corr < -0.7 or corr > 0.98:
            return True, 0.80, "Unnatural face part movement - AI generated"
        return False, 0.0, "normal"
    except Exception:
        return False, 0.0, "error"

def _check_frequency_anomalies(frame: np.ndarray) -> Tuple[bool, float]:
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        mag = 20 * np.log(np.abs(fshift) + 1e-8)
        h, w = mag.shape
        centre = mag[h//3:2*h//3, w//3:2*w//3]
        mean_freq = np.mean(centre)
        std_freq = np.std(centre)
        if mean_freq < 10 or std_freq < 4:
            return True, 0.75
        return False, 0.0
    except Exception:
        return False, 0.0

# ----------------------------------------------------------------------
# MAIN VERIFICATION FUNCTION with friendly deepfake messages
# ----------------------------------------------------------------------
def verify_user(data: Dict[str, Any]) -> Tuple[Dict[str, Any], int]:
    email = data.get("email", "").strip().lower()
    frames_b64 = data.get("frames", [])
    client_ip = request.headers.get("X-Forwarded-For", request.remote_addr)
    if client_ip:
        client_ip = client_ip.split(",")[0].strip()

    logger.info(f"Verification request: {email}, frames={len(frames_b64)}")

    # STEP 1: email validation
    if not email:
        return {"status": "ERROR", "reason": "Email is required"}, 400
    email_exists, user_id_from_db, full_name_from_db = _check_email_exists(email)
    if not email_exists:
        logger.warning(f"Email not registered: {email}")
        _log_verification(None, 0.0, "REJECTED", risk_score=0, details={"reason": "Email not registered"})
        return {"status": "REJECTED", "reason": "Email not registered. Please enroll first."}, 200

    # STEP 2: rate limiting
    attempts = record_attempt(client_ip)
    if is_blocked(client_ip):
        return {"status": "ERROR", "reason": "Too many attempts. Please try again later."}, 429

    # STEP 3: decode frames
    frames = []
    for b64 in frames_b64:
        frame = decode_base64_image(b64)
        if frame is not None:
            frames.append(frame)
    if len(frames) < Config.MIN_FRAMES_REQUIRED:
        return {"status": "ERROR", "reason": f"Need at least {Config.MIN_FRAMES_REQUIRED} frames"}, 400

    # STEP 4: black frame check
    black_count = 0
    for f in frames[:3]:
        if QualityService.is_black_frame(f): black_count += 1
    if black_count > len(frames[:3])//2:
        return {"status": "ERROR", "reason": "Camera is covered. Please uncover your camera."}, 200

    # STEP 5: face detection
    if not _has_face(frames):
        return {"status": "ERROR", "reason": "No face detected. Please look directly at the camera."}, 200

    best_frame = _get_best_frame(frames)
    if best_frame is None:
        return {"status": "ERROR", "reason": "Could not detect a valid face. Please ensure your face is centered."}, 200

    # STEP 6-7: face size & clarity
    detections = _detect_faces(best_frame)
    face_width_pct = None
    if detections:
        h, w = best_frame.shape[:2]
        det = max(detections, key=lambda d: d.location_data.relative_bounding_box.width * d.location_data.relative_bounding_box.height)
        bbox = det.location_data.relative_bounding_box
        face_width = int(bbox.width * w)
        face_width_pct = (face_width / w) * 100
        logger.info(f"Face size: {face_width_pct:.1f}%")
        if face_width_pct < 18:
            return {"status": "REJECTED", "reason": "Your face is too far from the camera. Please move closer."}, 200
        x = int(bbox.xmin * w); y = int(bbox.ymin * h)
        fw = int(bbox.width * w); fh = int(bbox.height * h)
        face_region = best_frame[y:y+fh, x:x+fw]
        if face_region.size > 0:
            clarity = cv2.Laplacian(cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
            if clarity < 35:
                return {"status": "REJECTED", "reason": "Your face is not clear. Please ensure good lighting and hold the camera steady."}, 200
    else:
        return {"status": "ERROR", "reason": "No face detected. Please look at the camera."}, 200

    # STEP 8: image quality (brightness, contrast, etc.)
    good_quality, quality_reason, quality_score = QualityService.check_image_quality(best_frame)
    if not good_quality:
        # Map the quality reason to a friendly message
        if "dark" in quality_reason.lower():
            friendly = "Image is too dark. Please turn on more lights."
        elif "bright" in quality_reason.lower():
            friendly = "Image is too bright. Please move away from direct light."
        elif "blurry" in quality_reason.lower():
            friendly = "Image is blurry. Please hold the camera steady."
        elif "glare" in quality_reason.lower():
            friendly = "Too much glare on your face. Please avoid bright lights behind you."
        else:
            friendly = quality_reason
        return {"status": "REJECTED", "reason": friendly}, 200
    if quality_score < 35:
        return {"status": "REJECTED", "reason": "Face quality is too low. Please ensure your face is well-lit and clear."}, 200

    # STEP 9: multiple faces
    if _has_multiple_faces(frames):
        return {"status": "REJECTED", "reason": "Multiple faces detected. Only one person allowed at a time."}, 200

    # STEP 10: quick liveness (blink)
    try:
        liveness_check = quick_liveness_check(frames)
        if not liveness_check.get("is_live", False):
            reason = liveness_check.get("reason", "")
            if "blink" in reason.lower():
                return {"status": "REJECTED", "reason": "No blink detected. Please blink naturally during the capture."}, 200
            else:
                return {"status": "SPOOF", "reason": "Liveness check failed. Please use your real face."}, 200
    except Exception as e:
        logger.error(f"Quick liveness error: {e}")

    # STEP 11: face recognition
    face_result = verify_identity(email, best_frame)
    if not face_result.get("face_detected", False):
        return {"status": "REJECTED", "reason": "No face detected in the captured image. Please look at the camera."}, 200
    if not face_result.get("success", False):
        similarity_score = face_result.get("similarity", 0.0)
        return {"status": "REJECTED", "reason": "Face does not match the registered user. Please use your own face."}, 200
    user_id = face_result["user_id"]
    similarity = face_result["similarity"]
    logger.info(f"Face matched: {email} (sim={similarity:.3f})")

    # ========== DEEPFAKE CHECKS – all return friendly message ==========

    # Frequency anomaly
    is_freq, freq_score = _check_frequency_anomalies(best_frame)
    if is_freq:
        logger.warning(f"Frequency anomaly: score={freq_score}")
        _log_verification(user_id, 0.0, "DEEPFAKE", risk_score=85, details={"reason": "frequency_anomaly"})
        _check_and_send_alerts(email, client_ip, "DEEPFAKE", "frequency anomaly", 85, attempts, similarity, 0.7, 0.85, freq_score)
        return {"status": "DEEPFAKE", "reason": "AI-generated face detected."}, 200

    # Temporal consistency
    is_temp, temp_score, temp_reason = _analyze_temporal_consistency(frames)
    if is_temp:
        logger.warning(f"Temporal anomaly: {temp_reason}")
        _log_verification(user_id, 0.0, "DEEPFAKE", risk_score=85, details={"reason": temp_reason})
        _check_and_send_alerts(email, client_ip, "DEEPFAKE", temp_reason, 85, attempts, similarity, 0.7, 0.85, temp_score)
        return {"status": "DEEPFAKE", "reason": "AI-generated face detected."}, 200

    # Face-swap
    try:
        swap = detect_face_swap(best_frame)
        if swap.get("is_fake", False):
            conf = swap.get("confidence", 0)
            reasons = swap.get("reasons", [])
            logger.warning(f"Face-swap detected: conf={conf}, reasons={reasons}")
            _log_verification(user_id, 0.0, "DEEPFAKE", risk_score=85, details={"reason": "face_swap"})
            _check_and_send_alerts(email, client_ip, "DEEPFAKE", "face_swap", 85, attempts, similarity, 0.7, 0.85, conf)
            return {"status": "DEEPFAKE", "reason": "AI-generated face detected."}, 200
    except Exception as e:
        logger.error(f"Face-swap error: {e}")

    # XceptionNet deepfake
    deepfake_confidence = 0.15
    try:
        deepfake_result = detect_deepfake_advanced(best_frame)
        if deepfake_result:
            deepfake_confidence = deepfake_result.get("confidence", 0.15)
            if deepfake_confidence > Config.DEEPFAKE_THRESHOLD:
                logger.warning(f"XceptionNet deepfake: conf={deepfake_confidence}")
                _log_verification(user_id, 0.0, "DEEPFAKE", risk_score=85, details={"deepfake_confidence": deepfake_confidence})
                _check_and_send_alerts(email, client_ip, "DEEPFAKE", "XceptionNet deepfake", 85, attempts, similarity, 0.7, 0.85, deepfake_confidence)
                return {"status": "DEEPFAKE", "reason": "AI-generated face detected."}, 200
    except Exception as e:
        logger.error(f"XceptionNet error: {e}")

    # Face consistency
    is_incon, cons_score, cons_reason = _check_face_consistency(frames)
    if is_incon:
        logger.warning(f"Face inconsistency: {cons_reason}")
        _log_verification(user_id, similarity, "DEEPFAKE", risk_score=85, details={"reason": cons_reason})
        _check_and_send_alerts(email, client_ip, "DEEPFAKE", cons_reason, 85, attempts, similarity, 0.7, 0.85, cons_score)
        return {"status": "DEEPFAKE", "reason": "AI-generated face detected."}, 200

    # Blink analysis (part of deepfake detection)
    try:
        liveness_check = quick_liveness_check(frames)
        blinks = liveness_check.get("blinks_detected", 0)
        if blinks == 0:
            return {"status": "DEEPFAKE", "reason": "AI-generated face detected."}, 200
        if blinks > 10:
            return {"status": "DEEPFAKE", "reason": "AI-generated face detected."}, 200
    except Exception:
        pass

    # Step 18: anti-spoof
    spoof_result = _ANTISPOOF_SERVICE.check_spoof(best_frame)
    if not spoof_result["is_live"]:
        reason = spoof_result.get('reason', 'Spoof detected')
        # Map generic spoof reasons to friendly messages
        if "screen" in reason.lower() or "moire" in reason.lower():
            friendly = "Screen reflection detected. Please do not use a screen or printed photo."
        elif "bezel" in reason.lower():
            friendly = "Phone bezel detected. Please hold the camera directly."
        elif "flat" in reason.lower():
            friendly = "Flat surface detected. Please use your real face."
        else:
            friendly = reason
        return {"status": "SPOOF", "reason": friendly}, 200

    # Step 19: passive liveness
    liveness_result, is_live = passive_liveness(frames)
    if not is_live:
        return {"status": "SPOOF", "reason": "Liveness check failed – please use your real face."}, 200
    liveness_conf = liveness_result.get("confidence", 0.7)

    # Step 20: risk assessment
    motion_mean = _frame_motion_score(frames)
    risk_score = calculate_risk(
        similarity=similarity, liveness_confidence=liveness_conf,
        antispoof_confidence=0.85, deepfake_vote_ratio=deepfake_confidence,
        motion_score=motion_mean, recent_attempts=attempts,
        deepfake_confidence=deepfake_confidence, face_width_pct=face_width_pct
    )
    decision = decide(risk_score)

    # Step 21: decision handling
    if decision == "APPROVED_PASSIVE":
        status = "VERIFIED"
        _log_verification(user_id, similarity, status.lower(), risk_score=risk_score, details={"decision": decision})
        try:
            jwt_service = get_jwt_service()
            user_data = {'user_id': user_id, 'email': email, 'full_name': full_name_from_db or email.split('@')[0]}
            access_token, refresh_token = jwt_service.create_tokens(user_data)
            return {
                "status": status, "decision": decision, "risk_score": risk_score,
                "similarity": round(similarity, 3), "liveness_confidence": round(liveness_conf, 2),
                "deepfake_confidence": round(deepfake_confidence, 3),
                "face_width_percentage": round(face_width_pct, 1) if face_width_pct else None,
                "access_token": access_token, "refresh_token": refresh_token,
                "token_type": "Bearer", "expires_in": Config.JWT_ACCESS_TOKEN_EXPIRES
            }, 200
        except Exception as e:
            logger.error(f"JWT error: {e}")
            return {
                "status": status, "decision": decision, "risk_score": risk_score,
                "similarity": round(similarity, 3), "liveness_confidence": round(liveness_conf, 2),
                "deepfake_confidence": round(deepfake_confidence, 3),
                "face_width_percentage": round(face_width_pct, 1) if face_width_pct else None
            }, 200
    elif decision == "REQUIRES_ACTIVE_LIVENESS":
        return {
            "status": "REQUIRES_ACTIVE", "decision": decision, "risk_score": risk_score,
            "similarity": round(similarity, 3), "liveness_confidence": round(liveness_conf, 2),
            "deepfake_confidence": round(deepfake_confidence, 3),
            "face_width_percentage": round(face_width_pct, 1) if face_width_pct else None
        }, 200
    else:
        _log_verification(user_id, similarity, "BLOCKED", risk_score=risk_score, details={"decision": decision})
        return {"status": "BLOCKED", "reason": "Risk score too high. Please try again."}, 200
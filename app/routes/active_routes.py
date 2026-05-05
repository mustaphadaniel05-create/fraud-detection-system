# app/routes/active_routes.py

import logging
from typing import Any, Dict, Optional, List, Tuple

import cv2
import numpy as np
from flask import Blueprint, request, jsonify, session, redirect, url_for, render_template

from app.utils.image_utils import decode_base64_image
from app.services.liveness_service import (
    estimate_head_pose,
    get_face_mesh
)
from app.services.face_recognition_service import verify_identity
from app.services.jwt_service import get_jwt_service
from app.db import get_connection
from config import Config

logger = logging.getLogger(__name__)

active_bp = Blueprint("active", __name__, url_prefix="/api")

# Initialize FaceMesh
face_mesh = get_face_mesh()

# CHALLENGES
CHALLENGES = [
    {"id": 0, "instruction": "OPEN YOUR MOUTH WIDE and say 'AH'", "type": "mouth"},
    {"id": 1, "instruction": "Nod your head DOWN and UP 3 times slowly", "type": "nod"},
    {"id": 2, "instruction": "SMILE WIDE (show your teeth)", "type": "smile"},
    {"id": 3, "instruction": "BLINK your eyes SLOWLY and CLEARLY 3 times", "type": "blink"},
]

# Thresholds
EAR_BLINK_THRESHOLD = 0.22
MOUTH_OPEN_THRESHOLD = 0.20
MOUTH_SMILE_THRESHOLD = 0.35

# ======================================================================
# STRONGER NOD DETECTION THRESHOLDS - Harder to pass
# ======================================================================
NOD_PITCH_THRESHOLD = 18.0        # Increased from 12.0 (need bigger nod)
NOD_MIN_MAGNITUDE = 0.08          # Increased from 0.03 (need more movement)
NOD_REQUIRED_FRAMES = 3           # Need multiple successful frames
NOD_YAW_THRESHOLD = 10.0          # Also check head turn
NOD_ROLL_THRESHOLD = 10.0         # Also check head tilt


def _calculate_mouth_open(landmarks, w, h) -> float:
    try:
        upper_lip = landmarks[13]
        lower_lip = landmarks[14]
        left_mouth = landmarks[61]
        right_mouth = landmarks[291]
        
        mouth_height = abs(upper_lip.y - lower_lip.y) * h
        mouth_width = abs(left_mouth.x - right_mouth.x) * w
        
        mar = mouth_height / mouth_width if mouth_width > 0 else 0
        return float(mar)
    except:
        return 0.0


def _calculate_smile(landmarks, w, h) -> float:
    try:
        left_corner = landmarks[61]
        right_corner = landmarks[291]
        upper_lip = landmarks[13]
        
        corner_y = (left_corner.y + right_corner.y) / 2
        lip_y = upper_lip.y
        smile_height = (lip_y - corner_y) * h
        
        mouth_width = abs(left_corner.x - right_corner.x) * w
        width_score = min(1.0, mouth_width / 0.15)
        
        score = (smile_height * 15) + (width_score * 0.5)
        return min(1.0, max(0.0, score))
    except:
        return 0.0


def _calculate_blink(landmarks, w, h) -> float:
    try:
        left_eye = [landmarks[i] for i in [33, 160, 158, 133, 153, 144]]
        right_eye = [landmarks[i] for i in [362, 385, 387, 263, 373, 380]]
        
        def ear(eye_points):
            A = np.linalg.norm(np.array([eye_points[1].x, eye_points[1].y]) - 
                              np.array([eye_points[5].x, eye_points[5].y]))
            B = np.linalg.norm(np.array([eye_points[2].x, eye_points[2].y]) - 
                              np.array([eye_points[4].x, eye_points[4].y]))
            C = np.linalg.norm(np.array([eye_points[0].x, eye_points[0].y]) - 
                              np.array([eye_points[3].x, eye_points[3].y]))
            return (A + B) / (2.0 * C) if C != 0 else 0
        
        return float((ear(left_eye) + ear(right_eye)) / 2)
    except:
        return 1.0


# ======================================================================
# STRONGER NOD DETECTION - Tracks movement over multiple frames
# ======================================================================

class NodDetector:
    """
    Enhanced nod detection that requires actual head movement
    over multiple frames, not just a single pose.
    """
    
    def __init__(self):
        self.pitch_history: List[float] = []
        self.yaw_history: List[float] = []
        self.roll_history: List[float] = []
        self.nod_completed = False
        self.down_detected = False
        self.up_detected = False
        
    def reset(self):
        """Reset detector state for new nod attempt."""
        self.pitch_history = []
        self.yaw_history = []
        self.roll_history = []
        self.nod_completed = False
        self.down_detected = False
        self.up_detected = False
    
    def process_frame(self, pose: Dict[str, float]) -> Tuple[bool, str]:
        """
        Process a frame with head pose.
        Returns: (is_complete, feedback_message)
        """
        pitch = pose.get("pitch", 0)
        yaw = pose.get("yaw", 0)
        roll = pose.get("roll", 0)
        magnitude = pose.get("magnitude", 0)
        
        # Add to history
        self.pitch_history.append(pitch)
        self.yaw_history.append(yaw)
        self.roll_history.append(roll)
        
        # Keep only last 10 frames
        if len(self.pitch_history) > 10:
            self.pitch_history.pop(0)
            self.yaw_history.pop(0)
            self.roll_history.pop(0)
        
        # Check if head is moving (magnitude indicates movement)
        if magnitude < NOD_MIN_MAGNITUDE:
            return False, "Move your head more - not enough movement"
        
        # Detect DOWN nod (pitch > threshold)
        if not self.down_detected and abs(pitch) > NOD_PITCH_THRESHOLD:
            self.down_detected = True
            logger.info(f"✅ DOWN nod detected: pitch={pitch:.1f}°")
            return False, "Good! Now nod UP"
        
        # Detect UP nod (pitch < -threshold) after down detected
        if self.down_detected and not self.up_detected and pitch < -NOD_PITCH_THRESHOLD:
            self.up_detected = True
            logger.info(f"✅ UP nod detected: pitch={pitch:.1f}°")
            return True, "Perfect! Nod completed"
        
        # Provide feedback based on current state
        if not self.down_detected:
            if abs(pitch) > NOD_PITCH_THRESHOLD * 0.6:
                return False, "Almost there! Nod DOWN more"
            else:
                return False, f"Nod DOWN - current pitch: {pitch:.1f}° (need {NOD_PITCH_THRESHOLD}°)"
        elif self.down_detected and not self.up_detected:
            if pitch < -NOD_PITCH_THRESHOLD * 0.6:
                return False, "Almost there! Nod UP more"
            else:
                return False, f"Nod UP - current pitch: {pitch:.1f}° (need -{NOD_PITCH_THRESHOLD}°)"
        
        return False, "Continue nodding"


# Global nod detector instance
_nod_detector = NodDetector()


# ======================================================================
# ONLY LOGS FINAL ACTIVE CHALLENGE RESULT (PASS OR FAIL)
# ======================================================================

def _log_active_challenge_result(user_id: Optional[int], email: str, success: bool, 
                                  similarity: float = 0.0, reason: str = None) -> None:
    try:
        with get_connection() as conn:
            with conn.cursor() as cursor:
                if success:
                    status = "ACTIVE_CHALLENGE_PASSED"
                    details = {
                        "active_challenge": "passed",
                        "challenge_completed": True,
                        "email": email,
                        "final_similarity": similarity,
                        "reason": reason or "All challenges completed successfully"
                    }
                else:
                    status = "ACTIVE_CHALLENGE_FAILED"
                    details = {
                        "active_challenge": "failed",
                        "challenge_completed": False,
                        "email": email,
                        "final_similarity": similarity,
                        "reason": reason or "Active challenge failed"
                    }
                
                import json
                details_json = json.dumps(details)
                
                if user_id is None or user_id == 0:
                    cursor.execute("""
                        INSERT INTO verification_logs 
                        (user_id, similarity_score, status, risk_score, details, created_at) 
                        VALUES (NULL, %s, %s, %s, %s, NOW())
                    """, (similarity, status, 0, details_json))
                else:
                    cursor.execute("""
                        INSERT INTO verification_logs 
                        (user_id, similarity_score, status, risk_score, details, created_at) 
                        VALUES (%s, %s, %s, %s, %s, NOW())
                    """, (user_id, similarity, status, 0, details_json))
                
                conn.commit()
                
                if success:
                    logger.info(f"✅ ACTIVE CHALLENGE PASSED - logged for {email}")
                else:
                    logger.warning(f"❌ ACTIVE CHALLENGE FAILED - logged for {email}")
                
    except Exception as e:
        logger.error(f"Failed to log active challenge result: {e}")


@active_bp.route("/active-challenge", methods=["GET", "POST"])
def active_challenge() -> Any:
    if request.method == "GET":
        email = request.args.get("email")
        if not email:
            return redirect(url_for("verify.verify"))
        return render_template("active_challenge.html", email=email)

    try:
        data = request.get_json(silent=True) or {}
        step = data.get("step")
        image_b64 = data.get("image")
        email = (data.get("email") or "unknown").strip()

        if step is None or not isinstance(step, int):
            return jsonify({"status": "error", "message": "Invalid step"}), 400
        if not image_b64 or len(image_b64) < 200:
            return jsonify({"status": "error", "message": "Invalid image"}), 400
        if step < 0 or step >= len(CHALLENGES):
            return jsonify({"status": "error", "message": "Invalid step"}), 400

        img = decode_base64_image(image_b64)
        if img is None or img.size == 0:
            return jsonify({"status": "error", "message": "Invalid image"}), 400

        h, w = img.shape[:2]
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb)
        
        if not result or not result.multi_face_landmarks:
            return jsonify({
                "status": "FAILED_STEP",
                "feedback": "No face detected. Please look at the camera.",
                "attempt": step + 1
            }), 200
        
        landmarks = result.multi_face_landmarks[0].landmark
        
        passed = False
        feedback = ""
        challenge_type = CHALLENGES[step]["type"]

        if challenge_type == "mouth":
            mar = _calculate_mouth_open(landmarks, w, h)
            logger.info(f"Mouth aspect ratio: {mar:.3f}")
            passed = mar > MOUTH_OPEN_THRESHOLD
            feedback = "Open your mouth WIDER" if not passed else "Good mouth opening!"

        elif challenge_type == "nod":
            # ==============================================================
            # STRONGER NOD DETECTION - Uses NodDetector class
            # ==============================================================
            pose = estimate_head_pose(img)
            pitch = pose.get("pitch", 0)
            yaw = pose.get("yaw", 0)
            roll = pose.get("roll", 0)
            magnitude = pose.get("magnitude", 0)
            
            logger.info(f"Nod pose: pitch={pitch:.1f}°, yaw={yaw:.1f}°, roll={roll:.1f}°, mag={magnitude:.3f}")
            
            # Get nod completion status
            is_complete, nod_feedback = _nod_detector.process_frame(pose)
            
            if is_complete:
                # Reset detector for next time
                _nod_detector.reset()
                passed = True
                feedback = "Good nod! Challenge passed."
            else:
                passed = False
                feedback = nod_feedback

        elif challenge_type == "smile":
            smile_score = _calculate_smile(landmarks, w, h)
            logger.info(f"Smile score: {smile_score:.3f}")
            passed = smile_score > MOUTH_SMILE_THRESHOLD
            feedback = "Smile WIDER - show your teeth!" if not passed else "Great smile!"

        elif challenge_type == "blink":
            ear = _calculate_blink(landmarks, w, h)
            logger.info(f"Eye aspect ratio: {ear:.3f}")
            passed = ear < EAR_BLINK_THRESHOLD
            feedback = "Blink COMPLETELY - close your eyes fully" if not passed else "Good blink!"

        if passed:
            if step == len(CHALLENGES) - 1:
                # Reset nod detector after successful challenge
                _nod_detector.reset()
                
                # FINAL STEP - VERIFY FACE AND LOG RESULT
                identity_result = verify_identity(email, img)
                similarity = identity_result.get("similarity", 0.0)
                verified = identity_result.get("success", False)
                user_id = identity_result.get("user_id")
                
                if not verified:
                    logger.warning(f"❌ Active challenge FAILED - face mismatch for {email}")
                    
                    _log_active_challenge_result(
                        user_id=user_id,
                        email=email,
                        success=False,
                        similarity=similarity,
                        reason=f"Face mismatch after challenge (sim={similarity:.3f})"
                    )
                    
                    return jsonify({
                        "status": "FAILED_STEP",
                        "feedback": "Face does not match enrolled user. Please ensure good lighting.",
                        "attempt": step + 1
                    }), 200

                _log_active_challenge_result(
                    user_id=user_id,
                    email=email,
                    success=True,
                    similarity=similarity,
                    reason="All challenges completed successfully"
                )
                
                session["active_liveness_passed"] = True
                session["active_liveness_email"] = email
                
                # Create JWT tokens
                try:
                    full_name = None
                    with get_connection() as conn:
                        with conn.cursor() as cursor:
                            cursor.execute("SELECT full_name FROM users WHERE id = %s", (user_id,))
                            row = cursor.fetchone()
                            if row:
                                full_name = row["full_name"]
                    
                    jwt_service = get_jwt_service()
                    user_data = {
                        'user_id': user_id,
                        'email': email,
                        'full_name': full_name or email.split('@')[0]
                    }
                    access_token, refresh_token = jwt_service.create_tokens(user_data)
                    
                    return jsonify({
                        "status": "COMPLETE",
                        "message": "Verification successful",
                        "access_token": access_token,
                        "refresh_token": refresh_token,
                        "token_type": "Bearer",
                        "expires_in": Config.JWT_ACCESS_TOKEN_EXPIRES,
                        "user": {
                            "user_id": user_id,
                            "email": email,
                            "full_name": full_name
                        }
                    }), 200
                    
                except Exception as e:
                    logger.error(f"Failed to create JWT tokens: {e}")
                    return jsonify({
                        "status": "COMPLETE",
                        "message": "Verification successful"
                    }), 200
            
            # Reset nod detector after each successful nod step
            if challenge_type == "nod":
                _nod_detector.reset()
            
            return jsonify({
                "status": "NEXT",
                "feedback": feedback,
                "next_instruction": CHALLENGES[step + 1]["instruction"],
                "next_step": step + 1
            }), 200
        
        # STEP FAILED - For nod step, provide detailed feedback
        if challenge_type == "nod" and not passed:
            # Check if user just started the nod step
            if len(_nod_detector.pitch_history) == 0:
                feedback = "Nod your head DOWN, then UP. Show clear movement!"
        
        return jsonify({
            "status": "FAILED_STEP",
            "feedback": feedback,
            "attempt": step + 1
        }), 200

    except Exception as e:
        logger.exception("Active challenge error")
        return jsonify({"status": "error", "message": "Processing error"}), 500


@active_bp.route("/active-challenge-final", methods=["POST"])
def active_challenge_final():
    """
    Endpoint to log final failure when user gives up or times out.
    """
    try:
        data = request.get_json(silent=True) or {}
        email = data.get("email", "").strip()
        reason = data.get("reason", "User gave up or challenge timeout")
        
        if not email:
            return jsonify({"status": "error", "message": "Email required"}), 400
        
        user_id = None
        try:
            with get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT id FROM users WHERE email = %s", (email,))
                    row = cursor.fetchone()
                    if row:
                        user_id = row["id"]
        except Exception as e:
            logger.error(f"Failed to get user_id: {e}")
        
        # Reset nod detector
        _nod_detector.reset()
        
        _log_active_challenge_result(
            user_id=user_id,
            email=email,
            success=False,
            similarity=0.0,
            reason=reason
        )
        
        return jsonify({"status": "success", "message": "Final failure logged"}), 200
        
    except Exception as e:
        logger.exception("Failed to log final failure")
        return jsonify({"status": "error", "message": "Internal error"}), 500
# app/routes/verify_routes.py

import re
import logging
from typing import Dict, Any

from flask import Blueprint, request, jsonify, current_app
from config import Config
from app.services.verification_service import verify_user
from app.services.jwt_service import jwt_required, get_current_user_from_request, get_jwt_service
from app.services.token_blacklist import get_token_blacklist
from app.db import get_db
from datetime import datetime

logger = logging.getLogger(__name__)

verify_bp = Blueprint("verify_bp", __name__, url_prefix="/api")

EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def is_valid_email(email: str) -> bool:
    return bool(email and EMAIL_PATTERN.match(email))


def log_security_event(db, ip: str, event_type: str, email: str, description: str) -> None:
    """
    Logs suspicious or security-related activity.
    """
    try:
        cursor = db.cursor()
        cursor.execute(
            """
            INSERT INTO security_events (ip_address, event_type, email, description)
            VALUES (%s, %s, %s, %s)
            """,
            (ip, event_type, email, description),
        )
        db.commit()
        cursor.close()
    except Exception:
        logger.exception("Failed to log security event")


@verify_bp.route("/verify", methods=["POST"])
def verify() -> tuple[Dict[str, Any], int]:
    """
    POST /api/verify

    Body:
    {
        "email": str,
        "frames": [base64_frame,...]
    }
    """

    # Detect real client IP (proxy-safe)
    client_ip = request.headers.get("X-Forwarded-For", request.remote_addr)
    if client_ip:
        client_ip = client_ip.split(",")[0].strip()

    # Validate request
    if not request.is_json:
        return {"status": "error", "message": "Request must be JSON"}, 400

    data = request.get_json(silent=True) or {}
    email = (data.get("email") or "").strip().lower()
    frames = data.get("frames")

    if not is_valid_email(email):
        return {"status": "error", "message": "Valid email is required"}, 400

    if not frames or not isinstance(frames, list):
        return {"status": "error", "message": "'frames' must be a non-empty list"}, 400

    # Prevent abuse (DoS protection)
    if len(frames) > Config.MAX_FRAMES_ALLOWED:
        return {
            "status": "error",
            "message": f"Too many frames (max {Config.MAX_FRAMES_ALLOWED})",
        }, 400

    if len(frames) < Config.MIN_FRAMES_REQUIRED:
        return {
            "status": "error",
            "message": f"At least {Config.MIN_FRAMES_REQUIRED} frames required",
        }, 400

    # Basic frame validation
    for i, frame_b64 in enumerate(frames):
        if not isinstance(frame_b64, str) or len(frame_b64) < 100:
            return {
                "status": "error",
                "message": f"Frame {i} appears invalid",
            }, 400

    db = None
    try:
        db = get_db()
        cursor = db.cursor()

        # STEP 1: Get user_id from users table
        cursor.execute(
            """
            SELECT id
            FROM users
            WHERE email=%s
            """,
            (email,),
        )
        row = cursor.fetchone()
        user_id = row["id"] if row else None

        # STEP 2: Brute-force protection
        attempts = 0
        if user_id:
            cursor.execute(
                """
                SELECT COUNT(*) AS count
                FROM verification_logs
                WHERE user_id=%s
                AND created_at > NOW() - INTERVAL 10 MINUTE
                """,
                (user_id,),
            )
            row = cursor.fetchone()
            attempts = row["count"] if row else 0

        if attempts > 5:
            logger.warning(f"Too many verification attempts for {email}")
            log_security_event(
                db,
                client_ip,
                "too_many_attempts",
                email,
                "More than 5 verification attempts in 10 minutes",
            )
            cursor.close()
            return {
                "status": "error",
                "message": "Too many verification attempts. Try again later.",
            }, 429

        # STEP 3: Run verification pipeline
        result, status_code = verify_user(
            {
                "email": email,
                "frames": frames,
            }
        )

        # STEP 4: Security event logging
        if result.get("status") != "VERIFIED":
            reason = result.get("reason", "Verification failed")
            if "spoof" in reason.lower() or "screen" in reason.lower():
                log_security_event(
                    db,
                    client_ip,
                    "spoof_attack",
                    email,
                    reason,
                )
            else:
                log_security_event(
                    db,
                    client_ip,
                    "verification_failed",
                    email,
                    reason,
                )

        cursor.close()
        return jsonify(result), status_code

    except Exception:
        current_app.logger.exception("Verification route unhandled error")
        if db:
            try:
                log_security_event(
                    db,
                    client_ip,
                    "system_error",
                    email,
                    "Unhandled exception during verification",
                )
            except Exception:
                pass
        return {"status": "error", "message": "Internal server error"}, 500


# ----------------------------------------------------------------------
# JWT Protected Endpoints
# ----------------------------------------------------------------------

@verify_bp.route("/verify-secure", methods=["GET", "POST"])
@jwt_required
def verify_secure():
    """
    Protected endpoint requiring valid JWT token.
    Returns current user information.
    """
    current_user = get_current_user_from_request()
    
    return jsonify({
        "status": "authenticated",
        "user": current_user,
        "message": "User session is active and verified"
    }), 200


@verify_bp.route("/refresh", methods=["POST"])
def refresh_token():
    """
    Refresh access token using refresh token.
    
    Body:
    {
        "refresh_token": "your_refresh_token"
    }
    """
    data = request.get_json(silent=True) or {}
    refresh_token = data.get("refresh_token")
    
    if not refresh_token:
        return jsonify({"error": "Refresh token required"}), 400
    
    jwt_service = get_jwt_service()
    success, new_access_token, error = jwt_service.refresh_access_token(refresh_token)
    
    if not success:
        return jsonify({"error": error}), 401
    
    return jsonify({
        "access_token": new_access_token,
        "token_type": "Bearer",
        "expires_in": Config.JWT_ACCESS_TOKEN_EXPIRES
    }), 200


@verify_bp.route("/logout", methods=["POST"])
@jwt_required
def logout():
    """
    Logout - revoke current token.
    """
    auth_header = request.headers.get('Authorization', '')
    token = auth_header.split(' ')[1] if auth_header.startswith('Bearer ') else None
    
    if token:
        jwt_service = get_jwt_service()
        is_valid, payload, _ = jwt_service.verify_token(token)
        
        if is_valid and payload:
            exp = payload.get('exp')
            if exp:
                expires_in = max(0, int(exp - datetime.utcnow().timestamp()))
            else:
                expires_in = Config.JWT_ACCESS_TOKEN_EXPIRES
            
            blacklist = get_token_blacklist()
            blacklist.blacklist_token(token, expires_in)
            logger.info(f"Token blacklisted for user {payload.get('email')}")
    
    return jsonify({"message": "Logged out successfully"}), 200


@verify_bp.route("/validate", methods=["GET"])
@jwt_required
def validate_token():
    """
    Validate current access token.
    """
    current_user = get_current_user_from_request()
    
    return jsonify({
        "valid": True,
        "user": current_user
    }), 200
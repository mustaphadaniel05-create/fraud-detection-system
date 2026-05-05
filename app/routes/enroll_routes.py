# app/routes/enroll_routes.py
import re
import logging
from typing import Dict, Any

from flask import Blueprint, request, jsonify, current_app
from app.services.enrollment_service import enroll_user

logger = logging.getLogger(__name__)

enroll_bp = Blueprint("enroll_bp", __name__, url_prefix="/api")

EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


@enroll_bp.route("/enroll", methods=["POST"])
def enroll() -> tuple[Dict[str, Any], int]:
    """
    POST /api/enroll
    Body: { "full_name": str, "email": str, "image": base64_str }
    """
    if not request.is_json:
        return {"status": "error", "message": "Request must be JSON"}, 400

    data = request.get_json(silent=True) or {}

    full_name = (data.get("full_name") or "").strip()
    email     = (data.get("email") or "").strip()
    image_b64 = data.get("image")

    # Validate inputs
    if not full_name:
        return {"status": "error", "message": "Full name is required"}, 400

    if not email or not EMAIL_PATTERN.match(email):
        return {"status": "error", "message": "Valid email is required"}, 400

    if not image_b64 or not isinstance(image_b64, str) or len(image_b64) < 100:
        return {"status": "error", "message": "Valid image is required"}, 400

    try:
        result, status_code = enroll_user(full_name, email, image_b64)
        return jsonify(result), status_code
        
    except Exception as e:
        current_app.logger.exception("Enrollment route unhandled error")
        return {"status": "error", "message": "Internal server error"}, 500
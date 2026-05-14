# app/routes/dashboard_routes.py

import logging
from flask import Blueprint, jsonify
from app.db import get_db
from app.auth import login_required
import json

logger = logging.getLogger(__name__)

dashboard_bp = Blueprint("dashboard_bp", __name__, url_prefix="/api")


@dashboard_bp.route("/dashboard", methods=["GET"])
@login_required
def dashboard_api():
    """
    Fraud monitoring dashboard API with correct counting.
    Active challenge passed counts as success, failed counts as failed.
    """
    try:
        db = get_db()
        cursor = db.cursor()

        # ---------------------------------------------------
        # TOTAL VERIFICATIONS
        # ---------------------------------------------------
        cursor.execute("SELECT COUNT(*) AS total FROM verification_logs")
        row = cursor.fetchone()
        total = row["total"] if row else 0

        # ---------------------------------------------------
        # SUCCESSFUL VERIFICATIONS (includes active challenge passed)
        # ---------------------------------------------------
        cursor.execute("""
            SELECT COUNT(*) AS success 
            FROM verification_logs 
            WHERE LOWER(status) IN (
                'verified', 'success', 'approved_passive', 'complete',
                'active_challenge_passed', 'active_challenge_success'
            )
        """)
        row = cursor.fetchone()
        success = row["success"] if row else 0

        # ---------------------------------------------------
        # FAILED VERIFICATIONS (includes active challenge failed)
        # ---------------------------------------------------
        cursor.execute("""
            SELECT COUNT(*) AS failed 
            FROM verification_logs 
            WHERE LOWER(status) IN (
                'rejected', 'spoof', 'blocked', 'deepfake', 
                'error', 'requires_active', 'requires_active_failed',
                'active_challenge_failed', 'active_challenge_failure'
            )
        """)
        row = cursor.fetchone()
        failed = row["failed"] if row else 0

        # ---------------------------------------------------
        # FRAUD ALERTS
        # ---------------------------------------------------
        try:
            # Check if risk_score column exists
            cursor.execute("""
                SELECT COUNT(*) AS col_exists 
                FROM information_schema.COLUMNS 
                WHERE TABLE_NAME = 'verification_logs' AND COLUMN_NAME = 'risk_score'
            """)
            col_check = cursor.fetchone()
            has_risk_column = col_check["col_exists"] > 0 if col_check else False
            
            if has_risk_column:
                cursor.execute("""
                    SELECT COUNT(*) AS alerts 
                    FROM verification_logs 
                    WHERE LOWER(status) IN ('spoof', 'deepfake', 'blocked', 'requires_active_failed', 'active_challenge_failed')
                       OR risk_score > 55
                """)
            else:
                cursor.execute("""
                    SELECT COUNT(*) AS alerts 
                    FROM verification_logs 
                    WHERE LOWER(status) IN ('spoof', 'deepfake', 'blocked', 'requires_active_failed', 'active_challenge_failed')
                """)
            row = cursor.fetchone()
            alerts = row["alerts"] if row else 0
        except Exception as e:
            logger.warning(f"Risk score column not found, using basic fraud detection: {e}")
            cursor.execute("""
                SELECT COUNT(*) AS alerts 
                FROM verification_logs 
                WHERE LOWER(status) IN ('spoof', 'deepfake', 'blocked', 'requires_active_failed', 'active_challenge_failed')
            """)
            row = cursor.fetchone()
            alerts = row["alerts"] if row else 0

        # ---------------------------------------------------
        # RECENT VERIFICATION LOGS (corrected DATE_FORMAT using raw string)
        # ---------------------------------------------------
        try:
            cursor.execute(r"""
                SELECT 
                    COALESCE(u.email, 'unknown') AS email,
                    v.status,
                    v.similarity_score AS similarity,
                    COALESCE(v.risk_score, 0) AS risk_score,
                    v.details,
                    DATE_FORMAT(v.created_at, '%Y-%m-%d %H:%i:%s') AS created_at
                FROM verification_logs v
                LEFT JOIN users u ON v.user_id = u.id
                ORDER BY v.created_at DESC
                LIMIT 50
            """)
        except Exception as e:
            logger.warning(f"Column error in logs query: {e}")
            cursor.execute(r"""
                SELECT 
                    COALESCE(u.email, 'unknown') AS email,
                    v.status,
                    v.similarity_score AS similarity,
                    0 AS risk_score,
                    v.details,
                    DATE_FORMAT(v.created_at, '%Y-%m-%d %H:%i:%s') AS created_at
                FROM verification_logs v
                LEFT JOIN users u ON v.user_id = u.id
                ORDER BY v.created_at DESC
                LIMIT 50
            """)
        
        logs = cursor.fetchall() or []
        cursor.close()
        db.close()

        # Format logs for frontend
        formatted_logs = []
        for log in logs:
            is_fraud = False
            status_lower = log["status"].lower() if log["status"] else ""
            
            if status_lower in ["spoof", "deepfake", "blocked", "requires_active_failed", "active_challenge_failed"]:
                is_fraud = True
            elif log.get("risk_score") and int(log["risk_score"]) > 55:
                is_fraud = True
            
            details_dict = {}
            try:
                if log.get("details"):
                    if isinstance(log["details"], str):
                        details_dict = json.loads(log["details"])
                    else:
                        details_dict = log["details"]
            except:
                pass
            
            active_challenge_status = "—"
            if details_dict:
                if details_dict.get("active_challenge") == "passed" or details_dict.get("challenge_completed") == True:
                    active_challenge_status = "Passed"
                elif details_dict.get("active_challenge") == "failed" or details_dict.get("challenge_completed") == False:
                    active_challenge_status = "Failed"
            
            formatted_logs.append({
                "email": log["email"] if log["email"] else "unknown",
                "status": log["status"].upper() if log["status"] else "UNKNOWN",
                "similarity": float(log["similarity"]) if log["similarity"] else 0.0,
                "risk_score": int(log["risk_score"]) if log.get("risk_score") else 0,
                "is_fraud": is_fraud,
                "details": details_dict,
                "active_challenge_status": active_challenge_status,
                "created_at": log["created_at"] if log["created_at"] else "N/A"
            })

        logger.info(f"Dashboard - Total: {total}, Success: {success}, Failed: {failed}, Alerts: {alerts}")

        return jsonify({
            "total": total,
            "success": success,
            "failed": failed,
            "alerts": alerts,
            "logs": formatted_logs
        }), 200

    except Exception as e:
        logger.exception(f"Dashboard API failed: {e}")
        return jsonify({
            "status": "error",
            "message": "Failed to load dashboard",
            "total": 0, 
            "success": 0, 
            "failed": 0, 
            "alerts": 0, 
            "logs": []
        }), 500
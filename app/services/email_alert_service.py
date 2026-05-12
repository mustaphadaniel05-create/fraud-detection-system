# app/services/email_alert_service.py – Resend API

import logging
import requests
import os
from datetime import datetime

logger = logging.getLogger(__name__)

def send_fraud_alert(alert_data):
    api_key = os.getenv("RESEND_API_KEY")
    if not api_key:
        logger.warning("Resend API key missing – alert not sent.")
        _log_alert(alert_data)
        return

    recipients = []
    user_email = alert_data.get('email')
    admin_email = os.getenv("ADMIN_EMAIL")
    if user_email and user_email != 'unknown':
        recipients.append(user_email)
    if admin_email and admin_email != user_email:
        recipients.append(admin_email)

    if not recipients:
        logger.warning("No valid recipients – alert not sent.")
        _log_alert(alert_data)
        return

    alert_type = alert_data.get('alert_type', 'unknown').replace('_', ' ').title()
    subject = f"Fraud Alert: {alert_type}"
    body = f"""
Fraud Detection System Alert

Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Alert Type: {alert_data.get('alert_type')}
User Email: {alert_data.get('email')}
IP Address: {alert_data.get('ip_address')}
Status: {alert_data.get('status')}
Reason: {alert_data.get('reason')}
Risk Score: {alert_data.get('risk_score')}
Attempt Number: {alert_data.get('attempt_number')}
Similarity: {alert_data.get('similarity')}
Liveness Confidence: {alert_data.get('liveness_confidence')}
Anti‑spoof Confidence: {alert_data.get('antispoof_confidence')}
Deepfake Score: {alert_data.get('deepfake_score')}
"""

    from_email = "on@resend.dev"  # Resend free tier sender

    for to_email in recipients:
        try:
            response = requests.post(
                "https://api.resend.com/emails",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "from": from_email,
                    "to": to_email,
                    "subject": subject,
                    "text": body
                },
                timeout=10
            )
            if response.status_code == 200:
                logger.info(f"Email alert sent to {to_email}")
            else:
                logger.error(f"Resend error {response.status_code}: {response.text}")
        except Exception as e:
            logger.error(f"Failed to send email to {to_email}: {e}")

def _log_alert(alert_data):
    logger.warning(f"ALERT (would be emailed): {alert_data.get('alert_type')} - {alert_data.get('email')} - {alert_data.get('reason')}")
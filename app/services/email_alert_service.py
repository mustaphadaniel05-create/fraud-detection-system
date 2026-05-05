# app/services/email_alert_service.py

import logging
from flask import current_app
from flask_mail import Message
from threading import Thread
from datetime import datetime
import time

# Import from extensions
from app.extensions import mail

logger = logging.getLogger(__name__)


def send_async_email(app, msg, retry_count=0):
    """Send email asynchronously with retry logic."""
    with app.app_context():
        try:
            # Log what we're about to do
            logger.info(f"📨 Attempting to send email to: {msg.recipients}")
            logger.info(f"📨 Subject: {msg.subject}")
            
            # Use the imported mail instance
            mail.send(msg)
            
            logger.info(f"✅ Email sent successfully to {msg.recipients}")
            
        except Exception as e:
            logger.error(f"❌ FAILED to send email: {e}")
            logger.error(f"   Recipients: {msg.recipients}")
            logger.error(f"   Subject: {msg.subject}")
            logger.error(f"   Error type: {type(e).__name__}")
            
            # Retry up to 3 times with delay
            if retry_count < 3:
                logger.info(f"⏳ Retrying in 5 seconds... (attempt {retry_count + 1}/3)")
                time.sleep(5)
                send_async_email(app, msg, retry_count + 1)


def send_user_alert(user_email, alert_data):
    """
    Send alert to the user who triggered the event.
    """
    try:
        app = current_app._get_current_object()
        
        logger.info(f"🔔 Preparing user alert for: {user_email}")
        
        # User-friendly message based on alert type
        if alert_data['alert_type'] == 'high_risk':
            subject = "Security Alert: Unusual Verification Attempt"
            body = f"""
Dear User,

SECURITY NOTICE

We detected an unusual verification attempt on your account.

Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Status: {alert_data.get('status', 'Unknown')}
Reason: {alert_data.get('reason', 'No reason provided')}

What to do:
- If this wasn't you: Your account may be under attack
- Change your password immediately
- Contact support if you need assistance

Fraud Detection System
"""
        elif alert_data['alert_type'] == 'spoof_attack':
            subject = "URGENT: Suspicious Activity Detected"
            body = f"""
Dear User,

URGENT SECURITY ALERT

We detected a suspicious verification attempt on your account that appears to be a SPOOF ATTACK (photo/video).

Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Status: {alert_data.get('status', 'Unknown')}
Reason: {alert_data.get('reason', 'No reason provided')}

IMMEDIATE ACTION REQUIRED:
- If this wasn't you: Your account may be under attack
- Change your password immediately
- Enable two-factor authentication if not already enabled
- Contact support right away

Fraud Detection System
"""
        else:  # multiple_attempts
            subject = "Multiple Failed Verification Attempts"
            body = f"""
Dear User,

SECURITY NOTICE

We detected multiple failed verification attempts on your account.

Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Attempts: {alert_data.get('attempt_number', 'N/A')}
Status: {alert_data.get('status', 'Unknown')}

What to do:
- If this wasn't you: Your account may be under attack
- Change your password if you suspect unauthorized access

Fraud Detection System
"""
        
        # Create message
        msg = Message(
            subject=subject,
            recipients=[user_email],
            body=body
        )
        
        logger.info(f"📧 Triggering async email to {user_email}")
        
        # Send asynchronously
        Thread(target=send_async_email, args=(app, msg)).start()
        
    except Exception as e:
        logger.error(f"Error preparing user alert: {e}")
        logger.error(f"Alert data: {alert_data}")


def send_admin_alert(alert_data):
    """
    Send detailed alert to system administrator.
    """
    try:
        app = current_app._get_current_object()
        
        subject = f"ADMIN ALERT: {alert_data.get('alert_type', 'Unknown')}"
        
        body = f"""
FRAUD DETECTION SYSTEM - ADMIN ALERT

ALERT DETAILS
Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Alert Type: {alert_data.get('alert_type', 'Unknown')}
Risk Score: {alert_data.get('risk_score', 'N/A')}

USER INFORMATION
Email: {alert_data.get('email', 'Unknown')}
IP Address: {alert_data.get('ip_address', 'Unknown')}
Status: {alert_data.get('status', 'Unknown')}
Reason: {alert_data.get('reason', 'No reason provided')}

TECHNICAL DETAILS
Attempt #: {alert_data.get('attempt_number', 'N/A')}
Similarity Score: {alert_data.get('similarity', 'N/A')}
Liveness Confidence: {alert_data.get('liveness_confidence', 'N/A')}
Anti-spoof Confidence: {alert_data.get('antispoof_confidence', 'N/A')}

Fraud Detection System
"""
        
        msg = Message(
            subject=subject,
            recipients=[app.config['ADMIN_EMAIL']],
            body=body
        )
        
        Thread(target=send_async_email, args=(app, msg)).start()
        logger.info(f"📧 Admin alert triggered for {app.config['ADMIN_EMAIL']}")
        
    except Exception as e:
        logger.error(f"Error sending admin alert: {e}")


def send_fraud_alert(alert_data):
    """
    Send alerts to both user and admin.
    """
    logger.info(f"🚨 FRAUD ALERT TRIGGERED: {alert_data.get('alert_type')}")
    logger.info(f"   Email: {alert_data.get('email')}")
    logger.info(f"   Reason: {alert_data.get('reason')}")
    
    # Send to user (if email exists and not 'unknown')
    if alert_data.get('email') and alert_data['email'] != 'unknown':
        send_user_alert(alert_data['email'], alert_data)
    else:
        logger.warning("⚠️ No valid user email to send alert to")
    
    # Always send to admin
    send_admin_alert(alert_data)
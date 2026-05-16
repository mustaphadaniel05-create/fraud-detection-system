import os
from datetime import timedelta
from dotenv import load_dotenv

load_dotenv()


class Config:
    """
    Central configuration for fraud detection system.
    Production-balanced thresholds.
    """

    # =========================================================
    # CORE SECURITY
    # =========================================================

    SECRET_KEY = os.getenv("SECRET_KEY") or os.urandom(32).hex()

    # =========================================================
    # DATABASE
    # =========================================================

    MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")
    MYSQL_USER = os.getenv("MYSQL_USER", "fraud_user")
    MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "")
    MYSQL_DB = os.getenv("MYSQL_DB", "fraud_detection_db")

    # =========================================================
    # FLASK / SESSION
    # =========================================================

    DEBUG = os.getenv("DEBUG", "False").lower() == "true"

    PERMANENT_SESSION_LIFETIME = timedelta(minutes=30)

    SESSION_COOKIE_SECURE = not DEBUG
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = "Strict"

    # =========================================================
    # FACE RECOGNITION
    # =========================================================

    ENROLLMENT_MODEL = "ArcFace"
    RECOGNITION_MODEL = "ArcFace"

    SIMILARITY_THRESHOLD = 0.75

    # =========================================================
    # DEEPFAKE DETECTION – STRICTER
    # =========================================================

    DEEPFAKE_MODEL_PATH = os.getenv(
        "DEEPFAKE_MODEL_PATH",
        "models/xception_deepfake.h5"
    )

    DEEPFAKE_THRESHOLD = 0.42  # was 0.56 – much stricter
    DEEPFAKE_HIGH_CONFIDENCE_THRESHOLD = 0.75  # was 0.85

    # =========================================================
    # LIVENESS
    # =========================================================

    PASSIVE_LIVENESS_THRESHOLD = 0.45

    ANTISPOOF_REAL_THRESHOLD = 0.80

    # =========================================================
    # FRAME REQUIREMENTS
    # =========================================================

    MIN_FRAMES_REQUIRED = 6
    MAX_FRAMES_ALLOWED = 15

    # =========================================================
    # ANTI-SPOOF INTERNALS
    # =========================================================

    FREQ_THRESHOLD = 11.0
    TEXTURE_THRESHOLD = 55.0
    MOIRE_THRESHOLD = 280
    REFLECTION_THRESHOLD = 85
    BEZEL_LINE_THRESHOLD = 35
    FLICKER_THRESHOLD = 8.0
    BRIGHTNESS_VAR_THRESHOLD = 3.0

    # =========================================================
    # MOTION
    # =========================================================

    MIN_MOTION_SCORE = 1.5
    MIN_MOTION_STD = 0.40

    # =========================================================
    # FACE SIZE VALIDATION
    # =========================================================

    MIN_FACE_WIDTH_PX = 100
    MIN_FACE_HEIGHT_PX = 100
    MIN_RELATIVE_FACE_AREA = 0.08

    # =========================================================
    # SCREEN / REPLAY SPOOF
    # =========================================================

    SCREEN_SPOOF_THRESHOLD = 0.80

    VIDEO_REPLAY_BRIGHTNESS_STD_THRESHOLD = 30
    VIDEO_REPLAY_TEXTURE_THRESHOLD = 100
    VIDEO_REPLAY_COLOR_BINS_THRESHOLD = 120
    VIDEO_REPLAY_EDGE_DENSITY_THRESHOLD = 0.10

    # =========================================================
    # FRAUD ENGINE
    # =========================================================

    RISK_LOW_THRESHOLD = 25
    RISK_MEDIUM_THRESHOLD = 50
    RISK_HIGH_THRESHOLD = 75

    # =========================================================
    # ATTACK PROTECTION
    # =========================================================

    MAX_VERIFICATION_ATTEMPTS = 5
    BLOCK_DURATION_SECONDS = 900

    # =========================================================
    # EMAIL
    # =========================================================

    MAIL_SERVER = os.getenv("MAIL_SERVER", "smtp.gmail.com")
    MAIL_PORT = int(os.getenv("MAIL_PORT", 587))

    MAIL_USE_TLS = os.getenv(
        "MAIL_USE_TLS",
        "True"
    ).lower() == "true"

    MAIL_USE_SSL = os.getenv(
        "MAIL_USE_SSL",
        "False"
    ).lower() == "true"

    MAIL_USERNAME = os.getenv("MAIL_USERNAME")
    MAIL_PASSWORD = os.getenv("MAIL_PASSWORD")

    MAIL_DEFAULT_SENDER = os.getenv("MAIL_DEFAULT_SENDER")
    ADMIN_EMAIL = os.getenv("ADMIN_EMAIL")

    ALERT_RISK_THRESHOLD = 60
    ALERT_ATTEMPT_THRESHOLD = 5

    # =========================================================
    # REDIS
    # =========================================================

    REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
    REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None)
    REDIS_DB = int(os.getenv("REDIS_DB", 0))

    # =========================================================
    # JWT
    # =========================================================

    JWT_SECRET_KEY = os.getenv(
        "JWT_SECRET_KEY",
        os.urandom(32).hex()
    )

    JWT_ACCESS_TOKEN_EXPIRES = int(
        os.getenv("JWT_ACCESS_TOKEN_EXPIRES", 1800)
    )

    JWT_REFRESH_TOKEN_EXPIRES = int(
        os.getenv("JWT_REFRESH_TOKEN_EXPIRES", 604800)
    )

    JWT_ALGORITHM = "HS256"

    # =========================================================
    # HARDWARE LIVENESS
    # =========================================================

    HARDWARE_LIVENESS_ENABLED = False

    DEPTH_CAMERA_TYPE = os.getenv(
        "DEPTH_CAMERA_TYPE",
        "realsense"
    )

    REQUIRE_DEPTH_LIVENESS = False

    DEPTH_THRESHOLD_MM = int(
        os.getenv("DEPTH_THRESHOLD_MM", 50)
    )

    IR_HEAT_THRESHOLD = float(
        os.getenv("IR_HEAT_THRESHOLD", 28.0)
    )

    def __repr__(self):
        return f"<Config debug={self.DEBUG}>"
    
    # Dashboard authentication
    DASHBOARD_USERNAME = os.getenv('DASHBOARD_USERNAME', 'admin')
    DASHBOARD_PASSWORD = os.getenv('DASHBOARD_PASSWORD', 'admin')
import logging
import os
from logging.handlers import RotatingFileHandler
from datetime import timedelta

from flask import Flask, jsonify, current_app
from flask_cors import CORS

# Import extensions from separate file
from app.extensions import limiter, mail

# Blueprints
from app.routes.page_routes import page_bp
from app.routes.enroll_routes import enroll_bp
from app.routes.verify_routes import verify_bp
from app.routes.active_routes import active_bp
from app.routes.dashboard_routes import dashboard_bp

from config import Config

# Database utilities
from app.db import get_connection, create_tables


def create_app(config_object=Config) -> Flask:

    app = Flask(__name__, instance_relative_config=False)
    app.config.from_object(config_object)

    # Secure session settings
    app.config.update(
        SESSION_COOKIE_SECURE=not app.config.get("DEBUG", False),
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Strict",
        PERMANENT_SESSION_LIFETIME=timedelta(minutes=30),
    )

    # Enable CORS for frontend
    CORS(
        app,
        resources={
            r"/api/*": {
                "origins": [
                    "http://localhost:3000",
                    "http://127.0.0.1:3000",
                ]
            }
        },
    )

    # Initialize rate limiter with app
    limiter.init_app(app)
    
    # Initialize mail with app
    mail.init_app(app)

    # -----------------------------
    # Logging configuration
    # -----------------------------

    log_level = logging.DEBUG if app.debug else logging.INFO

    formatter = logging.Formatter(
        '{"time": "%(asctime)s", "level": "%(levelname)s", "name": "%(name)s", "message": "%(message)s"}'
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)

    app.logger.addHandler(console_handler)
    app.logger.setLevel(log_level)

    # File logging in production
    if not app.debug and not app.testing:

        os.makedirs("logs", exist_ok=True)

        file_handler = RotatingFileHandler(
            "logs/fraud_detection.log",
            maxBytes=10 * 1024 * 1024,
            backupCount=10,
        )

        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.INFO)

        app.logger.addHandler(file_handler)

    app.logger.info("Application startup")

    # -----------------------------
    # Register blueprints
    # -----------------------------

    app.register_blueprint(page_bp)
    app.register_blueprint(enroll_bp)
    app.register_blueprint(verify_bp)
    app.register_blueprint(active_bp)
    app.register_blueprint(dashboard_bp)

    # -----------------------------
    # Apply rate limits
    # -----------------------------

    limiter.limit("10/minute")(enroll_bp)
    limiter.limit("15/minute")(verify_bp)
    limiter.limit("20/minute")(active_bp)

    # -----------------------------
    # Create database tables (if not exist)
    # -----------------------------
    with app.app_context():
        create_tables()

    # -----------------------------
    # Health check endpoint
    # -----------------------------

    @app.route("/health")
    def health():
        return jsonify({"status": "healthy"}), 200

    # -----------------------------
    # Error handlers
    # -----------------------------

    @app.errorhandler(429)
    def ratelimit_handler(e):
        return jsonify({"error": "rate limit exceeded"}), 429

    @app.errorhandler(500)
    def server_error(e):
        current_app.logger.exception("Server error")
        return jsonify({"error": "internal server error"}), 500

    return app
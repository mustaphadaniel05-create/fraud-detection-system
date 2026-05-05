"""
SQLAlchemy models — kept for reference/migrations.
The active DB layer uses raw PyMySQL via app/db.py.

SQL to create tables (run once):

    CREATE TABLE users (
        id             INT AUTO_INCREMENT PRIMARY KEY,
        full_name      VARCHAR(120)  NOT NULL,
        email          VARCHAR(255)  NOT NULL UNIQUE,
        face_embedding LONGTEXT      NOT NULL,
        created_at     DATETIME      DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE verification_logs (
        id               INT AUTO_INCREMENT PRIMARY KEY,
        user_id          INT           NOT NULL,
        similarity_score FLOAT,
        status           VARCHAR(50),
        created_at       DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(id)
    );
"""

from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


class User(db.Model):
    __tablename__ = "users"

    id             = db.Column(db.Integer,     primary_key=True)
    full_name      = db.Column(db.String(120), nullable=False)
    email          = db.Column(db.String(255), nullable=False, unique=True)
    face_embedding = db.Column(db.Text,        nullable=False)
    created_at     = db.Column(db.DateTime,    server_default=db.func.now())


class VerificationLog(db.Model):
    __tablename__ = "verification_logs"

    id               = db.Column(db.Integer, primary_key=True)
    user_id          = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    similarity_score = db.Column(db.Float)
    status           = db.Column(db.String(50))
    created_at       = db.Column(db.DateTime, server_default=db.func.now())
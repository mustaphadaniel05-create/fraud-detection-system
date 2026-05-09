import os
import pymysql
import logging
from flask import current_app
from contextlib import contextmanager
import tempfile

logger = logging.getLogger(__name__)

def _get_ssl_config():
    ca_cert = os.getenv("MYSQL_SSL_CA")
    if ca_cert and ca_cert.startswith("-----BEGIN CERTIFICATE-----"):
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".pem", delete=False) as f:
                f.write(ca_cert)
                return {"ca": f.name}
        except Exception as e:
            logger.error(f"Failed to write CA cert: {e}")
    logger.warning("MYSQL_SSL_CA not set or invalid. SSL disabled.")
    return None

@contextmanager
def get_connection():
    conn = None
    try:
        try:
            host = current_app.config["MYSQL_HOST"]
            user = current_app.config["MYSQL_USER"]
            password = current_app.config["MYSQL_PASSWORD"]
            database = current_app.config["MYSQL_DB"]
        except (RuntimeError, KeyError):
            host = os.getenv("MYSQL_HOST")
            user = os.getenv("MYSQL_USER")
            password = os.getenv("MYSQL_PASSWORD")
            database = os.getenv("MYSQL_DB")

        port = int(os.getenv("MYSQL_PORT", 3306))
        ssl_config = _get_ssl_config()

        conn = pymysql.connect(
            host=host,
            user=user,
            password=password,
            database=database,
            port=port,
            cursorclass=pymysql.cursors.DictCursor,
            ssl=ssl_config,
            connect_timeout=10,
            read_timeout=30,
            write_timeout=30,
            autocommit=False,
            charset='utf8mb4'
        )
        yield conn
        conn.commit()
    except Exception as e:
        if conn:
            conn.rollback()
        logger.error(f"Database connection error: {e}")
        raise
    finally:
        if conn:
            conn.close()

def get_db():
    try:
        try:
            host = current_app.config["MYSQL_HOST"]
            user = current_app.config["MYSQL_USER"]
            password = current_app.config["MYSQL_PASSWORD"]
            database = current_app.config["MYSQL_DB"]
        except (RuntimeError, KeyError):
            host = os.getenv("MYSQL_HOST")
            user = os.getenv("MYSQL_USER")
            password = os.getenv("MYSQL_PASSWORD")
            database = os.getenv("MYSQL_DB")

        port = int(os.getenv("MYSQL_PORT", 3306))
        ssl_config = _get_ssl_config()

        conn = pymysql.connect(
            host=host,
            user=user,
            password=password,
            database=database,
            port=port,
            cursorclass=pymysql.cursors.DictCursor,
            ssl=ssl_config,
            connect_timeout=10,
            read_timeout=30,
            write_timeout=30,
            autocommit=True,
            charset='utf8mb4'
        )
        return conn
    except Exception as e:
        logger.error(f"Failed to connect to database: {e}")
        raise

def test_connection():
    try:
        with get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT 1")
                return cursor.fetchone() is not None
    except Exception as e:
        logger.error(f"Database connection test failed: {e}")
        return False

def create_tables():
    try:
        with get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS users (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        full_name VARCHAR(120) NOT NULL,
                        email VARCHAR(255) NOT NULL UNIQUE,
                        face_embedding LONGTEXT NOT NULL,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS verification_logs (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        user_id INT NULL,
                        similarity_score FLOAT DEFAULT 0,
                        status VARCHAR(50) NOT NULL,
                        risk_score INT DEFAULT 0,
                        details JSON NULL,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL
                    )
                """)
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS security_events (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        ip_address VARCHAR(45) NOT NULL,
                        event_type VARCHAR(50) NOT NULL,
                        email VARCHAR(255) NULL,
                        description TEXT NULL,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                # Add missing columns (safe)
                try:
                    cursor.execute("ALTER TABLE verification_logs ADD COLUMN risk_score INT DEFAULT 0")
                except: pass
                try:
                    cursor.execute("ALTER TABLE verification_logs ADD COLUMN details JSON NULL")
                except: pass
                conn.commit()
                logger.info("Tables verified/created successfully.")
    except Exception as e:
        logger.error(f"Error creating tables: {e}")
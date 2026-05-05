#!/usr/bin/env python3
"""
Migration script: Convert existing Facenet embeddings (128-dim) to ArcFace (512-dim).
Run this once after switching to ArcFace.

Usage:
    python scripts/migrate_to_arcface.py --reembed  (if you have original images)
    python scripts/migrate_to_arcface.py --check    (just check dimensions)
    python scripts/migrate_to_arcface.py --delete   (delete all users and start fresh)
"""

import sys
import os
import argparse
import json
import logging
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
from deepface import DeepFace
import tempfile

from app.db import get_connection
from config import Config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# TensorFlow/Keras fix
os.environ['TF_USE_LEGACY_KERAS'] = '1'


def check_current_embeddings():
    """Check all users and their embedding dimensions."""
    try:
        with get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT id, email, full_name, face_embedding FROM users")
                users = cursor.fetchall()
                
                if not users:
                    logger.info("No users found in database")
                    return []
                
                logger.info(f"Found {len(users)} users:")
                
                facenet_count = 0
                arcface_count = 0
                unknown_count = 0
                
                for user in users:
                    user_id = user["id"]
                    email = user["email"]
                    full_name = user["full_name"]
                    
                    try:
                        embedding = json.loads(user["face_embedding"])
                        dim = len(embedding)
                        
                        if dim == 128:
                            model = "Facenet"
                            facenet_count += 1
                        elif dim == 512:
                            model = "ArcFace"
                            arcface_count += 1
                        else:
                            model = f"Unknown ({dim}-dim)"
                            unknown_count += 1
                            
                        logger.info(f"  • {email} ({full_name}): {model}")
                        
                    except Exception as e:
                        logger.error(f"  • {email}: Error parsing embedding - {e}")
                        unknown_count += 1
                
                logger.info(f"\nSummary:")
                logger.info(f"  Facenet (128-dim): {facenet_count}")
                logger.info(f"  ArcFace (512-dim): {arcface_count}")
                logger.info(f"  Unknown: {unknown_count}")
                logger.info(f"  Total: {len(users)}")
                
                return users
                
    except Exception as e:
        logger.error(f"Database error: {e}")
        return []


def delete_all_users(confirm=False):
    """Delete all users from database (WARNING: IRREVERSIBLE)."""
    if not confirm:
        logger.warning("This will DELETE ALL USERS. Use --confirm to proceed.")
        return False
    
    try:
        with get_connection() as conn:
            with conn.cursor() as cursor:
                # Delete verification logs first (foreign key constraint)
                cursor.execute("DELETE FROM verification_logs")
                logger.info("Deleted all verification logs")
                
                # Delete users
                cursor.execute("DELETE FROM users")
                logger.info("Deleted all users")
                
            conn.commit()
            
        logger.info("✅ Database cleared successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error deleting users: {e}")
        return False


def generate_arcface_embedding(image_path):
    """Generate ArcFace embedding from image file."""
    try:
        representations = DeepFace.represent(
            img_path=image_path,
            model_name="ArcFace",
            enforce_detection=False,
            detector_backend="opencv",
            align=True,
            normalization="base"
        )
        
        if not representations:
            return None
            
        embedding = representations[0]["embedding"]
        return np.array(embedding, dtype=np.float32).tolist()
        
    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        return None


def reembed_users():
    """
    Re-embed all users with ArcFace.
    NOTE: This requires the original enrollment images.
    If you don't have them, users must re-enroll manually.
    """
    logger.warning("⚠️  This requires original enrollment images")
    logger.warning("If you don't have the images, users must re-enroll manually")
    
    # Check if images directory exists
    images_dir = Path("data/enrollment_images")
    if not images_dir.exists():
        logger.error(f"Images directory not found: {images_dir}")
        logger.info("Please ensure enrollment images are stored in data/enrollment_images/")
        return False
    
    try:
        with get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT id, email, full_name FROM users")
                users = cursor.fetchall()
                
                if not users:
                    logger.info("No users to re-embed")
                    return True
                
                for user in users:
                    user_id = user["id"]
                    email = user["email"]
                    
                    # Look for image file (assuming naming convention)
                    image_path = images_dir / f"{email}.jpg"
                    if not image_path.exists():
                        image_path = images_dir / f"{user_id}.jpg"
                    
                    if not image_path.exists():
                        logger.warning(f"No image found for {email}, skipping")
                        continue
                    
                    logger.info(f"Processing {email}...")
                    
                    # Generate new ArcFace embedding
                    new_embedding = generate_arcface_embedding(str(image_path))
                    
                    if new_embedding:
                        # Update database
                        cursor.execute(
                            "UPDATE users SET face_embedding = %s WHERE id = %s",
                            (json.dumps(new_embedding), user_id)
                        )
                        logger.info(f"  ✅ Updated {email} with ArcFace embedding")
                    else:
                        logger.error(f"  ❌ Failed to generate embedding for {email}")
                
                conn.commit()
                
        logger.info("✅ Re-embedding complete")
        return True
        
    except Exception as e:
        logger.error(f"Error during re-embedding: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Migrate from Facenet to ArcFace")
    parser.add_argument("--check", action="store_true", help="Check current embeddings")
    parser.add_argument("--delete", action="store_true", help="Delete all users")
    parser.add_argument("--confirm", action="store_true", help="Confirm deletion")
    parser.add_argument("--reembed", action="store_true", help="Re-embed users (requires images)")
    
    args = parser.parse_args()
    
    if args.check:
        logger.info("Checking current embeddings...")
        check_current_embeddings()
        
    elif args.delete:
        logger.warning("⚠️  DANGER: This will delete ALL users!")
        if args.confirm:
            delete_all_users(confirm=True)
        else:
            delete_all_users(confirm=False)
            
    elif args.reembed:
        logger.info("Re-embedding users with ArcFace...")
        reembed_users()
        
    else:
        # Default: show menu
        logger.info("=" * 50)
        logger.info("ArcFace Migration Tool")
        logger.info("=" * 50)
        logger.info("Options:")
        logger.info("  1. Check current embeddings (--check)")
        logger.info("  2. Delete all users (--delete --confirm)")
        logger.info("  3. Re-embed users (--reembed) - requires images")
        logger.info("")
        logger.info("Example:")
        logger.info("  python scripts/migrate_to_arcface.py --check")
        logger.info("  python scripts/migrate_to_arcface.py --delete --confirm")
        logger.info("=" * 50)
        
        # Show current status
        check_current_embeddings()


if __name__ == "__main__":
    main()
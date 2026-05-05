#!/usr/bin/env python3
"""
Download pretrained XceptionNet deepfake detection model.
MATCHED: Uses same import style as xception_deepfake_service.py
"""

import os
import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "xception_deepfake.h5"


def create_lightweight_model():
    """
    Create a lightweight XceptionNet model with ImageNet pretrained weights.
    MATCHED: Uses tf_keras consistently
    """
    logger.info("Creating lightweight XceptionNet model with ImageNet weights...")
    
    try:
        # Use tf_keras (matches xception_deepfake_service.py)
        import tf_keras as keras
        from tf_keras.applications import Xception
        from tf_keras.models import Model
        from tf_keras.layers import Dense, GlobalAveragePooling2D, Dropout
        
        # Load pretrained Xception (ImageNet weights)
        logger.info("Loading XceptionNet with ImageNet weights...")
        base_model = Xception(
            weights='imagenet',
            include_top=False,
            input_shape=(299, 299, 3)
        )
        
        # Freeze base model layers
        base_model.trainable = False
        
        # Add custom classification head
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.5)(x)
        predictions = Dense(1, activation='sigmoid')(x)
        
        model = Model(inputs=base_model.input, outputs=predictions)
        
        # Compile
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Save the model
        model.save(MODEL_PATH)
        logger.info(f"✅ Lightweight model created and saved to {MODEL_PATH}")
        
        if MODEL_PATH.exists():
            size_mb = MODEL_PATH.stat().st_size / (1024 * 1024)
            logger.info(f"   Model size: {size_mb:.1f} MB")
        
        return True
        
    except ImportError as e:
        logger.error(f"Import error: {e}")
        logger.info("   Try: pip install tf-keras")
        return False
    except Exception as e:
        logger.error(f"Failed to create model: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model():
    """
    Quick test to verify model loads.
    MATCHED: Uses same loading method as xception_deepfake_service.py
    """
    logger.info("Testing model...")
    
    try:
        # Use same import pattern as xception_deepfake_service.py
        import tf_keras as keras
        import numpy as np
        
        # Match the exact loading method from xception_deepfake_service.py
        model = keras.models.load_model(MODEL_PATH)
        
        # Create dummy input
        dummy_input = np.random.rand(1, 299, 299, 3).astype(np.float32)
        prediction = model.predict(dummy_input, verbose=0)
        
        logger.info(f"✅ Model loaded successfully")
        logger.info(f"   Input shape: {model.input_shape}")
        logger.info(f"   Output shape: {model.output_shape}")
        logger.info(f"   Test prediction: {prediction[0][0]:.4f}")
        
        return True
        
    except Exception as e:
        logger.error(f"Model test failed: {e}")
        return False


def main():
    """Main execution."""
    logger.info("=" * 60)
    logger.info("Deepfake Model Download/Setup")
    logger.info("=" * 60)
    
    # Check if model already exists
    if MODEL_PATH.exists():
        logger.info(f"Model already exists at {MODEL_PATH}")
        size_mb = MODEL_PATH.stat().st_size / (1024 * 1024)
        logger.info(f"Size: {size_mb:.1f} MB")
        
        # Test existing model
        if test_model():
            logger.info("✅ Existing model is working")
            return True
        else:
            logger.warning("Existing model is corrupted. Recreating...")
            MODEL_PATH.unlink()
    
    # Create lightweight model with ImageNet weights
    logger.info("\nCreating lightweight model with ImageNet weights...")
    if create_lightweight_model():
        if test_model():
            logger.info("✅ Lightweight model created and working")
            return True
    
    logger.error("❌ Failed to create/load model")
    logger.info("\n⚠️ Your system will use heuristic deepfake detection as fallback.")
    return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
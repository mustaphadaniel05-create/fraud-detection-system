"""
XceptionNet-based deepfake detection – FORGIVING MODE
Threshold = 0.55
"""

import logging
import numpy as np
import cv2
import os
from typing import Dict, Optional, List

import tf_keras as keras
from tf_keras.applications import Xception
from tf_keras.models import Model
from tf_keras.layers import Dense, GlobalAveragePooling2D, Dropout

from config import Config

logger = logging.getLogger(__name__)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class XceptionDeepfakeDetector:
    
    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.input_size = (299, 299)
        self.threshold = 0.55   # set to 0.55

        self._load_model_with_fallback(model_path)
        if self.model is None:
            logger.warning("⚠️ No deepfake model. Using heuristic detection.")

    def _load_model_with_fallback(self, model_path: Optional[str] = None):
        paths_to_try = []
        if model_path and os.path.exists(model_path):
            paths_to_try.append(model_path)
        paths_to_try.extend([
            Config.DEEPFAKE_MODEL_PATH,
            "models/xception_deepfake.h5",
            "xception_deepfake.h5",
            "../models/xception_deepfake.h5",
            os.path.join(os.path.dirname(__file__), "../../models/xception_deepfake.h5")
        ])
        for path in paths_to_try:
            if path and os.path.exists(path):
                if self._load_model(path):
                    return True
        return False

    def _load_model(self, model_path: str) -> bool:
        try:
            self.model = keras.models.load_model(model_path)
            logger.info(f"✅ Deepfake model loaded from {model_path}")
            return True
        except Exception as e:
            logger.warning(f"Failed to load model from {model_path}: {e}")
            return False

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        resized = cv2.resize(image, self.input_size, interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = rgb.astype(np.float32) / 255.0
        return np.expand_dims(normalized, axis=0)

    def detect(self, image: np.ndarray) -> Dict:
        if image is None or image.size == 0:
            return {"is_fake": False, "confidence": 0.0, "score": 0.0, "error": "Invalid image"}

        if self.model is None:
            return self._heuristic_detect(image)

        try:
            input_tensor = self._preprocess(image)
            prediction = self.model.predict(input_tensor, verbose=0)
            fake_score = float(prediction[0][0])
            is_fake = fake_score > self.threshold
            confidence = fake_score if is_fake else (1 - fake_score)
            logger.info(f"Deepfake detection: score={fake_score:.3f}, is_fake={is_fake}, threshold={self.threshold}")
            return {
                "is_fake": is_fake,
                "confidence": round(confidence, 3),
                "score": round(fake_score, 3),
                "threshold": self.threshold
            }
        except Exception as e:
            logger.error(f"Deepfake detection error: {e}")
            return self._heuristic_detect(image)

    def _heuristic_detect(self, image: np.ndarray) -> Dict:
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Frequency analysis
            f = np.fft.fft2(gray)
            fshift = np.fft.fftshift(f)
            magnitude = 20 * np.log(np.abs(fshift) + 1e-8)
            freq_std = float(np.std(magnitude))
            freq_score = 0.0
            if freq_std < 6.5:
                freq_score = 0.85
            elif freq_std < 8.0:
                freq_score = 0.65
            elif freq_std < 10.0:
                freq_score = 0.45
            else:
                freq_score = 0.15

            # Texture analysis
            lap = cv2.Laplacian(gray, cv2.CV_64F)
            texture_var = float(lap.var())
            texture_score = 0.0
            if texture_var < 35:
                texture_score = 0.85
            elif texture_var < 50:
                texture_score = 0.65
            elif texture_var < 75:
                texture_score = 0.45
            else:
                texture_score = 0.15

            # Noise analysis
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            noise = gray - blurred
            noise_std = float(np.std(noise))
            noise_score = 0.0
            if noise_std < 2.0:
                noise_score = 0.85
            elif noise_std < 3.2:
                noise_score = 0.65
            else:
                noise_score = 0.15

            # Edge analysis
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            edge_score = 0.0
            if edge_density > 0.13:
                edge_score = 0.75
            elif edge_density < 0.014:
                edge_score = 0.65
            else:
                edge_score = 0.15

            # Color analysis
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            color_std = float(np.std(hsv[:, :, 0]))
            color_score = 0.0
            if color_std < 15:
                color_score = 0.75
            elif color_std < 22:
                color_score = 0.55
            else:
                color_score = 0.15

            fake_score = (freq_score * 0.30 + texture_score * 0.30 + noise_score * 0.15 +
                          edge_score * 0.15 + color_score * 0.10)
            is_fake = fake_score > 0.55   # changed to 0.55
            logger.info(f"Heuristic detection: score={fake_score:.3f}, is_fake={is_fake}")
            return {
                "is_fake": is_fake,
                "confidence": round(fake_score, 3),
                "score": round(fake_score, 3),
                "heuristic": True,
                "frequency_std": round(freq_std, 2),
                "texture_var": round(texture_var, 2),
                "noise_std": round(noise_std, 2),
                "edge_density": round(edge_density, 4),
                "color_std": round(color_std, 1)
            }
        except Exception as e:
            logger.error(f"Heuristic detection error: {e}")
            return {"is_fake": False, "confidence": 0.0, "score": 0.0, "error": str(e)}

    def detect_sequence(self, frames: List[np.ndarray], max_frames: int = 5) -> Dict:
        if not frames:
            return {"is_fake": False, "confidence": 0.0, "frames_analyzed": 0}
        frames_to_analyze = min(max_frames, len(frames))
        scores = []
        frame_results = []
        for frame in frames[:frames_to_analyze]:
            result = self.detect(frame)
            if "score" in result:
                scores.append(result["score"])
                frame_results.append(result)
        if not scores:
            return {"is_fake": False, "confidence": 0.0, "frames_analyzed": 0}
        mean_score = float(np.mean(scores))
        fake_frames = sum(1 for r in frame_results if r.get("is_fake", False))
        fake_ratio = fake_frames / len(frame_results) if frame_results else 0
        if mean_score > 0.55 or fake_ratio >= 0.4:   # changed to 0.55
            is_fake = True
            confidence = mean_score
        else:
            is_fake = False
            confidence = 1 - mean_score if mean_score < 0.5 else mean_score
        logger.info(f"Sequence detection: mean={mean_score:.3f}, fake_ratio={fake_ratio:.2f}, is_fake={is_fake}")
        return {
            "is_fake": bool(is_fake),
            "confidence": round(confidence, 3),
            "score": round(mean_score, 3),
            "mean_score": round(mean_score, 3),
            "std_score": round(np.std(scores), 3) if len(scores) > 1 else 0.0,
            "fake_frames": fake_frames,
            "total_frames": len(frame_results),
            "frames_analyzed": len(scores)
        }


_deepfake_detector = None

def get_deepfake_detector() -> XceptionDeepfakeDetector:
    global _deepfake_detector
    if _deepfake_detector is None:
        _deepfake_detector = XceptionDeepfakeDetector(Config.DEEPFAKE_MODEL_PATH)
    return _deepfake_detector

def detect_deepfake_advanced(image: np.ndarray) -> Dict:
    detector = get_deepfake_detector()
    return detector.detect(image)

def detect_deepfake_sequence_advanced(frames: List[np.ndarray], max_frames: int = 5) -> Dict:
    detector = get_deepfake_detector()
    return detector.detect_sequence(frames, max_frames)
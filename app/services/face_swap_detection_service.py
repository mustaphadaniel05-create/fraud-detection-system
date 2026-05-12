"""
Specialized face-swap deepfake detection for high-quality deepfakes.
Now more sensitive to catch deepfakes (lowered threshold to 0.55).
"""

import logging
import cv2
import numpy as np
from typing import Dict, Tuple, List, Optional

logger = logging.getLogger(__name__)


class FaceSwapDetector:
    """
    Detector for face-swap deepfakes. Now stricter to block fakes.
    """
    
    @staticmethod
    def detect_face_swap_artifacts(image: np.ndarray) -> Dict:
        if image is None or image.size == 0:
            return {"is_fake": False, "confidence": 0.0, "reasons": []}
        
        reasons = []
        confidence_score = 0.0
        
        try:
            # 1. Edge artifacts (low edge density)
            edge_score, edge_artifacts = FaceSwapDetector._check_edge_artifacts(image)
            if edge_artifacts:
                reasons.append("unnatural_edge_blending")
                confidence_score += 0.30
            logger.info(f"Edge artifacts score: {edge_score}")
            
            # 2. Color mismatch (lower threshold)
            color_mismatch, color_score = FaceSwapDetector._check_color_mismatch(image)
            if color_mismatch:
                reasons.append("face_background_color_mismatch")
                confidence_score += 0.35
            logger.info(f"Color mismatch score: {color_score}")
            
            # 3. Skin texture (too smooth)
            texture_score, is_smooth = FaceSwapDetector._check_skin_texture(image)
            if is_smooth:
                reasons.append("unnaturally_smooth_skin")
                confidence_score += 0.25
            logger.info(f"Skin texture score: {texture_score}")
            
            # 4. Lighting inconsistency
            lighting_score, is_inconsistent = FaceSwapDetector._check_lighting_consistency(image)
            if is_inconsistent:
                reasons.append("inconsistent_lighting")
                confidence_score += 0.20
            logger.info(f"Lighting consistency score: {lighting_score}")
            
            # Lower threshold to 0.55 (was 0.70) – easier to flag fakes
            is_fake = confidence_score > 0.55
            
            logger.info(f"Face-swap detection: is_fake={is_fake}, confidence={confidence_score:.2f}, reasons={reasons}")
            
            return {
                "is_fake": is_fake,
                "confidence": round(confidence_score, 3),
                "reasons": reasons,
                "edge_score": edge_score,
                "color_score": color_score,
                "texture_score": texture_score,
                "lighting_score": lighting_score
            }
            
        except Exception as e:
            logger.error(f"Face-swap detection error: {e}")
            return {"is_fake": False, "confidence": 0.0, "reasons": []}
    
    @staticmethod
    def _check_edge_artifacts(image: np.ndarray) -> Tuple[float, bool]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        # Real faces typically 0.01-0.05; deepfakes often lower (<0.008) or higher (>0.12)
        if edge_density < 0.008 or edge_density > 0.12:
            return edge_density, True
        return edge_density, False
    
    @staticmethod
    def _check_color_mismatch(image: np.ndarray) -> Tuple[bool, float]:
        h, w = image.shape[:2]
        face_region = image[h//4:3*h//4, w//4:3*w//4]
        if face_region.size == 0:
            return False, 0.0
        face_color = np.mean(face_region, axis=(0, 1))
        
        top_border = image[0:h//8, :]
        bottom_border = image[7*h//8:, :]
        left_border = image[:, 0:w//8]
        right_border = image[:, 7*w//8:]
        
        border_colors = []
        for border in [top_border, bottom_border, left_border, right_border]:
            if border.size > 0:
                border_colors.append(np.mean(border, axis=(0, 1)))
        if not border_colors:
            return False, 0.0
        
        avg_border_color = np.mean(border_colors, axis=0)
        color_diff = np.linalg.norm(face_color - avg_border_color)
        # Lower threshold to 45 (was 60) to catch more mismatches
        if color_diff > 45:
            return True, color_diff
        return False, color_diff
    
    @staticmethod
    def _check_skin_texture(image: np.ndarray) -> Tuple[float, bool]:
        h, w = image.shape[:2]
        face_region = image[h//4:3*h//4, w//4:3*w//4]
        if face_region.size == 0:
            return 0.0, False
        gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
        texture = cv2.Laplacian(gray_face, cv2.CV_64F).var()
        # Deepfake skin often too smooth (<30); real face often >30
        if texture < 30:
            return texture, True
        return texture, False
    
    @staticmethod
    def _check_lighting_consistency(image: np.ndarray) -> Tuple[float, bool]:
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        left_face = gray[h//4:3*h//4, w//4:w//2]
        right_face = gray[h//4:3*h//4, w//2:3*w//4]
        if left_face.size == 0 or right_face.size == 0:
            return 0.0, False
        left_brightness = np.mean(left_face)
        right_brightness = np.mean(right_face)
        brightness_diff = abs(left_brightness - right_brightness)
        # Real faces have natural asymmetry (5-15). Flag if too symmetrical (<4) or too different (>30)
        if brightness_diff < 4 or brightness_diff > 30:
            return brightness_diff, True
        return brightness_diff, False


# Singleton
_face_swap_detector = None

def get_face_swap_detector() -> FaceSwapDetector:
    global _face_swap_detector
    if _face_swap_detector is None:
        _face_swap_detector = FaceSwapDetector()
    return _face_swap_detector

def detect_face_swap(image: np.ndarray) -> Dict:
    detector = get_face_swap_detector()
    return detector.detect_face_swap_artifacts(image)
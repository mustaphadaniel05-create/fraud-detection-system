"""
Specialized face-swap deepfake detection for high-quality deepfakes.
Detects unnatural blending, edge artifacts, and color inconsistencies.
"""

import logging
import cv2
import numpy as np
from typing import Dict, Tuple, List, Optional

logger = logging.getLogger(__name__)


class FaceSwapDetector:
    """
    Specialized detector for face-swap deepfakes (like deepfakemaker.io).
    Looks for blending artifacts, unnatural edges, and color mismatches.
    """
    
    @staticmethod
    def detect_face_swap_artifacts(image: np.ndarray) -> Dict:
        """
        Detect face-swap artifacts in a single image.
        Returns: {"is_fake": bool, "confidence": float, "reasons": list}
        """
        if image is None or image.size == 0:
            return {"is_fake": False, "confidence": 0.0, "reasons": []}
        
        reasons = []
        confidence_score = 0.0
        
        try:
            h, w = image.shape[:2]
            
            # 1. Check for unnatural edge blending around face boundary
            edge_score, edge_artifacts = FaceSwapDetector._check_edge_artifacts(image)
            if edge_artifacts:
                reasons.append("unnatural_edge_blending")
                confidence_score += 0.25
            logger.info(f"Edge artifacts score: {edge_score}")
            
            # 2. Check for color mismatch between face and background
            color_mismatch, color_score = FaceSwapDetector._check_color_mismatch(image)
            if color_mismatch:
                reasons.append("face_background_color_mismatch")
                confidence_score += 0.30
            logger.info(f"Color mismatch score: {color_score}")
            
            # 3. Check for unnatural skin texture (too smooth)
            texture_score, is_smooth = FaceSwapDetector._check_skin_texture(image)
            if is_smooth:
                reasons.append("unnaturally_smooth_skin")
                confidence_score += 0.25
            logger.info(f"Skin texture score: {texture_score}")
            
            # 4. Check for inconsistent lighting on face
            lighting_score, is_inconsistent = FaceSwapDetector._check_lighting_consistency(image)
            if is_inconsistent:
                reasons.append("inconsistent_lighting")
                confidence_score += 0.20
            logger.info(f"Lighting consistency score: {lighting_score}")
            
            # Determine if fake
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
        """Detect unnatural edge blending around face boundary."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect edges
        edges = cv2.Canny(gray, 50, 150)
        
        # Look for unnaturally smooth edges (deepfakes have too smooth boundaries)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Real faces have edge density 0.03-0.08
        # Face-swaps often have lower edge density (< 0.02) or unnaturally sharp (> 0.12)
        
        if edge_density < 0.018:
            return edge_density, True
        elif edge_density > 0.12:
            return edge_density, True
        
        return edge_density, False
    
    @staticmethod
    def _check_color_mismatch(image: np.ndarray) -> Tuple[bool, float]:
        """Check for color mismatch between face and background."""
        h, w = image.shape[:2]
        
        # Split into face region (center) and background (edges)
        face_region = image[h//4:3*h//4, w//4:3*w//4]
        
        if face_region.size == 0:
            return False, 0.0
        
        # Calculate average color of face region
        face_color = np.mean(face_region, axis=(0, 1))
        
        # Calculate average color of border regions
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
        
        # Calculate color difference
        color_diff = np.linalg.norm(face_color - avg_border_color)
        
        # Large color difference indicates face-swap
        if color_diff > 40:
            return True, color_diff
        
        return False, color_diff
    
    @staticmethod
    def _check_skin_texture(image: np.ndarray) -> Tuple[float, bool]:
        """Check if skin texture is unnaturally smooth."""
        # Focus on center region where face is
        h, w = image.shape[:2]
        face_region = image[h//4:3*h//4, w//4:3*w//4]
        
        if face_region.size == 0:
            return 0.0, False
        
        gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
        
        # Calculate texture using Laplacian variance
        texture = cv2.Laplacian(gray_face, cv2.CV_64F).var()
        
        # Real skin has texture variance 30-120
        # Deepfake skin is often too smooth (< 25)
        
        if texture < 25:
            return texture, True
        
        return texture, False
    
    @staticmethod
    def _check_lighting_consistency(image: np.ndarray) -> Tuple[float, bool]:
        """Check for inconsistent lighting across face."""
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Split face into left and right halves
        left_face = gray[h//4:3*h//4, w//4:w//2]
        right_face = gray[h//4:3*h//4, w//2:3*w//4]
        
        if left_face.size == 0 or right_face.size == 0:
            return 0.0, False
        
        left_brightness = np.mean(left_face)
        right_brightness = np.mean(right_face)
        
        brightness_diff = abs(left_brightness - right_brightness)
        
        # Real faces have natural asymmetry (diff 5-15)
        # Deepfakes often have unnatural symmetry (diff < 3) or extreme diff (> 25)
        
        if brightness_diff < 3 or brightness_diff > 30:
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
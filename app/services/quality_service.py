"""
Image quality service for face verification - EXTRA FORGIVING MODE.
Accepts clear real faces, rejects only truly problematic images.
"""

import logging
import cv2
import numpy as np
from typing import Tuple, Optional, Dict, Any

logger = logging.getLogger(__name__)


class QualityService:
    """
    Extra forgiving quality service – accepts more real faces, especially with lighting variations.
    """
    
    # ======================================================================
    # EXTRA FORGIVING QUALITY THRESHOLDS
    # ======================================================================
    
    # Blur detection (Laplacian variance)
    BLUR_EXCELLENT = 100      
    BLUR_GOOD = 70            
    BLUR_ACCEPTABLE = 38      
    BLUR_REJECT = 15          # only reject extremely blurry
    
    # Brightness (0-255 range) – MUCH MORE FORGIVING
    BRIGHTNESS_IDEAL_MIN = 40      # lowered from 60
    BRIGHTNESS_IDEAL_MAX = 230     # increased from 210
    BRIGHTNESS_ACCEPTABLE_MIN = 25 # lowered from 45
    BRIGHTNESS_ACCEPTABLE_MAX = 245 # increased from 230
    BRIGHTNESS_REJECT_DARK = 15     # lowered from 25
    BRIGHTNESS_REJECT_BRIGHT = 250  # increased from 245
    
    # Contrast (standard deviation)
    CONTRAST_EXCELLENT = 50   
    CONTRAST_GOOD = 35        
    CONTRAST_ACCEPTABLE = 22  
    CONTRAST_REJECT = 6        # was 8 – more forgiving
    
    # Face size (% of frame)
    FACE_SIZE_IDEAL_MIN = 22      
    FACE_SIZE_IDEAL_MAX = 48      
    FACE_SIZE_ACCEPTABLE_MIN = 18     
    FACE_SIZE_REJECT_SMALL = 15    
    FACE_SIZE_REJECT_LARGE = 60    
    
    # Glare detection – MORE FORGIVING
    GLARE_THRESHOLD = 240
    GLARE_PERCENT_MAX = 20          # increased from 12
    GLARE_PERCENT_REJECT = 30       # increased from 18
    
    # ======================================================================
    
    @classmethod
    def check_image_quality(cls, frame: np.ndarray) -> Tuple[bool, str, float]:
        """
        Extra forgiving image quality check.
        Returns: (is_good_quality, reason_message, quality_score)
        """
        if frame is None or frame.size == 0:
            return False, "Invalid image", 0.0
        
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Calculate metrics
            brightness = cls._fast_brightness(gray)
            contrast = cls._fast_contrast(gray)
            blur_score = cls._fast_blur(gray)
            glare_percent = cls._fast_glare(frame, gray)
            
            logger.debug(f"Quality: blur={blur_score:.1f}, bright={brightness:.1f}, contrast={contrast:.1f}, glare={glare_percent:.1f}%")
            
            # ========== EXTRA FORGIVING CHECKS ==========
            
            # 1. Brightness check – only reject extreme cases
            if brightness < cls.BRIGHTNESS_REJECT_DARK:
                return False, f"Image too dark ({brightness:.0f}). Please turn on lights.", brightness
            if brightness > cls.BRIGHTNESS_REJECT_BRIGHT:
                return False, f"Image too bright ({brightness:.0f}). Please move from bright lights.", brightness
            
            # 2. Blur check – only reject if EXTREMELY blurry
            if blur_score < cls.BLUR_REJECT:
                return False, f"Image too blurry ({blur_score:.0f}). Please hold camera steady.", blur_score
            
            # 3. Contrast check – only reject if extremely low
            if contrast < cls.CONTRAST_REJECT:
                return False, f"Very low contrast ({contrast:.0f}). Please ensure even lighting.", contrast
            
            # 4. Glare check – only reject if severe glare
            if glare_percent > cls.GLARE_PERCENT_REJECT:
                return False, f"Too much glare ({glare_percent:.0f}%). Please avoid bright lights facing camera.", glare_percent
            
            # Calculate quality score (0-100)
            quality_score = cls._calculate_quality_score(blur_score, brightness, contrast, glare_percent)
            
            # Log warnings for borderline cases but don't reject
            if blur_score < cls.BLUR_ACCEPTABLE:
                logger.info(f"⚠️ Slightly blurry ({blur_score:.0f}) - accepting anyway")
            
            if brightness < cls.BRIGHTNESS_ACCEPTABLE_MIN or brightness > cls.BRIGHTNESS_ACCEPTABLE_MAX:
                logger.info(f"⚠️ Suboptimal brightness ({brightness:.0f}) - accepting anyway")
            
            logger.info(f"✅ Quality PASS: score={quality_score:.0f}")
            return True, "Good quality", quality_score
            
        except Exception as e:
            logger.error(f"Quality check error: {e}")
            return True, "Unable to check quality", 70.0  # Default to pass on error
    
    # ======================================================================
    # FAST CALCULATION METHODS (unchanged)
    # ======================================================================
    
    @staticmethod
    def _fast_brightness(gray: np.ndarray) -> float:
        return float(np.mean(gray))
    
    @staticmethod
    def _fast_contrast(gray: np.ndarray) -> float:
        return float(np.std(gray))
    
    @staticmethod
    def _fast_blur(gray: np.ndarray) -> float:
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())
    
    @classmethod
    def _fast_glare(cls, frame: np.ndarray, gray: np.ndarray) -> float:
        if gray is None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        overexposed = np.sum(gray > cls.GLARE_THRESHOLD)
        return (overexposed / gray.size) * 100
    
    @classmethod
    def _calculate_quality_score(cls, blur: float, brightness: float, contrast: float, glare: float) -> float:
        """Calculate overall quality score 0-100."""
        score = 100.0
        
        # Blur penalty – more forgiving
        if blur < cls.BLUR_GOOD:
            score -= 10
        elif blur < cls.BLUR_EXCELLENT:
            score -= 5
        
        # Brightness penalty – more forgiving
        if brightness < cls.BRIGHTNESS_IDEAL_MIN or brightness > cls.BRIGHTNESS_IDEAL_MAX:
            score -= 5
        
        # Contrast penalty – more forgiving
        if contrast < cls.CONTRAST_GOOD:
            score -= 8
        elif contrast < cls.CONTRAST_EXCELLENT:
            score -= 4
        
        # Glare penalty – more forgiving
        if glare > cls.GLARE_PERCENT_MAX:
            score -= 8
        elif glare > 10:
            score -= 4
        
        return max(0, min(100, score))
    
    # ======================================================================
    # FACE SIZE CHECK (unchanged)
    # ======================================================================
    
    @classmethod
    def check_face_size(cls, face_width_pct: float) -> Tuple[bool, str]:
        if face_width_pct < cls.FACE_SIZE_REJECT_SMALL:
            return False, f"Face too far ({face_width_pct:.0f}%). Please move closer."
        if face_width_pct > cls.FACE_SIZE_REJECT_LARGE:
            return False, f"Face too close ({face_width_pct:.0f}%). Please move back."
        if face_width_pct < cls.FACE_SIZE_ACCEPTABLE_MIN:
            return True, f"Face a bit far ({face_width_pct:.0f}%). Move closer for better results."
        if face_width_pct > cls.FACE_SIZE_IDEAL_MAX:
            return True, f"Face a bit close ({face_width_pct:.0f}%). Move back slightly."
        return True, f"Face distance good ({face_width_pct:.0f}%)."
    
    # ======================================================================
    # QUICK CHECKS (unchanged but using new thresholds)
    # ======================================================================
    
    @classmethod
    def is_blurry(cls, frame: np.ndarray, threshold: float = None) -> Tuple[bool, float]:
        if threshold is None:
            threshold = cls.BLUR_REJECT
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur_score = cls._fast_blur(gray)
            return blur_score < threshold, blur_score
        except:
            return True, 0.0
    
    @classmethod
    def is_too_dark(cls, frame: np.ndarray, threshold: float = None) -> Tuple[bool, float]:
        if threshold is None:
            threshold = cls.BRIGHTNESS_REJECT_DARK
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            brightness = cls._fast_brightness(gray)
            return brightness < threshold, brightness
        except:
            return True, 0.0
    
    @classmethod
    def is_too_bright(cls, frame: np.ndarray, threshold: float = None) -> Tuple[bool, float]:
        if threshold is None:
            threshold = cls.BRIGHTNESS_REJECT_BRIGHT
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            brightness = cls._fast_brightness(gray)
            return brightness > threshold, brightness
        except:
            return True, 255.0
    
    @classmethod
    def is_black_frame(cls, frame: np.ndarray, threshold: float = None) -> bool:
        if frame is None or frame.size == 0:
            return True
        if threshold is None:
            threshold = cls.BRIGHTNESS_REJECT_DARK
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return np.mean(gray) < threshold
        except:
            return True
    
    @classmethod
    def is_face_clear(cls, frame: np.ndarray, threshold: float = 30) -> Tuple[bool, float]:
        if frame is None or frame.size == 0:
            return False, 0.0
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            clarity_score = cv2.Laplacian(gray, cv2.CV_64F).var()
            return clarity_score >= threshold, clarity_score
        except Exception as e:
            logger.error(f"Clarity check error: {e}")
            return True, 50.0
    
    # ======================================================================
    # COMPREHENSIVE QUALITY REPORT
    # ======================================================================
    
    @classmethod
    def get_quality_report(cls, frame: np.ndarray) -> Dict[str, Any]:
        if frame is None or frame.size == 0:
            return {"error": "Invalid frame"}
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur_score = cls._fast_blur(gray)
            brightness = cls._fast_brightness(gray)
            contrast = cls._fast_contrast(gray)
            glare = cls._fast_glare(frame, gray)
            return {
                "blur_score": round(blur_score, 1),
                "brightness": round(brightness, 1),
                "contrast": round(contrast, 1),
                "glare_percent": round(glare, 1),
                "quality_score": round(cls._calculate_quality_score(blur_score, brightness, contrast, glare), 1),
                "blur_status": cls._get_blur_status(blur_score),
                "brightness_status": cls._get_brightness_status(brightness),
                "contrast_status": cls._get_contrast_status(contrast),
                "glare_status": cls._get_glare_status(glare),
                "is_acceptable": cls.check_image_quality(frame)[0]
            }
        except Exception as e:
            return {"error": str(e)}
    
    @classmethod
    def _get_blur_status(cls, blur_score: float) -> str:
        if blur_score >= cls.BLUR_EXCELLENT:
            return "excellent"
        if blur_score >= cls.BLUR_GOOD:
            return "good"
        if blur_score >= cls.BLUR_ACCEPTABLE:
            return "acceptable"
        if blur_score >= cls.BLUR_REJECT:
            return "poor"
        return "reject"
    
    @classmethod
    def _get_brightness_status(cls, brightness: float) -> str:
        if cls.BRIGHTNESS_IDEAL_MIN <= brightness <= cls.BRIGHTNESS_IDEAL_MAX:
            return "ideal"
        if cls.BRIGHTNESS_ACCEPTABLE_MIN <= brightness <= cls.BRIGHTNESS_ACCEPTABLE_MAX:
            return "acceptable"
        if brightness < cls.BRIGHTNESS_REJECT_DARK or brightness > cls.BRIGHTNESS_REJECT_BRIGHT:
            return "reject"
        return "poor"
    
    @classmethod
    def _get_contrast_status(cls, contrast: float) -> str:
        if contrast >= cls.CONTRAST_EXCELLENT:
            return "excellent"
        if contrast >= cls.CONTRAST_GOOD:
            return "good"
        if contrast >= cls.CONTRAST_ACCEPTABLE:
            return "acceptable"
        if contrast >= cls.CONTRAST_REJECT:
            return "poor"
        return "reject"
    
    @classmethod
    def _get_glare_status(cls, glare_percent: float) -> str:
        if glare_percent <= cls.GLARE_PERCENT_MAX:
            return "good"
        if glare_percent <= cls.GLARE_PERCENT_REJECT:
            return "acceptable"
        return "reject"
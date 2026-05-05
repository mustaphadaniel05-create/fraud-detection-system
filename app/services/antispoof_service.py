"""
COMPLETE anti-spoofing service - FORGIVING ON REAL FACES, AGGRESSIVE ON PHONES.
Accepts real faces, aggressively blocks screen replays and deepfake videos.
FIXED: Lowered thresholds for better real face acceptance.
"""

import logging
from typing import Dict, Any, List, Tuple

import cv2
import numpy as np

from config import Config

logger = logging.getLogger(__name__)


class AntiSpoofService:
    """
    Anti-spoofing with FORGIVING detection for real faces.
    Aggressively blocks phone screens and deepfake videos.
    FIXED: More forgiving texture and flatness thresholds.
    """
    
    def __init__(self):
        self.frame_history: List[np.ndarray] = []
        logger.info("=" * 60)
        logger.info("🛡️🛡️🛡️ ANTI-SPOOF SERVICE INITIALIZED - FORGIVING REAL FACE MODE 🛡️🛡️🛡️")
        logger.info("=" * 60)

    def check_spoof(self, image: np.ndarray) -> Dict[str, Any]:
        """Check if image is a real face or attack - FORGIVING FOR REAL FACES."""
        
        logger.info("=" * 50)
        logger.info("🔍🔍🔍 ANTI-SPOOF CHECK STARTED (FORGIVING REAL FACE MODE) 🔍🔍🔍")
        
        if image is None or image.size == 0:
            logger.error("❌ ANTI-SPOOF: Invalid image")
            return self._result(False, 0.0, "Invalid image")

        try:
            self._update_history(image)
            
            # ======================================================================
            # LAYER 1: TEXTURE ANALYSIS - VERY FORGIVING (FIXED)
            # ======================================================================
            texture = self._calc_texture(image)
            logger.info(f"📊 LAYER 1 - Texture value: {texture:.2f}")
            
            # Only block extremely low texture (obvious printed photos)
            # FIXED: Lowered from 12 to 4 for real faces
            if texture < 4:
                logger.warning(f"🚫 LAYER 1 BLOCK: Very low texture ({texture:.2f}) - printed photo")
                return self._result(False, 0.0, f"Printed photo detected (low texture)")
            
            # FIXED: Added high-quality fast path
            if texture > 25 and self._calc_flatness(image) < 0.6:
                logger.info(f"✅ High quality real face - texture={texture:.1f}")
                return self._result(True, 0.92, None)
            
            # Just warning, not blocking (FIXED: lowered warning threshold)
            if texture < 15:
                logger.warning(f"⚠️ LAYER 1 WARNING: Low texture ({texture:.2f})")
            
            # ======================================================================
            # LAYER 1.5: FLATNESS DETECTION - VERY FORGIVING (FIXED)
            # ======================================================================
            flatness = self._calc_flatness(image)
            logger.info(f"📊 LAYER 1.5 - Flatness score: {flatness:.3f}")
            
            # Only block extremely flat surfaces (FIXED: increased from 0.90 to 0.95)
            if flatness > 0.95:
                logger.warning(f"🚫 LAYER 1.5 BLOCK: Flat surface detected - printed photo (flatness={flatness:.3f})")
                return self._result(False, 0.0, f"Printed photo detected (flat surface)")
            
            # ======================================================================
            # LAYER 1.6: AI-GENERATED FACE DETECTION
            # ======================================================================
            ai_score, ai_reason = self._calc_ai_artifacts(image)
            logger.info(f"📊 LAYER 1.6 - AI Artifact score: {ai_score:.3f} - {ai_reason}")
            
            if ai_score > 0.70:  # FIXED: Increased from 0.65 to 0.70
                logger.warning(f"🚫 LAYER 1.6 BLOCK: AI-generated face detected - {ai_reason}")
                return self._result(False, 0.0, f"DEEPFAKE detected - {ai_reason}")
            
            # ======================================================================
            # LAYER 2: MOIRÉ PATTERN DETECTION - AGGRESSIVE FOR PHONES
            # ======================================================================
            moire = self._calc_moire(image)
            logger.info(f"📊 LAYER 2 - Moire value: {moire:.2f}")
            
            # Aggressive for phone screens
            if moire > 280:  # FIXED: Increased from 260 to 280
                logger.warning(f"🚫 LAYER 2 BLOCK: Screen pattern detected (moire={moire:.1f})")
                return self._result(False, 0.0, f"Screen pattern detected - possible phone photo")
            
            # ======================================================================
            # LAYER 3: BEZEL DETECTION - AGGRESSIVE FOR PHONES
            # ======================================================================
            bezel = self._calc_bezel_lines(image)
            logger.info(f"📊 LAYER 3 - Bezel lines: {bezel}")
            
            # Aggressive for phone bezels (FIXED: increased from 30 to 35)
            if bezel > 35:
                logger.warning(f"🚫 LAYER 3 BLOCK: Phone bezel detected (lines={bezel})")
                return self._result(False, 0.0, f"Phone bezel detected - possible phone photo")
            
            # ======================================================================
            # LAYER 4: FREQUENCY ANALYSIS - FORGIVING
            # ======================================================================
            frequency = self._calc_frequency(image)
            logger.info(f"📊 LAYER 4 - Frequency value: {frequency:.2f}")
            
            # FIXED: Lowered from 5.0 to 4.0
            if frequency < 4.0:
                logger.warning(f"🚫 LAYER 4 BLOCK: Unusual frequency ({frequency:.2f}) - AI generated")
                return self._result(False, 0.0, f"DEEPFAKE detected - frequency anomaly")
            
            # ======================================================================
            # LAYER 5: COMBINATION DETECTION - AGGRESSIVE FOR PHONES
            # ======================================================================
            
            # Screen pattern + bezel = phone photo (FIXED: increased thresholds)
            if moire > 220 and bezel > 12:  # Was moire>200, bezel>10
                logger.warning(f"🚫 LAYER 5 BLOCK: Screen+bezel combo (moire={moire:.1f}, bezel={bezel})")
                return self._result(False, 0.0, f"Phone photo detected (screen pattern + bezel)")
            
            # Low texture + screen pattern = photo on screen (FIXED: more forgiving)
            if texture < 18 and moire > 230:  # Was texture<15, moire>220
                logger.warning(f"🚫 LAYER 5 BLOCK: Low texture+screen (texture={texture:.1f}, moire={moire:.1f})")
                return self._result(False, 0.0, f"Photo on screen detected")
            
            # High moire + low frequency = screen replay (FIXED: increased thresholds)
            if moire > 230 and frequency < 9.0:  # Was moire>220, freq<8.5
                logger.warning(f"🚫 LAYER 5 BLOCK: Screen replay detected (moire={moire:.1f}, freq={frequency:.1f})")
                return self._result(False, 0.0, f"Screen replay detected")
            
            logger.info(f"✅ LAYER 5 PASS: No dangerous combinations")
            
            # ======================================================================
            # LAYER 6: TEMPORAL ANALYSIS (FLICKER) - FORGIVING
            # ======================================================================
            flicker = self._calc_flicker()
            logger.info(f"📊 LAYER 6 - Flicker value: {flicker:.2f}")
            
            # FIXED: Increased threshold
            if flicker > 9.0 and moire > 180:  # Was 8.0
                logger.warning(f"🚫 LAYER 6 BLOCK: Screen flicker detected (flicker={flicker:.1f})")
                return self._result(False, 0.0, f"Screen flicker detected")
            
            # ======================================================================
            # LAYER 7: REFLECTION DETECTION - AGGRESSIVE FOR PHONES
            # ======================================================================
            reflection_score = self._calc_reflection(image)
            logger.info(f"📊 LAYER 7 - Reflection score: {reflection_score:.2f}")
            
            # FIXED: Increased from 70 to 85
            if reflection_score > 85:
                logger.warning(f"🚫 LAYER 7 BLOCK: Screen reflection detected (score={reflection_score:.1f})")
                return self._result(False, 0.0, f"Screen reflection detected")
            
            # ======================================================================
            # LAYER 8: EDGE SHARPNESS - FORGIVING
            # ======================================================================
            edge_sharpness = self._calc_edge_sharpness(image)
            logger.info(f"📊 LAYER 8 - Edge sharpness: {edge_sharpness:.2f}")
            
            # FIXED: Increased from 40 to 50
            if edge_sharpness > 50:
                logger.warning(f"🚫 LAYER 8 BLOCK: Unnatural edge sharpness - possible screen")
                return self._result(False, 0.0, f"Screen artifact detected")
            
            # ======================================================================
            # LAYER 9: TEMPORAL CONSISTENCY - FORGIVING
            # ======================================================================
            temporal_consistency = self._calc_temporal_consistency()
            logger.info(f"📊 LAYER 9 - Temporal consistency: {temporal_consistency:.3f}")
            
            # FIXED: Lowered from 0.5 to 0.4
            if temporal_consistency < 0.4 and len(self.frame_history) > 5:
                logger.warning(f"🚫 LAYER 9 BLOCK: Unnatural temporal consistency - possible video replay (var={temporal_consistency:.3f})")
                return self._result(False, 0.0, f"Video replay detected - unnatural smoothness")
            
            # ======================================================================
            # LAYER 10: SATURATION VARIATION - FORGIVING
            # ======================================================================
            saturation_var = self._calc_saturation_variation(image)
            logger.info(f"📊 LAYER 10 - Saturation variation: {saturation_var:.2f}")
            
            # FIXED: Lowered from 6 to 5
            if saturation_var < 5 and texture < 35:
                logger.warning(f"🚫 LAYER 10 BLOCK: Unnatural saturation - possible screen")
                return self._result(False, 0.0, f"Screen color anomaly detected")
            
            # ======================================================================
            # LAYER 11: VIDEO COMPRESSION ARTIFACTS - AGGRESSIVE FOR PHONES
            # ======================================================================
            compression_artifacts = self._calc_compression_artifacts(image)
            logger.info(f"📊 LAYER 11 - Compression artifacts: {compression_artifacts:.2f}")
            
            # FIXED: Increased from 7.5 to 8.5
            if compression_artifacts > 8.5:
                logger.warning(f"🚫 LAYER 11 BLOCK: Video compression artifacts detected - possible replay")
                return self._result(False, 0.0, f"Video replay detected - compression artifacts")
            
            # ======================================================================
            # LAYER 12: SCREEN GLARE PATTERN - AGGRESSIVE FOR PHONES
            # ======================================================================
            glare_pattern = self._calc_glare_pattern(image)
            logger.info(f"📊 LAYER 12 - Glare pattern score: {glare_pattern:.2f}")
            
            # FIXED: Increased from 0.65 to 0.75
            if glare_pattern > 0.75:
                logger.warning(f"🚫 LAYER 12 BLOCK: Screen glare pattern detected")
                return self._result(False, 0.0, f"Screen glare detected - possible replay")
            
            # ======================================================================
            # LAYER 13: FORCED SCREEN DETECTION - BALANCED
            # ======================================================================
            screen_score = 0
            if moire > 200:
                screen_score += 1
            if bezel > 15:
                screen_score += 1
            if reflection_score > 60:
                screen_score += 1
            if edge_sharpness > 40:
                screen_score += 1
            if compression_artifacts > 7:
                screen_score += 1
            
            # FIXED: Increased from 3 to 4
            if screen_score >= 4:
                logger.warning(f"🚫 LAYER 13 BLOCK: Multiple screen indicators detected (score={screen_score})")
                return self._result(False, 0.0, f"Screen replay detected - multiple indicators")
            
            # ======================================================================
            # LAYER 14: DEEPFAKE VIDEO DETECTION (AI-generated face artifacts)
            # ======================================================================
            deepfake_video_score = self._calc_deepfake_video_artifacts(image)
            logger.info(f"📊 LAYER 14 - Deepfake video artifacts: {deepfake_video_score:.3f}")
            
            # FIXED: Increased from 0.68 to 0.72
            if deepfake_video_score > 0.72:
                logger.warning(f"🚫 LAYER 14 BLOCK: Deepfake video artifacts detected - AI-generated face")
                return self._result(False, 0.0, f"DEEPFAKE video detected - AI-generated face")
            
            # ======================================================================
            # LAYER 15: VIDEO REPLAY DETECTION - VERY FORGIVING FOR REAL FACES
            # ======================================================================
            is_replay, replay_confidence = self._is_video_replay(image)
            logger.info(f"📊 LAYER 15 - Video replay score: {replay_confidence:.2f}")
            
            # Only block if VERY confident (FIXED: increased from 0.85 to 0.88)
            if is_replay and replay_confidence > 0.88:
                logger.warning(f"🚫 LAYER 15 BLOCK: Video replay detected on screen")
                return self._result(False, 0.0, f"Video replay detected - screen playback")
            
            # ======================================================================
            # ALL CHECKS PASSED - REAL FACE
            # ======================================================================
            logger.info("=" * 50)
            logger.info(f"✅✅✅ ANTI-SPOOF: REAL FACE ACCEPTED")
            logger.info(f"📊 Final metrics - Texture: {texture:.1f}, Moire: {moire:.1f}, Bezel: {bezel}, "
                       f"Flatness: {flatness:.3f}, AI Score: {ai_score:.3f}, Temporal: {temporal_consistency:.3f}, "
                       f"Compression: {compression_artifacts:.2f}, Screen Score: {screen_score}, "
                       f"Deepfake Video: {deepfake_video_score:.3f}, Replay Score: {replay_confidence:.2f}")
            logger.info("=" * 50)
            
            # Calculate confidence
            confidence = 0.75
            if texture > 60:
                confidence += 0.10
            if moire < 150:
                confidence += 0.10
            if flatness < 0.3:
                confidence += 0.05
            
            return {
                "is_live": True,
                "confidence": min(confidence, 0.98),
                "reason": None,
                "texture": round(texture, 1),
                "moire": round(moire, 1),
                "bezel": bezel,
                "flatness": round(flatness, 3),
                "ai_score": round(ai_score, 3),
                "temporal_consistency": round(temporal_consistency, 3),
                "compression_artifacts": round(compression_artifacts, 2),
                "screen_score": screen_score,
                "deepfake_video_score": round(deepfake_video_score, 3),
                "replay_confidence": round(replay_confidence, 2)
            }
            
        except Exception as e:
            logger.error(f"❌ ANTI-SPOOF ERROR: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return self._result(True, 0.5, None)

    # ======================================================================
    # _is_video_replay - VERY FORGIVING FOR REAL FACES (FIXED)
    # ======================================================================
    def _is_video_replay(self, image: np.ndarray) -> Tuple[bool, float]:
        """
        Specialized detection for video replays on screens.
        VERY FORGIVING - only blocks extreme cases.
        """
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # 1. Check brightness uniformity (only block if extremely uniform)
            brightness_std = np.std(gray)
            if brightness_std < 15:  # FIXED: increased from 12 to 15
                logger.warning(f"⚠️ Unnatural brightness uniformity: {brightness_std:.1f}")
                return True, 0.85
            
            # 2. Check texture detail (only block if extremely low)
            lap = cv2.Laplacian(gray, cv2.CV_64F)
            texture_var = lap.var()
            if texture_var < 45:  # FIXED: increased from 35 to 45
                logger.warning(f"⚠️ Low texture detail: {texture_var:.1f}")
                return True, 0.80
            
            # 3. Check color banding (only block if severe)
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            hue_hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
            hue_hist = hue_hist / (hue_hist.sum() + 1e-8)
            non_zero_bins = np.sum(hue_hist > 0.001)
            if non_zero_bins < 60:  # FIXED: increased from 50 to 60
                logger.warning(f"⚠️ Severe color banding detected: {non_zero_bins} bins")
                return True, 0.90
            
            # 4. Check edge density (only block if extremely high)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            if edge_density > 0.22:  # FIXED: increased from 0.18 to 0.22
                logger.warning(f"⚠️ Unnatural edge density: {edge_density:.3f}")
                return True, 0.75
            
            return False, 0.0
            
        except Exception as e:
            logger.error(f"Video replay detection error: {e}")
            return False, 0.0

    # ======================================================================
    # _calc_bezel_lines - AGGRESSIVE FOR PHONES BUT FORGIVING FOR FACES
    # ======================================================================
    def _calc_bezel_lines(self, image: np.ndarray) -> int:
        """Count straight lines (phone bezels) - aggressive for phones."""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Use medium blur for balance
            blurred = cv2.GaussianBlur(gray, (3, 3), 0)
            edges = cv2.Canny(blurred, 100, 220)
            lines = cv2.HoughLinesP(
                edges, 
                rho=1, 
                theta=np.pi/180,
                threshold=100,
                minLineLength=100,
                maxLineGap=20
            )
            
            # Count ALL lines (not just edge lines) for better phone detection
            return len(lines) if lines is not None else 0
            
        except Exception as e:
            logger.error(f"Bezel calculation error: {e}")
            return 0

    # ======================================================================
    # All other helper methods (standard versions)
    # ======================================================================
    
    def _calc_texture(self, image: np.ndarray) -> float:
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            lap = cv2.Laplacian(gray, cv2.CV_64F)
            return float(lap.var())
        except Exception as e:
            logger.error(f"Texture calculation error: {e}")
            return 0.0

    def _calc_flatness(self, image: np.ndarray) -> float:
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            grad_mag = np.sqrt(grad_x**2 + grad_y**2)
            grad_std = np.std(grad_mag)
            flatness = 1.0 - min(1.0, grad_std / 50.0)
            return float(flatness)
        except Exception as e:
            logger.error(f"Flatness calculation error: {e}")
            return 0.0

    def _calc_ai_artifacts(self, image: np.ndarray) -> tuple:
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            f = np.fft.fft2(gray)
            fshift = np.fft.fftshift(f)
            magnitude = 20 * np.log(np.abs(fshift) + 1e-8)
            freq_std = np.std(magnitude)
            lap = cv2.Laplacian(gray, cv2.CV_64F)
            texture_uniformity = np.std(lap)
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            color_std = np.std(hsv[:, :, 0])
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            
            ai_score = 0.0
            reasons = []
            
            if freq_std < 7.0:
                ai_score += 0.35
                reasons.append("low_frequency_variance")
            elif freq_std < 8.5:
                ai_score += 0.20
                reasons.append("moderate_frequency_anomaly")
            
            if texture_uniformity < 9:
                ai_score += 0.30
                reasons.append("uniform_texture")
            elif texture_uniformity < 16:
                ai_score += 0.15
                reasons.append("smooth_texture")
            
            if color_std < 22:
                ai_score += 0.20
                reasons.append("low_color_variation")
            
            if edge_density < 0.02 or edge_density > 0.08:
                ai_score += 0.15
                reasons.append("unnatural_edge_density")
            
            ai_score = min(ai_score, 1.0)
            reason_str = ", ".join(reasons[:2]) if reasons else "normal"
            return ai_score, reason_str
            
        except Exception as e:
            logger.error(f"AI artifact calculation error: {e}")
            return 0.0, "error"

    def _calc_deepfake_video_artifacts(self, image: np.ndarray) -> float:
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape
            face_roi = gray[h//4:3*h//4, w//4:3*w//4]
            
            edges = cv2.Canny(face_roi, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            edge_anomaly = abs(edge_density - 0.05) > 0.03
            
            lap = cv2.Laplacian(face_roi, cv2.CV_64F)
            texture_uniformity = np.std(lap)
            texture_too_uniform = texture_uniformity < 10
            
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            skin_hue = hsv[h//4:3*h//4, w//4:3*w//4, 0]
            hue_std = np.std(skin_hue)
            hue_anomaly = hue_std < 7 or hue_std > 24
            
            f = np.fft.fft2(face_roi)
            fshift = np.fft.fftshift(f)
            magnitude = 20 * np.log(np.abs(fshift) + 1e-8)
            freq_std = np.std(magnitude)
            freq_anomaly = freq_std < 7.0
            
            face_edges = cv2.Canny(face_roi, 30, 100)
            edge_variation = np.std(face_edges) if np.sum(face_edges) > 0 else 0
            edge_variation_anomaly = edge_variation < 15
            
            score = 0.0
            if edge_anomaly:
                score += 0.20
            if texture_too_uniform:
                score += 0.25
            if hue_anomaly:
                score += 0.20
            if freq_anomaly:
                score += 0.20
            if edge_variation_anomaly:
                score += 0.15
            
            return min(score, 1.0)
            
        except Exception as e:
            logger.error(f"Deepfake video artifact error: {e}")
            return 0.0

    def _calc_moire(self, image: np.ndarray) -> float:
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            f = np.fft.fft2(gray)
            fshift = np.fft.fftshift(f)
            magnitude = 20 * np.log(np.abs(fshift) + 1e-8)
            h, w = magnitude.shape
            center = magnitude[h//2-16:h//2+16, w//2-16:w//2+16]
            return float(np.mean(center))
        except Exception as e:
            logger.error(f"Moire calculation error: {e}")
            return 500.0

    def _calc_frequency(self, image: np.ndarray) -> float:
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            f = np.fft.fft2(gray)
            fshift = np.fft.fftshift(f)
            magnitude = 20 * np.log(np.abs(fshift) + 1e-8)
            return float(np.std(magnitude))
        except Exception as e:
            logger.error(f"Frequency calculation error: {e}")
            return 12.0

    def _calc_flicker(self) -> float:
        if len(self.frame_history) < 3:
            return 0.0
        try:
            brightness = []
            for frame in self.frame_history[-3:]:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                brightness.append(np.mean(gray))
            return float(np.std(brightness))
        except Exception as e:
            logger.error(f"Flicker calculation error: {e}")
            return 0.0

    def _calc_temporal_consistency(self) -> float:
        if len(self.frame_history) < 5:
            return 0.0
        try:
            variations = []
            for i in range(1, len(self.frame_history)):
                prev = cv2.cvtColor(self.frame_history[i-1], cv2.COLOR_BGR2GRAY)
                curr = cv2.cvtColor(self.frame_history[i], cv2.COLOR_BGR2GRAY)
                diff = np.mean(cv2.absdiff(prev, curr))
                variations.append(diff)
            std_var = np.std(variations) if variations else 0
            return float(std_var)
        except Exception as e:
            logger.error(f"Temporal consistency error: {e}")
            return 0.0

    def _calc_saturation_variation(self, image: np.ndarray) -> float:
        try:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            saturation = hsv[:, :, 1]
            return float(np.std(saturation))
        except Exception as e:
            logger.error(f"Saturation variation error: {e}")
            return 0.0

    def _calc_reflection(self, image: np.ndarray) -> float:
        try:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            saturation = hsv[:, :, 1]
            sat_mean = np.mean(saturation)
            sat_std = np.std(saturation)
            if sat_mean > 80 and sat_std < 50:
                return sat_mean
            return 0.0
        except Exception as e:
            logger.error(f"Reflection calculation error: {e}")
            return 0.0

    def _calc_edge_sharpness(self, image: np.ndarray) -> float:
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 100, 200)
            edge_intensity = np.mean(edges) if np.sum(edges) > 0 else 0
            return edge_intensity
        except Exception as e:
            logger.error(f"Edge sharpness calculation error: {e}")
            return 0.0

    def _calc_compression_artifacts(self, image: np.ndarray) -> float:
        try:
            _, encoded = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 85])
            decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            diff = cv2.absdiff(image.astype(np.float32), decoded.astype(np.float32))
            diff_mean = np.mean(diff)
            
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            block_size = 8
            h, w = gray.shape
            blockiness = 0
            for y in range(0, h - block_size, block_size):
                for x in range(0, w - block_size, block_size):
                    block = gray[y:y+block_size, x:x+block_size]
                    blockiness += np.std(block)
            blockiness = blockiness / ((h // block_size) * (w // block_size) + 1e-8)
            
            artifact_score = diff_mean * 2 + (blockiness / 100)
            return float(artifact_score)
        except Exception as e:
            logger.error(f"Compression artifacts error: {e}")
            return 0.0

    def _calc_glare_pattern(self, image: np.ndarray) -> float:
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            glare_score = 0.0
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 100:
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = max(w, h) / (min(w, h) + 0.01)
                    if aspect_ratio > 3:
                        glare_score += 0.2
            return min(glare_score, 1.0)
        except Exception as e:
            logger.error(f"Glare pattern error: {e}")
            return 0.0

    def _update_history(self, frame: np.ndarray) -> None:
        self.frame_history.append(frame.copy())
        if len(self.frame_history) > 10:
            self.frame_history.pop(0)

    def _result(self, is_live: bool, confidence: float, reason: str = None) -> Dict[str, Any]:
        return {
            "is_live": is_live,
            "confidence": confidence,
            "reason": reason
        }

    def reset(self) -> None:
        self.frame_history.clear()
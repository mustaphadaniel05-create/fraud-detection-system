# app/services/hardware_liveness_service.py
"""
Hardware-based liveness detection using depth and IR cameras.
Supports Intel RealSense, iPhone TrueDepth, and thermal cameras.
"""

import logging
import numpy as np
from typing import Dict, Any, Optional, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


class DepthCameraType(Enum):
    """Supported depth camera types."""
    REALSENSE = "realsense"
    IPHONE_DEPTH = "iphone_depth"
    IR_CAMERA = "ir_camera"
    NONE = "none"


class HardwareLivenessService:
    """
    Hardware-based liveness detection.
    Uses depth maps and IR signatures to detect real faces.
    """
    
    def __init__(self):
        self.enabled = False
        self.camera_type = DepthCameraType.NONE
        self.depth_camera = None
        self.ir_camera = None
        self.pipeline = None
        
        self._init_camera()
    
    def _init_camera(self):
        """Initialize depth/IR camera if available."""
        from config import Config
        
        if not Config.HARDWARE_LIVENESS_ENABLED:
            logger.info("Hardware liveness disabled in config")
            return
        
        camera_type = Config.DEPTH_CAMERA_TYPE
        
        if camera_type == "realsense":
            self._init_realsense()
        elif camera_type == "iphone_depth":
            self._init_iphone_depth()
        elif camera_type == "ir_camera":
            self._init_ir_camera()
        else:
            logger.warning(f"Unknown depth camera type: {camera_type}")
    
    def _init_realsense(self):
        """Initialize Intel RealSense camera."""
        try:
            import pyrealsense2 as rs
            
            # Configure pipeline
            self.pipeline = rs.pipeline()
            config = rs.config()
            
            # Enable depth and color streams
            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            
            # Start streaming
            self.pipeline.start(config)
            
            # Get device info
            profile = self.pipeline.get_active_profile()
            device = profile.get_device()
            logger.info(f"RealSense camera initialized: {device.get_info(rs.camera_info.name)}")
            
            self.enabled = True
            self.camera_type = DepthCameraType.REALSENSE
            
        except ImportError:
            logger.warning("pyrealsense2 not installed. RealSense support disabled.")
        except Exception as e:
            logger.error(f"Failed to initialize RealSense: {e}")
    
    def _init_iphone_depth(self):
        """Initialize iPhone TrueDepth camera (via AVFoundation)."""
        logger.info("iPhone TrueDepth camera support requires iOS device")
        self.enabled = False
    
    def _init_ir_camera(self):
        """Initialize IR/thermal camera."""
        logger.info("IR camera support requires specific hardware")
        self.enabled = False
    
    def is_available(self) -> bool:
        """Check if hardware liveness is available."""
        return self.enabled
    
    def get_depth_frame(self) -> Optional[np.ndarray]:
        """Get current depth frame."""
        if not self.enabled:
            return None
        
        try:
            if self.camera_type == DepthCameraType.REALSENSE:
                import pyrealsense2 as rs
                
                # Wait for frames
                frames = self.pipeline.wait_for_frames(timeout_ms=1000)
                if not frames:
                    return None
                
                depth_frame = frames.get_depth_frame()
                if not depth_frame:
                    return None
                
                # Convert to numpy array
                depth_image = np.asanyarray(depth_frame.get_data())
                return depth_image
                
        except Exception as e:
            logger.error(f"Failed to get depth frame: {e}")
        
        return None
    
    def get_color_frame(self) -> Optional[np.ndarray]:
        """Get current color frame."""
        if not self.enabled:
            return None
        
        try:
            if self.camera_type == DepthCameraType.REALSENSE:
                import pyrealsense2 as rs
                
                frames = self.pipeline.wait_for_frames(timeout_ms=1000)
                if not frames:
                    return None
                
                color_frame = frames.get_color_frame()
                if not color_frame:
                    return None
                
                # Convert to numpy array
                color_image = np.asanyarray(color_frame.get_data())
                return color_image
                
        except Exception as e:
            logger.error(f"Failed to get color frame: {e}")
        
        return None
    
    def get_face_depth_metrics(self, face_region: Tuple[int, int, int, int]) -> Dict[str, Any]:
        """
        Analyze depth metrics for a face region.
        
        Args:
            face_region: (x, y, width, height) in image coordinates
            
        Returns:
            Dict with depth statistics
        """
        depth_frame = self.get_depth_frame()
        if depth_frame is None:
            return {"available": False}
        
        x, y, w, h = face_region
        
        # Extract face region from depth map
        face_depth = depth_frame[y:y+h, x:x+w]
        
        # Filter out zero values (invalid depth)
        valid_depth = face_depth[face_depth > 0]
        
        if len(valid_depth) < 100:
            return {
                "available": True,
                "valid_pixels": len(valid_depth),
                "error": "Insufficient depth data"
            }
        
        # Calculate metrics
        depth_mean = float(np.mean(valid_depth))
        depth_std = float(np.std(valid_depth))
        depth_min = float(np.min(valid_depth))
        depth_max = float(np.max(valid_depth))
        depth_range = depth_max - depth_min
        
        # Calculate face curvature (3D shape)
        # Real faces have natural curvature, screens are flat
        curvature_score = self._calculate_curvature(face_depth)
        
        return {
            "available": True,
            "depth_mean_mm": round(depth_mean, 1),
            "depth_std_mm": round(depth_std, 1),
            "depth_range_mm": round(depth_range, 1),
            "curvature_score": round(curvature_score, 3),
            "valid_pixels": len(valid_depth)
        }
    
    def _calculate_curvature(self, depth_map: np.ndarray) -> float:
        """
        Calculate 3D curvature score.
        Real faces have natural curvature, screens are flat.
        """
        try:
            # Ensure we have valid data
            if depth_map is None or depth_map.size == 0:
                return 0.0
            
            # Filter out zero or invalid depth values
            valid_depth = depth_map[depth_map > 0]
            if len(valid_depth) < 100:
                return 0.0
            
            # Convert to float for gradient calculation
            depth_float = depth_map.astype(np.float32)
            
            # Calculate gradient
            gy, gx = np.gradient(depth_float)
            
            # Calculate curvature magnitude
            curvature = np.sqrt(gx**2 + gy**2)
            
            # Get mean curvature of valid pixels only
            valid_curvature = curvature[depth_map > 0]
            if len(valid_curvature) == 0:
                return 0.0
            
            mean_curvature = float(np.mean(valid_curvature))
            
            # Normalize to 0-1 range
            # Real faces typically have curvature between 0.5-2.0
            # Screens/photos have curvature near 0
            normalized = min(1.0, mean_curvature / 3.0)
            
            return normalized
            
        except Exception as e:
            logger.debug(f"Curvature calculation error: {e}")
            return 0.0
    
    def validate_depth_liveness(self, depth_metrics: Dict[str, Any]) -> Tuple[bool, str, float]:
        """
        Validate if face is real based on depth metrics.
        
        Returns:
            (is_live, reason, confidence)
        """
        from config import Config
        
        if not depth_metrics.get("available", False):
            return False, "Depth data not available", 0.0
        
        depth_range = depth_metrics.get("depth_range_mm", 0)
        curvature = depth_metrics.get("curvature_score", 0)
        depth_std = depth_metrics.get("depth_std_mm", 0)
        
        # Check 1: Depth variation (real faces have 3D shape)
        if depth_range < Config.DEPTH_THRESHOLD_MM:
            return False, f"Flat face detected (depth range {depth_range:.1f}mm)", 0.2
        
        # Check 2: Curvature (real faces have natural curvature)
        if curvature < 0.3:
            return False, f"Unnatural face curvature (score {curvature:.2f})", 0.3
        
        # Calculate confidence
        confidence = 0.5
        
        # Higher depth range = more confidence
        if depth_range > 80:
            confidence += 0.3
        elif depth_range > 60:
            confidence += 0.2
        elif depth_range > Config.DEPTH_THRESHOLD_MM:
            confidence += 0.1
        
        # Higher curvature = more confidence
        if curvature > 0.7:
            confidence += 0.2
        elif curvature > 0.5:
            confidence += 0.1
        
        confidence = min(1.0, confidence)
        
        return True, "Valid 3D face detected", confidence
    
    def capture_with_depth(self) -> Dict[str, Any]:
        """
        Capture a frame with both color and depth data.
        Returns combined result for verification.
        """
        color_frame = self.get_color_frame()
        if color_frame is None:
            return {"success": False, "error": "No color frame"}
        
        depth_frame = self.get_depth_frame()
        
        return {
            "success": True,
            "color_frame": color_frame,
            "depth_frame": depth_frame,
            "depth_available": depth_frame is not None
        }
    
    def close(self):
        """Close camera connection."""
        if self.camera_type == DepthCameraType.REALSENSE and hasattr(self, 'pipeline') and self.pipeline:
            self.pipeline.stop()
            logger.info("RealSense pipeline stopped")


# Singleton instance
_hardware_liveness = None


def get_hardware_liveness() -> HardwareLivenessService:
    """Get or create global hardware liveness instance."""
    global _hardware_liveness
    if _hardware_liveness is None:
        _hardware_liveness = HardwareLivenessService()
    return _hardware_liveness


def check_hardware_liveness(face_region: Tuple[int, int, int, int]) -> Dict[str, Any]:
    """Convenience function for hardware liveness check."""
    service = get_hardware_liveness()
    
    if not service.is_available():
        return {
            "is_live": None,
            "reason": "Hardware liveness not available",
            "confidence": 0.0,
            "available": False
        }
    
    depth_metrics = service.get_face_depth_metrics(face_region)
    is_live, reason, confidence = service.validate_depth_liveness(depth_metrics)
    
    return {
        "is_live": is_live,
        "reason": reason,
        "confidence": confidence,
        "available": True,
        "depth_metrics": depth_metrics
    }
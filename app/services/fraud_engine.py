"""
BALANCED fraud risk engine with XceptionNet deepfake integration.
More forgiving thresholds for real enrolled faces.
"""

import logging
from typing import Optional
from config import Config

logger = logging.getLogger(__name__)


def calculate_risk(
    similarity: float,
    liveness_confidence: float,
    antispoof_confidence: float,
    deepfake_vote_ratio: float,
    motion_score: Optional[float] = None,
    recent_attempts: int = 0,
    deepfake_confidence: float = 0,
    face_width_pct: Optional[float] = None,
) -> int:
    """
    BALANCED risk calculation with more forgiving thresholds for real faces.
    
    Returns:
        risk_score: 0-100 (higher = more risky)
    """
    similarity = max(0.0, min(1.0, similarity))
    liveness_confidence = max(0.0, min(1.0, liveness_confidence))
    deepfake_vote_ratio = max(0.0, min(1.0, deepfake_vote_ratio))
    deepfake_confidence = max(0.0, min(1.0, deepfake_confidence))
    motion_score = max(0.0, motion_score or 0.0)
    recent_attempts = max(0, recent_attempts)

    risk = 0

    # 0. FACE SIZE ADJUSTMENT - More forgiving for normal distances
    if face_width_pct is not None:
        if face_width_pct < 12:
            risk -= 10  # Slightly far - reduce risk
            logger.info(f"📏 Far face adjustment: -10 (face size {face_width_pct:.1f}%)")
        elif face_width_pct < 15:
            risk -= 5   # A bit far - small reduction
            logger.info(f"📏 Slightly far adjustment: -5 (face size {face_width_pct:.1f}%)")
        elif 20 <= face_width_pct <= 45:
            risk -= 0   # Good distance - no adjustment
        elif face_width_pct > 50:
            risk += 5   # Too close - slight penalty
            logger.info(f"📏 Too close adjustment: +5 (face size {face_width_pct:.1f}%)")
    
    # Ensure risk doesn't go negative
    risk = max(0, risk)

    # 1. Identity similarity (0-40 points) - MORE FORGIVING
    if similarity >= 0.82:
        risk += 0      # Excellent match (lowered from 0.85)
    elif similarity >= 0.75:
        risk += 5      # Very good (lowered from 0.78)
    elif similarity >= 0.68:
        risk += 10     # Good (lowered from 0.72)
    elif similarity >= 0.60:
        risk += 18     # Borderline (lowered from 0.65)
    elif similarity >= 0.50:
        risk += 30     # Poor (lowered from 0.55)
    else:
        risk += 45     # Very poor (lowered from 50)

    # 2. Liveness confidence (0-35 points) - MORE FORGIVING
    if liveness_confidence >= 0.80:
        risk += 0
    elif liveness_confidence >= 0.70:
        risk += 5
    elif liveness_confidence >= 0.60:
        risk += 12
    elif liveness_confidence >= 0.50:
        risk += 22
    else:
        risk += 35

    # 3. Deepfake detection - MORE FORGIVING for real faces
    if deepfake_confidence > 0:
        # Using XceptionNet
        if deepfake_confidence >= 0.85:
            risk += 45   # Very likely deepfake (lowered from 50)
        elif deepfake_confidence >= 0.75:
            risk += 35
        elif deepfake_confidence >= 0.65:
            risk += 25
        elif deepfake_confidence >= 0.55:
            risk += 15
        elif deepfake_confidence >= 0.45:
            risk += 8
        else:
            risk += 0
    else:
        # Fallback to heuristic vote ratio
        if deepfake_vote_ratio <= 0.25:
            risk += 0
        elif deepfake_vote_ratio <= 0.35:
            risk += 5
        elif deepfake_vote_ratio <= 0.45:
            risk += 10
        elif deepfake_vote_ratio <= 0.60:
            risk += 20
        else:
            risk += 35

    # 4. Motion analysis (0-30 points) - MORE FORGIVING
    if motion_score >= 3.5:
        risk += 0      # Good natural movement
    elif motion_score >= 2.5:
        risk += 5
    elif motion_score >= 1.5:
        risk += 10
    elif motion_score >= 0.8:
        risk += 18
    else:
        risk += 30     # Static or minimal movement

    # 5. Recent attempts (0-25 points) - MORE FORGIVING
    if recent_attempts >= 10:
        risk += 25
    elif recent_attempts >= 7:
        risk += 18
    elif recent_attempts >= 4:
        risk += 10
    elif recent_attempts >= 2:
        risk += 5
    # else: 0

    # Cap at 100
    risk_score = min(int(risk), 100)
    
    logger.info(
        f"📊 BALANCED RISK: {risk_score} | "
        f"sim={similarity:.2f} | liveness={liveness_confidence:.2f} | "
        f"df_advanced={deepfake_confidence:.2f} | df_heuristic={deepfake_vote_ratio:.2f} | "
        f"motion={motion_score:.1f} | attempts={recent_attempts} | "
        f"face_size={face_width_pct:.1f}%"
    )

    return risk_score


def decide(risk_score: int) -> str:
    """
    BALANCED decision thresholds - MORE FORGIVING:
    - 0-30: APPROVED_PASSIVE (real faces, high confidence)
    - 31-55: REQUIRES_ACTIVE (borderline, need active liveness)
    - 56-80: HIGH_RISK_BLOCK (potential fraud)
    - 81+: CRITICAL_BLOCK (immediate rejection + permanent flag)
    """
    if risk_score <= 30:
        logger.info(f"✅ PASSIVE APPROVAL: risk={risk_score}")
        return "APPROVED_PASSIVE"
    
    if risk_score <= 55:
        logger.info(f"⚠️ ACTIVE CHALLENGE REQUIRED: risk={risk_score}")
        return "REQUIRES_ACTIVE_LIVENESS"
    
    if risk_score <= 80:
        logger.warning(f"🚫 HIGH RISK - BLOCKED: risk={risk_score}")
        return "BLOCKED_HIGH_RISK"
    
    logger.error(f"💀 CRITICAL RISK - PERMANENT FLAG: risk={risk_score}")
    return "CRITICAL_BLOCK"


def is_critical_risk(risk_score: int) -> bool:
    """Check if risk score warrants permanent flagging."""
    return risk_score >= 81


def get_risk_level(risk_score: int) -> str:
    """Get human-readable risk level."""
    if risk_score <= 30:
        return "LOW"
    elif risk_score <= 55:
        return "MEDIUM"
    elif risk_score <= 80:
        return "HIGH"
    else:
        return "CRITICAL"
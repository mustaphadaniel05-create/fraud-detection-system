"""
FRAUD RISK ENGINE – STRICTER FOR DEEPFAKES.
Higher penalties for deepfake scores, lower passive approval threshold.
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
    STRICTER risk calculation – deepfakes get high risk scores.
    """
    similarity = max(0.0, min(1.0, similarity))
    liveness_confidence = max(0.0, min(1.0, liveness_confidence))
    deepfake_vote_ratio = max(0.0, min(1.0, deepfake_vote_ratio))
    deepfake_confidence = max(0.0, min(1.0, deepfake_confidence))
    motion_score = max(0.0, motion_score or 0.0)
    recent_attempts = max(0, recent_attempts)

    risk = 0

    # 0. FACE SIZE ADJUSTMENT
    if face_width_pct is not None:
        if face_width_pct < 12:
            risk -= 10
            logger.info(f"Far face adjustment: -10 (face size {face_width_pct:.1f}%)")
        elif face_width_pct < 15:
            risk -= 5
        elif face_width_pct > 50:
            risk += 5
            logger.info(f"Too close adjustment: +5 (face size {face_width_pct:.1f}%)")
    risk = max(0, risk)

    # 1. Identity similarity (0-40)
    if similarity >= 0.82:
        risk += 0
    elif similarity >= 0.75:
        risk += 5
    elif similarity >= 0.68:
        risk += 10
    elif similarity >= 0.60:
        risk += 18
    elif similarity >= 0.50:
        risk += 30
    else:
        risk += 45

    # 2. Liveness confidence (0-35)
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

    # 3. Deepfake detection – STRICTER (higher penalties)
    if deepfake_confidence > 0:
        if deepfake_confidence >= 0.70:
            risk += 55   # was 45
        elif deepfake_confidence >= 0.55:
            risk += 45   # was 35
        elif deepfake_confidence >= 0.45:
            risk += 35   # was 25
        elif deepfake_confidence >= 0.35:
            risk += 20   # was 15
        else:
            risk += 10   # was 8
    else:
        # Fallback heuristic
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

    # 4. Motion analysis (0-30) – penalize static videos more
    if motion_score >= 3.5:
        risk += 0
    elif motion_score >= 2.5:
        risk += 5
    elif motion_score >= 1.5:
        risk += 10
    elif motion_score >= 0.8:
        risk += 18
    else:
        risk += 30   # static → high risk

    # 5. Recent attempts (0-25)
    if recent_attempts >= 10:
        risk += 25
    elif recent_attempts >= 7:
        risk += 18
    elif recent_attempts >= 4:
        risk += 10
    elif recent_attempts >= 2:
        risk += 5

    risk_score = min(int(risk), 100)

    logger.info(
        f"STRICT RISK: {risk_score} | sim={similarity:.2f} | liveness={liveness_confidence:.2f} | "
        f"df={deepfake_confidence:.2f} | motion={motion_score:.1f} | attempts={recent_attempts}"
    )
    return risk_score


def decide(risk_score: int) -> str:
    """
    STRICTER thresholds:
    - 0-25: APPROVED_PASSIVE (very high confidence)
    - 26-50: REQUIRES_ACTIVE
    - 51-80: HIGH_RISK_BLOCK
    - 81+: CRITICAL_BLOCK
    """
    if risk_score <= 25:
        logger.info(f"PASSIVE APPROVAL: risk={risk_score}")
        return "APPROVED_PASSIVE"

    if risk_score <= 50:
        logger.info(f"ACTIVE CHALLENGE REQUIRED: risk={risk_score}")
        return "REQUIRES_ACTIVE_LIVENESS"

    if risk_score <= 80:
        logger.warning(f"HIGH RISK - BLOCKED: risk={risk_score}")
        return "BLOCKED_HIGH_RISK"

    logger.error(f"CRITICAL RISK - PERMANENT FLAG: risk={risk_score}")
    return "CRITICAL_BLOCK"


def is_critical_risk(risk_score: int) -> bool:
    return risk_score >= 81


def get_risk_level(risk_score: int) -> str:
    if risk_score <= 25:
        return "LOW"
    elif risk_score <= 50:
        return "MEDIUM"
    elif risk_score <= 80:
        return "HIGH"
    else:
        return "CRITICAL"
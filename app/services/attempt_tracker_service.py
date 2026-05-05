# app/services/attempt_tracker_service.py

"""
Production-grade attempt tracking with Redis backend.
Falls back to in-memory storage if Redis is unavailable.
"""

import time
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Configuration
WINDOW_SECONDS = 300   # 5 minutes
MAX_ATTEMPTS = 10

# Fallback in-memory store (used when Redis is unavailable)
_attempt_store: Dict[str, List[float]] = {}

# Redis service (lazy-loaded)
_redis_service = None


def _get_redis():
    """Lazy load Redis service."""
    global _redis_service
    if _redis_service is None:
        try:
            from app.services.redis_service import get_redis_service
            _redis_service = get_redis_service()
        except ImportError:
            logger.warning("Redis service not available, using in-memory fallback")
            _redis_service = None
    return _redis_service


def _cleanup_attempts(identifier: str) -> None:
    """
    Remove old attempts outside the time window (in-memory fallback).
    """
    now = time.time()

    if identifier not in _attempt_store:
        return

    _attempt_store[identifier] = [
        t for t in _attempt_store[identifier]
        if now - t < WINDOW_SECONDS
    ]


def record_attempt(identifier: str) -> int:
    """
    Record a verification attempt using Redis (with fallback).

    Args:
        identifier: user_id or client_ip

    Returns:
        number of recent attempts
    """
    redis = _get_redis()
    
    # Use Redis if available
    if redis and redis.is_available():
        try:
            attempt_count = redis.record_attempt(identifier, WINDOW_SECONDS)
            logger.info(f"Attempt recorded (Redis) | id={identifier} attempts={attempt_count}")
            return attempt_count
        except Exception as e:
            logger.error(f"Redis record_attempt failed: {e}, falling back to memory")
    
    # Fallback to in-memory
    now = time.time()

    if identifier not in _attempt_store:
        _attempt_store[identifier] = []

    _cleanup_attempts(identifier)

    _attempt_store[identifier].append(now)

    attempt_count = len(_attempt_store[identifier])

    logger.info(f"Attempt recorded (Memory) | id={identifier} attempts={attempt_count}")

    return attempt_count


def get_recent_attempts(identifier: str) -> int:
    """
    Get number of recent attempts inside time window.
    """
    redis = _get_redis()
    
    # Use Redis if available
    if redis and redis.is_available():
        try:
            return redis.get_recent_attempts(identifier, WINDOW_SECONDS)
        except Exception as e:
            logger.error(f"Redis get_recent_attempts failed: {e}, falling back to memory")
    
    # Fallback to in-memory
    _cleanup_attempts(identifier)
    return len(_attempt_store.get(identifier, []))


def is_blocked(identifier: str) -> bool:
    """
    Check if identifier exceeded allowed attempts.
    Also checks for manual blocks.
    """
    redis = _get_redis()
    
    # Check Redis manual blocks if available
    if redis and redis.is_available():
        try:
            if redis.is_manually_blocked(identifier):
                logger.warning(f"Identifier manually blocked: {identifier}")
                return True
        except Exception as e:
            logger.error(f"Redis is_manually_blocked failed: {e}")

    attempts = get_recent_attempts(identifier)
    blocked = attempts >= MAX_ATTEMPTS

    if blocked:
        logger.warning(
            f"Identifier blocked due to excessive attempts | id={identifier} attempts={attempts}"
        )

    return blocked


def reset_attempts(identifier: str) -> None:
    """
    Reset attempts after successful verification.
    """
    redis = _get_redis()
    
    # Use Redis if available
    if redis and redis.is_available():
        try:
            redis.reset_attempts(identifier)
            logger.info(f"Attempts reset (Redis) | id={identifier}")
            return
        except Exception as e:
            logger.error(f"Redis reset_attempts failed: {e}, falling back to memory")
    
    # Fallback to in-memory
    if identifier in _attempt_store:
        del _attempt_store[identifier]

    logger.info(f"Attempts reset (Memory) | id={identifier}")


def block_identifier(identifier: str, duration_seconds: int = 600) -> None:
    """
    Explicitly block an identifier for suspicious activity.
    Only works with Redis backend.
    """
    redis = _get_redis()
    
    if redis and redis.is_available():
        try:
            redis.block_identifier(identifier, duration_seconds)
            logger.warning(f"Manually blocked {identifier} for {duration_seconds}s")
        except Exception as e:
            logger.error(f"Redis block_identifier failed: {e}")
    else:
        logger.warning(f"Cannot manually block {identifier} - Redis unavailable")
        logger.info("Manual blocking requires Redis. Falling back to rate limiting only.")


def is_manually_blocked(identifier: str) -> bool:
    """
    Check if identifier is manually blocked.
    Only works with Redis backend.
    """
    redis = _get_redis()
    
    if redis and redis.is_available():
        try:
            return redis.is_manually_blocked(identifier)
        except Exception as e:
            logger.error(f"Redis is_manually_blocked failed: {e}")
    
    return False


def get_block_status(identifier: str) -> Dict[str, any]:
    """
    Get detailed block status for an identifier.
    Returns dict with blocked status, attempts, and remaining time if blocked.
    """
    redis = _get_redis()
    
    result = {
        "blocked": False,
        "attempts": 0,
        "max_attempts": MAX_ATTEMPTS,
        "window_seconds": WINDOW_SECONDS,
        "remaining_attempts": MAX_ATTEMPTS,
        "manually_blocked": False,
        "time_remaining": None
    }
    
    # Check manual block first
    if redis and redis.is_available():
        try:
            result["manually_blocked"] = redis.is_manually_blocked(identifier)
            if result["manually_blocked"]:
                result["blocked"] = True
                # Get remaining time (approximate)
                key = f"block:{identifier}"
                ttl = redis.client.ttl(key) if redis.client else None
                if ttl and ttl > 0:
                    result["time_remaining"] = ttl
                return result
        except Exception as e:
            logger.error(f"Redis manual block check failed: {e}")
    
    # Check rate limiting
    attempts = get_recent_attempts(identifier)
    result["attempts"] = attempts
    result["remaining_attempts"] = max(0, MAX_ATTEMPTS - attempts)
    result["blocked"] = attempts >= MAX_ATTEMPTS
    
    return result
# app/services/redis_service.py
"""
Redis-based rate limiting and attempt tracking for production.
Persistent across server restarts and shared across workers.
"""

import logging
import redis
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import json

from config import Config

logger = logging.getLogger(__name__)


class RedisService:
    """
    Redis service for rate limiting, attempt tracking, and caching.
    """
    
    def __init__(self):
        self.client = None
        self._connect()
    
    def _connect(self):
        """Establish Redis connection."""
        try:
            self.client = redis.Redis(
                host=Config.REDIS_HOST,
                port=Config.REDIS_PORT,
                password=Config.REDIS_PASSWORD or None,
                db=Config.REDIS_DB,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5
            )
            # Test connection
            self.client.ping()
            logger.info(f"✅ Connected to Redis at {Config.REDIS_HOST}:{Config.REDIS_PORT}")
        except Exception as e:
            logger.error(f"❌ Redis connection failed: {e}")
            logger.warning("Falling back to in-memory rate limiting")
            self.client = None
    
    def is_available(self) -> bool:
        """Check if Redis is available."""
        return self.client is not None
    
    def record_attempt(self, identifier: str, window_seconds: int = 300) -> int:
        """
        Record a verification attempt.
        
        Args:
            identifier: user_id or client_ip
            window_seconds: Time window for attempts (default: 5 minutes)
        
        Returns:
            Number of attempts in current window
        """
        if not self.client:
            return self._fallback_record(identifier, window_seconds)
        
        try:
            key = f"attempt:{identifier}"
            now = datetime.now().timestamp()
            
            # Add current attempt
            self.client.zadd(key, {str(now): now})
            
            # Remove attempts outside window
            cutoff = now - window_seconds
            self.client.zremrangebyscore(key, 0, cutoff)
            
            # Set expiry on key
            self.client.expire(key, window_seconds)
            
            # Get count
            count = self.client.zcard(key)
            return count
            
        except Exception as e:
            logger.error(f"Redis record_attempt error: {e}")
            return self._fallback_record(identifier, window_seconds)
    
    def get_recent_attempts(self, identifier: str, window_seconds: int = 300) -> int:
        """Get number of recent attempts."""
        if not self.client:
            return self._fallback_get(identifier)
        
        try:
            key = f"attempt:{identifier}"
            cutoff = datetime.now().timestamp() - window_seconds
            return self.client.zcount(key, cutoff, '+inf')
        except Exception as e:
            logger.error(f"Redis get_recent_attempts error: {e}")
            return self._fallback_get(identifier)
    
    def is_blocked(self, identifier: str, max_attempts: int = 5, window_seconds: int = 300) -> bool:
        """Check if identifier is blocked."""
        attempts = self.get_recent_attempts(identifier, window_seconds)
        blocked = attempts >= max_attempts
        
        if blocked:
            logger.warning(f"Blocked: {identifier} has {attempts} attempts")
        
        return blocked
    
    def reset_attempts(self, identifier: str) -> bool:
        """Reset attempts for an identifier (after successful verification)."""
        if not self.client:
            self._fallback_reset(identifier)
            return True
        
        try:
            key = f"attempt:{identifier}"
            self.client.delete(key)
            logger.info(f"Reset attempts for {identifier}")
            return True
        except Exception as e:
            logger.error(f"Redis reset_attempts error: {e}")
            return False
    
    def block_identifier(self, identifier: str, duration_seconds: int = 600) -> bool:
        """
        Explicitly block an identifier for a duration.
        Used for suspicious activity.
        """
        if not self.client:
            return False
        
        try:
            key = f"block:{identifier}"
            self.client.setex(key, duration_seconds, "blocked")
            logger.warning(f"Blocked {identifier} for {duration_seconds}s")
            return True
        except Exception as e:
            logger.error(f"Redis block_identifier error: {e}")
            return False
    
    def is_manually_blocked(self, identifier: str) -> bool:
        """Check if identifier is manually blocked."""
        if not self.client:
            return False
        
        try:
            key = f"block:{identifier}"
            return self.client.exists(key) > 0
        except Exception as e:
            logger.error(f"Redis is_manually_blocked error: {e}")
            return False
    
    def cache_set(self, key: str, value: Any, expiry_seconds: int = 300) -> bool:
        """Cache a value with expiry."""
        if not self.client:
            return False
        
        try:
            serialized = json.dumps(value)
            self.client.setex(key, expiry_seconds, serialized)
            return True
        except Exception as e:
            logger.error(f"Redis cache_set error: {e}")
            return False
    
    def cache_get(self, key: str) -> Optional[Any]:
        """Get cached value."""
        if not self.client:
            return None
        
        try:
            value = self.client.get(key)
            if value:
                return json.loads(value)
            return None
        except Exception as e:
            logger.error(f"Redis cache_get error: {e}")
            return None
    
    # Fallback methods (in-memory when Redis unavailable)
    _fallback_store: Dict[str, list] = {}
    
    def _fallback_record(self, identifier: str, window_seconds: int) -> int:
        """Fallback in-memory recording."""
        now = datetime.now().timestamp()
        
        if identifier not in self._fallback_store:
            self._fallback_store[identifier] = []
        
        # Clean old attempts
        cutoff = now - window_seconds
        self._fallback_store[identifier] = [
            t for t in self._fallback_store[identifier] if t > cutoff
        ]
        
        self._fallback_store[identifier].append(now)
        return len(self._fallback_store[identifier])
    
    def _fallback_get(self, identifier: str) -> int:
        """Fallback in-memory get."""
        return len(self._fallback_store.get(identifier, []))
    
    def _fallback_reset(self, identifier: str) -> None:
        """Fallback in-memory reset."""
        if identifier in self._fallback_store:
            del self._fallback_store[identifier]


# Singleton instance
_redis_service = None


def get_redis_service() -> RedisService:
    """Get or create global Redis service instance."""
    global _redis_service
    if _redis_service is None:
        _redis_service = RedisService()
    return _redis_service
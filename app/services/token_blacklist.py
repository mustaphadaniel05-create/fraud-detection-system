# app/services/token_blacklist.py
"""
Token blacklist for revoked JWTs.
Uses Redis for distributed token revocation.
"""

import logging
from datetime import datetime
from typing import Optional

from app.services.redis_service import get_redis_service

logger = logging.getLogger(__name__)


class TokenBlacklist:
    """
    Redis-based token blacklist for revoked JWT tokens.
    """
    
    def __init__(self):
        self.redis = get_redis_service()
    
    def blacklist_token(self, token: str, expires_in: int) -> bool:
        """
        Add token to blacklist.
        
        Args:
            token: JWT token to blacklist
            expires_in: Seconds until token expires (blacklist entry lives as long)
        """
        if not self.redis.is_available():
            logger.warning("Redis unavailable, token blacklist disabled")
            return False
        
        try:
            key = f"blacklist:{token}"
            self.redis.client.setex(key, expires_in, "revoked")
            logger.info(f"Token blacklisted (expires in {expires_in}s)")
            return True
        except Exception as e:
            logger.error(f"Failed to blacklist token: {e}")
            return False
    
    def is_blacklisted(self, token: str) -> bool:
        """Check if token is blacklisted."""
        if not self.redis.is_available():
            return False
        
        try:
            key = f"blacklist:{token}"
            return self.redis.client.exists(key) > 0
        except Exception as e:
            logger.error(f"Failed to check blacklist: {e}")
            return False
    
    def revoke_user_sessions(self, user_id: int) -> bool:
        """
        Revoke all sessions for a user.
        Note: This requires storing session IDs per user.
        """
        # This would require storing active session IDs in Redis
        # Implementation depends on requirements
        logger.warning(f"Revoke all sessions for user {user_id} - not implemented")
        return False


# Singleton instance
_token_blacklist = None


def get_token_blacklist() -> TokenBlacklist:
    """Get or create global token blacklist instance."""
    global _token_blacklist
    if _token_blacklist is None:
        _token_blacklist = TokenBlacklist()
    return _token_blacklist
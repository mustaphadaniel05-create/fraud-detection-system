# app/services/jwt_service.py
"""
JWT token management for session binding.
Provides cryptographically signed tokens for verified sessions.
"""

import logging
import jwt
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple
from functools import wraps
from flask import request, jsonify, current_app

from config import Config

logger = logging.getLogger(__name__)


class JWTService:
    """
    JWT token service for secure session management.
    """
    
    def __init__(self):
        self.secret_key = Config.JWT_SECRET_KEY
        self.algorithm = Config.JWT_ALGORITHM
        self.access_expires = Config.JWT_ACCESS_TOKEN_EXPIRES
        self.refresh_expires = Config.JWT_REFRESH_TOKEN_EXPIRES
    
    def create_tokens(self, user_data: Dict[str, Any]) -> Tuple[str, str]:
        """
        Create access and refresh tokens for a user.
        
        Args:
            user_data: Dictionary with user_id, email, full_name
            
        Returns:
            (access_token, refresh_token)
        """
        # Create unique session ID
        session_id = str(uuid.uuid4())
        
        # Convert user_id to string for JWT sub field
        user_id_str = str(user_data.get('user_id'))
        
        # Access token payload (short-lived)
        access_payload = {
            'sub': user_id_str,
            'email': user_data.get('email'),
            'full_name': user_data.get('full_name'),
            'session_id': session_id,
            'type': 'access',
            'iat': datetime.utcnow(),
            'exp': datetime.utcnow() + timedelta(seconds=self.access_expires)
        }
        
        # Refresh token payload (longer-lived)
        refresh_payload = {
            'sub': user_id_str,
            'email': user_data.get('email'),
            'session_id': session_id,
            'type': 'refresh',
            'iat': datetime.utcnow(),
            'exp': datetime.utcnow() + timedelta(seconds=self.refresh_expires)
        }
        
        access_token = jwt.encode(
            access_payload,
            self.secret_key,
            algorithm=self.algorithm
        )
        
        refresh_token = jwt.encode(
            refresh_payload,
            self.secret_key,
            algorithm=self.algorithm
        )
        
        logger.info(f"Created tokens for user {user_data.get('email')} (session: {session_id})")
        
        return access_token, refresh_token
    
    def verify_token(self, token: str) -> Tuple[bool, Optional[Dict], Optional[str]]:
        """
        Verify and decode a JWT token.
        
        Returns:
            (is_valid, payload, error_message)
        """
        if not token:
            return False, None, "No token provided"
        
        try:
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm]
            )
            
            # Check token type
            if payload.get('type') not in ['access', 'refresh']:
                return False, None, "Invalid token type"
            
            # Check expiration (jwt already does this, but double-check)
            exp = payload.get('exp')
            if exp and datetime.fromtimestamp(exp) < datetime.utcnow():
                return False, None, "Token expired"
            
            return True, payload, None
            
        except jwt.ExpiredSignatureError:
            return False, None, "Token expired"
        except jwt.InvalidTokenError as e:
            return False, None, f"Invalid token: {str(e)}"
    
    def refresh_access_token(self, refresh_token: str) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Refresh an expired access token using a valid refresh token.
        
        Returns:
            (success, new_access_token, error_message)
        """
        is_valid, payload, error = self.verify_token(refresh_token)
        
        if not is_valid:
            return False, None, error
        
        if payload.get('type') != 'refresh':
            return False, None, "Invalid token type for refresh"
        
        # Create new access token
        new_access_payload = {
            'sub': payload.get('sub'),
            'email': payload.get('email'),
            'full_name': payload.get('full_name'),
            'session_id': payload.get('session_id'),
            'type': 'access',
            'iat': datetime.utcnow(),
            'exp': datetime.utcnow() + timedelta(seconds=self.access_expires)
        }
        
        new_access_token = jwt.encode(
            new_access_payload,
            self.secret_key,
            algorithm=self.algorithm
        )
        
        logger.info(f"Refreshed access token for user {payload.get('email')}")
        
        return True, new_access_token, None
    
    def get_current_user(self, token: str) -> Optional[Dict]:
        """Get current user from token."""
        is_valid, payload, _ = self.verify_token(token)
        
        if not is_valid or payload.get('type') != 'access':
            return None
        
        return {
            'user_id': int(payload.get('sub')) if payload.get('sub') else None,
            'email': payload.get('email'),
            'full_name': payload.get('full_name'),
            'session_id': payload.get('session_id')
        }
    
    def revoke_token(self, token: str) -> bool:
        """
        Revoke a token (for logout).
        Note: JWT is stateless, so revocation requires a blacklist.
        """
        logger.info("Token revocation requested (stateless JWT)")
        return True


# Singleton instance
_jwt_service = None


def get_jwt_service() -> JWTService:
    """Get or create global JWT service instance."""
    global _jwt_service
    if _jwt_service is None:
        _jwt_service = JWTService()
    return _jwt_service


# Flask decorator for protected routes
def jwt_required(f):
    """
    Decorator to require JWT authentication for routes.
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        auth_header = request.headers.get('Authorization', '')
        
        if not auth_header.startswith('Bearer '):
            return jsonify({'error': 'Missing or invalid Authorization header'}), 401
        
        token = auth_header.split(' ')[1]
        
        jwt_service = get_jwt_service()
        is_valid, payload, error = jwt_service.verify_token(token)
        
        if not is_valid:
            return jsonify({'error': error}), 401
        
        if payload.get('type') != 'access':
            return jsonify({'error': 'Invalid token type'}), 401
        
        # Attach user info to request
        request.current_user = {
            'user_id': int(payload.get('sub')) if payload.get('sub') else None,
            'email': payload.get('email'),
            'full_name': payload.get('full_name'),
            'session_id': payload.get('session_id')
        }
        
        return f(*args, **kwargs)
    
    return decorated_function


def get_current_user_from_request():
    """Get current user from request (set by jwt_required decorator)."""
    return getattr(request, 'current_user', None)
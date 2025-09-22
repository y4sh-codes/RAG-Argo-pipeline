"""
Security utilities for authentication, authorization, and API protection.
Industry-ready security implementation with JWT, rate limiting, and input validation.
"""

import secrets
import hashlib
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any
import logging

import jwt
from passlib.context import CryptContext
from fastapi import HTTPException, status, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from slowapi import Limiter
from slowapi.util import get_remote_address

from ..config import settings

logger = logging.getLogger(__name__)

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT configuration
ALGORITHM = settings.algorithm
SECRET_KEY = settings.secret_key

# Security schemes
security = HTTPBearer(auto_error=False)

# Rate limiter
limiter = Limiter(key_func=get_remote_address)


class SecurityManager:
    """Centralized security management."""
    
    def __init__(self):
        self.active_tokens = set()  # In production, use Redis
        self.blocked_ips = set()    # In production, use Redis
        
    def hash_password(self, password: str) -> str:
        """Hash a password using bcrypt."""
        return pwd_context.hash(password)
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash."""
        return pwd_context.verify(plain_password, hashed_password)
    
    def create_access_token(self, data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
        """Create a JWT access token."""
        to_encode = data.copy()
        
        if expires_delta:
            expire = datetime.now(timezone.utc) + expires_delta
        else:
            expire = datetime.now(timezone.utc) + timedelta(minutes=settings.access_token_expire_minutes)
        
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        
        # Track active token (simplified - use Redis in production)
        self.active_tokens.add(encoded_jwt)
        
        return encoded_jwt
    
    def verify_token(self, token: str) -> Dict[str, Any]:
        """Verify and decode a JWT token."""
        try:
            if token not in self.active_tokens:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Token has been revoked"
                )
            
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            return payload
        
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired"
            )
        except jwt.JWTError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials"
            )
    
    def revoke_token(self, token: str) -> bool:
        """Revoke a token."""
        try:
            self.active_tokens.discard(token)
            return True
        except Exception as e:
            logger.error(f"Error revoking token: {str(e)}")
            return False
    
    def generate_api_key(self, user_id: str) -> str:
        """Generate a secure API key."""
        # Create a unique identifier
        unique_data = f"{user_id}_{datetime.now(timezone.utc).isoformat()}_{secrets.token_hex(16)}"
        
        # Hash to create API key
        api_key = hashlib.sha256(unique_data.encode()).hexdigest()[:32]
        
        return f"rag_{api_key}"
    
    def validate_api_key(self, api_key: str) -> bool:
        """Validate an API key format."""
        if not api_key.startswith("rag_"):
            return False
        
        if len(api_key) != 36:  # rag_ + 32 hex chars
            return False
        
        return True
    
    def is_ip_blocked(self, ip: str) -> bool:
        """Check if an IP address is blocked."""
        return ip in self.blocked_ips
    
    def block_ip(self, ip: str) -> None:
        """Block an IP address."""
        self.blocked_ips.add(ip)
        logger.warning(f"Blocked IP address: {ip}")
    
    def unblock_ip(self, ip: str) -> None:
        """Unblock an IP address."""
        self.blocked_ips.discard(ip)
        logger.info(f"Unblocked IP address: {ip}")


# Global security manager instance
security_manager = SecurityManager()


# Authentication dependencies
async def get_current_user(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)):
    """Get the current authenticated user from JWT token."""
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header required"
        )
    
    token = credentials.credentials
    payload = security_manager.verify_token(token)
    
    user_id = payload.get("sub")
    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials"
        )
    
    return {"user_id": user_id, "payload": payload}


async def get_api_key(request: Request) -> Optional[str]:
    """Extract API key from headers."""
    api_key = request.headers.get("X-API-Key")
    
    if api_key and security_manager.validate_api_key(api_key):
        return api_key
    
    return None


async def require_api_key(api_key: Optional[str] = Depends(get_api_key)):
    """Require a valid API key for access."""
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Valid API key required"
        )
    
    return api_key


async def optional_auth(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)):
    """Optional authentication - returns user if authenticated, None otherwise."""
    if not credentials:
        return None
    
    try:
        token = credentials.credentials
        payload = security_manager.verify_token(token)
        user_id = payload.get("sub")
        
        if user_id:
            return {"user_id": user_id, "payload": payload}
    except HTTPException:
        pass  # Ignore auth errors for optional auth
    
    return None


# IP blocking middleware dependency
async def check_ip_not_blocked(request: Request):
    """Check if the client IP is not blocked."""
    client_ip = request.client.host if request.client else "unknown"
    
    if security_manager.is_ip_blocked(client_ip):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied from this IP address"
        )
    
    return client_ip


# Input validation utilities
def validate_query_length(query: str, max_length: int = 1000) -> str:
    """Validate query length to prevent abuse."""
    if len(query) > max_length:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Query too long. Maximum {max_length} characters allowed."
        )
    
    return query


def validate_file_size(file_size: int, max_size_mb: int = None) -> int:
    """Validate file size."""
    max_size = (max_size_mb or settings.max_file_size_mb) * 1024 * 1024
    
    if file_size > max_size:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Maximum {max_size_mb or settings.max_file_size_mb}MB allowed."
        )
    
    return file_size


def sanitize_filename(filename: str) -> str:
    """Sanitize filename to prevent path traversal."""
    import os
    import re
    
    # Remove any path components
    filename = os.path.basename(filename)
    
    # Remove potentially dangerous characters
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    
    # Limit length
    if len(filename) > 255:
        name, ext = os.path.splitext(filename)
        filename = name[:250 - len(ext)] + ext
    
    return filename


# Rate limiting decorators
def rate_limit_by_ip(requests_per_minute: int = 60):
    """Rate limit by IP address."""
    def decorator(func):
        return limiter.limit(f"{requests_per_minute}/minute")(func)
    return decorator


def rate_limit_by_user(requests_per_minute: int = 100):
    """Rate limit by user (requires authentication)."""
    def decorator(func):
        return limiter.limit(f"{requests_per_minute}/minute", key_func=lambda r: get_user_id_from_request(r))(func)
    return decorator


def get_user_id_from_request(request: Request) -> str:
    """Extract user ID from request for rate limiting."""
    auth_header = request.headers.get("Authorization")
    
    if auth_header and auth_header.startswith("Bearer "):
        try:
            token = auth_header.split(" ")[1]
            payload = security_manager.verify_token(token)
            return payload.get("sub", "anonymous")
        except Exception:
            pass
    
    # Fallback to IP-based limiting
    return request.client.host if request.client else "unknown"


# Security headers middleware
async def add_security_headers(request: Request, call_next):
    """Add security headers to responses."""
    response = await call_next(request)
    
    # Security headers
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    response.headers["Content-Security-Policy"] = "default-src 'self'"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    
    # Remove server header
    if "server" in response.headers:
        del response.headers["server"]
    
    return response


# Permission checking
class Permission:
    """Permission constants."""
    READ_DOCUMENTS = "read:documents"
    WRITE_DOCUMENTS = "write:documents"
    DELETE_DOCUMENTS = "delete:documents"
    GENERATE_REPORTS = "generate:reports"
    ADMIN_ACCESS = "admin:access"


def require_permission(permission: str):
    """Require specific permission for access."""
    async def permission_checker(current_user: dict = Depends(get_current_user)):
        user_permissions = current_user.get("payload", {}).get("permissions", [])
        
        if permission not in user_permissions and Permission.ADMIN_ACCESS not in user_permissions:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission '{permission}' required"
            )
        
        return current_user
    
    return permission_checker


# Audit logging
def log_security_event(event_type: str, details: Dict[str, Any], request: Request = None):
    """Log security-related events for auditing."""
    log_data = {
        "event_type": event_type,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "details": details
    }
    
    if request:
        log_data.update({
            "client_ip": request.client.host if request.client else "unknown",
            "user_agent": request.headers.get("user-agent", "unknown"),
            "method": request.method,
            "url": str(request.url)
        })
    
    logger.warning("Security event", extra=log_data)


# Session management (simplified - use Redis in production)
class SessionManager:
    """Manage user sessions."""
    
    def __init__(self):
        self.sessions = {}  # In production, use Redis
    
    def create_session(self, user_id: str, session_data: Dict[str, Any]) -> str:
        """Create a new session."""
        session_id = secrets.token_urlsafe(32)
        
        self.sessions[session_id] = {
            "user_id": user_id,
            "created_at": datetime.now(timezone.utc),
            "last_accessed": datetime.now(timezone.utc),
            **session_data
        }
        
        return session_id
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session data."""
        session = self.sessions.get(session_id)
        
        if session:
            # Update last accessed time
            session["last_accessed"] = datetime.now(timezone.utc)
        
        return session
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a session."""
        return self.sessions.pop(session_id, None) is not None
    
    def cleanup_expired_sessions(self, max_age_hours: int = 24):
        """Clean up expired sessions."""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
        
        expired_sessions = [
            session_id for session_id, session_data in self.sessions.items()
            if session_data["last_accessed"] < cutoff_time
        ]
        
        for session_id in expired_sessions:
            del self.sessions[session_id]
        
        return len(expired_sessions)


# Global session manager
session_manager = SessionManager()


# Content Security Policy
CSP_POLICY = {
    "default-src": ["'self'"],
    "script-src": ["'self'", "'unsafe-inline'"],
    "style-src": ["'self'", "'unsafe-inline'"],
    "img-src": ["'self'", "data:", "https:"],
    "font-src": ["'self'"],
    "connect-src": ["'self'"],
    "frame-ancestors": ["'none'"]
}


def generate_csp_header() -> str:
    """Generate Content Security Policy header."""
    policies = []
    
    for directive, sources in CSP_POLICY.items():
        policy = f"{directive} {' '.join(sources)}"
        policies.append(policy)
    
    return "; ".join(policies)

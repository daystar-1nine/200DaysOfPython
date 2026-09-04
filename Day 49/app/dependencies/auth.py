# ==============================================================================
# Program    : Authentication & Authorization Dependencies (auth.py)
# Objective  : Implement get_current_user (401 check) and require_admin (403 check) dependencies using custom exceptions.
# Concept    : Security Dependencies & Custom Domain Exception Raising
# Why Used   : Raises AuthenticationError and AuthorizationError caught by global exception handlers.
# ==============================================================================

import os
import sys
import jwt
from fastapi import Depends
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from app.database import get_db
from app.models.user import User
from app.repositories.user_repository import UserRepository
from app.security import decode_access_token
from app.exceptions import AuthenticationError, AuthorizationError

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
) -> User:
    """Extract Bearer token, decode JWT claims, and return the authenticated User model.

    Raises:
        AuthenticationError (401): If token is missing, invalid, or expired.
    """
    try:
        payload = decode_access_token(token)
        sub: str = payload.get("sub")
        if sub is None:
            raise AuthenticationError("Invalid token payload: missing subject claim.")
        user_id = int(sub)
    except jwt.ExpiredSignatureError:
        raise AuthenticationError("Access token has expired. Please log in again.")
    except (jwt.PyJWTError, ValueError):
        raise AuthenticationError("Could not validate authentication credentials.")

    repo = UserRepository(db)
    user = repo.get_by_id(user_id)
    if user is None:
        raise AuthenticationError("Authenticated user account no longer exists.")

    return user

def require_admin(current_user: User = Depends(get_current_user)) -> User:
    """Verify that current authenticated user possesses the 'admin' role.

    Raises:
        AuthorizationError (403): If authenticated user is not an administrator.
    """
    if current_user.role != "admin":
        raise AuthorizationError("Administrator privileges are required for this action.")
    return current_user

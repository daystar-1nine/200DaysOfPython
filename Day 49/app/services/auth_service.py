# ==============================================================================
# Program    : Authentication Service Layer (auth_service.py)
# Objective  : Business logic for user registration, credential verification, and token issuance with logging.
# Concept    : Business Logic Encapsulation & Safe Logging
# Why Used   : Logs security events without printing sensitive password attributes.
# ==============================================================================

import logging
from sqlalchemy.orm import Session
from app.models.user import User
from app.schemas.auth import RegisterRequest, LoginRequest, TokenResponse
from app.repositories.user_repository import UserRepository
from app.security import hash_password, verify_password, create_access_token
from app.exceptions import DuplicateEmailError, InvalidCredentialsError

logger = logging.getLogger("app.services.auth_service")

class AuthService:
    def __init__(self, db: Session):
        self.repo = UserRepository(db)

    def register_user(self, req: RegisterRequest) -> User:
        logger.info(f"User registration requested for email='{req.email}'")
        existing = self.repo.get_by_email(req.email)
        if existing:
            logger.warning(f"Registration failed: duplicate email '{req.email}'")
            raise DuplicateEmailError(req.email)

        hashed = hash_password(req.password)
        new_user = User(
            name=req.name,
            email=req.email,
            age=req.age,
            phone=req.phone,
            password_hash=hashed,
            role=req.role or "user"
        )
        created = self.repo.create(new_user)
        logger.info(f"User successfully registered with id={created.id}, role='{created.role}'")
        return created

    def authenticate_user(self, req: LoginRequest) -> TokenResponse:
        logger.info(f"Login attempt for email='{req.email}'")
        user = self.repo.get_by_email(req.email)
        if not user or not verify_password(req.password, user.password_hash):
            logger.warning(f"Authentication failed for email='{req.email}'")
            raise InvalidCredentialsError()

        token = create_access_token(data={"sub": str(user.id), "email": user.email, "role": user.role})
        logger.info(f"Authentication successful for user_id={user.id}, role='{user.role}'")
        return TokenResponse(access_token=token, token_type="bearer")

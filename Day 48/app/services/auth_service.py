# ==============================================================================
# Program    : Authentication Service Layer (auth_service.py)
# Objective  : Business logic for user registration, password hashing, and authentication token issuance.
# Concept    : Separation of Concerns & Business Logic Encapsulation
# Why Used   : Keeps route handlers focused strictly on HTTP protocol concerns.
# ==============================================================================

from sqlalchemy.orm import Session
from app.models.user import User
from app.schemas.auth import RegisterRequest, LoginRequest, TokenResponse
from app.repositories.user_repository import UserRepository
from app.security import hash_password, verify_password, create_access_token
from app.exceptions import UserAlreadyExistsError, InvalidCredentialsError

class AuthService:
    def __init__(self, db: Session):
        self.repo = UserRepository(db)

    def register_user(self, req: RegisterRequest) -> User:
        existing = self.repo.get_by_email(req.email)
        if existing:
            raise UserAlreadyExistsError(req.email)

        hashed = hash_password(req.password)
        new_user = User(
            name=req.name,
            email=req.email,
            age=req.age,
            phone=req.phone,
            password_hash=hashed,
            role=req.role or "user"
        )
        return self.repo.create(new_user)

    def authenticate_user(self, req: LoginRequest) -> TokenResponse:
        user = self.repo.get_by_email(req.email)
        if not user or not verify_password(req.password, user.password_hash):
            raise InvalidCredentialsError()

        token = create_access_token(data={"sub": str(user.id), "email": user.email, "role": user.role})
        return TokenResponse(access_token=token, token_type="bearer")

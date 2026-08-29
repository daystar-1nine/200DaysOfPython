# ==============================================================================
# Program    : Dependency Factory Providers (providers.py)
# Objective  : Provide singleton/factory functions for UserRepository, UserService, ProductRepository, and ProductService.
# Concept    : Dependency Injection Factories (Day 43 requirement)
# Why Used   : Instantiates repositories and injects them into services for FastAPI Depends().
# ==============================================================================

import os
import sys
from fastapi import Depends

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from app.repositories.user_repository import UserRepository
from app.repositories.product_repository import ProductRepository
from app.services.user_service import UserService
from app.services.product_service import ProductService

# Singleton instances for in-memory persistence
_user_repository_instance = UserRepository()
_product_repository_instance = ProductRepository()

def get_user_repository() -> UserRepository:
    """Dependency provider returning UserRepository instance."""
    return _user_repository_instance

def get_user_service(repo: UserRepository = Depends(get_user_repository)) -> UserService:
    """Dependency provider injecting UserRepository into UserService."""
    return UserService(repository=repo)

def get_product_repository() -> ProductRepository:
    """Dependency provider returning ProductRepository instance."""
    return _product_repository_instance

def get_product_service(repo: ProductRepository = Depends(get_product_repository)) -> ProductService:
    """Dependency provider injecting ProductRepository into ProductService."""
    return ProductService(repository=repo)

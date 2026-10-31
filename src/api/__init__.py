"""
Module: src.api
Chức năng: Tầng Giao tiếp API - Cung cấp FastAPI application factory và REST API routes.
"""

from src.api.app import create_app
from src.api.routes import router

__all__ = ["create_app", "router"]

"""ASGI entry point so `uvicorn main:app` works from the backend root."""

from app.main import app

application = app

__all__ = ("app", "application")
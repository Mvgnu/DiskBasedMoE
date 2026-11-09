"""Local HTTP interface; importing it does not load ML dependencies."""

from .app import create_app

__all__ = ["create_app"]

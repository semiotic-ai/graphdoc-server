"""GraphDoc Server package."""
from .app import create_app, main
from .keys import *

__all__ = ["create_app", "main"]

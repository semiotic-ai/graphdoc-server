# system packages 
import os
import json
import secrets
import logging
import functools
from pathlib import Path
from typing import Optional, Callable

# internal packages 

# external packages 
from flask import request, jsonify, Response

# logging 
log = logging.getLogger(__name__)

def get_api_config_path() -> Path:
    """Get the path to the API configuration file."""
    # Get the directory where app.py is located
    app_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    # Create the keys directory if it doesn't exist
    keys_dir = app_dir / "keys"
    keys_dir.mkdir(exist_ok=True)
    # Return the path to the API key config file
    return keys_dir / "api_key_config.json"


def load_api_keys() -> None:
    """Load API keys from configuration file."""
    global api_keys, api_config
    config_path = get_api_config_path()
    
    try:
        if config_path.exists():
            with open(config_path, 'r') as f:
                api_config = json.load(f)
                api_keys = set(api_config.get("api_keys", []))
                log.info(f"Loaded {len(api_keys)} API keys from {config_path}")
        else:
            log.warning(f"API config file not found at {config_path}")
    except Exception as e:
        log.error(f"Error loading API keys: {str(e)}")


def save_api_keys() -> None:
    """Save API keys to configuration file."""
    global api_config
    config_path = get_api_config_path()
    
    try:
        # Update the api_keys in config
        api_config["api_keys"] = list(api_keys)
        
        # Save to file
        with open(config_path, 'w') as f:
            json.dump(api_config, f, indent=2)
        
        log.info(f"Saved {len(api_keys)} API keys to {config_path}")
    except Exception as e:
        log.error(f"Error saving API keys: {str(e)}")


def generate_api_key() -> str:
    """Generate a new API key."""
    # Generate a secure random key (32 bytes = 64 hex chars)
    new_key = secrets.token_hex(32)
    api_keys.add(new_key)
    save_api_keys()
    return new_key


def get_admin_key() -> Optional[str]:
    """Get the admin key from configuration."""
    return api_config.get("admin_key")


def set_admin_key(key: str) -> None:
    """Set the admin key in configuration."""
    api_config["admin_key"] = key
    save_api_keys()


def require_api_key(func: Callable) -> Callable:
    """Decorator to require API key authentication."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Response:
        api_key = request.headers.get("X-API-Key")
        if not api_key:
            return jsonify({"error": "API key required"}), 401
        if api_key not in api_keys:
            return jsonify({"error": "Invalid API key"}), 403
        return func(*args, **kwargs)
    return wrapper


def require_admin_key(func: Callable) -> Callable:
    """Decorator to require admin API key authentication."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Response:
        admin_key = get_admin_key()
        if not admin_key:
            return jsonify({"error": "Admin key not configured on server"}), 500
            
        api_key = request.headers.get("X-API-Key")
        if not api_key:
            return jsonify({"error": "API key required"}), 401
        if api_key != admin_key:
            return jsonify({"error": "Admin access required"}), 403
        return func(*args, **kwargs)
    return wrapper
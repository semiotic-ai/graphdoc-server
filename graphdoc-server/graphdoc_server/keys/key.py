# system packages 
import os
import json
import secrets
import logging
import functools
from pathlib import Path
from typing import Optional, Callable, Set, Dict, Any, Union

# external packages 
from flask import request, jsonify, Response

# logging 
log = logging.getLogger(__name__)

class KeyManager:
    """
    Manages API keys and authentication for the GraphDoc server.

    This class is a singleton that manages API keys and authentication for the GraphDoc server.
    It provides methods to load, save, and generate API keys, as well as to require and validate API keys.
    """
    
    _instance = None  # Class variable for singleton pattern

    def __init__(self, config_path: Union[Path, str]):
        """
        Initialize the KeyManager with optional custom config path.
        
        :param config_path: Optional path to the API configuration file. If not provided, the default path will be used.
        :type config_path: Optional[Path]
        """
        self.api_keys: Set[str] = set()
        self.api_config: Dict[str, Any] = {
            "api_keys": [],
            "admin_key": None
        }
        self.config_path = config_path
        self.load_api_keys()
    
    ##################
    # class methods  #
    ##################
    @classmethod
    def get_instance(cls, config_path: Union[Path, str]) -> 'KeyManager':
        """
        Get the singleton instance of KeyManager.

        :param config_path: Optional path to the API configuration file. If not provided, the default path will be used.
        :type config_path: Optional[Path]
        :return: The singleton instance of KeyManager.
        :rtype: KeyManager
        """
        if cls._instance is None:
            cls._instance = KeyManager(config_path)
        elif config_path is not None:
            cls._instance.config_path = config_path
            cls._instance.load_api_keys()
        return cls._instance
    
    ####################
    # instance methods #
    ####################
    def load_api_keys(self) -> None:
        """
        Load API keys from configuration file.

        :return: None
        :rtype: None
        """
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    self.api_config = json.load(f)
                    self.api_keys = set(self.api_config.get("api_keys", []))
                    log.info(f"Loaded {len(self.api_keys)} API keys from {self.config_path}")
            else:
                log.warning(f"API config file not found at {self.config_path}")
        except Exception as e:
            log.error(f"Error loading API keys: {str(e)}")

    def save_api_keys(self) -> None:
        """
        Save API keys to configuration file.

        :return: None
        :rtype: None
        """
        try:
            # Update the api_keys in config
            self.api_config["api_keys"] = list(self.api_keys)
            
            # Save to file
            with open(self.config_path, 'w') as f:
                json.dump(self.api_config, f, indent=2)
            
            log.info(f"Saved {len(self.api_keys)} API keys to {self.config_path}")
        except Exception as e:
            log.error(f"Error saving API keys: {str(e)}")

    def generate_api_key(self) -> str:
        """
        Generate a new API key.

        :return: The new API key.
        :rtype: str
        """
        # Generate a secure random key (32 bytes = 64 hex chars)
        new_key = secrets.token_hex(32)
        self.api_keys.add(new_key)
        self.save_api_keys()
        return new_key

    def get_admin_key(self) -> Optional[str]:
        """
        Get the admin key from configuration.

        :return: The admin key.
        :rtype: Optional[str]
        """
        return self.api_config.get("admin_key")

    def set_admin_key(self, key: str) -> None:
        """
        Set the admin key in configuration.

        :param key: The admin key to set.
        :type key: str
        """
        self.api_config["admin_key"] = key
        self.save_api_keys()
    
    def _get_test_key(self) -> Optional[str]:
        """
        Get the test key from configuration.

        :return: The test key.
        :rtype: Optional[str]
        """
        return self.api_config.get("test_key")
    
    #####################
    # decorator methods #
    #####################
    def require_api_key(self, func: Callable) -> Callable:
        """
        Decorator to require API key authentication.

        :param func: The function to decorate.
        :type func: Callable
        :return: The decorated function.
        """
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Response:
            api_key = request.headers.get("X-API-Key")
            if not api_key:
                return jsonify({"error": "API key required"}), 401
            if api_key not in self.api_keys:
                return jsonify({"error": "Invalid API key"}), 403
            return func(*args, **kwargs)
        return wrapper

    def require_admin_key(self, func: Callable) -> Callable:
        """
        Decorator to require admin API key authentication.

        :param func: The function to decorate.
        :type func: Callable
        :return: The decorated function.
        """
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Response:
            admin_key = self.get_admin_key()
            if not admin_key:
                return jsonify({"error": "Admin key not configured on server"}), 500
                
            api_key = request.headers.get("X-API-Key")
            if not api_key:
                return jsonify({"error": "API key required"}), 401
            if api_key != admin_key:
                return jsonify({"error": "Admin access required"}), 403
            return func(*args, **kwargs)
        return wrapper

# Create global functions that use the singleton for backward compatibility
# def get_api_config_path() -> Path:
#     return KeyManager.get_instance().config_path

# def load_api_keys() -> None:
#     KeyManager.get_instance().load_api_keys()

# def save_api_keys() -> None:
#     KeyManager.get_instance().save_api_keys()

# def generate_api_key() -> str:
#     return KeyManager.get_instance().generate_api_key()

# def get_admin_key() -> Optional[str]:
#     return KeyManager.get_instance().get_admin_key()

# def set_admin_key(key: str) -> None:
#     KeyManager.get_instance().set_admin_key(key)

# def require_api_key(func: Callable) -> Callable:
#     return KeyManager.get_instance().require_api_key(func)

# def require_admin_key(func: Callable) -> Callable:
#     return KeyManager.get_instance().require_admin_key(func)
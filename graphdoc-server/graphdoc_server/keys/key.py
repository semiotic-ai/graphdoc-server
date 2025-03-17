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

    def __init__(
        self,
        config_path: Union[Path, str],
        require_api_key: bool = True,
        require_admin_key: bool = True,
    ):
        """
        Initialize the KeyManager with optional custom config path.

        :param config_path: Optional path to the API configuration file. If not provided, the default path will be used.
        :type config_path: Optional[Path]
        """
        self.api_keys: Set[str] = set()
        self.api_config: Dict[str, Any] = {"api_keys": [], "admin_key": None}
        self.admin_key: Optional[str] = None
        self.config_path = config_path
        self.require_api_key_flag: bool = require_api_key
        self.require_admin_key_flag: bool = require_admin_key
        self.load_api_keys()

    ##################
    # class methods  #
    ##################
    @classmethod
    def get_instance(
        cls,
        config_path: Union[Path, str],
        require_api_key: bool = True,
        require_admin_key: bool = True,
    ) -> "KeyManager":
        """
        Get the singleton instance of KeyManager.

        :param config_path: Optional path to the API configuration file. If not provided, the default path will be used.
        :type config_path: Optional[Path]
        :return: The singleton instance of KeyManager.
        :rtype: KeyManager
        """
        if cls._instance is None:
            cls._instance = KeyManager(config_path, require_api_key, require_admin_key)
            cls._instance.load_api_keys()
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
            with open(self.config_path, "r") as f:
                self.api_config = json.load(f)
                self.api_keys = set(self.api_config.get("api_keys", []))
                self.admin_key = self.api_config.get("admin_key")
                if self.admin_key:
                    self.api_keys.add(self.admin_key)
                log.info(
                    f"Loaded {len(self.api_keys)} API keys from {self.config_path}"
                )
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
            with open(self.config_path, "w") as f:
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

    def delete_api_key(self, key: str) -> None:
        """
        Delete an API key from the configuration.

        :param key: The API key to delete.
        :type key: str
        """
        self.load_api_keys()
        self.api_keys.remove(key)
        self.save_api_keys()

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
        def wrapper(*args, **kwargs):
            if not self.require_api_key_flag:
                return func(*args, **kwargs)

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
        def wrapper(*args, **kwargs):
            if not self.require_admin_key_flag:
                return func(*args, **kwargs)

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

# system
import os
import json
import logging
from pathlib import Path
import werkzeug.exceptions
from typing import Optional, Dict, Any, List, Set, Callable
import secrets
import functools

# internal
from graphdoc import GraphDoc, load_yaml_config

# external 
import dspy
import mlflow
from mlflow import MlflowClient
from flask import Flask, request, jsonify, Response

# Set up logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Global variables to store our loaded objects
graph_doc: Optional[GraphDoc] = None
module: Optional[Any] = None
config: Optional[Dict[str, Any]] = None
api_keys: Set[str] = set()  # Store API keys in memory
api_config: Dict[str, Any] = {
    "api_keys": [],
    "admin_key": None
}


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

# def init_model(config_path: str, metric_config_path: str) -> bool:
def init_model(config_path: str) -> bool:
    """Initialize the GraphDoc and load the module."""
    global graph_doc, module, config

    try:
        # Load configs
        loaded_config = load_yaml_config(config_path)
        if not loaded_config:
            raise ValueError("Failed to load config")

        # Initialize GraphDoc
        graph_doc = GraphDoc.from_dict(loaded_config)
        
        # Load the module
        module = graph_doc.doc_generator_module_from_yaml(config_path)

        # Only set the global config if everything succeeded
        config = loaded_config
        log.info("Successfully initialized model and loaded module")
        return True
    except Exception as e:
        log.error(f"Error initializing model: {str(e)}")
        return False


def create_app() -> Flask:
    """Create and configure the Flask application."""
    app = Flask(__name__)

    config_path = os.getenv("GRAPHDOC_CONFIG_PATH")
    log.info(f"Config path: {config_path}")

    # Read and log the YAML config file contents
    try:
        if config_path:
            with open(config_path, 'r') as file:
                config_contents = file.read()
                log.info(f"Config file contents from {config_path}:\n{config_contents}")
        else:
            log.warning("Config path is not set, cannot read config file")
            
    except Exception as e:
        log.error(f"Error reading config files: {str(e)}")

    if not config_path:
        raise ValueError(
            "Environment variables GRAPHDOC_CONFIG_PATH must be set"
        )

    # Initialize the model
    if not init_model(config_path):
        raise RuntimeError("Failed to initialize model")

    if not config:  # This should never happen due to the init_model check above
        raise RuntimeError("Config is not initialized")
        
    # Load API keys
    load_api_keys()

    @app.route("/health", methods=["GET"])
    def health_check():
        """Health check endpoint."""
        return jsonify({"status": "healthy", "model_loaded": module is not None})

    @app.route("/model/version", methods=["GET"])
    @require_api_key
    def model_version():
        """Get model version information."""
        if not module or not config:
            return jsonify({"error": "Model not loaded"}), 503

        assert config is not None 
        return jsonify( # TODO: we can expand this more as we add tighter coupling between mlflow and the server
            {
                "model_name": config["prompt"]["prompt"],
            }
        )

    @app.route("/inference", methods=["POST"])
    @require_api_key
    def inference():
        """Run inference on the loaded model."""
        if not module:
            return jsonify({"error": "Model not loaded"}), 503

        try:
            # First try to parse the JSON data
            try:
                data = request.get_json()
            except (json.JSONDecodeError, werkzeug.exceptions.BadRequest):
                return jsonify({"error": "Invalid JSON in request"}), 400

            # Check for required fields
            if not data or "database_schema" not in data:
                return jsonify({"error": "Missing database_schema in request"}), 400

            # Run inference
            prediction = module.document_full_schema(data["database_schema"])

            # Convert prediction to string if it's not already
            if hasattr(prediction, "prediction"):
                prediction = prediction.prediction
            elif not isinstance(prediction, (str, int, float, bool, list, dict)):
                prediction = str(prediction)

            return jsonify({"prediction": prediction, "status": "success"})
        except Exception as e:
            log.error(f"Error during inference: {str(e)}")
            return jsonify({"error": str(e), "status": "error"}), 500
            
    @app.route("/api-keys/generate", methods=["POST"])
    @require_admin_key
    def create_api_key():
        """Create a new API key (admin only)."""
        new_key = generate_api_key()
        return jsonify({"status": "success", "api_key": new_key})
        
    @app.route("/api-keys/list", methods=["GET"])
    @require_admin_key
    def list_api_keys():
        """List all API keys (admin only)."""
        return jsonify({"status": "success", "api_keys": list(api_keys)})

    return app


def main():
    """Main entry point for the Flask development server."""
    import argparse

    parser = argparse.ArgumentParser(description="Start the GraphDoc API server.")
    parser.add_argument(
        "--config-path",
        type=str,
        required=True,
        help="Path to the configuration YAML file.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=6000,
        help="Port to run the server on.",
    )
    parser.add_argument(
        "--admin-key",
        type=str,
        help="Admin API key for managing other API keys.",
    )

    args = parser.parse_args()

    # Set environment variables for the app factory
    os.environ["GRAPHDOC_CONFIG_PATH"] = args.config_path
    
    # Load existing API keys
    load_api_keys()
    
    # Set admin key if provided
    if args.admin_key:
        set_admin_key(args.admin_key)
        log.info("Admin key set from command line argument")
    
    # Create initial API key if none exists
    if not api_keys:
        initial_key = generate_api_key()
        log.info(f"Created initial API key: {initial_key}")
        
    # Create initial admin key if none exists
    if not get_admin_key():
        admin_key = secrets.token_hex(32)
        set_admin_key(admin_key)
        log.info(f"Created initial admin key: {admin_key}")

    # Create and run the app
    app = create_app()
    app.run(host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()

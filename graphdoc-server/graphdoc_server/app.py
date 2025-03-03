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
from .keys import KeyManager

# from .key import load_api_keys, generate_api_key, require_api_key, require_admin_key, set_admin_key, get_admin_key

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
app_dir = Path(os.path.dirname(os.path.abspath(__file__)))
keys_dir = app_dir / "keys"
keys_dir.mkdir(exist_ok=True)
key_path = keys_dir / "api_key_config.json"

# api keys
# TODO: we would like to move this to a database in the future
# api_keys: Set[str] = set()
# api_config: Dict[str, Any] = {
#     "api_keys": [],
#     "admin_key": None
# }


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

    # initialize the KeyManager
    key_manager = KeyManager.get_instance(key_path)

    # Read and log the YAML config file contents
    try:
        if config_path:
            config_contents = load_yaml_config(config_path)
            log.info(f"Config file contents from {config_path}:\n{config_contents}")
        else:
            log.warning("Config path is not set, cannot read config file")

    except Exception as e:
        log.error(f"Error reading config files: {str(e)}")

    if not config_path:
        raise ValueError("Environment variables GRAPHDOC_CONFIG_PATH must be set")

    # Initialize the model
    if not init_model(config_path):
        raise RuntimeError("Failed to initialize model")
    
    # make sure we have the correct authentication environment variables set (TODO: this should be redundant given the mdh, but we are having issues)
    graph_doc.mdh.set_auth_env_vars()

    # Set dspy and mlflow tracking for traces
    mlflow.dspy.autolog()
    mlflow.set_experiment(config_contents["server"]["mlflow_experiment_name"])

    # Load API keys
    # load_api_keys()

    @app.route("/health", methods=["GET"])
    def health_check():
        """Health check endpoint."""
        return jsonify({"status": "healthy", "model_loaded": module is not None})

    @app.route("/model/version", methods=["GET"])
    @key_manager.require_api_key
    def model_version():
        """Get model version information."""
        if not module or not config:
            return jsonify({"error": "Model not loaded"}), 503

        assert config is not None
        return jsonify(  # TODO: we can expand this more as we add tighter coupling between mlflow and the server
            {
                "model_name": config["prompt"]["prompt"],
            }
        )

    @app.route("/inference", methods=["POST"])
    @key_manager.require_api_key
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

            # make sure we have a client initialized
            if graph_doc.mdh is None: 
                raise ValueError("Ensure that GraphDoc is initialized with mlflow_tracking_uri, mlflow_tracking_username, and mlflow_tracking_password")

            # run the inference with tracing
            prediction = module.document_full_schema(
                database_schema=data["database_schema"],
                trace=True,
                client=graph_doc.mdh.mlflow_client,
                expirement_name=config_contents["server"]["mlflow_experiment_name"],
                api_key=request.headers["X-API-Key"], # record the api key that made the request
            )

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
    @key_manager.require_admin_key
    def create_api_key():
        """Create a new API key (admin only)."""
        new_key = key_manager.generate_api_key()
        return jsonify({"status": "success", "api_key": new_key})

    @app.route("/api-keys/list", methods=["GET"])
    @key_manager.require_admin_key
    def list_api_keys():
        """List all API keys (admin only)."""
        return jsonify({"status": "success", "api_keys": list(key_manager.api_keys)})

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

    # initialize the KeyManager
    key_manager = KeyManager.get_instance(key_path)
    log.info(f"Keys: {key_manager.api_keys}")
    log.info(f"Admin key: {key_manager.get_admin_key()}")

    # Load existing API keys
    # load_api_keys()

    # Set admin key if provided
    if args.admin_key:
        key_manager.set_admin_key(args.admin_key)
        log.info("Admin key set from command line argument")

    # Create initial API key if none exists
    if not key_manager.api_keys:
        initial_key = key_manager.generate_api_key()
        log.info(f"Created initial API key: {initial_key}")

    # Create initial admin key if none exists
    if not key_manager.get_admin_key():
        admin_key = secrets.token_hex(32)
        key_manager.set_admin_key(admin_key)
        log.info(f"Created initial admin key: {admin_key}")

    # Create and run the app
    app = create_app()
    app.run(host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()

# system packages
import os
import json
import logging
from pathlib import Path
import werkzeug.exceptions
from typing import Optional, Dict, Any, List, Set, Callable
import secrets
import functools

# internal packages
from .keys import KeyManager
from graphdoc import GraphDoc, load_yaml_config

# external packages
import dspy
import mlflow
from mlflow import MlflowClient
from flask import Flask, request, jsonify, Response
from flask_restx import Api, Resource, fields, Namespace

# logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# global variables
graph_doc: Optional[GraphDoc] = None
module: Optional[Any] = None
config: Optional[Dict[str, Any]] = None
app_dir = Path(os.path.dirname(os.path.abspath(__file__)))
keys_dir = app_dir / "keys"
keys_dir.mkdir(exist_ok=True)
key_path = keys_dir / "api_key_config.json"


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

    ###################################
    # flask-restx api with swagger ui #
    ###################################
    authorizations = {"apikey": {"type": "apiKey", "in": "header", "name": "X-API-Key"}}

    api = Api(
        app,
        version="1.0",
        title="GraphDoc API",
        description="API for the GraphDoc server",
        doc="/swagger",
        authorizations=authorizations,
        security="apikey",
    )

    # namespaces
    health_ns = Namespace("health", description="Health check operations")
    model_ns = Namespace("model", description="Model information operations")
    inference_ns = Namespace("inference", description="Inference operations")
    api_keys_ns = Namespace("api-keys", description="API key management operations")

    api.add_namespace(health_ns, path="")
    api.add_namespace(model_ns, path="")
    api.add_namespace(inference_ns, path="")
    api.add_namespace(api_keys_ns, path="")

    # models for request/response
    inference_request = api.model(
        "InferenceRequest",
        {
            "database_schema": fields.String(
                required=True, description="Database schema to document"
            )
        },
    )

    inference_response = api.model(
        "InferenceResponse",
        {
            "prediction": fields.Raw(description="Model prediction"),
            "status": fields.String(description="Status of the inference"),
        },
    )

    error_response = api.model(
        "ErrorResponse",
        {
            "error": fields.String(description="Error message"),
            "status": fields.String(description="Status of the operation"),
        },
    )

    key_response = api.model(
        "KeyResponse",
        {
            "status": fields.String(description="Status of the operation"),
            "api_key": fields.String(description="Generated API key"),
        },
    )

    keys_list_response = api.model(
        "KeysListResponse",
        {
            "status": fields.String(description="Status of the operation"),
            "api_keys": fields.List(fields.String, description="List of API keys"),
        },
    )

    ###################################
    # read in config                  #
    ###################################
    config_path = os.getenv("GRAPHDOC_CONFIG_PATH")
    log.info(f"Config path: {config_path}")

    # read and log the YAML config file contents
    try:
        if config_path:
            config_contents = load_yaml_config(config_path)
            log.info(f"Config file contents from {config_path}:\n{config_contents}")
        else:
            raise ValueError("Config path is not set, cannot read config file")
    except Exception as e:
        log.error(f"Error reading config files: {str(e)}")
        raise ValueError("Environment variables GRAPHDOC_CONFIG_PATH must be set")

    ###################################
    # initialize objects              #
    ###################################

    # initialize the KeyManager
    key_manager = KeyManager.get_instance(key_path)

    # Initialize the model
    if not init_model(config_path):
        raise RuntimeError("Failed to initialize model")

    # make sure we have the correct authentication environment variables set (TODO: this should be redundant given the mdh, but we are having issues)
    if graph_doc.mdh is not None: # type: ignore # we explicitely check for graphdoc.mdh is not None
        graph_doc.mdh.set_auth_env_vars()  # type: ignore # we explicitely check for graphdoc.mdh is not None
    else:
        raise ValueError(
            "GraphDoc is not initialized with a MlflowDataHelper and therefore cannot connect to MLflow"
        )

    # Set dspy and mlflow tracking for traces
    mlflow.dspy.autolog()
    mlflow.set_experiment(config_contents["server"]["mlflow_experiment_name"])

    @health_ns.route("")
    class HealthCheck(Resource):
        @health_ns.doc("health_check")
        def get(self):
            """Health check endpoint."""
            return {"status": "healthy", "model_loaded": module is not None}

    @model_ns.route("/version")
    class ModelVersion(Resource):
        @model_ns.doc("model_version")
        @key_manager.require_api_key
        def get(self):
            """Get model version information."""
            if not module or not config:
                return {"error": "Model not loaded"}, 503

            assert config is not None
            return {  # TODO: we can expand this more as we add tighter coupling between mlflow and the server
                "model_name": config["prompt"]["prompt"],
            }

    @inference_ns.route("")
    class Inference(Resource):
        @inference_ns.doc("run_inference")
        @inference_ns.expect(inference_request)
        @inference_ns.response(200, "Success", inference_response)
        @inference_ns.response(400, "Bad Request", error_response)
        @inference_ns.response(500, "Internal Server Error", error_response)
        @inference_ns.response(503, "Service Unavailable", error_response)
        @key_manager.require_api_key
        def post(self):
            """Run inference on the loaded model."""
            if not module:
                return {"error": "Model not loaded"}, 503

            try:
                # try to parse the JSON data
                try:
                    data = request.get_json()
                except (json.JSONDecodeError, werkzeug.exceptions.BadRequest):
                    return {"error": "Invalid JSON in request"}, 400

                # check for required fields
                if not data or "database_schema" not in data:
                    return {"error": "Missing database_schema in request"}, 400

                # make sure we have a client initialized
                if graph_doc.mdh is None: # type: ignore # we explicitely check for graphdoc.mdh is not None
                    raise ValueError(
                        "Ensure that GraphDoc is initialized with mlflow_tracking_uri, mlflow_tracking_username, and mlflow_tracking_password"
                    )

                # run the inference with tracing
                prediction = module.document_full_schema(
                    database_schema=data["database_schema"],
                    trace=True,
                    client=graph_doc.mdh.mlflow_client, # type: ignore # we explicitely check for graphdoc.mdh is not None
                    expirement_name=config_contents["server"]["mlflow_experiment_name"],
                    api_key=request.headers[
                        "X-API-Key"
                    ],  # record the api key that made the request
                )

                # convert prediction to string if it's not already
                if hasattr(prediction, "prediction"):
                    prediction = prediction.prediction
                elif not isinstance(prediction, (str, int, float, bool, list, dict)):
                    prediction = str(prediction)

                return {"prediction": prediction, "status": "success"}
            except Exception as e:
                log.error(f"Error during inference: {str(e)}")
                return {"error": str(e), "status": "error"}, 500

    @api_keys_ns.route("/generate")
    class CreateApiKey(Resource):
        @api_keys_ns.doc("create_api_key")
        @api_keys_ns.response(200, "Success", key_response)
        @key_manager.require_admin_key
        def post(self):
            """Create a new API key (admin only)."""
            new_key = key_manager.generate_api_key()
            return {"status": "success", "api_key": new_key}

    @api_keys_ns.route("/list")
    class ListApiKeys(Resource):
        @api_keys_ns.doc("list_api_keys")
        @api_keys_ns.response(200, "Success", keys_list_response)
        @key_manager.require_admin_key
        def get(self):
            """List all API keys (admin only)."""
            return {"status": "success", "api_keys": list(key_manager.api_keys)}

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

    # set environment variables for the app factory
    os.environ["GRAPHDOC_CONFIG_PATH"] = args.config_path

    # initialize the KeyManager
    key_manager = KeyManager.get_instance(key_path)
    log.info(f"Keys: {key_manager.api_keys}")
    log.info(f"Admin key: {key_manager.get_admin_key()}")

    # set admin key if provided
    if args.admin_key:
        key_manager.set_admin_key(args.admin_key)
        log.info("Admin key set from command line argument")

    # create initial API key if none exists
    if not key_manager.api_keys:
        initial_key = key_manager.generate_api_key()
        log.info(f"Created initial API key: {initial_key}")

    # create initial admin key if none exists
    if not key_manager.get_admin_key():
        admin_key = secrets.token_hex(32)
        key_manager.set_admin_key(admin_key)
        log.info(f"Created initial admin key: {admin_key}")

    # create and run the app
    app = create_app()
    app.run(host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()

# system
import os
import json
import logging
from pathlib import Path
import werkzeug.exceptions
from typing import Optional, Dict, Any

# internal
from graphdoc import GraphDoc, load_yaml_config

# external 
import dspy
import mlflow
from mlflow import MlflowClient
from flask import Flask, request, jsonify

# Set up logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Global variables to store our loaded objects
graph_doc: Optional[GraphDoc] = None
module: Optional[Any] = None
config: Optional[Dict[str, Any]] = None


def init_model(config_path: str, metric_config_path: str) -> bool:
    """Initialize the GraphDoc and load the module."""
    global graph_doc, module, config

    try:
        # Load configs
        loaded_config = load_yaml_config(config_path)
        if not loaded_config:
            raise ValueError("Failed to load config")

        metric_config = load_yaml_config(metric_config_path)
        if not metric_config:
            raise ValueError("Failed to load metric config")

        # Set up MLflow
        mlflow_tracking_uri = loaded_config["trainer"]["mlflow_tracking_uri"]
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow.set_experiment(loaded_config["module"]["experiment_name"])
        log.info(f"MLflow tracking URI: {mlflow_tracking_uri}")
        log.info(f"MLflow experiment name: {loaded_config['module']['experiment_name']}")

        # Initialize GraphDoc
        graph_doc = GraphDoc(
            model=loaded_config["language_model"]["lm_model_name"],
            api_key=loaded_config["language_model"]["lm_api_key"],
            hf_api_key=loaded_config["data"]["hf_api_key"],
            cache=loaded_config["language_model"]["cache"],
            mlflow_tracking_uri=mlflow_tracking_uri,
        )

        # Load the module
        module = graph_doc.doc_generator_module_from_mlflow(
            config_path, metric_config_path
        )

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
    metric_config_path = os.getenv("GRAPHDOC_METRIC_CONFIG_PATH")
    log.info(f"Config path: {config_path}")
    log.info(f"Metric config path: {metric_config_path}")

    # Read and log the YAML config file contents
    try:
        if config_path:
            with open(config_path, 'r') as file:
                config_contents = file.read()
                log.info(f"Config file contents from {config_path}:\n{config_contents}")
        else:
            log.warning("Config path is not set, cannot read config file")
            
        if metric_config_path:
            with open(metric_config_path, 'r') as file:
                metric_config_contents = file.read()
                log.info(f"Metric config file contents from {metric_config_path}:\n{metric_config_contents}")
        else:
            log.warning("Metric config path is not set, cannot read metric config file")
    except Exception as e:
        log.error(f"Error reading config files: {str(e)}")

    if not config_path or not metric_config_path:
        raise ValueError(
            "Environment variables GRAPHDOC_CONFIG_PATH and GRAPHDOC_METRIC_CONFIG_PATH must be set"
        )

    # Initialize the model
    if not init_model(config_path, metric_config_path):
        raise RuntimeError("Failed to initialize model")

    if not config:  # This should never happen due to the init_model check above
        raise RuntimeError("Config is not initialized")

    @app.route("/health", methods=["GET"])
    def health_check():
        """Health check endpoint."""
        return jsonify({"status": "healthy", "model_loaded": module is not None})

    @app.route("/model/version", methods=["GET"])
    def model_version():
        """Get model version information."""
        if not module or not config:
            return jsonify({"error": "Model not loaded"}), 503

        assert config is not None  # Help pyright understand config is not None
        mlflow_module_path = (
            Path(config["trainer"]["mlflow_tracking_uri"].replace("file://", ""))
            / "modules"
            / config["module"]["module_name"]
        )

        return jsonify(
            {
                "module_path": str(mlflow_module_path),
                "model_name": config["module"]["module_name"],
                "experiment_name": config["module"]["experiment_name"],
            }
        )

    @app.route("/inference", methods=["POST"])
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
            prediction = module.forward(data["database_schema"])

            # Convert prediction to string if it's not already
            if hasattr(prediction, "prediction"):
                prediction = prediction.prediction
            elif not isinstance(prediction, (str, int, float, bool, list, dict)):
                prediction = str(prediction)

            return jsonify({"prediction": prediction, "status": "success"})
        except Exception as e:
            log.error(f"Error during inference: {str(e)}")
            return jsonify({"error": str(e), "status": "error"}), 500

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
        "--metric-config-path",
        type=str,
        required=True,
        help="Path to the metric configuration YAML file.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Port to run the server on.",
    )

    args = parser.parse_args()

    # Set environment variables for the app factory
    os.environ["GRAPHDOC_CONFIG_PATH"] = args.config_path
    os.environ["GRAPHDOC_METRIC_CONFIG_PATH"] = args.metric_config_path

    # Create and run the app
    app = create_app()
    with mlflow.start_run():

        uri = "http://localhost:5000"
        client = MlflowClient(tracking_uri=uri)
        registered_models = client.search_registered_models()
        log.info(f"Registered models: {registered_models}")
        for model in registered_models:
            log.info(f"Registered model: {model.name}")

        app.run(host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()

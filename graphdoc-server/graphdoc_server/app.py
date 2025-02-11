from flask import Flask, request, jsonify
import logging
from pathlib import Path
import os
from typing import Optional
import json

from graphdoc import GraphDoc, load_yaml_config
import mlflow
import dspy

# Set up logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Global variables to store our loaded objects
graph_doc = None
module = None
config = None

def init_model(config_path: str, metric_config_path: str):
    """Initialize the GraphDoc and load the module."""
    global graph_doc, module, config
    
    try:
        # Load configs
        config = load_yaml_config(config_path)
        metric_config = load_yaml_config(metric_config_path)

        # Set up MLflow
        mlflow_tracking_uri = config["trainer"]["mlflow_tracking_uri"]
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow.set_experiment(config["module"]["experiment_name"])

        # Initialize GraphDoc
        graph_doc = GraphDoc(
            model=config["language_model"]["lm_model_name"],
            api_key=config["language_model"]["lm_api_key"],
            hf_api_key=config["data"]["hf_api_key"],
            cache=config["language_model"]["cache"],
            mlflow_tracking_uri=mlflow_tracking_uri,
        )

        # Load the module
        module = graph_doc.doc_generator_module_from_mlflow(config_path, metric_config_path)
        
        log.info("Successfully initialized model and loaded module")
        return True
    except Exception as e:
        log.error(f"Error initializing model: {str(e)}")
        return False

def create_app():
    """Create and configure the Flask application."""
    app = Flask(__name__)
    
    config_path = os.getenv('GRAPHDOC_CONFIG_PATH')
    metric_config_path = os.getenv('GRAPHDOC_METRIC_CONFIG_PATH')
    
    if not config_path or not metric_config_path:
        raise ValueError("Environment variables GRAPHDOC_CONFIG_PATH and GRAPHDOC_METRIC_CONFIG_PATH must be set")
    
    # Initialize the model
    if not init_model(config_path, metric_config_path):
        raise RuntimeError("Failed to initialize model")
    
    @app.route("/health", methods=["GET"])
    def health_check():
        """Health check endpoint."""
        return jsonify({
            "status": "healthy",
            "model_loaded": module is not None
        })

    @app.route("/model/version", methods=["GET"])
    def model_version():
        """Get model version information."""
        if not module:
            return jsonify({"error": "Model not loaded"}), 503
        
        mlflow_module_path = Path(config["trainer"]["mlflow_tracking_uri"].replace("file://", "")) / "modules" / config["module"]["module_name"]
        
        return jsonify({
            "module_path": str(mlflow_module_path),
            "model_name": config["module"]["module_name"],
            "experiment_name": config["module"]["experiment_name"]
        })

    @app.route("/inference", methods=["POST"])
    def inference():
        """Run inference on the loaded model."""
        if not module:
            return jsonify({"error": "Model not loaded"}), 503

        try:
            # Get the database schema from the request
            data = request.get_json()
            if not data or "database_schema" not in data:
                return jsonify({"error": "Missing database_schema in request"}), 400

            # Run inference
            prediction = module.forward(data["database_schema"])
            
            # Convert prediction to string if it's not already
            if hasattr(prediction, 'prediction'):
                prediction = prediction.prediction
            elif not isinstance(prediction, (str, int, float, bool, list, dict)):
                prediction = str(prediction)
            
            return jsonify({
                "prediction": prediction,
                "status": "success"
            })
        except json.JSONDecodeError:
            return jsonify({"error": "Invalid JSON in request"}), 400
        except Exception as e:
            log.error(f"Error during inference: {str(e)}")
            return jsonify({
                "error": str(e),
                "status": "error"
            }), 500
    
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
    os.environ['GRAPHDOC_CONFIG_PATH'] = args.config_path
    os.environ['GRAPHDOC_METRIC_CONFIG_PATH'] = args.metric_config_path
    
    # Create and run the app
    app = create_app()
    app.run(host="0.0.0.0", port=args.port)

if __name__ == "__main__":
    main() 
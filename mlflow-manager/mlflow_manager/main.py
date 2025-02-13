# system packages 
import argparse
import tempfile
import shutil
from pathlib import Path
import os
from typing import Any

# external packages 
import dspy
import mlflow
from mlflow_export_import.model.import_model import import_model as mlflow_import_model
from mlflow_export_import.model.export_model import export_model as mlflow_export_model

# logging
import logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

class MLFlowManager:
    def __init__(self, source_tracking_uri: str, target_tracking_uri: str):
        self.source_tracking_uri = source_tracking_uri
        self.target_tracking_uri = target_tracking_uri

        self.source_client = mlflow.MlflowClient(tracking_uri=source_tracking_uri)
        self.target_client = mlflow.MlflowClient(tracking_uri=target_tracking_uri)

        log.info(f"Source tracking URI: {self.source_tracking_uri}")
        log.info(f"Target tracking URI: {self.target_tracking_uri}")
        log.info(f"Source client: {self.source_client}")
        log.info(f"Target client: {self.target_client}")

        self.qual_ex = dspy.Example(
            database_schema="database_schema",
            category="category",
            rating=5,
        ).with_inputs("database_schema")

        self.gen_ex = dspy.Example(
            database_schema="database_schema",
            documented_schema="documented_schema",
        ).with_inputs("database_schema")

        self.qual_ex_signature = mlflow.models.infer_signature({"database_schema": "database_schema"})
        self.gen_ex_signature = mlflow.models.infer_signature({"database_schema": "database_schema"})

    
    def load_latest_version(self, client: mlflow.MlflowClient, model_name: str):
        model_latest_version = client.get_latest_versions(model_name)        
        return mlflow.dspy.load_model(model_latest_version[0].source)
    
    def save_model(self, client: mlflow.MlflowClient, model_name: str, model_signature: str, model: Any):
        if model_signature == "qual_ex_signature":
            signature = self.qual_ex_signature
        elif model_signature == "gen_ex_signature":
            signature = self.gen_ex_signature
        else:
            raise ValueError(f"Invalid model signature: {model_signature}")
        
        mlflow.dspy.log_model(
            dspy_model=model,
            artifact_path="model",
            signature=signature,
            task=None,
            registered_model_name=model_name,
        )  # TODO: add metadata related to trainset and evalset

    def copy_model(
        self,
        model_name: str,
        target_model_name: str = None,
        target_experiment_name: str = None,
        stages: list = None,
        versions: list = None,
        export_latest_versions: bool = False,
        export_version_model: bool = True,
        delete_model: bool = True,
        import_permissions: bool = False,
        import_source_tags: bool = False,
        await_creation_for: int = None,
    ):
        """
        Copy a model from source MLflow server to target MLflow server.
        
        Args:
            model_name: Name of the model to copy from source
            target_model_name: Name to use in target server (defaults to source name)
            target_experiment_name: Name of experiment to use in target (defaults to model name)
            stages: List of stages to export (e.g. ["Production", "Staging"])
            versions: List of versions to export
            export_latest_versions: If True, export only the latest versions
            export_version_model: If True, export the model artifact (default: True)
            delete_model: If True, delete existing model in target before import
            import_permissions: If True, import Databricks permissions
            import_source_tags: If True, import source information as tags
            await_creation_for: Number of seconds to wait for model version creation
        """
        target_model_name = target_model_name or model_name
        target_experiment_name = target_experiment_name or model_name
        log.info(f"Copying model {model_name} to {target_model_name} in experiment {target_experiment_name}")
        log.info(f"Source tracking URI: {self.source_tracking_uri}")
        log.info(f"Target tracking URI: {self.target_tracking_uri}")

        # Create a temporary directory for the export/import process
        with tempfile.TemporaryDirectory() as temp_dir:
            # Set tracking URI for export
            mlflow.set_tracking_uri(self.source_tracking_uri)
            
            # Export the model with artifacts
            export_path = Path(temp_dir) / "model_export"
            export_path.mkdir(parents=True, exist_ok=True)
            
            log.info(f"Exporting model and artifacts to {export_path}")
            success, exported_model = mlflow_export_model(
                model_name=model_name,
                output_dir=str(export_path),
                stages=stages,
                versions=versions,
                export_latest_versions=export_latest_versions,
                export_version_model=export_version_model,
                mlflow_client=self.source_client,
            )
            if not success:
                raise Exception(f"Failed to export model {model_name}")
            log.info(f"Exported model {exported_model} from {self.source_tracking_uri} to {export_path}")

            # Set tracking URI for import
            mlflow.set_tracking_uri(self.target_tracking_uri)
            
            log.info(f"Importing model and artifacts from {export_path}")
            mlflow_import_model(
                model_name=target_model_name,
                experiment_name=target_experiment_name,
                input_dir=str(export_path),
                delete_model=delete_model,
                import_permissions=import_permissions,
                import_source_tags=import_source_tags,
                await_creation_for=await_creation_for,
                mlflow_client=self.target_client,
            )
            log.info(f"Imported model to {self.target_tracking_uri} from {export_path}")
        return True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_tracking_uri", type=str, required=True)
    parser.add_argument("--target_tracking_uri", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    args = parser.parse_args()

    mm = MLFlowManager(
        source_tracking_uri=args.source_tracking_uri,
        target_tracking_uri=args.target_tracking_uri
    )

    mlflow.set_tracking_uri(args.source_tracking_uri)
    model = mm.load_latest_version(mm.source_client, args.model_name)
    mlflow.set_tracking_uri(args.target_tracking_uri)
    mm.save_model(mm.target_client, args.model_name, "qual_ex_signature", model)

if __name__ == "__main__":
    main()
    # source_tracking_uri: http://localhost:4000
    # target_tracking_uri: http://localhost:5000
    # model_name: doc_generator_model
    # poetry run python mlflow_manager/main.py --source_tracking_uri http://localhost:4000 --target_tracking_uri http://localhost:5000 --model_name doc_generator_model

# import-model \
#   --model sklearn_wine \
#   --experiment-name sklearn_wine_imported \
#   --input-dir out  \
#   --delete-model True

# import-model --help

# Options:
#   --input-dir TEXT              Input directory  [required]
#   --model TEXT                  Registered model name.  [required]
#   --experiment-name TEXT        Destination experiment name  [required]
#   --delete-model BOOLEAN        If the model exists, first delete the model
#                                 and all its versions.  [default: False]
#   --import-permissions BOOLEAN  Import Databricks permissions using the HTTP
#                                 PATCH method.  [default: False]
#   --import-source-tags BOOLEAN  Import source information for registered model
#                                 and its versions ad tags in destination
#                                 object.  [default: False]
#   --await-creation-for INTEGER  Await creation for specified seconds.


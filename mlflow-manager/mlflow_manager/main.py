# system packages 

# external packages 
import mlflow
from mlflow_export_import.model.import_model import import_model as mlflow_import_model
from mlflow_export_import.model.export_model import export_model as mlflow_export_model

class MLFlowManager:
    def __init__(self, source_tracking_uri: str, target_tracking_uri: str):
        self.source_tracking_uri = source_tracking_uri
        self.target_tracking_uri = target_tracking_uri

        self.source_client = mlflow.MlflowClient(tracking_uri=source_tracking_uri)
        self.target_client = mlflow.MlflowClient(tracking_uri=target_tracking_uri)

    def copy_experiment(self, experiment_name: str):
        pass

    def import_model(
        self,
        model_name: str,
        experiment_name: str,
        input_dir: str,
        delete_model: bool = False,
        import_permissions: bool = False,
        import_source_tags: bool = False,
        await_creation_for: int = None,
        verbose: bool = False
    ):
        """
        Import a model from a directory into MLflow.
        
        Args:
            model_name: Name of the model to import
            experiment_name: Name of the experiment to import the model into
            input_dir: Directory containing the exported model
            delete_model: If True, delete existing model before import
            import_permissions: If True, import Databricks permissions
            import_source_tags: If True, import source information as tags
            await_creation_for: Number of seconds to wait for model version creation
            verbose: If True, print verbose output
        """
        # Set the MLflow tracking URI to the target
        mlflow.set_tracking_uri(self.target_tracking_uri)
        
        return mlflow_import_model(
            model_name=model_name,
            experiment_name=experiment_name,
            input_dir=input_dir,
            delete_model=delete_model,
            import_permissions=import_permissions,
            import_source_tags=import_source_tags,
            await_creation_for=await_creation_for,
            verbose=verbose
        )

    def export_model(
        self,
        model_name: str,
        output_dir: str,
        stages: list = None,
        versions: list = None,
        export_latest_versions: bool = False,
        export_version_model: bool = False,
        export_all_runs: bool = False
    ):
        """
        Export a model to a directory.
        
        Args:
            model_name: Name of the model to export
            output_dir: Directory to export the model to
            stages: List of stages to export (e.g. ["Production", "Staging"])
            versions: List of versions to export
            export_latest_versions: If True, export only the latest versions
            export_version_model: If True, export the model artifact
            export_all_runs: If True, export all runs associated with the model versions
        """
        # Set the MLflow tracking URI to the source
        mlflow.set_tracking_uri(self.source_tracking_uri)
        
        return mlflow_export_model(
            model_name=model_name,
            output_dir=output_dir,
            stages=stages,
            versions=versions,
            export_latest_versions=export_latest_versions,
            export_version_model=export_version_model,
            export_all_runs=export_all_runs
        )

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


# system packages
import logging
from pathlib import Path
from typing import List, Literal, Optional, Union

# internal packages
from .evaluate import DocQuality
from .loader.helper import load_yaml_config, setup_logging
from .train import TrainerFactory
from .prompts import PromptFactory, SinglePrompt
from .modules import DocGeneratorModule
from .data import DataHelper
from .loader import FlowLoader

# external packages
import dspy
from datasets import Dataset, IterableDataset

# logging
log = logging.getLogger(__name__)


class GraphDoc:
    def __init__(
        self,
        model: str,
        api_key: str,
        hf_api_key: str,
        cache: bool = True,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO",
        mlflow_tracking_uri: Optional[str] = None,
    ) -> None:
        setup_logging(log_level)
        log.info(
            f"GraphDoc initialized with model: {model}, cache: {cache}, log_level: {log_level}"
        )

        # initialize base dspy config
        kwargs = {
            "model": model,
            "api_key": api_key,
            "cache": cache,
        }
        if temperature is not None:
            kwargs["temperature"] = temperature
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

        self.lm = dspy.LM(**kwargs)
        dspy.configure(lm=self.lm)

        # initialize modules
        self.doc_eval = dspy.Predict(DocQuality)

        # initialize data helper
        self.dh = DataHelper(hf_api_key=hf_api_key)

        # initialize flow loader
        if mlflow_tracking_uri is not None:
            self.fl = FlowLoader(mlflow_tracking_uri)
        else: 
            self.fl = None

    ############
    # TRAINING #
    ############
    def update_graphdoc_dataset(
        self, local_file: bool = True, repo_card: bool = False
    ) -> None:
        """
        Updates the GraphDoc dataset either from a local file or from Hugging Face.

        Args:
            local_file (bool): If True, updates from a local file. Defaults to True.
            repo_card (bool): If True, creates and uploads a repo card. Defaults to False.
        """
        if not local_file:
            log.warning("Only local file updates are currently supported")
            return

        try:
            hf_ds = self.dh._load_from_hf()
            if hf_ds is None:
                log.error("Failed to load dataset from Hugging Face")
                return

            local_ds = self.dh._folder_of_folders_to_dataset()
            if local_ds is None:
                log.error("Failed to load dataset from local folder")
                return

            if not isinstance(hf_ds, dict) or "train" not in hf_ds:
                log.error("Invalid dataset format from Hugging Face")
                return

            train_dataset = hf_ds["train"]
            if not isinstance(train_dataset, Dataset):
                log.error("Invalid dataset type from Hugging Face")
                return

            ds = self.dh._add_to_graph_doc_dataset(train_dataset, local_ds)
            if ds is None:
                log.error("Failed to add data to GraphDoc dataset")
                return

            ds = self.dh._drop_dataset_duplicates(ds)
            if ds is None:
                log.error("Failed to drop duplicates from dataset")
                return

            if len(train_dataset) != len(ds):
                try:
                    self.dh._upload_to_hf(ds)
                    log.info(f"Dataset uploaded to Hugging Face (length: {len(ds)})")
                except Exception as e:
                    log.error(f"Failed to upload dataset to Hugging Face: {e}")
                    raise ValueError(f"Failed to upload dataset to Hugging Face: {e}")
            else:
                log.info(f"No new data to upload (length: {len(ds)})")

            if repo_card:
                try:
                    self.dh._create_and_upload_repo_card()
                    log.info("Repo card created and uploaded")
                except Exception as e:
                    log.error(f"Failed to create and upload repo card: {e}")
                    raise ValueError(f"Failed to create and upload repo card: {e}")

        except Exception as e:
            log.error(f"An unexpected error occurred: {e}")
            raise

    def _get_single_prompt(self, config_path: Union[str, Path]):
        config = load_yaml_config(config_path)
        try:
            prompt = config["prompt"]["prompt"]
            prompt_class = config["prompt"]["class"]
            prompt_type = config["prompt"]["type"]
            prompt_metric = config["prompt"]["metric"]
            prompt = PromptFactory.get_single_prompt(
                prompt, prompt_class, prompt_type, prompt_metric
            )
        except Exception as e:
            raise ValueError(f"Failed to initialize prompt class: {e}")
        return prompt

    def _get_nested_single_prompt(
        self, config_path: Union[str, Path], metric_config_path: Union[str, Path]
    ):
        """
        This is a single prompt that utilizes another single prompt as a metric.
        """  # the builder pattern could be used here to make this more readable and avoid this separation of the get_single_prompt and get_nested_single_prompt
        config = load_yaml_config(config_path)
        try:
            metric_prompt = self._get_single_prompt(metric_config_path)
            prompt = config["prompt"]["prompt"]
            prompt_class = config["prompt"]["class"]
            prompt_type = config["prompt"]["type"]
            prompt_metric = metric_prompt
            prompt = PromptFactory.get_single_prompt(
                prompt, prompt_class, prompt_type, prompt_metric
            )
        except Exception as e:
            raise ValueError(f"Failed to initialize nested single prompt: {e}")
        return prompt

    def _get_single_trainer(
        self,
        config_path: Union[str, Path],
        trainset: List[dspy.Example],
        evalset: List[dspy.Example],
        prompt: Optional[SinglePrompt] = None,
    ):
        config = load_yaml_config(config_path)
        try:
            trainer_class = config["trainer"]["class"]
            optimizer_type = config["trainer"]["optimizer_type"]
            mlflow_tracking_uri = config["trainer"]["mlflow_tracking_uri"]
            mlflow_model_name = config["trainer"]["mlflow_model_name"]
            mlflow_experiment_name = config["trainer"]["mlflow_experiment_name"]

            if prompt is None:
                prompt = self._get_single_prompt(config_path)

            optimizer_kwargs = config["optimizer"]

            trainer = TrainerFactory.get_single_prompt_trainer(
                trainer_class=trainer_class,
                prompt=prompt,
                optimizer_type=optimizer_type,
                optimizer_kwargs=optimizer_kwargs,
                mlflow_tracking_uri=mlflow_tracking_uri,
                mlflow_model_name=mlflow_model_name,
                mlflow_experiment_name=mlflow_experiment_name,
                trainset=trainset,
                evalset=evalset,
            )
        except Exception as e:
            raise ValueError(f"Failed to initialize trainer class: {e}")
        return trainer

    #######################
    # Loaging from MLFlow #
    #######################
    def _load_prompt_from_mlflow(self, config: dict):
        mlflow_model_uri = config["prompt"].get("mlflow_uri")
        mlflow_model_name = config["prompt"].get("mlflow_model_name")
        mlflow_model_version = config["prompt"].get("mlflow_model_version")
        
        if mlflow_model_uri is not None:
            prompt = self.fl.load_model_by_uri(mlflow_model_uri)
        elif mlflow_model_name is not None and mlflow_model_version is not None:
            prompt = self.fl.load_model_by_name_and_version(mlflow_model_name, mlflow_model_version)
        elif mlflow_model_name is not None:
            prompt = self.fl.load_latest_version(mlflow_model_name)
        else: 
            log.warning("No MLFlow model URI or name and version provided")
            return None
        
        return prompt


    def prompt_from_mlflow(self, config_path: Union[str, Path], default_prompt: bool = True):
        # load_from_uri: false # Whether to load the prompt from an MLFlow URI
        # mlflow_uri: null # The tracking URI for MLflow
        config = load_yaml_config(config_path)
        try:
            if config["prompt"]["load_from_uri"]:
                if self.fl is None: 
                    raise ValueError("MLFlow tracking URI not set")
            prompt = self._load_prompt_from_mlflow(config)
            
            # TODO: we should refactor this
            if prompt: 
                prompt_signature = self.fl.get_prompt_signature(prompt)
            
            elif prompt is None and default_prompt:
                prompt = self._get_single_prompt(config_path)
                prompt_signature = prompt.prompt
            
            prompt = PromptFactory.get_single_prompt(prompt_signature, config["prompt"]["class"], config["prompt"]["type"], config["prompt"]["metric"])
            return prompt
        except Exception as e:
            raise ValueError(f"Failed to load prompt from MLFlow: {e}")
    
    def nested_prompt_from_mlflow(self, config_path: Union[str, Path], metric_config_path: Union[str, Path], default_prompt: bool = True):
        config = load_yaml_config(config_path)
        try:
            metric_prompt = self.prompt_from_mlflow(metric_config_path, default_prompt)
            prompt = self._load_prompt_from_mlflow(config)

            # TODO: we should refactor this
            if prompt: 
                prompt_signature = self.fl.get_prompt_signature(prompt)
            
            elif prompt is None and default_prompt:
                prompt = self._get_single_prompt(config_path)
                prompt_signature = prompt.prompt
            # prompt_signature = self.fl.get_prompt_signature(prompt)
            
            prompt = PromptFactory.get_single_prompt(prompt_signature, config["prompt"]["class"], config["prompt"]["type"], metric_prompt)
            return prompt
        except Exception as e:
            raise ValueError(f"Failed to load nested prompt from MLFlow: {e}")
        
    def doc_generator_module_from_mlflow(self, config_path: Union[str, Path], metric_config_path: Union[str, Path], default_prompt: bool = True):
        doc_generator_prompt = self.nested_prompt_from_mlflow(config_path, metric_config_path, default_prompt)
        doc_generator_module = DocGeneratorModule(doc_generator_prompt)
        return doc_generator_module
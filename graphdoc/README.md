# graphdoc

This project is aimed at generating documentation given a graphql schema. 

## .env

Your `.env` file should look like the following: 

```
OPENAI_API_KEY=<your openai api key>
HF_DATASET_KEY=<a huggingface api key with access to datasets>
MLFLOW_TRACKING_URI=<the path to your mlflow tracking instance>
```

## Installation 

Ensure you have `pyenv` and `poetry` installed on your local machine. The instructions below are for `macOs`

```bash
# Install pyenv and poetry via Homebrew
brew install pyenv poetry

# ⚠️ WARNING: If you previously installed Python via Homebrew, you might want to:
brew unlink python
# This prevents conflicts between Homebrew's Python and pyenv's Python
```

Ensure you have the proper version of python installed for this project. 

```bash
# First, check if the .python-version file exists and read its content
cat .python-version  # Should show something like "3.11.8"

# Install the Python version specified in .python-version
pyenv install $(cat .python-version)

# Verify the installation
pyenv versions
```

Install the poetry package and its dependencies. 

```bash
# Ensure you're in the project directory
cd graphdoc

# Install dependencies from pyproject.toml
poetry install

# Verify the virtual environment
poetry env info

# Activate the virtual environment (optional)
poetry shell
```

To utilize the Jupyter notebook, we will need to initialize a kernel. 

```bash
poetry run python -m ipykernel install --user --name=graphdoc
```

## Dataset 

We have one primary dataset for the `GraphDoc` program, which contains schemas and an associated rating. The table is as follows: 

| category | rating | schema_name | schema_type | schema_str | 
|----------|--------|-------------|-------------|------------|
| [ perfect, almost perfect, poor but correct, incorrect ] | [4, 3, 2, 1] | str | [full schema, table schema] | str |

A public dataset can be found at [semiotic/graphdoc_schemas](https://huggingface.co/datasets/semiotic/graphdoc_schemas). 

## Runners

The `runners` directory contains early implementations of the `GraphDoc` program. These will largely be replaced by the `GraphDoc` program, but are useful for testing, experimentation, and documentation. 

```bash
poetry run python runners/evaluate.py
poetry run python runners/generate.py
```

## DSPy

### Optimizers 

One thing from the docs: "For prompt optimizers, we suggest starting with a 20% split for training and 80% for validation, which is often the opposite of what one does for DNNs."

## MLFlow

```bash
mlflow ui --port 5000
```

### Model Versioning

We are using MLFlow to version our models. We want to be as careful as possible to ensure that the most recent version of our model is the best version of our model. To this end, we will want to cover a couple of things: 

1. Tracking Training and Evaluation data for each model version
2. Tracking the model's performance on the training and evaluation data

We are going to use HuggingFace datasets to track the training and evaluation data for each model version. We will track the dataset version in the `metadata` field of the model. So that we can easily track the dataset version for each model version. One thing we will need to handle is the fact that the signature in our codebase will not always map the signature of our saved model. To this end, we will want to treat each model as a signature type. For now, this is going to be either `zero shot` or `few shot`. Going forward, we can imagine a scenarior where we will enable `tools` and other various methods to affect the signature of our model. 

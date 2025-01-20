# graphdoc

This project is aimed at generating documentation given a graphql schema. 

## .env

Your `.env` file should look like the following: 

```
OPENAI_API_KEY=<your openai api key>
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
| [ Perfect, Almost Perfect, Somewhat Correct, Incorrect ] | [4, 3, 2, 1] | str | [Full Schema, Table Schema] | str |

A public dataset can be found at [semiotic/graphdoc_schemas](https://huggingface.co/datasets/semiotic/graphdoc_schemas). 
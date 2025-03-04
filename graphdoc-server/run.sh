#!/bin/bash

# exit on error
set -e

# load variables from .env file if it exists
if [ -f "./.env" ]; then
  echo "Loading environment from ./.env"
  set -a  # automatically export all variables
  source "./.env"
  set +a
elif [ -f "../.env" ]; then
  echo "Loading environment from ../.env"
  set -a  # automatically export all variables
  source "../.env"
  set +a
fi

# default values
CONFIG_PATH="../assets/configs/server/single_prompt_schema_doc_generator_module.yaml"
METRIC_CONFIG_PATH="../assets/configs/server/single_prompt_schema_doc_quality_trainer.yaml"
MLFLOW_TRACKING_URI="http://localhost:5001"
MLFLOW_DOCKER_URI="http://host.docker.internal:5001"
MLFLOW_TRACKING_USERNAME=admin
MLFLOW_TRACKING_PASSWORD=password
PORT=8080
WORKERS=4

# run development server
run_dev() {
    if [ -z "$CONFIG_PATH" ]; then
        echo "Error: Config path is required for dev mode"
        exit 1
    fi
    export GRAPHDOC_CONFIG_PATH="$CONFIG_PATH"
    export MLFLOW_TRACKING_URI="$MLFLOW_TRACKING_URI"
    python -m graphdoc_server.app --config-path "$CONFIG_PATH" --port "$PORT"
}

# run production server
run_prod() {
    if [ -z "$CONFIG_PATH" ]; then
        echo "Error: Config path is required for prod mode"
        exit 1
    fi
    export GRAPHDOC_CONFIG_PATH="$CONFIG_PATH"
    gunicorn "graphdoc_server.app:create_app()" \
        --bind "0.0.0.0:$PORT" \
        --workers "$WORKERS" \
        --access-logfile - \
        --error-logfile -
}

# run docker build 
docker_prod_build() {
    cd ..
    docker build -t graphdoc-server:local -f graphdoc-server/Dockerfile.prod .
}

docker_prod_run() {
    docker run -p 8080:8080 \
        -e PORT=8080 \
        -e GRAPHDOC_CONFIG_PATH=/app/configs/server/single_prompt_schema_doc_generator_module.yaml \
        -e GRAPHDOC_METRIC_CONFIG_PATH=/app/configs/server/single_prompt_schema_doc_quality_trainer.yaml \
        -e MLFLOW_TRACKING_URI=$MLFLOW_DOCKER_URI \
        -e MLFLOW_TRACKING_USERNAME=$MLFLOW_TRACKING_USERNAME \
        -e MLFLOW_TRACKING_PASSWORD=$MLFLOW_TRACKING_PASSWORD \
        -e OPENAI_API_KEY=${OPENAI_API_KEY:-""} \
        -e HF_DATASET_KEY=${HF_DATASET_KEY:-""} \
        graphdoc-server:local
}

docker_prod_build_and_run() {
    docker_prod_build
    docker_prod_run
}

# formatting and linting 
format_command() {
    poetry run black .
}

lint_command() {
    poetry run pyright .
}

# run tests
run_tests() {
    poetry run pytest -v -W ignore
}

# run code quality checks
run_checks() {
    format_command
    run_tests
    lint_command
}

# parse command line arguments
COMMAND=""
while [[ $# -gt 0 ]]; do
    case $1 in
        dev|prod|test|commit|format|lint|docker-prod|docker-prod-run)
            COMMAND="$1"
            shift
            ;;
        -c|--config)
            CONFIG_PATH="$2"
            shift 2
            ;;
        -p|--port)
            PORT="$2"
            shift 2
            ;;
        -w|--workers)
            WORKERS="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# function to display usage
show_help() {
    echo "Usage: ./run.sh [command] [options]"
    echo ""
    echo "Commands:"
    echo "  dev             Run in development mode"
    echo "  prod            Run in production mode"
    echo "  test            Run tests"
    echo "  format          Format code (black)"
    echo "  lint            Run linting (pyright)"
    echo "  commit          Run code quality checks (black, pyright, tests)"
    echo "  docker-prod     Run docker prod build and run"
    echo "  docker-prod-run Run docker prod build and run"
    echo "Options:"
    echo "  -c, --config PATH           Path to config file"
    echo "  -m, --metric-config PATH    Path to metric config file"
    echo "  -p, --port PORT            Port to run on (default: 6000)"
    echo "  -w, --workers NUM          Number of workers for production mode (default: 4)"
    echo "  -h, --help                 Show this help message"
    echo ""
    echo "Examples:"
    echo "  ./run.sh dev -c config.yaml -m metric_config.yaml"
    echo "  ./run.sh prod -c config.yaml -m metric_config.yaml -p 8000 -w 8"
    echo "  ./run.sh test"
    echo "  ./run.sh check"
}

# execute the appropriate command
case "$COMMAND" in
    # build and run commands
    "dev") run_dev ;;
    "prod") run_prod ;;

    # docker commands
    "docker-prod") docker_prod_build_and_run ;;
    "docker-prod-run") docker_prod_run ;;

    # test commands
    "test") run_tests ;;
    "format") format_command ;;
    "lint") lint_command ;;
    "commit") run_checks ;;
    
    "")
        echo "Error: No command specified"
        show_help
        exit 1
        ;;
esac 
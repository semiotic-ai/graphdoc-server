#!/bin/bash

# exit on error
set -e

# default values
CONFIG_PATH="../assets/configs/server/single_prompt_schema_doc_generator_module.yaml"
METRIC_CONFIG_PATH="../assets/configs/server/single_prompt_schema_doc_quality_trainer.yaml"
MLFLOW_TRACKING_URI="http://localhost:5001"
PORT=6000
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
# TODO: add run_docker_build()

# run tests
run_tests() {
    pytest -v
}

# run code quality checks
run_checks() {
    echo "🔍 Running Black formatter..."
    black .

    echo "🔍 Running Pyright type checker..."
    pyright

    echo "🧪 Running tests..."
    pytest -v

    echo "✅ All checks passed!"
}

# parse command line arguments
COMMAND=""
while [[ $# -gt 0 ]]; do
    case $1 in
        dev|prod|test|check)
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
    echo "  dev         Run in development mode"
    echo "  prod        Run in production mode"
    echo "  test        Run tests"
    echo "  check       Run code quality checks (black, pyright, tests)"
    echo ""
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

    # test commands
    "test") run_tests ;;
    "check") run_checks ;;
    
    "")
        echo "Error: No command specified"
        show_help
        exit 1
        ;;
esac 
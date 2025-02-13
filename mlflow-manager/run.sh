python_command() {
    poetry run python
}

shell_command() {
    echo "Spawning shell within virtual environment..."
    poetry shell
}

get_mlruns_path() {
    SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
    MLRUNS_PATH="$SCRIPT_DIR/../mlruns"
    if command -v realpath &> /dev/null; then
        MLRUNS_PATH=$(realpath "$MLRUNS_PATH")
    elif command -v readlink &> /dev/null; then
        MLRUNS_PATH=$(readlink -f "$MLRUNS_PATH")
    else
        echo "Warning: 'realpath' or 'readlink' not found. Using unresolved path."
    fi
    echo "$MLRUNS_PATH"
}

start_mlflow_ui() {
    # Ensure MLflow is installed
    if ! poetry run mlflow --version &> /dev/null; then
        echo "MLflow is not installed. Installing MLflow..."
        poetry add mlflow
    fi

    # Get the mlruns path
    MLRUNS_PATH=$(get_mlruns_path)
    echo "Starting MLflow UI with backend store URI: $MLRUNS_PATH"

    # Check if port 4000 is already in use
    if lsof -i :4000 &> /dev/null; then
        echo "Port 4000 is already in use. Killing existing processes..."
        kill_mlflow_ui
    fi

    # Start MLflow UI
    poetry run mlflow ui --backend-store-uri "$MLRUNS_PATH" --port 4000 &
    echo "MLflow UI started. Access it at http://localhost:4000"
}

kill_mlflow_ui() {
    # Find and kill the MLflow UI process
    PID=$(ps aux | grep 'mlflow ui' | grep -v grep | awk '{print $2}')

    if [ -z "$PID" ]; then
        echo "No running MLflow UI process found."
    else
        echo "Killing MLflow UI process with PID: $PID"
        kill "$PID"
        if ps -p "$PID" > /dev/null; then
            echo "Process $PID is still running. Forcefully killing it."
            kill -9 "$PID"
        else
            echo "Process $PID has been terminated."
        fi
    fi

    # Ensure port 4000 is freed
    if lsof -i :4000 &> /dev/null; then
        echo "Port 4000 is still in use. Killing remaining processes..."
        lsof -ti :4000 | xargs kill -9
    fi
}

show_help() {
    echo "Usage: ./run.sh [option]"
    echo "Options:"
    echo "  python                 Run Python"
    echo "  shell                  Run a shell in the virtual environment"
    echo "  mlruns-path            Show the full path to the mlruns directory"
    echo "  start-mlflow-ui        Start the MLflow UI"
    echo "  kill-mlflow-ui         Kill the MLflow UI"
}

load_env_variables() {
    if [ -f "../.env" ]; then
        export $(grep -v '^#' ../.env | xargs)
    else
        echo ".env file not found. Please ensure it exists in the root directory."
        exit 1
    fi
}

print_env_variables() {
    load_env_variables
    printenv
}

if [ -z "$1" ]; then
    show_help
else
    case "$1" in
        "python") python_command ;;
        "shell") shell_command ;;

        # run mlflow ui locally 
        "mlruns-path") get_mlruns_path ;;
        "start-mlflow-ui") start_mlflow_ui ;;
        "kill-mlflow-ui") kill_mlflow_ui ;;

        # env variables 
        "load-env-variables") load_env_variables ;;
        "print-env-variables") print_env_variables ;;
        *) show_help ;;
    esac
fi
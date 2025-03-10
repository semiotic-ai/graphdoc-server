#!/bin/bash

# Exit on error
set -e

# Check if required environment variables are set
if [ -z "$GRAPHDOC_CONFIG_PATH" ] || [ -z "$GRAPHDOC_METRIC_CONFIG_PATH" ]; then
    echo "Error: GRAPHDOC_CONFIG_PATH and GRAPHDOC_METRIC_CONFIG_PATH must be set"
    exit 1
fi

# Verify config files exist
if [ ! -f "$GRAPHDOC_CONFIG_PATH" ]; then
    echo "Error: Config file not found at $GRAPHDOC_CONFIG_PATH"
    exit 1
fi

if [ ! -f "$GRAPHDOC_METRIC_CONFIG_PATH" ]; then
    echo "Error: Metric config file not found at $GRAPHDOC_METRIC_CONFIG_PATH"
    exit 1
fi

echo "Config files found and validated"
echo "Using config: $GRAPHDOC_CONFIG_PATH"
echo "Using metric config: $GRAPHDOC_METRIC_CONFIG_PATH"

# Run the server in production mode
exec gunicorn "graphdoc_server.app:create_app()" \
    --bind "0.0.0.0:${PORT:-6000}" \
    --workers "${WORKERS:-4}" \
    --timeout "${TIMEOUT:-60}" \
    --log-level=debug \
    --access-logfile - \
    --error-logfile - 
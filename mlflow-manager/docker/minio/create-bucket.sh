#!/bin/bash

# SPDX-FileCopyrightText: 2025 Semiotic AI, Inc.
#
# SPDX-License-Identifier: Apache-2.0

set -e

echo "Waiting for MinIO to be ready..."
until mc alias set minioserver http://minio:${MINIO_PORT} ${MINIO_ROOT_USER} ${MINIO_ROOT_PASSWORD} 2>&1; do
    echo "MinIO is not ready - waiting..."
    sleep 1
done

echo "MinIO is ready. Creating bucket..."

# Create the MLFlow bucket if it doesn't exist
if ! mc ls minioserver/mlflow 2>/dev/null; then
    mc mb minioserver/mlflow
    echo "Bucket 'mlflow' created successfully"
else
    echo "Bucket 'mlflow' already exists"
fi

# Verify bucket exists
mc ls minioserver/mlflow
echo "Setup completed successfully"

<!--
SPDX-FileCopyrightText: 2025 Semiotic AI, Inc.

SPDX-License-Identifier: Apache-2.0
-->

# GraphDoc Server

This is a Flask application that serves as a GraphDoc server. It is used to serve the GraphDoc API.

## Setup

Ensure that you have an MLflow server running. This can be done by running the `mlflow-manager` service. Checkout out the [mlflow-manager README](../mlflow-manager/README.md) for more information.

Running the server can be done using the following command (depending on your desired setup):

```bash
# From the parent directory
docker compose -f graphdoc-server/docker-compose.yml --profile dev up --build

# From the parent directory
docker compose -f graphdoc-server/docker-compose.yml --profile prod up --build
```

```bash
# From the root directory
docker compose -f docker-compose.yml --profile dev up --build

# From the parent directory
docker compose -f docker-compose.yml --profile prod up --build
```
# MLFlow Manager

This module is meant to assist in setting up a local mlflow server and interacting with both local and remote mlflow servers. This is useful for sending training runs to a remote server, as well as for storing and serving models. 


## Docker 

Thanks to `violincoding` for their useful template for setting up a mlflow server with a postgres and minio backend. Their repository can be found [here](https://github.com/violincoding/mlflow-setup.git). We make slight modifications to the original template to suit our needs, but largely things are unchanges. 

### Prerequisites

Docker and docker-compose should be installed on your machine, either through [Docker Desktop](https://www.docker.com/products/docker-desktop/), or its alternatives such as [Orbstack](https://orbstack.dev/)

### Configure environment variables

Make a copy of the `docker/.env.example` file and rename it to `docker/.env`. Then, update the environment variables in the `.env` file as per your requirements.

### Build and start the services

Note, if you are using `pip3`, you will need to alias `pip` to `pip3` in order to install the dependencies.

```bash
alias pip=pip3
```

Build and start the services.

```bash
docker compose up -d --build
```

If everything is setup properly, you should be able to access the services at the following URLs:

- MLFlow Tracking Server: [http://localhost:5001](http://localhost:5001)
- MinIO Console UI: [http://localhost:9001](http://localhost:9001)

## Local MLFlow Server 

To start the mlflow server locally, run the following command. 

```bash 
mlflow ui --backend-store-uri <mlruns-path>
```

## Scripts 

Ensure the `run.sh` file is executable 

```bash
chmod +x run.sh
```

These are useful commands for setting up and tearing down the mlflow ui. 

```bash
# starts the mlflow ui locally, pointing towards the mlruns directory in the repo
./run.sh start-mlflow-ui 

# kills the running mlflow processes running locally
./run.sh kill-mlflow-ui 
```
## MLFlow Manager

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
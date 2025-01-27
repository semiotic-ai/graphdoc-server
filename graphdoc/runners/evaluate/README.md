# Evaluate

This is primarily an PoC of using DSPy and MLflow together. We will go through the basic steps of using MLFlow for monitoring our training and evaluation of a DSPy model. We will go through the basic steps of using MLFlow to save and load a DSPy model. We will go through the basic steps of using MLFlow to trace a DSPy model. From there, we will have the foundation to implement this in a more modular fashion within the graphdoc project.

## Basic Steps

1. Managing the DSPy / MLFlow relationship
    - [x] Create a DSPy model
    - [x] Create a MLFlow model that tracks to that DSPy model 
    - [x] Save the DSPy model to MLFlow
2. Training / Evaluation pipeline 
    - [x] load in a given MLFlow model
    - [ ] assess if it was trained on our most recent dataset
    - [ ] if not trained, train it on our most recent dataset
    - [ ] assess performance of the trained model on our most recent dataset
    - [x] store the training and evaluation results to MLFlow 
    - [x] save the trained model to MLFlow
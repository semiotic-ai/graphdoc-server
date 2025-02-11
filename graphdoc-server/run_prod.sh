#!/bin/bash
export GRAPHDOC_CONFIG_PATH="../assets/configs/single_prompt_schema_doc_generator_module.yaml"
export GRAPHDOC_METRIC_CONFIG_PATH="../assets/configs/single_prompt_schema_doc_quality_trainer.yaml"

poetry run gunicorn \
    --workers 4 \
    --bind 0.0.0.0:6000 \
    --log-level info \
    "graphdoc_server.app:create_app()" 
#!/bin/bash
poetry run serve \
    --config-path ../assets/configs/single_prompt_schema_doc_generator_module.yaml \
    --metric-config-path ../assets/configs/single_prompt_schema_doc_quality_trainer.yaml \
    --port 6000 
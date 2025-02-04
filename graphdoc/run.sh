python_command() {
    poetry run python
}

shell_command() {
    poetry shell
}

format_command() {
    poetry run black .
}

lint_command() {
    poetry run pyright .
}

test_fire_command() {
    poetry run pytest --fire 
}

test_dry_command() {
    poetry run pytest --dry-fire 
}

test_command() {
    poetry run pytest 
}

install_command() {
    poetry install --without dev
}

dev_command() {
    poetry install --with dev
}

commit_command() {
    format_command
    lint_command
    test_command
}

commit_fire_command() {
    format_command
    lint_command
    test_fire_command
}

commit_dry_command() {
    format_command
    lint_command
    test_dry_command
}

commit_ci_command() {
    format_command
    test_dry_command
}

# training scripts 
train_single_prompt_quality_train_command() {
    poetry run python runners/train/doc_quality_trainer.py --config-path ../assets/configs/single_prompt_schema_doc_quality_trainer.yaml
}

eval_single_prompt_doc_generator_command() {
    poetry run python runners/eval/eval_doc_generator_prompt.py --config-path ../assets/configs/single_prompt_schema_doc_generator_trainer.yaml --metric-config-path ../assets/configs/single_prompt_schema_doc_quality_trainer.yaml
}

# data scripts
local_data_update_command() {
    poetry run python runners/data/local_data_update.py --repo-card False
}

generate_bad_data_command() {
    poetry run python runners/data/bad_data_generator.py --config-path ../assets/configs/single_prompt_schema_bad_doc_generator_trainer.yaml --metric-config-path ../assets/configs/single_prompt_schema_doc_quality_trainer.yaml
}

generate_poor_data_command() {
    poetry run python runners/data/poor_data_generator.py --config-path ../assets/configs/single_prompt_schema_bad_doc_generator_trainer.yaml --metric-config-path ../assets/configs/single_prompt_schema_doc_quality_trainer.yaml
}

show_help() {
    echo "Usage: ./nli [option]"
    echo "Options:"
    echo "  python                 Run Python"
    echo "  shell                  Run a shell in the virtual environment"
    echo "  format                 Format the code"
    echo "  lint                   Lint the code"
    echo "  test                   Run tests"
    echo "  test-fire              Run tests with external calls"
    echo "  test-dry               Run tests without external calls"
    echo "  commit                 Run format, lint and test"
    echo "  commit-fire            Run format, lint and test-fire"
    echo "  commit-dry             Run format, lint and test-dry"
    echo "  commit-ci              Run format and test-dry"
    echo "  install                Install dependencies"
    echo "  dev                    Install development dependencies"

    # training scripts
    echo "  train-single-prompt-quality-train Run single prompt quality training"

    # eval scripts
    echo "  eval-single-prompt-doc-generator Run single prompt doc generator evaluation"

    # data scripts
    echo "  local-data-update       Upload local data to the Hugging Face Hub"
    echo "  generate-bad-data       Generate bad data for evaluation"
    echo "  generate-poor-data      Generate poor data for evaluation"
}

if [ -z "$1" ]; then
    show_help
else
    case "$1" in
        "python") python_command ;;
        "shell") shell_command ;;
        "format") format_command ;;
        "lint") lint_command ;;
        "test") test_command ;;
        "test-fire") test_fire_command ;;
        "test-dry") test_dry_command ;;
        "commit") commit_command ;;
        "commit-fire") commit_fire_command ;;
        "commit-dry") commit_dry_command ;;
        "commit-ci") commit_ci_command ;;
        "install") install_command ;;
        "dev") dev_command ;;

        # training scripts
        "train-single-prompt-quality-train") train_single_prompt_quality_train_command ;;

        # eval scripts
        "eval-single-prompt-doc-generator") eval_single_prompt_doc_generator_command ;;

        # data scripts
        "local-data-update") local_data_update_command ;;
        "generate-bad-data") generate_bad_data_command ;;
        "generate-poor-data") generate_poor_data_command ;;
        *) show_help ;;
    esac
fi
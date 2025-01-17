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
}

commit_ci_command() {
    format_command
    test_dry_command
}

show_help() {
    echo "Usage: ./nli [option]"
    echo "Options:"
    echo "  python                 Run Python"
    echo "  shell                  Run a shell in the virtual environment"
    echo "  format                 Format the code"
    echo "  lint                   Lint the code"
    echo "  test-fire              Run tests with external calls"
    echo "  test-dry               Run tests without external calls"
    echo "  commit-ci              Run format and test-dry"
    echo "  install                Install dependencies"
    echo "  dev                    Install development dependencies"
}

if [ -z "$1" ]; then
    show_help
else
    case "$1" in
        "python") python_command ;;
        "shell") shell_command ;;
        "format") format_command ;;
        "lint") lint_command ;;
        "test-fire") test_fire_command ;;
        "test-dry") test_dry_command ;;
        "commit") commit_command ;;
        "commit-ci") commit_ci_command ;;
        "install") install_command ;;
        "dev") dev_command ;;
        *) show_help ;;
    esac
fi
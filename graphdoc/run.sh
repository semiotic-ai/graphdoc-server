python_command() {
    poetry run python
}

shell_command() {
    poetry shell
}

format_command() {
    poetry run black .
}

# lint_command() {
#     poetry run pyright .
# }

test_fire_command() {
    poetry run pytest --fire
}

test_dry_command() {
    poetry run pytest --dry-fire
}

show_help() {
    echo "Usage: ./nli [option]"
    echo "Options:"
    echo "  python                 Run Python"
    echo "  shell                  Run a shell in the virtual environment"
    echo "  format                 Format the code"
    # echo "  lint                   Lint the code"
    echo "  test-fire              Run tests with external calls"
    echo "  test-dry               Run tests without external calls"
}

if [ -z "$1" ]; then
    show_help
else
    case "$1" in
        "python") python_command ;;
        "shell") shell_command ;;
        "format") format_command ;;
        # "lint") lint_command ;;
        "test-fire") test_fire_command ;;
        "test-dry") test_dry_command ;;
        *) show_help ;;
    esac
fi
# system packages
from pathlib import Path

# internal packages

# external packages


def check_directory_path(directory_path: str) -> None:
    _directory_path = Path(directory_path).resolve()
    if not _directory_path.is_dir():
        raise ValueError(
            f"The provided path does not resolve to a valid directory: {directory_path}"
        )


def check_file_path(file_path: str) -> None:
    _file_path = Path(file_path).resolve()
    if not _file_path.is_file():
        raise ValueError(
            f"The provided path does not resolve to a valid file: {file_path}"
        )

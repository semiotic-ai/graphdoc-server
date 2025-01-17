# system packages 
from pathlib import Path

# internal packages 

# external packages 

def check_directory_path(directory_path: str) -> None:
    directory_path = Path(directory_path).resolve()
    if not directory_path.is_dir(): 
        raise ValueError(
            f"The provided path does not resolve to a valid directory: {directory_path}"
        )

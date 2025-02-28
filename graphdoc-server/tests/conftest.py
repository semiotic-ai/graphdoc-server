# system packages
import os
import time
import signal
import logging
import requests
import subprocess
from pathlib import Path

# internal packages
from graphdoc_server import KeyManager

# external packages
from pytest import fixture

# logging
log = logging.getLogger(__name__)

# global variables
key_path = Path(__file__).parent / "keys" / "api_key_config.json"

####################
# fixtures         #
####################


@fixture(scope="session")
def server():
    """Start the server for all integration tests across multiple files."""
    server_process = subprocess.Popen(
        ["./run.sh", "dev"],
        preexec_fn=os.setsid,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    max_attempts = 30
    for attempt in range(max_attempts):
        try:
            response = requests.get("http://localhost:6000/health", timeout=1)
            if response.status_code == 200:
                break
        except requests.exceptions.RequestException:
            pass
        time.sleep(1)
    else:
        os.killpg(os.getpgid(server_process.pid), signal.SIGTERM)
        log.warning(f"Server failed to start after {max_attempts} seconds")
        log.warning(
            f"You may want to check to make sure that the mlflow server is running"
        )
        raise Exception(f"Server failed to start after {max_attempts} seconds")

    yield server_process

    os.killpg(os.getpgid(server_process.pid), signal.SIGTERM)


@fixture
def key_manager(server) -> KeyManager:
    """Returns an instance of the KeyManager class."""
    key_manager = KeyManager.get_instance(key_path)
    return key_manager


@fixture
def admin(server, key_manager):
    """Returns the admin key for the server. We import the server fixture so that the admin key is set up before this fixture is used."""
    return key_manager.get_admin_key()

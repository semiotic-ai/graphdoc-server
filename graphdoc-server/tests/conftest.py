# system packages 
import os
import time
import signal
import logging
import requests
import subprocess

# internal packages 

# external packages 
import pytest

# logging 
log = logging.getLogger(__name__)

####################
# fixtures         #
####################


@pytest.fixture(scope="session")
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
        log.warning(f"You may want to check to make sure that the mlflow server is running")
        raise Exception(f"Server failed to start after {max_attempts} seconds")

    yield server_process

    os.killpg(os.getpgid(server_process.pid), signal.SIGTERM)

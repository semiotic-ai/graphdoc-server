import requests
import pytest
import time
import subprocess
import signal
import os
from pathlib import Path


@pytest.fixture(scope="module")
def server():
    """Start the server for integration tests."""
    # Start the server
    server_process = subprocess.Popen(
        ["./run_prod.sh"], preexec_fn=os.setsid  # Creates a new process group
    )

    # Wait for server to start
    time.sleep(2)

    yield server_process

    # Cleanup: Kill the server and all its children
    os.killpg(os.getpgid(server_process.pid), signal.SIGTERM)


def test_server_health(server):
    """Test the health endpoint on the running server."""
    response = requests.get("http://localhost:6000/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


def test_server_inference(server):
    """Test the inference endpoint on the running server."""
    test_schema = "CREATE TABLE users (id INT PRIMARY KEY, name VARCHAR(255));"
    response = requests.post(
        "http://localhost:6000/inference", json={"database_schema": test_schema}
    )
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert data["status"] == "success"

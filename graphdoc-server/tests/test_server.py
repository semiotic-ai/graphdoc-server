import pytest
import json
from unittest.mock import patch


def test_health_check(client):
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = json.loads(response.data)
    assert "status" in data
    assert "model_loaded" in data
    assert data["status"] == "healthy"


def test_model_version(client):
    """Test the model version endpoint."""
    # First ensure model is loaded
    with patch("graphdoc_server.app.module", not None):
        response = client.get("/model/version")
        assert response.status_code == 200
        data = json.loads(response.data)
        assert "module_path" in data
        assert "model_name" in data
        assert "experiment_name" in data


def test_model_version_no_model(client):
    """Test model version endpoint when no model is loaded."""
    with patch("graphdoc_server.app.module", None):
        response = client.get("/model/version")
        assert response.status_code == 503
        data = json.loads(response.data)
        assert "error" in data
        assert data["error"] == "Model not loaded"


def test_inference(client):
    """Test the inference endpoint with valid input."""
    test_schema = "CREATE TABLE users (id INT PRIMARY KEY, name VARCHAR(255));"

    with patch("graphdoc_server.app.module") as mock_module:
        mock_module.forward.return_value = "Test documentation"

        response = client.post("/inference", json={"database_schema": test_schema})

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["status"] == "success"
        assert "prediction" in data
        assert data["prediction"] == "Test documentation"


def test_inference_no_model(client):
    """Test inference endpoint when no model is loaded."""
    with patch("graphdoc_server.app.module", None):
        response = client.post("/inference", json={"database_schema": "test"})
        assert response.status_code == 503
        data = json.loads(response.data)
        assert data["error"] == "Model not loaded"


def test_inference_missing_schema(client):
    """Test inference endpoint with missing schema."""
    with patch("graphdoc_server.app.module", not None):
        response = client.post("/inference", json={})
        assert response.status_code == 400
        data = json.loads(response.data)
        assert data["error"] == "Missing database_schema in request"


def test_inference_invalid_json(client):
    """Test inference endpoint with invalid JSON."""
    with patch("graphdoc_server.app.module", not None):
        response = client.post(
            "/inference", data="invalid json", content_type="application/json"
        )
        assert response.status_code == 400


def test_create_app_missing_env_vars():
    """Test app creation with missing environment variables."""
    with pytest.raises(ValueError) as exc_info:
        with patch.dict("os.environ", clear=True):
            from graphdoc_server.app import create_app

            create_app()
    assert (
        "Environment variables GRAPHDOC_CONFIG_PATH and GRAPHDOC_METRIC_CONFIG_PATH must be set"
        in str(exc_info.value)
    )

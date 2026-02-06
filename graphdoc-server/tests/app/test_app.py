# SPDX-FileCopyrightText: 2025 Semiotic AI, Inc.
#
# SPDX-License-Identifier: Apache-2.0

# system packages
import logging
import requests

# internal packages

# external packages

# logging
log = logging.getLogger(__name__)


class TestApp:
    def test_health_check(self, server):
        """Test the health check endpoint."""
        response = requests.get("http://localhost:8080/health")
        assert response.status_code == 200
        assert response.json() == {"status": "healthy", "model_loaded": True}

    def test_model_version(self, server, admin):
        """Test the model version endpoint."""
        response = requests.get(
            "http://localhost:8080/model/version", headers={"X-API-Key": admin}
        )
        assert response.status_code == 200
        assert response.json() == {"model_name": "base_doc_gen"}

    def test_inference(self, server, key_manager, admin):
        """Test the inference endpoint."""
        test_schema = """
        type UserEntity @entity {
            id: Bytes!
            first_name: String!
            last_name: String!
            email: String!
        }
        """
        response = requests.post(
            "http://localhost:8080/inference",
            headers={"X-API-Key": admin},
            json={"database_schema": test_schema},
        )
        assert response.status_code == 200
        assert response.json()["status"] == "success"
        response = requests.post(
            "http://localhost:8080/inference",
            headers={"X-API-Key": next(iter(key_manager.api_keys))},
            json={"database_schema": test_schema},
        )
        assert response.status_code == 200
        assert response.json()["status"] == "success"

    # def test_create_api_key(self):
    #     pass

    def test_list_api_keys(self, server, admin):
        """Test the list API keys endpoint."""
        response = requests.get(
            "http://localhost:8080/api-keys/list", headers={"X-API-Key": admin}
        )
        assert response.status_code == 200
        assert response.json()["status"] == "success"

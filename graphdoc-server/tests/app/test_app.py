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
        response = requests.get("http://localhost:6000/health")
        assert response.status_code == 200
        assert response.json() == {"status": "healthy", "model_loaded": True}

    def test_model_version(self, server, key_manager, admin):
        """Test the model version endpoint."""
        response = requests.get("http://localhost:6000/model/version", headers={"X-API-Key": admin})
        assert response.status_code == 200
        # assert response.json() == {"model_name": "test_model"}
    
    # def test_inference(self):
    #     pass 

    # def test_create_api_key(self):
    #     pass 

    # def test_list_api_keys(self):
    #     pass 
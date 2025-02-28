# system packages 
import logging
import requests
# internal packages 

# external packages 

# logging 
log = logging.getLogger(__name__)

class TestConftest:
    """Test the conftest file."""

    def test_server(self, server):
        """Test the server fixture. Ensure that we get the health check that we are expecting."""
        response = requests.get("http://localhost:6000/health")
        assert response.status_code == 200
        assert response.json() == {"status": "healthy", "model_loaded": True}

# system packages 
import logging
import requests

# internal packages 
from graphdoc_server import KeyManager

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

    def test_key_manager(self, key_manager):
        """Test that the key manager fixture returns a valid KeyManager instance."""
        assert key_manager is not None
        assert isinstance(key_manager, KeyManager)
        assert key_manager._get_test_key() == "test_key"

    def test_admin(self, admin):
        """Test that the admin fixture returns a valid admin key."""
        assert admin is not None
        assert isinstance(admin, str)
        assert len(admin) == 64

#!/usr/bin/env python3
"""
Test script for GraphDoc Server API endpoints with API key authentication.
This script will execute the API requests and log the responses.
"""

import json
import requests
import os
import sys
from pathlib import Path

# Configuration
API_BASE_URL = "http://127.0.0.1:6000"
CONFIG_FILE_PATH = Path(__file__).parents[1] / "graphdoc_server" / "keys" / "api_key_config.json"

def load_api_keys():
    """Load API keys from configuration file."""
    if not CONFIG_FILE_PATH.exists():
        print(f"Error: API key configuration file not found at {CONFIG_FILE_PATH}")
        sys.exit(1)
        
    try:
        with open(CONFIG_FILE_PATH, 'r') as f:
            config = json.load(f)
            if not config.get("api_keys"):
                print("Warning: No API keys found in configuration file.")
            return config
    except Exception as e:
        print(f"Error loading API keys: {str(e)}")
        sys.exit(1)

def test_inference(api_key):
    """Test the inference endpoint."""
    print("\n=== Testing Inference Endpoint ===")
    url = f"{API_BASE_URL}/inference"
    headers = {
        "Content-Type": "application/json",
        "X-API-Key": api_key
    }
    data = {
        "database_schema": "type User { id: ID!, name: String, email: String }"
    }
    
    try:
        response = requests.post(url, headers=headers, json=data)
        
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        
        if response.status_code != 200:
            print(f"ERROR: Inference request failed with status code {response.status_code}")
    except Exception as e:
        print(f"Error making inference request: {str(e)}")

def test_model_version(api_key):
    """Test the model version endpoint."""
    print("\n=== Testing Model Version Endpoint ===")
    url = f"{API_BASE_URL}/model/version"
    headers = {
        "X-API-Key": api_key
    }
    
    try:
        response = requests.get(url, headers=headers)
        
        print(f"Status Code: {response.status_code}")
        
        try:
            json_data = response.json()
            print(f"Response: {json.dumps(json_data, indent=2)}")
        except json.JSONDecodeError:
            print(f"Error: Response is not valid JSON")
            print(f"Raw response: {response.text}")
            
            if response.status_code == 500:
                print("This appears to be a server-side error. Check the server logs for more details.")
        
        if response.status_code != 200:
            print(f"ERROR: Model version request failed with status code {response.status_code}")
    except Exception as e:
        print(f"Error making model version request: {str(e)}")

def test_generate_api_key(admin_key):
    """Test generating a new API key."""
    print("\n=== Testing Generate API Key Endpoint ===")
    url = f"{API_BASE_URL}/api-keys/generate"
    headers = {
        "X-API-Key": admin_key
    }
    
    try:
        response = requests.post(url, headers=headers)
        
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        
        if response.status_code != 200:
            print(f"ERROR: Generate API key request failed with status code {response.status_code}")
            return None
        else:
            return response.json().get("api_key")
    except Exception as e:
        print(f"Error making generate API key request: {str(e)}")
        return None

def test_list_api_keys(admin_key):
    """Test listing all API keys."""
    print("\n=== Testing List API Keys Endpoint ===")
    url = f"{API_BASE_URL}/api-keys/list"
    headers = {
        "X-API-Key": admin_key
    }
    
    try:
        response = requests.get(url, headers=headers)
        
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        
        if response.status_code != 200:
            print(f"ERROR: List API keys request failed with status code {response.status_code}")
    except Exception as e:
        print(f"Error making list API keys request: {str(e)}")

def main():
    """Main function to run all tests."""
    print("Loading API keys from configuration file...")
    config = load_api_keys()
    
    admin_key = config.get("admin_key")
    if not admin_key:
        print("Error: No admin key found in configuration file.")
        sys.exit(1)
    print(f"Admin key: {admin_key}")
    
    api_keys = config.get("api_keys", [])
    if not api_keys:
        print("No API keys found. Generating a new one with the admin key...")
        new_api_key = test_generate_api_key(admin_key)
        if new_api_key:
            api_key = new_api_key
            print(f"Successfully generated new API key: {api_key}")
        else:
            print("Failed to generate a new API key. Exiting.")
            sys.exit(1)
    else:
        api_key = api_keys[0]
        print(f"Using API key: {api_key}")
    
    test_model_version(api_key)
    test_inference(api_key)
    test_list_api_keys(admin_key)
    
    print("\nAll tests completed.")

if __name__ == "__main__":
    main() 
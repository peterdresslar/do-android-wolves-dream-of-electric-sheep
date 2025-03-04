# init_utils.py
"""
Initialization utilities for the model.
This module handles loading environment variables and other initialization tasks.
"""

import os
from dotenv import load_dotenv

# Flag to track if environment variables have been loaded
_env_loaded = False

def load_environment():
    """
    Load environment variables from .env.local file.
    This function is idempotent - it will only load the environment once.
    """
    global _env_loaded
    if not _env_loaded:
        # Load keys from .env file. See .env.local.example
        # First try relative path from this file
        dotenv_path = os.path.join(os.path.dirname(__file__), "..", ".env.local")
        
        # If file doesn't exist, try from current working directory
        if not os.path.exists(dotenv_path):
            dotenv_path = os.path.join(os.getcwd(), ".env.local")
            
        # If still doesn't exist, try one directory up from current working directory
        if not os.path.exists(dotenv_path):
            dotenv_path = os.path.join(os.getcwd(), "..", ".env.local")
        
        if os.path.exists(dotenv_path):
            load_dotenv(dotenv_path)
            _env_loaded = True
            return True
        else:
            print(f"Warning: Could not find .env.local file. Tried paths: {os.path.join(os.path.dirname(__file__), '..', '.env.local')}, {os.path.join(os.getcwd(), '.env.local')}, {os.path.join(os.getcwd(), '..', '.env.local')}")
            return False

def initialize_model():
    """
    Initialize the model by loading environment variables and performing other setup tasks.
    This should be called once at the start of the application.
    """
    # Load environment variables
    load_environment()
    
    # Import and configure API clients
    import openai
    openai.api_key = os.getenv("OPENAI_API_KEY")
    
    # Add other initialization tasks as needed
    
    return True 
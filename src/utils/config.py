"""
Configuration module for loading environment variables.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Get the project root directory
ROOT_DIR = Path(__file__).parent.parent.parent

# Load environment variables
load_dotenv(ROOT_DIR / '.env')

def get_openai_api_key():
    """
    Get the OpenAI API key from environment variables.
    
    Returns:
        str: The OpenAI API key
    
    Raises:
        ValueError: If OPENAI_API_KEY is not set in environment
    """
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables. "
                        "Please create a .env file with your API key.")
    return api_key 
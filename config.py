"""
Configuration file - Manage API Keys and other settings
"""

import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# NVIDIA API settings
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "")
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
NVIDIA_MODEL = "nvidia/llama-3.3-nemotron-super-49b-v1.5"

# Agent settings
AGENT_CONFIG = {
    "technical_expert": {
        "temperature": 0.6,
        "max_tokens": 4096,
        "description": "Technical Expert - Rational analysis, data-driven"
    },
    "hater": {
        "temperature": 0.9,
        "max_tokens": 4096,
        "description": "Hater - Emotional, bandwagon, negative"
    }
}

# Budget settings (virtual budget for each Agent)
DEFAULT_BUDGET = 1000

def validate_config():
    """Validate configuration"""
    if not NVIDIA_API_KEY:
        print("⚠️  Warning: NVIDIA_API_KEY not set")
        print("Please create a .env file in the project root directory and add:")
        print("NVIDIA_API_KEY=your_api_key_here")
        return False
    return True

if __name__ == "__main__":
    if validate_config():
        print("✅ Configuration validation successful!")
        print(f"API Base URL: {NVIDIA_BASE_URL}")
        print(f"Model: {NVIDIA_MODEL}")
    else:
        print("❌ Configuration validation failed, please check .env file")

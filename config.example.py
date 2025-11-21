"""
Configuration file for LogiCue Runner

SETUP INSTRUCTIONS:
1. Copy this file to 'config.py' in the same directory
2. Add your API keys below
3. Adjust paths as needed
4. Add config.py to .gitignore (it should never be committed!)
"""

# API Keys
# Get your keys from:
# - OpenAI: https://platform.openai.com/api-keys
# - Anthropic: https://console.anthropic.com/settings/keys
# - DeepSeek: https://platform.deepseek.com/api_keys
# - Google AI Studio: https://aistudio.google.com/app/apikey

API_KEYS = {
    'openai': 'your-openai-api-key-here',
    'anthropic': 'your-anthropic-api-key-here',
    'deepseek': 'your-deepseek-api-key-here',
    'gemini': 'your-gemini-api-key-here'
}

# Paths configuration
PATHS = {
    'repo_path': '.',  # Current directory, or specify full path
    'ollama_url': 'http://localhost:11434'  # Change if Ollama runs elsewhere
}

import os
from dotenv import load_dotenv

# Load environment variables from a .env file if it exists.
# This allows for easy local development without setting system-wide env vars.
load_dotenv()

# Fetch the OpenAI API key from the environment.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Provide a clear warning if the API key is not configured,
# as this is essential for the ASR module to function.
if not OPENAI_API_KEY:
    print("WARNING: The 'OPENAI_API_KEY' environment variable is not set. The ASR module will fail.")

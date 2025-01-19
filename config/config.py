import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("API_KEY")
BASE_PROMPT = os.getenv("BASE")
TEMPERATURE = 0.5
MAX_TOKENS = 200

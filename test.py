from dotenv import load_dotenv
import os

load_dotenv()

key = os.getenv("OPENWEATHER_API_KEY")

print("Key loaded:", key)
print("Key length:", len(key) if key else 0)
import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
key = os.getenv("GEMINI_API_KEY")
print("Using key:", key[:10] + "..." if key else "No key found!")

genai.configure(api_key=key)

try:
    model = genai.GenerativeModel('gemini-2.5-flash')
    response = model.generate_content("Say hello world")
    print("Success! Response:", response.text)
except Exception as e:
    print("Failed:", str(e))
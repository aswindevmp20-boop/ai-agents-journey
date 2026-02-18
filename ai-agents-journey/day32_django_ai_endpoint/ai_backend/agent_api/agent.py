from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def run_agent(query):
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role":"system", "content": "You are a helpful AI assistant."},
            {"role":"user", "content": query }
        ]
    )
    return response.choices[0].message.content
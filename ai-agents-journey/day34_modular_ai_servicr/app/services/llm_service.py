from groq import Groq
from app.config import settings

client = Groq(api_key = settings.GROQ_API_KEY)

def generate_response(messages):
    response = client.chat.completions.create(
        model = settings.MODEL_NAME,
        messages = messages
    )
    return response.choices[0].message.content
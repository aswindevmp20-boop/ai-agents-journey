from groq import Groq
from app.config import settings
import logging

logger = logging.getLogger("ai_service")

client = Groq(api_key = settings.GROQ_API_KEY)

def generate_response(messages):
    try:
        response = client.chat.completions.create(
            model = settings.MODEL_NAME,
            messages = messages
        )
        logging.info("LLM call successful.")
        return response.choices[0].message.content
    
    except Exception as e:
        logger.error(f"LLM call failed: {str(e)}")
        raise Exception("AI service temporarily unavailable.")
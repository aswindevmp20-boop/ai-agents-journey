import os
from groq import Groq
from dotenv import load_dotenv
from shared.circuit_breaker import CiruitBreaker
from shared.model_router import select_model
load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

breaker = CiruitBreaker(failure_threshold=3, recovery_time=20)

def generate_response(messages):

    query = messages[-1]["content"]
    model_name = select_model(query)

    if not breaker.call_allowed():
        raise Exception("Circuit breaker open. LLM temporarily unavailable.")

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages
        )

        breaker.record_success()
        return response.choices[0].message.content

    except Exception as e:
        breaker.record_failure()
        raise e
from app.services.llm_service import generate_response

def run_agent(query):
    messages = [
        {"role":"system","content":"You are a helpful AI assistant."},
        {"role":"user","content": query}
    ]

    return generate_response(messages)
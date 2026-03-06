def select_model(query: str):
    query_lower = query.lower()

    #Simple rule-based routing(can later be LLM-based)
    if "summarize" in query_lower:
        return "llama-3.1-8b-instant"
    
    if "urgent" in query_lower or "analyze" in query_lower:
        return "llama-3.3-70b-versatile"
    
    if len(query) < 50:
        return "llama-3.1-8b-instant"
    
    #Default
    return "llama-3.3-70b-versatile"
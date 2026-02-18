from fastapi import FastAPI
from app.api.routes import router

app = FastAPI(title="AI Agent Service")

app.include_router(router)



# Expected Output:- 

# uvicorn app.main:app --reload

# {
#   "query": "What do you meant by LLM ?",
#   "response": "LLM stands for Large Language Model. It refers to a type of artificial intelligence (AI) model that is designed to process and understand human language at a large scale. LLMs are trained on vast amounts of text data, which enables them to learn patterns, relationships, and structures of language.\n\nLarge Language Models are typically characterized by their ability to:\n\n1. Understand natural language input: LLMs can comprehend and interpret human language, including nuances, context, and subtleties.\n2. Generate human-like text: LLMs can produce coherent, readable, and sometimes even creative text that mimics human writing.\n3. Learn from large datasets: LLMs are trained on massive datasets, which allows them to learn from a vast range of texts, including books, articles, conversations, and more.\n\nSome common applications of LLMs include:\n\n1. Language translation\n2. Text summarization\n3. Sentiment analysis\n4. Chatbots and conversational AI\n5. Content generation (e.g., writing articles, creating dialogues)\n\nI am an example of an LLM, and I'm here to assist you with any questions or tasks you may have!"
# }
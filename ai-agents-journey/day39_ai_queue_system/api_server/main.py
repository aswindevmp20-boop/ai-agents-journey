from fastapi import FastAPI
from pydantic import BaseModel
import redis
import json
import uuid

app=FastAPI()

r = redis.Redis(host="redis", port=6379, decode_responses=True)

class QueryRequest(BaseModel):
    query: str

@app.post("/submit/")
def submit_job(request: QueryRequest):
    job_id = str(uuid.uuid4())

    payload = {
        "job_id": job_id,
        "query": request.query
    }

    r.rpush("task_queue", json.dumps(payload))

    return{
        "job_id": job_id,
        "status": "queued"
    }

@app.get("/result/{job_id}")
def get_result(job_id: str):
    result = r.get(f"result:{job_id}")

    if not result:
        return {"status": "processing"}

    return {
        "status":"completed",
        "result": result
    }



# Expected_output:

# {
#   "status": "completed",
#   "result": "LLM stands for Large Language Model. It refers to a type of artificial intelligence (AI) designed to process and understand human language at a large scale. LLMs are trained on vast amounts of text data, which enables them to learn patterns, relationships, and structures of language, allowing them to generate human-like text, answer questions, and even converse with humans.\n\nLarge Language Models are typically based on deep learning architectures, such as transformer models, and are trained using massive datasets of text from various sources, including books, articles, research papers, and websites. This training process allows LLMs to develop a broad understanding of language, including grammar, syntax, semantics, and pragmatics.\n\nSome key features of LLMs include:\n\n1. **Language understanding**: LLMs can comprehend and interpret human language, including nuances and context.\n2. **Text generation**: LLMs can generate coherent and natural-sounding text, such as articles, stories, or even entire books.\n3. **Conversational capabilities**: LLMs can engage in conversation, answering questions and responding to prompts in a human-like way.\n4. **Knowledge retrieval**: LLMs can retrieve and provide information on a wide range of topics, from science and history to entertainment and culture.\n\nLLMs have many potential applications, including:\n\n1. **Virtual assistants**: LLMs can power virtual assistants, such as chatbots and voice assistants, to provide helpful and informative responses to user queries.\n2. **Language translation**: LLMs can be used to improve machine translation systems, enabling more accurate and fluent translation of text from one language to another.\n3. **Content creation**: LLMs can assist with content creation, such as generating articles, social media posts, or even entire books.\n4. **Research and education**: LLMs can aid researchers and students by providing access to vast amounts of information and helping with tasks such as summarization, citation, and referencing.\n\nExamples of LLMs include transformer-based models like BERT, RoBERTa, and XLNet, as well as other models like transformer-XL and Longformer. These models have achieved state-of-the-art results in various natural language processing (NLP) tasks and have many potential applications in areas like customer service, content creation, and education."
# }
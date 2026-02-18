from groq import Groq
from sentence_transformers import SentenceTransformer
import faiss
import os,json
from dotenv import load_dotenv
import numpy as np

load_dotenv()

client = Groq(api_key = os.getenv("GROQ_API_KEY"))

DOCS_PATH = "./docs"
CHUNK_SIZE = 120
TOP_K = 3

embedder = SentenceTransformer("all-MiniLm-L6-v2")

def chunk_text(text):
    words = text.split()
    return[
        " ".join(words[i:i + CHUNK_SIZE])
        for i in range(0,len(words), CHUNK_SIZE)
    ]

def load_documents():
    chunks = []
    for file in os.listdir(DOCS_PATH):
        if file.endswith(".txt"):
            with open(os.path.join(DOCS_PATH, file), "r", encoding="utf-8") as f:
                text = f.read()
            for chunk in chunk_text(text):
                chunks.append({
                    "file": file,
                    "content": chunk
                })
    return chunks

DOCUMENT_CHUNKS = load_documents()

embeddings = embedder.encode(
    [c["content"] for c in DOCUMENT_CHUNKS],
    convert_to_numpy = True
)

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

def retrieve_chunks(query, top_k=TOP_K):
    query_embedding = embedder.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, top_k)

    results = []
    for idx in indices[0]:
        results.append(DOCUMENT_CHUNKS[idx])
    return results

TOOL_FUNCTIONS = {
    "retrieve_chunks": retrieve_chunks
}

tools = [
    {
        "type":"function",
        "function": {
            "name": "retrieve_chunks",
            "description": "Retreive relevant document chunks using FAISS vector search.",
            "parameters": {
                "type":"object",
                "properties":{
                    "query":{"type":"string"},
                    "top_k":{"type": "integer"}
                },
                "required": ["query"]
            }
        }
    }
]

def call_model(messages):
    return client.chat.completions.create(
        model = "llama-3.3-70b-versatile",
        messages = messages,
        tools = tools
    )

def run_agent(query):
    messages = [{"role":"user", "content": query}]

    while True:
        response = call_model(messages)
        msg = response.choices[0].message

        if msg.content:
            print("\n Final Answer: ", msg.content)
            break

        if msg.tool_calls:
            for call in msg.tool_calls:
                fn_name = call.function.name
                args = json.loads(call.function.arguments)
                results = retrieve_chunks(**args)

                context = "\n\n".join(
                    f"[{r['file']}]\n{r['content']}"
                    for r in results
                )

                messages.append({"role":"assistant","tool_calls": msg.tool_calls})
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": context
                })

query = "How Marine ecosystems are incredibly diverse ?."
run_agent(query)

#  Expected Output

#  Final Answer:  Marine ecosystems are incredibly diverse, ranging from shallow coral reefs to deep-sea trenches. They are home to millions of species, many of which are still undiscovered,
#  and play a crucial role in regulating global climate by absorbing large amounts of heat and carbon dioxide. However, ocean health is under serious threat due to overfishing,
#  plastic pollution, chemical waste, and rising sea temperatures caused by climate change. It is essential to take steps to protect ocean ecosystems,
#  such as adopting sustainable fishing practices, reducing plastic use, and taking climate action, to preserve the long-term health of the oceans and life on Earth.
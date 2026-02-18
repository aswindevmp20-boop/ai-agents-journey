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
    word = text.split()
    return[
        " ".join(word[i:i + CHUNK_SIZE])
        for i in range(0, len(word), CHUNK_SIZE)
    ]

def load_documents():
    chunks = []
    for file in os.listdir(DOCS_PATH):
        if file.endswith(".txt"):
            with open(os.path.join(DOCS_PATH, file), "r", encoding = "utf-8") as f:
                text = f.read()
            for chunk in chunk_text(text):
                chunks.append({
                    "file":file,
                    "content": chunk
                })
    return chunks

DOCUMENT_CHUNKS = load_documents() 

def keyword_score(text, query):
    score = 0
    for word in query.lower().split():
        score +=text.lower().count(word)
    return score

embeddings = embedder.encode(
    [c["content"] for c in DOCUMENT_CHUNKS],
    convert_to_numpy=True
)

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

#Hybrid Retrieval

def retrieve_chunks(query, top_k=TOP_K):
    # vector search
    query_vec =  embedder.encode([query], convert_to_numpy = True)
    distances, indices = index.search(query_vec, top_k * 2)

    candidates = []
    for idx, dist in zip(indices[0], distances[0]):
        vec_score = 1 / (1 + dist)
        key_score = keyword_score(DOCUMENT_CHUNKS[idx]["content"], query)
        final_score = (0.7 * vec_score) + (0.3 * key_score)

        candidates.append((final_score, DOCUMENT_CHUNKS[idx]))

    candidates.sort(key = lambda x:x[0], reverse=True)
    return [chunk for score, chunk in candidates[:top_k]]

tools = [
    {
        "type": "function",
        "function":{
            "name" : "retrieve_chunks",
            "description": " hybrid retrieval using keyword + vector search.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type":"string"},
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
    messages = [{"role":"user", "content": query }]

    while True:
        response = call_model(messages)
        msg = response.choices[0].message

        if msg.content:
            print("\nFinal Answer:", msg.content)
            break

        if msg.tool_calls:
            for call in msg.tool_calls:
                args = json.loads(call.function.arguments)
                results = retrieve_chunks(**args)

                context = "\n\n".join(
                    f"[{r['file']}]\n{r['content']}"
                    for r in results
                )

                messages.append({"role": "assistant", "tool_calls": msg.tool_calls})
                messages.append({
                    "role":"tool",
                    "tool_call_id":call.id,
                    "content": context
                })

query = "What are the major threats to ocean health?"
run_agent(query)


# Expected Output:

# Final Answer: The major threats to ocean health include:

# 1. Overfishing: This has significantly reduced fish populations in many regions.
# 2. Plastic pollution: Plastic waste contaminates marine habitats and harms wildlife.
# 3. Chemical waste: Chemical pollutants contaminate marine ecosystems and harm wildlife.
# 4. Rising sea temperatures: Caused by climate change, leading to coral bleaching and habitat loss.
# 5. Climate change: Affects ocean currents, weather patterns, and the overall health of marine ecosystems.

# It is essential to address these threats through sustainable practices, such as reducing plastic use, 
# implementing sustainable fishing practices, and taking climate action, to protect the long-term health of the oceans and life on Earth.
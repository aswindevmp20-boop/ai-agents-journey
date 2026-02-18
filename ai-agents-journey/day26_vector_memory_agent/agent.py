import faiss
from groq import Groq
from sentence_transformers import SentenceTransformer
import os, json
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

embedder = SentenceTransformer("all-MiniLM-L6-v2")
EMBEDDING_DIM = 384

INDEX_FILE = "vector_store.index"
MEMORY_FILE = "memory.json"

# Load or create memory

memory_data = []
if os.path.exists(MEMORY_FILE):
    try:
        with open(MEMORY_FILE, "r") as f:
            memory_data = json.load(f)
    except json.JSONDecodeError:
        memory_data = []

# Load or create FAISS index

if os.path.exists(INDEX_FILE):
    try:
        index = faiss.read_index(INDEX_FILE)
    except RuntimeError:
        index = faiss.IndexFlatL2(EMBEDDING_DIM)
else:
    index = faiss.IndexFlatL2(EMBEDDING_DIM)

# Save memory

def save_memory(text):
    embedding = embedder.encode([text]).astype("float32")
    index.add(embedding)

    memory_data.append(text)

    faiss.write_index(index, INDEX_FILE)
    with open(MEMORY_FILE, "w") as f:
        json.dump(memory_data, f, indent=2)

# Retrieve memory

def retrieve_memory(query: str, top_k: int = 3):
    if len(memory_data) == 0 or index.ntotal == 0:
        return []

    query_vec = embedder.encode([query]).astype("float32")
    distances, indices = index.search(query_vec, top_k)

    results = []
    for idx in indices[0]:
        if idx < len(memory_data):
            results.append(memory_data[idx])

    return results

def run_agent(query: str):
    relevant_memory = retrieve_memory(query)

    messages = [
        {
            "role": "system",
            "content": "You are an AI assistant with vector memory."
        }
    ]

    if relevant_memory:
        memory_text = "\n".join(relevant_memory)
        messages.append({
            "role": "assistant",
            "content": f"Relevant past memory:\n{memory_text}"
        })

    messages.append({
        "role": "user",
        "content": query
    })

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages
    )

    answer = response.choices[0].message.content
    print("\nFinal Answer:\n", answer)

    save_memory(f"User: {query} | Assistant: {answer}")


while True:
    q = input("\nAsk something (or exit): ")
    if q.lower() == "exit":
        break
    run_agent(q)


# Expected Output:-

# Ask something (or exit): My name is Aswin

# Final Answer:
#  Hello Aswin, it's nice to meet you. Is there something I can help you with or would you like to chat?      

# Ask something (or exit): my fathers name is sajith

# Final Answer:
#  I've stored that information. So, to recap:

# * Your name is Aswin
# * Your father's name is Sajith

# Is there anything else you'd like to share or talk about?

# Ask something (or exit): exit



# Ask something (or exit): Wht's my father's name ?

# Final Answer:
#  I recall that your father's name is Sajith.

# Ask something (or exit): and my name is ?

# Final Answer:
#  I recall that your name is Aswin.

# Ask something (or exit): good great job

# Final Answer:
#  I'm glad I could recall the information correctly. Vector memory allows me to store and retrieve associations between pieces of information, so I can build a knowledge graph of our conversation. If you'd like to add more information or ask me to recall something, feel 
# free to do so!

# Current memory:
# * Your name is Aswin
# * Your father's name is Sajith

# What's next?

# Ask something (or exit):exit

from groq import Groq
from sentence_transformers import SentenceTransformer
import faiss
import os,json
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key = os.getenv("GROQ_API_KEY"))

DOCS_PATH = "./docs"
CHUNK_SIZE = 120
TOP_K = 3

embedder = SentenceTransformer("all-MiniLm-L6-v2")

#Memory
conversation_history = []

def chunk_text(text):
    word = text.split()
    return [
        " ".join(word[i:i + CHUNK_SIZE])
        for i in range(0, len(word), CHUNK_SIZE)
    ]

def load_documents():
    chunks = []
    for file in os.listdir(DOCS_PATH):
        if file.endswith(".txt"):
            with open(os.path.join(DOCS_PATH,file), "r", encoding="utf-8") as f:
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
    convert_to_numpy=True
)

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

def retrieve_chunks(query, top_k=TOP_K):
    query_vec = embedder.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_vec, top_k)

    results = []
    for idx in indices[0]:
        results.append(DOCUMENT_CHUNKS[idx])
    return results

TOOLS_FUNCTION = {
    "retrieve_chunks": retrieve_chunks
}

tools = [
    {
        "type": "function",
        "function":{
            "name": "retrieve_chunks",
            "description": "retrieve relevant document chunks.",
            "parameters":{
                "type":"object",
                "properties":{
                    "query":{"type":"string"},
                    "top_k":{"type":"integer"}
                },
                "required":["query"]
            }
        }
    }
]

SYSTEM = {
    "role": "system",
    "content": (
        "You are a conversational RAG assistant.\n"
        "Use conversation history and retrieved context to answer.\n"
        "Always retrieve document chunks before answering.\n"
        "Answer only from retrieved context."
    )
}

def call_model(messages):
    return client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages =messages,
        tools=tools
    )

def run_agent(query):
    global conversation_history

    messages = [SYSTEM]

    for msg in conversation_history:
        messages.append(msg)

    messages.append({"role": "user", "content": query})

    while True:
        response = call_model(messages)
        msg = response.choices[0].message

        if msg.content:
            print("\nFinal Answer: ", msg.content)

            #save memory
            conversation_history.append({"role": "user", "content": query})
            conversation_history.append({"role":"assistant", "content": msg.content})
            break

        if msg.tool_calls:
            for call in msg.tool_calls:
                args = json.loads(call.function.arguments)
                results = retrieve_chunks(**args)

                context = "\n\n".join(
                    f"[{r['file']}]\n{r['content']}"
                    for r in results
                )

                messages.append({"role":"assistant", "tool_calls": msg.tool_calls})
                messages.append({
                    "role":"tool",
                    "tool_call_id":call.id,
                    "content": context
                })


while True:
    user_query = input("\nAsk a question (or type 'exit'): ")
    if user_query.lower() == "exit":
        break
    run_agent(user_query)


# Expected Output:

# Ask a question (or type 'exit'): Opinion about cultural exchange

# Final Answer:  Based on the retrieved context, cultural exchange is viewed positively as it allows people to experience new cultures, traditions, and perspectives, helping individuals develop empathy and a broader understanding of the world. 
# Traveling and exploring different countries can be both educational and personally transformative, and cultural exchange plays a key role in global understanding. However, 
# it's also important to consider responsible travel practices to reduce environmental impact and preserve destinations for future generations.
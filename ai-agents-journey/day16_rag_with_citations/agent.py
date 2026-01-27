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

def chunk_text(text):
    words = text.split()
    return [
        " ".join(words[i:i + CHUNK_SIZE])
        for i in range(0,len(words), CHUNK_SIZE)
    ]

def load_documents():
    chunks =[]
    for file in os.listdir(DOCS_PATH):
        if file.endswith(".txt"):
            with open(os.path.join(DOCS_PATH,file),"r", encoding="utf-8") as f:
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
    query_vec = embedder.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_vec, top_k)

    results = []
    for idx in indices[0]:
        results.append(DOCUMENT_CHUNKS[idx])
    return results

tools = [
    {
        "type":"function",
        "function":{
            "name":"retrieve_chunks",
            "description":"Retrieve relevant chunks from documents with source info.",
            "parameters":{
                "type":"object",
                "properties":{
                    "query":{"type":"string"},
                    "top_k":{"type":"integer"},
                },
                "required":["query"]
            }
        }
    }
]

SYSTEM = {
    "role": "system",
    "content": (
        "You are a RAG assistant with citations.\n"
        "Use retrieved context only.\n"
        "Always cite sources as [filename] in your answer."
    )
}

def call_model(messages):
    return client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        tools=tools
    )

def run_agent(query):
    messages = [SYSTEM, {"role":"user", "content": query}]

    while True:
        response = call_model(messages)
        msg = response.choices[0].message

        if msg.content:
            print("\nFinal Answer with Sources:\n", msg.content)
            break

        if msg.tool_calls:
            for call in msg.tool_calls:
                args = json.loads(call.function.arguments)
                results = retrieve_chunks(**args)

                context = "\n\n".join(
                    f"[{r['file']}]\n{r['content']}"
                    for r in results
                )

                messages.append({"role":"assistant", "tool_calls":msg.tool_calls})
                messages.append({
                    "role":"tool",
                    "tool_call_id": call.id,
                    "content": context
                })


query = "What are the major threats to ocean health?"
run_agent(query)


# Expected Output:

# Final Answer with Sources:
#  The major threats to ocean health include overfishing, plastic pollution, chemical waste, and rising sea temperatures caused by climate change, which leads to coral bleaching and habitat loss [doc1.txt, doc2.txt].
#  To protect ocean ecosystems, global cooperation is required, and steps such as sustainable fishing practices, reduced plastic use, and climate action are critical [doc1.txt, doc2.txt].
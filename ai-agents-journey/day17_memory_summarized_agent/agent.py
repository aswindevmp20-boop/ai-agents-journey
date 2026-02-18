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

memory_summary = ""

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
            with open(os.path.join(DOCS_PATH,file), "r", encoding="utf-8") as f:
                text = f.read()
            for chunk in chunk_text(text):
                chunks.append({
                    "file": file,
                    "content": chunk
                })
    return chunks

DOCUMENT_CHUNKS = load_documents()

embeddings= embedder.encode([c["content"] for c in DOCUMENT_CHUNKS], convert_to_numpy=True)

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

def retrieve_chunks(query, top_k=TOP_K):
    q_vec = embedder.encode([query], convert_to_numpy=True)
    distances, indices = index.search(q_vec, top_k)
    return [DOCUMENT_CHUNKS[i] for i in indices[0]]

# Memory Summarizer

def update_memory(old_memory, user_msg, assistant_msg):
    prompt = f"""

    Current Memory:
    {old_memory}

    New interaction:
    User: {user_msg}
    Assistant: {assistant_msg}

    Update the memory summary briefly:
    """

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages = [{"role":"user", "content": prompt}]
    )
    return response.choices[0].message.content


tools = [
    {
        "type":"function",
        "function":{
            "name":"retrieve_chunks",
            "description":"Retrieve relevant document chunks.",
            "parameters":{
                "type":"object",
                "properties": {
                    "query":{"type":"string"},
                    "top_k":{"type":"integer"}
                },
                "required": ["query"]
            }
        }
    }
]

SYSTEM = {
    "role": "system",
    "content": (
        "You are a RAG assistant with long-term memory.\n"
        "Use memory summary + retrieved context to answer.\n"
        "Only answer from retrieved context."
    )
}

def call_model(messages):
    return client.chat.completions.create(
        model = "llama-3.3-70b-versatile",
        messages = messages,
        tools = tools
    )

def run_agent(query):
    global memory_summary
    
    messages = [SYSTEM]
    if memory_summary:
        messages.append({"role":"assistant", "content": f"Memory summary: {memory_summary}"})

    messages.append({"role":"user", "content":query})

    while True:
        response = call_model(messages)
        msg = response.choices[0].message

        if msg.content:
            print("\n Final Answer: ", msg.content)
            
            memory_summary = update_memory(memory_summary,query, msg.content)
            print("\n Updated Memory Summary:", memory_summary)
            break

        if msg.tool_calls:
            for call in msg.tool_calls:
                args = json.loads(call.function.arguments)
                results = retrieve_chunks(**args)

                context = "\n\n".join(
                    f"[{r['file']}]\n{r['content']}" for r in results
                )

                messages.append({"role":"assistant", "tool_calls": msg.tool_calls})
                messages.append({
                    "role":"tool",
                    "tool_call_id": call.id,
                    "content": context
                })

while True:
    q= input("\n Ask a question (or type 'exit'): ")
    if q.lower() =="exit":
        break
    run_agent(q)


# Expected Output:-

#  Ask a question (or type 'exit'): What are threats to ocean health?

#  Final Answer:  Threats to ocean health include climate change, which leads to coral bleaching and habitat loss, overfishing, plastic pollution, and chemical waste contaminating marine habitats and harming wildlife. 
#  Rising sea temperatures caused by climate change also pose a significant threat to ocean health. Additionally, human activities such as sustainable fishing practices, reduced plastic use, and climate action are critical steps to protecting ocean ecosystems.

#  Updated Memory Summary: Current Memory: Threats to ocean health include climate change, overfishing, plastic pollution, and chemical waste, while human activities like sustainable fishing and reduced plastic use can help protect ocean ecosystems.

#  Ask a question (or type 'exit'): Why are they dangerous?

#  Final Answer:  They are dangerous because overfishing has significantly reduced fish populations in many regions, plastic pollution and chemical waste contaminate marine habitats and harm wildlife, and rising sea temperatures caused by climate change lead to coral bleaching and habitat loss.

#  Updated Memory Summary: Current Memory: Threats to ocean health, including climate change, overfishing, plastic pollution, and chemical waste, are dangerous as they harm marine habitats, reduce fish populations, and lead to coral bleaching and habitat loss.

#  Ask a question (or type 'exit'): exit
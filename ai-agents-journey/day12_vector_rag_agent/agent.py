from groq import Groq
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os,json
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key = os.getenv("GROQ_API_KEY"))

DOCS_PATH = "./docs"
CHUNK_SIZE = 120

embedder = SentenceTransformer("all-MiniLm-L6-v2")

def chunk_text(text):
    words = text.split()
    return [
        " ".join(words[i:i + CHUNK_SIZE])
        for i in range(0, len(words), CHUNK_SIZE)
    ]

def load_and_embed():
    chunks = []
    embeddings = []

    for file in os.listdir(DOCS_PATH):
        if file.endswith(".txt"):
            with open(os.path.join(DOCS_PATH,file), "r", encoding="utf-8") as f:
                text = f.read()

            for chunk in chunk_text(text):
                chunks.append({
                    "file": file,
                    "content": chunk
                })
                embeddings.append(embedder.encode(chunk))

    return chunks, embeddings

DOCUMENT_CHUNKS, CHUNK_EMBEDDINGS = load_and_embed()

def retrieve_chunks(query, top_k=3):
    query_embedding = embedder.encode(query)
    scores = cosine_similarity([query_embedding], CHUNK_EMBEDDINGS)[0]

    ranked = sorted(
        zip(scores, DOCUMENT_CHUNKS),
        key = lambda x: x[0],
        reverse = True
    )

    return [chunk for score, chunk in ranked[:top_k]]


TOOLS_FUNCTION = {
    "retrieve_chunks": retrieve_chunks
}

tools = [
    {
        "type": "function",
        "function": {
            "name": "retrieve_chunks",
            "description": "Retrieve semantically relevant document chunks using embeddings.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "top_k": {"type": "integer"}
                },
                "required": ["query"]
            }
        }
    }
]

def call_models(messages):
    return client.chat.completions.create(
        model = "llama-3.3-70b-versatile",
        messages=messages,
        tools=tools
    )

def run_agent(query):
    messages = [{"role":"user", "content": query}]

    while True:
        response = call_models(messages)
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

                messages.append({"role": "assistant", "tool_calls":msg.tool_calls})
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": context
                })


query = "Why are oceans important for climate regulation?"
run_agent(query)


#  Expected Output :

#  Final Answer:  The oceans play a crucial role in regulating the Earth's climate.
#  They absorb large amounts of heat and carbon dioxide, which helps to distribute heat across different regions and influence weather patterns worldwide. 
#  The oceans are also home to diverse marine ecosystems, including coral reefs, deep-sea trenches, and complex food chains that maintain ecological balance.
#  However, ocean health is under threat due to overfishing, plastic pollution, chemical waste, and rising sea temperatures caused by climate change. 
#  It is essential to take steps to protect the oceans, such as adopting sustainable fishing practices, reducing plastic use, and taking climate action, to preserve the long-term health of the oceans and life on Earth.
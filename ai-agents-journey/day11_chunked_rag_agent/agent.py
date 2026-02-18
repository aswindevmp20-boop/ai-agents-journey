# Instead of this:
# Read entire document → hope the answer is inside

# We do this:
# Split documents into small chunks
# Search only relevant chunks
# Pass only those chunks to the LLM
# Generate an answer grounded in evidence

import os,json
from groq import Groq
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

client = Groq(api_key = os.getenv("GROQ_API_KEY"))

DOCS_PATH = "./docs"

CHUNK_SIZE = 120

# chunk logic
def chunk_text(text, size=CHUNK_SIZE):
    words = text.split()
    return [
        " ".join(words[i:i + size])
        for i in range(0, len(words), size)
    ]

# load + chunk all documents
def load_documents():
    chunks = []
    for filename in os.listdir(DOCS_PATH):
        if filename.endswith(".txt"):
            with open(os.path.join(DOCS_PATH, filename), "r", encoding = "utf-8") as f:
                text = f.read()
            for chunk in chunk_text(text):
                chunks.append({
                    "file": filename,
                    "content": chunk
                })
    return chunks

DOCUMENT_CHUNKS = load_documents()

# simple relevance scoring
def score_chunk(chunk, query):
    score = 0
    for word in query.lower().split():
        if word in chunk.lower():
            score +=1
    return score

def retrieve_chunks(query, top_k=3):
    scored = [
        (score_chunk(c["content"], query), c)
        for c in DOCUMENT_CHUNKS
    ]
    scored.sort(key=lambda x:x[0], reverse=True)
    return [c for score, c in scored if score > 0][:top_k]

TOOLS_FUNCTION = {
    "retrieve_chunks": retrieve_chunks
}

tools = [
    {
        "type": "function",
        "function":{
            "name": "retrieve_chunks",
            "description": "Retrieve relevant chunks for a query.",
            "parameters":{
                "type":"object",
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
    }
]

def call_model(messages):
    return client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        tools=tools
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

                print("\n Tool Called: ", fn_name, "args", args)
                result = TOOLS_FUNCTION[fn_name](**args)

                print("Result: ",result)

                context = "\n\n".join(
                    f"[{r['file']}]\n{r['content']}"
                    for r in result
                )

                messages.append({"role":"assistant", "tool_calls": msg.tool_calls})
                messages.append({
                    "role":"tool",
                    "tool_call_id": call.id,
                    "content": context
                })


query = "What are the major threats to ocean health?"
run_agent(query)



# Expected Output:


# Tool Called:  retrieve_chunks args {'query': 'major threats to ocean health', 'top_k': 5}
# Result:  [{'file': 'doc1.txt', 'content': 'Oceans cover more than seventy percent of the Earth’s surface and are essential to life on the planet. They play a major role in regulating global climate by absorbing large amounts of heat and carbon dioxide. Ocean currents help distribute heat across different regions, influencing weather patterns worldwide. Marine ecosystems are incredibly diverse, ranging from shallow coral reefs to deep-sea trenches. Millions of species live in the oceans, many of which are still undiscovered. Fish, plankton, corals, and marine mammals form complex food chains that maintain ecological balance. However, ocean health is under serious threat. Overfishing has significantly reduced fish populations in many regions. Plastic pollution and chemical waste contaminate marine habitats and harm wildlife. Rising sea temperatures caused by'}, {'file': 'doc2.txt', 'content': 'Oceans cover more than seventy percent of the Earth’s surface and are essential to life on the planet. They play a major role in regulating global climate by absorbing large amounts of heat and carbon dioxide. Ocean currents help distribute heat across different regions, influencing weather patterns worldwide. Marine ecosystems are incredibly diverse, ranging from shallow coral reefs to deep-sea trenches. Millions of species live in the oceans, many of which are still undiscovered. Fish, plankton, corals, and marine mammals form complex food chains that maintain ecological balance. However, ocean health is under serious threat. Overfishing has significantly reduced fish populations in many regions. Plastic pollution and chemical waste contaminate marine habitats and harm wildlife. Rising sea temperatures caused by'}, {'file': 'doc1.txt', 'content': 'climate change lead to coral bleaching and habitat loss. Protecting ocean ecosystems requires global cooperation. Sustainable fishing practices, reduced plastic use, and climate action are critical steps. Without intervention, the long-term health of the oceans — and life on Earth — is at risk.'}, {'file': 'doc2.txt', 'content': 'climate change lead to coral bleaching and habitat loss. Protecting ocean ecosystems requires global cooperation. Sustainable fishing practices, reduced plastic use, and climate action are critical steps. Without intervention, the long-term health of the oceans — and life on Earth — is at risk.'}, {'file': 'doc3.txt', 'content': 'Traveling allows people to experience new cultures, traditions, and perspectives. Exploring different countries helps individuals develop empathy and a broader understanding of the world. Many travelers enjoy visiting historical landmarks, museums, and natural wonders. Cultural exchange plays a key role in global understanding. Local food, festivals, and daily customs provide insight into how communities live. Travel can be both educational and personally transformative. In recent years, sustainable tourism has become increasingly important. Overtourism can damage fragile ecosystems and heritage sites. Responsible travel practices aim to reduce environmental impact while supporting local economies. By choosing sustainable options, travelers can help preserve destinations for future generations.'}]

#  Final Answer:  The major threats to ocean health include:

# 1. Overfishing: Significant reduction of fish populations in many regions.
# 2. Plastic pollution: Contamination of marine habitats and harm to wildlife.
# 3. Chemical waste: Pollution of oceans with harmful chemicals.
# 4. Rising sea temperatures: Coral bleaching and habitat loss due to climate change.

# These threats require global cooperation to address, with critical steps including sustainable fishing practices, reduced plastic use, and climate action.
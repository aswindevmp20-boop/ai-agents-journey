# Build an Retrieval-Augmented Generation (RAG) that:

# Reads multiple text files
# Searches for relevant chunks
# Returns only relevant content to the LLM
# Provides a summary or answer based on retrieved text


# A RAG agent has three steps:

# 1) Retrieve

# Find which file(s) contain relevant text.

# 2) Read

# Load the file content.

# 3) Generate

# LLM summarizes or answers using retrieved content.

# We’ll rely on keyword search (not embeddings yet).


import os,json
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key = os.getenv("GROQ_API_KEY"))


# Simple retrieval function

def search_docs(query):
    docs_folder = "./docs"

    for filename in os.listdir(docs_folder):
        if filename.endswith(".txt"):
            path = os.path.join(docs_folder, filename)

            with open(path, "r", encoding="utf-8") as f:
                text = f.read()

            if any(word.lower() in text.lower() for word in query.split()):
                return {"filename": filename}

    return {"filename": None}


# Tools

def read_document(filename):
    file_path = os.path.join("./docs", filename)
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except:
        return "FILE NOT FOUND."


TOOL_FUNCTIONS = {
    "search_docs": search_docs,
    "read_document": read_document,
}

tools = [
    {
        "type": "function",
        "function": {
            "name": "search_docs",
            "description": "Search for relevant documents based on user query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_document",
            "description": "Read a specific document by filename.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {"type": "string"}
                },
                "required": ["filename"]
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


def run_agent(user_query):
    messages = [{"role":"user", "content": user_query}]

    while True:
        response = call_model(messages)
        msg = response.choices[0].message

        if msg.content:
            print("\n FINAL ANSWER: \n",msg.content)
            break

        if msg.tool_calls:
            for call in msg.tool_calls:
                fn_name = call.function.name
                raw_args = call.function.arguments

                args =json.loads(raw_args)
                if args is None:
                    args = {}
                if isinstance(args,list):
                    args = args[0]

                print("\nTool Called: ", fn_name, "args:", args)

                result = TOOL_FUNCTIONS[fn_name](**args)
                print("Tool Results: ", result)

                messages.append({"role":"assistant", "tool_calls": msg.tool_calls})
                messages.append({
                    "role":"tool",
                    "tool_call_id": call.id,
                    "content": json.dumps(result)
                })

query = "Find documents about oceans and summarize the relevant ones."

run_agent(query)




# Expected Output:

# Tool Called:  search_docs args: {'query': 'oceans'}
# Tool Results:  {'filename': 'doc1.txt'}

# Tool Called:  read_document args: {'filename': 'doc1.txt'}
# Tool Results:  Oceans cover more than 70% of the Earth's surface and play a vital role in regulating global climate patterns. 
# They absorb a large portion of the carbon dioxide released into the atmosphere and distribute heat around the planet. 
# Marine ecosystems are home to millions of species, many of which are still undiscovered. 
# Overfishing, pollution, and rising sea temperatures pose major threats to ocean health.


#  FINAL ANSWER: 
#  The search for documents about oceans yielded several results. One document, "doc1.txt", provided an overview of the importance of oceans in regulating global climate patterns, their role in absorbing carbon dioxide, and the diversity of marine ecosystems. However, it also highlighted the major threats facing ocean health, including overfishing, pollution, and rising sea temperatures.
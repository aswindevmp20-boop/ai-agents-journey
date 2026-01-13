import json
from dotenv import load_dotenv
load_dotenv()

from groq import Groq
import os

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Python tool to read files

def read_file(path):
    try:
        with open(path,"r", encoding = "utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "ERROR: File not found."


TOOL_FUNCTIONS = {
    "read_file": read_file,
}

# Declare Tool Schema

tools = [
    {
        "type":"function",
        "function":{
            "name":"read_file",
            "description":"Read text from a file and return its content.",
            "parameters":{
                "type":"object",
                "properties":{
                    "path":{"type":"string"}
                },
                "required":["path"]
            }
        }
    }
]

# User Task

query = (
    "Read the file './sample.txt' and summarize it in 4 bullet points."
)

response = client.chat.completions.create(
    model = "llama-3.3-70b-versatile",
    messages = [{"role":"user","content": query}],
    tools=tools
)

print("\nRAW RESPONSE:\n", response, "\n")

msg = response.choices[0].message

# Handle Tool call

if msg.tool_calls:
    for call in msg.tool_calls:
        fn_name = call.function.name
        raw_args = call.function.arguments

        print("Raw args:", raw_args)

        args = json.loads(raw_args)
        result = TOOL_FUNCTIONS[fn_name](**args)

        print("\nTool result (file content):\n", result)

        # Send Result back to the model

        followup = client.chat.completions.create(
            model = "llama-3.3-70b-versatile",
            messages=[
                {"role":"user", "content": query},
                {"role":"assistant", "tool_calls": msg.tool_calls},
                {
                    "role": "tool",
                    "tool_call_id": call.id,   
                    "content": result
                }
            ]
        )

        print("\n FINAL SUMMARY: \n", followup.choices[0].message.content)
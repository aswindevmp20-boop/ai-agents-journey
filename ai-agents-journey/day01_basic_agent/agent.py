import json
from dotenv import load_dotenv
load_dotenv()

from groq import Groq
import os

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


# Tools
def add(a, b):
    return a + b

def subtract(a, b):
    return a - b

TOOL_FUNCTIONS = {
    "add": add,
    "subtract": subtract,
}

# Tool Schema
tools = [
    {
        "type": "function",
        "function": {
            "name": "add",
            "description": "Add two numbers",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number"},
                    "b": {"type": "number"},
                },
                "required": ["a", "b"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "subtract",
            "description": "Subtract b from a",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number"},
                    "b": {"type": "number"},
                },
                "required": ["a", "b"],
            },
        },
    },
]

# Query
query = (
    "Task: Calculate 55 + 21, then subtract 10 from the result."
    "You must complete the entire task using multiple tool calls if needed."
    "After executing the first tool call, continue reasoning and call the next tool."
    "Do not stop until the final result is reached."
)

# Calling groq
response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[{"role": "user", "content": query}],
    tools=tools,
)

print("\nRAW RESPONSE:\n", response, "\n")

#Handling Tool calls
msg = response.choices[0].message

if msg.tool_calls:
    for call in msg.tool_calls:

        fn_name = call.function.name
        raw_args = call.function.arguments

        print("Raw args string:", raw_args)

        try:
            args = json.loads(raw_args)

            # If model returns array instead of object
            if isinstance(args, list) and len(args) == 1 and isinstance(args[0], dict):
                args = args[0]

        except json.JSONDecodeError:
            raise ValueError("Model returned invalid JSON for tool arguments.")

        print(f"Tool called: {fn_name}, args={args}")

        # Execute tool
        result = TOOL_FUNCTIONS[fn_name](**args)
        print("Tool result:", result)

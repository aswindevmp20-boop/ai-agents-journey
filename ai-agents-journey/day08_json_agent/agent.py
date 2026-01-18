# Messy text and extract clean structured JSON

import os,json
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key = os.getenv("GROQ_API_KEY"))

def save_json(**data):
    return {
        "status": "success",
        "parsed_data": data
    }

TOOL_FUNCTIONS = {
    "save_json": save_json
}

tools = [
    {
        "type": "function",
        "function": {
            "name": "save_json",
            "description": "Extract structured information from natural language text and return it.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"},
                    "city": {"type": "string"},
                    "purchased_item": {"type": "string"},
                    "price": {"type": "number"},
                },
                "required": []
            }
        }
    }
]

def call_model(messages):
    return client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        tools = tools
    )

def run_agent(query):
    messages = [{"role":"user", "content":query}]

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

                print("\n Tool called: ", fn_name, "args", args)

                result = TOOL_FUNCTIONS[fn_name](**args)
                print("Tool Result", result)

                messages.append({"role": "assistant", "tool_calls": msg.tool_calls})
                messages.append({
                    "role":"tool",
                    "tool_call_id": call.id,
                    "content": json.dumps(result)
                })

query = """
Hey, can you help me record this customer detail?

So yesterday a guy named Rohit Sharma (32 years old, from Mumbai)
came to the store and bought a pair of noise-cancelling headphones.
He paid around 3.5k for it.

Thanks.

"""

run_agent(query)


# Expected Output:

# Tool called:  save_json args {'age': 32, 'city': 'Mumbai', 'name': 'Rohit Sharma', 'price': 3500, 'purchased_item': 'noise-cancelling headphones'}
# Tool Result {'status': 'success', 'parsed_data': {'age': 32, 'city': 'Mumbai', 'name': 'Rohit Sharma', 'price': 3500, 'purchased_item': 'noise-cancelling headphones'}}

# Final Answer:  The customer's details have been recorded successfully.
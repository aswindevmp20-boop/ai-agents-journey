# Day 1 → Single LLM call → You must force multi-step thinking in the PROMPT.
# Day 3 → Agent loop handles multi-step behavior automatically → No need to instruct it in the prompt.


from groq import Groq
import json
import os
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key = os.getenv("GROQ_API_KEY"))

def add(a,b):
    return a + b

def subtract(a,b):
    return a - b

def multiply(a,b):
    return a * b

TOOL_FUNCTIONS = {
    "add":add,
    "subtract": subtract,
    "multiply": multiply
}

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
                    "b": {"type": "number"}
                    },
                "required": ["a", "b"]
            }
        }
    },
    {
        "type":"function",
        "function":{
            "name":"subtract",
            "descriptions": "Subtract b from a",
            "parameters":{
                "type":"object",
                "properties":{
                    "a":{"type":"number"},
                    "b":{"type":"number"}
                },
                "required":["a","b"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "multiply",
            "description": "Multiply two numbers",
            "parameters": {
                "type": "object",
                "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
                "required": ["a", "b"]
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

def run_agent(user_query):
    messages = [{"role":"user", "content": user_query}]

    while True:
        response = call_model(messages)
        message = response.choices[0].message

        if message.content:
            print("\n FINAL ANSWER:")
            print(message.content)
            break

        if message.tool_calls:
            for tool_call in message.tool_calls:
                fn_name = tool_call.function.name
                raw_args = tool_call.function.arguments
                args = json.loads(raw_args)

                result = TOOL_FUNCTIONS[fn_name](**args)

                print(f"\n Tool called: {fn_name}, args = {args}, result = {result}")

                messages.append({"role":"assistant", "tool_calls":message.tool_calls})
                messages.append({
                    "role":"tool",
                    "tool_call_id": tool_call.id,
                    "content": str(result)
                })
        else:
            break

query = "Do (100+1000)., then multiply the result by 3, then subtract 10 from the result."
run_agent(query)




############################################

# Example: A User Asks

# User:

# “What is (100 + 1000)?”

# Your loop starts with:

# messages = [
#   {"role": "user", "content": "What is (100 + 1000)?"}
# ]

# Step 1 — Model Responds With a Tool Call

# The model returns something like:

# {
#   "role": "assistant",
#   "tool_calls": [
#     {
#       "id": "abc123",
#       "function": {
#         "name": "add",
#         "arguments": "{\"a\":100,\"b\":1000}"
#       }
#     }
#   ]
# }


# This means:

# “I (the assistant) need you to run the tool add with these arguments.”

# Now your loop sees message.tool_calls and runs the tool.

# Step 2 — Append the Tool Call Back into History
# messages.append({
#     "role": "assistant",
#     "tool_calls": message.tool_calls
# })


# Now messages looks like:

# [
#   {"role":"user", "content": "What is (100 + 1000)?"},
#   {
#     "role":"assistant",
#     "tool_calls":[
#       {
#         "id":"abc123",
#         "function":{
#           "name":"add",
#           "arguments":"{\"a\":100,\"b\":1000}"
#         }
#       }
#     ]
#   }
# ]


# This tells the model “Remember: you just asked for this tool call.”
# Without this, the model forgets that it made a tool call.

# Step 3 — Append the Tool’s Result Message

# Suppose your tool returns:

# result = 1100


# Then this line runs:

# messages.append({
#     "role": "tool",
#     "tool_call_id": "abc123",
#     "content": "1100"
# })


# Now messages becomes:

# [
#   {"role":"user","content":"What is (100 + 1000)?"},
#   {
#     "role":"assistant",
#     "tool_calls":[
#       {
#         "id":"abc123",
#         "function":{
#           "name":"add",
#           "arguments":"{\"a\":100,\"b\":1000}"
#         }
#       }
#     ]
#   },
#   {
#     "role":"tool",
#     "tool_call_id":"abc123",
#     "content":"1100"
#   }
# ]


# This tells the model:

# “Here is the result of the tool call you asked for.”

# Step 4 — The Model Continues Normally

# Now when you call call_model(messages) again, the model sees:

# It asked for a tool call

# It got the tool result

# So it continues the conversation:

# {
#   "role": "assistant",
#   "content": "The result of 100 + 1000 is 1100."
# }

# Final Output

# Your print statement shows:

# FINAL ANSWER:
# The result of 100 + 1000 is 1100.






# Expected Output:-

#  Tool called: add, args = {'a': 100, 'b': 1000}, result = 1100

#  Tool called: multiply, args = {'a': 1100, 'b': 3}, result = 3300

#  Tool called: subtract, args = {'a': 3300, 'b': 10}, result = 3290

#  FINAL ANSWER:
# The result of the calculation is 3290.
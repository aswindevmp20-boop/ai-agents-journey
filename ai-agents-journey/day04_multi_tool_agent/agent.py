import os, json, datetime, random
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key = os.getenv("GROQ_API_KEY"))

def add(a,b):
    return a + b

def subtract(a,b):
    return a - b

def multiply(a,b):
    return a * b

def read_file(path):
    try:
        with open(path, "r") as f:
            return f.read()
    except FileNotFoundError:
        return "ERROR: File not found"

def get_time():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def random_number(min_val, max_val):
    return random.randint(min_val, max_val)


TOOL_FUNCTIONS = {
    "add": add, 
    "subtract": subtract,
    "multiply": multiply,
    "read_file": read_file,
    "get_time": get_time,
    "random_number": random_number
}

tools = [
    {
        "type": "function",
        "function":{
            "name": "add",
            "description": "Add two numbers",
            "parameters":{
                "type": "object",
                "properties":{
                    "a":{"type":"number"},
                    "b":{"type":"number"}
                },
                "required": ["a","b"]
            }
        }
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
                "required": ["a", "b"]
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
    },
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
    },
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "Get the current date and time.",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "random_number",
            "description": "Generate a random number between min and max.",
            "parameters": {
                "type": "object",
                "properties": {
                    "min_val": {"type": "number"},
                    "max_val": {"type": "number"},
                },
                "required": ["min_val", "max_val"]
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
            print("\n Final Answer: \n", message.content)
            return message.content

        if message.tool_calls:
            for call in message.tool_calls:
                fn_name = call.function.name
                args = json.loads(call.function.arguments)

                print("Tool called:",fn_name, "args:", args)

                result = TOOL_FUNCTIONS[fn_name](**args)
                print("Tool Result:", result)

                messages.append({"role":"assistant","tool_calls":message.tool_calls})
                messages.append({
                    "role":"tool",
                    "tool_call_id":call.id,
                    "content":json.dumps(result)
                })

        else:
            break

query = (
    "1) Add 10 and 45 as a quick test. "
    "2) Then read the file './sample.txt' and summarize it in 4 bullet points. "
    "3) Finally generate a random number from 1 to 100."
)

run_agent(query)


# Expected Output:-

# Tool called: add args: {'a': 10, 'b': 45}
# Tool Result: 55
# Tool called: read_file args: {'path': './sample.txt'}
# Tool Result: Life in the small coastal town moved at a rhythm that felt almost separate from the rest of the world. Each morning, the gulls announced the sunrise long before the first boats drifted out of the harbor, their calls echoing across the water like impatient reminders of the day ahead. Locals wandered the cobblestone streets with a familiarity born from generations living in the same place, exchanging quiet nods or warm greetings as they passed. And though nothing remarkable ever seemed to happen there, the town held a quiet magic—one woven from salty air, worn wooden docks, and stories whispered between waves.
# Tool called: random_number args: {'max_val': 100, 'min_val': 1}
# Tool Result: 14

#  Final Answer: 
#  The results of the function calls are as follows:
# 1. The sum of 10 and 45 is 55.
# 2. The content of the file './sample.txt' is: "Life in the small coastal town moved at a rhythm that felt almost separate from the rest of the world. Each morning, the gulls announced the sunrise long before the first boats drifted out of the harbor, their calls echoing across the water like impatient reminders of the day ahead. Locals wandered the cobblestone streets with a familiarity born from generations living in the same place, exchanging quiet nods or warm greetings as they passed. And though nothing remarkable ever seemed to happen there, the town held a quiet magic—one woven from salty air, worn wooden docks, and stories whispered between waves."
# Here's a summary of the file content in 4 bullet points:
# * The town has a unique rhythm that feels separate from the rest of the world.
# * The town's daily life is marked by the sounds of gulls and the activities of the locals and fishermen.
# * The town has a strong sense of community, with locals who are familiar with each other and exchange greetings.
# * The town has a quiet magic that is woven from its natural surroundings and the stories of its people.
# 3. The generated random number between 1 and 100 is 14.


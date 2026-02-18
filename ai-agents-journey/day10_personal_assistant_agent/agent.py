# This agent can automatically decide what type of task the user wants and then call the correct tool:

# Weather
# Time
# Calculator
# Simple To-Do List (in-memory)
# Greetings / general chat
# AND you’ll see how the agent intelligently routes tasks


import os,json
from groq import Groq
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

client = Groq(api_key = os.getenv("GROQ_API_KEY"))

CITY_COORDS = {
    "delhi": {"lat": 28.6, "lon": 77.2},
    "mumbai": {"lat": 19.07, "lon": 72.87},
    "new york": {"lat": 40.71, "lon": -74.01},
    "london": {"lat": 51.50, "lon": -0.12},
}

def get_weather(city):
    key = city.lower()
    if key not in CITY_COORDS:
        return "City not supported."
    
    lat = CITY_COORDS[key]["lat"]
    lon = CITY_COORDS[key]["lon"]

    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
    data = requests.get(url).json()
    return data.get("current_weather", "Weather not available.")

def get_time():
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return {"current_time": now}

def calculate(expression:str):
    try:
        result = eval(expression, {"__builtins__":{}})
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}

TODO_LIST = []

def add_todo(task:str):
    TODO_LIST.append(task)
    return {"todo_list": TODO_LIST}


TOOL_FUNCTIONS = {
    "get_weather": get_weather,
    "get_time": get_time,
    "calculate": calculate,
    "add_todo": add_todo
}

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for a city.",
            "parameters":{
                "type":"object",
                "parameters":{
                    "type":"object",
                    "properties":{"city":{"type":"string"}},
                    "required":["city"]
                }
            }
        }
    },
    {
        "type":"function",
        "function":{
            "name":"get_time",
            "description":"Get current system time.",
            "parameters":{"type": "object", "properties":{}}
        }
    },
    {
        "type":"function",
        "function":{
            "name":"calculate",
            "description": "Solve a math expression.",
            "parameters":{
                "type":"object",
                "properties": {"expression":{"type":"string"}},
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "add_todo",
            "description": "Add a task to the todo list.",
            "parameters": {
                "type": "object",
                "properties": {"task": {"type": "string"}},
                "required": ["task"]
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
    messages = [{"role":"user","content": query}]

    while True:
        response = call_model(messages)
        msg = response.choices[0].message

        if msg.content:
            print("\n Final Answer:", msg.content)
            break

        if msg.tool_calls:
            for call in msg.tool_calls:
                fn_name = call.function.name
                args = json.loads(call.function.arguments)

                print("\n Tool called:", fn_name, "args", args)

                result = TOOL_FUNCTIONS[fn_name](**args)
                print("\nResult:", result)

                messages.append({"role":"assistant","tool_calls": msg.tool_calls})
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": json.dumps(result)
                })
 
query = "Add a task to buy groceries and then tell me the current time."
run_agent(query)


# Expected Output:

# Tool called: add_todo args {'task': 'buy groceries'}

# Result: {'todo_list': ['buy groceries']}

# Tool called: get_time args {}

# Result: {'current_time': '2026-01-19 16:54:10'}

# Final Answer: The current time is 2026-01-19 16:54:10. 

# You have added "buy groceries" to your todo list.
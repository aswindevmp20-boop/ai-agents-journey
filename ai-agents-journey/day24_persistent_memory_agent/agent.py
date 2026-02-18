from groq import Groq
import os, json, requests
from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

MEMORY_FILE = "memory.json"

def load_memory():
  if os.path.exists(MEMORY_FILE):
    try:
      with open(MEMORY_FILE,"r") as f:
        return json.load(f)
    except:
      print("memory.json is empty or corrupted. resetting memory.")
      return []
  return []

def save_memory(memory):
  with open(MEMORY_FILE, "w") as f:
    json.dump(memory, f, indent=2) #indent-2 makes it human readable.

memory = load_memory()

def calculator(expression):
  try:
    return str(eval(expression))
  except:
    return "invalid calculation"

def get_weather(city):
  return f"The weather in {city} is sunny with 30 degree celsius."

def web_search(query):
  try:
    result = tavily_client.search(
      query = query,
      max_result = 5
    )
    return json.dumps(result)
  except Exception as e:
    return f"Web search failed: {str(e)}"


TOOL_FUNCTION = {
  "calculator": calculator,
  "get_weather": get_weather,
  "web_search": web_search
}

tools = [
  {
    "type":"function",
    "function":{
      "name":"get_weather",
      "description": "Get weather of a city.",
      "parameters":{
        "type": "object",
        "properties":{
          "city": {"type":"string"}
        },
        "required":["city"]
      }
    }
  },
  {
    "type":"function",
    "function":{
      "name":"web_search",
      "description": "Search the web.",
      "parameters":{
        "type":"object",
        "properties":{
          "query":{"type":"string"}
        },
        "required":["query"]
      }
    }
  },
  {
    "type":"function",
    "function":{
      "name": "calculator",
      "description": "solve math expression",
      "parameters":{
        "type":"object",
        "properties":{
          "expression":{"type":"string"}
        },
        "required":["expression"]
      }
    }
  }
]

SYSTEM = {
  "role": "system",
  "content": (
    "You are a persistent personal assistant. "
    "Use memory from previous interactions when helpful. "
    "Choose the correct tool when needed. "
    "Call tools using valid JSON only."
  )
}

def call_model(messages):
  return client.chat.completions.create(
    model = "llama-3.3-70b-versatile",
    messages = messages,
    tools = tools
  )

def run_agent(query):
  global memory

  messages = [SYSTEM]

  if memory:
    memory_text = "\n".join(
      f"User: {m['user']} | Assistant: {m['assistant']}"
      for m in memory[-5:]
    )
    messages.append({
      "role":"assistant",
      "content": f"Past memory:\n{memory_text}"
    })

  messages.append({"role":"user", "content":query})

  while True:
    response = call_model(messages)
    msg = response.choices[0].message

    if msg.content:
      print("\n Final Answer: ", msg.content)

      memory.append({
        "user": query,
        "assistant": msg.content
      })

      save_memory(memory)
      break

    if msg.tool_calls:
      for call in msg.tool_calls:
        args = json.loads(call.function.arguments)
        tool_name = call.function.name

        result = TOOL_FUNCTION[tool_name](**args)

        messages.append({"role":"assistant", "tool_calls":msg.tool_calls})
        messages.append({
          "role":"tool",
          "tool_call_id": call.id , 
          "content": str(result)
        })

while True:
    q = input("\nAsk something (or exit): ")
    if q.lower() == "exit":
        break
    run_agent(q)



# Expected Output:

# $ python agent.py

# Ask something (or exit): what is my name ?

#  Final Answer:  Your name is Aswin.

# Ask something (or exit): calculate 55*9

#  Final Answer:  The result of 55*9 is 495.

# Ask something (or exit): exit

# Administrator@WIN-B6LHO8CCTMU MINGW64 /d/ai-agents-journey/ai-agents-journey/day24_persistent_memory_agent (main)

# Ask something (or exit): give me the answer of the calculation done

#  Final Answer:  The result of the calculation is 495.

# Ask something (or exit): exit

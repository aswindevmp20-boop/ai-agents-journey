from groq import Groq
from tavily import TavilyClient
import os, json
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

def web_search(query):
  try:
    result = tavily_client.search(
      query=query,
      max_results=5
    )
    return json.dumps(result)
  except:
    return f"Web search failed: {str(e)}"

def calculator(expression):
  try:
    return str(eval(expression))
  except:
    return "Invalid calculation"

def get_weather(city):
  return f"The weather in {city} is sunny with 30 degree celsius."

TOOL_FUNCTION = {
  "web_search" : web_search,
  "calculator" : calculator,
  "get_weather" : get_weather
}

tools = [
  {
    "type":"function",
    "function":{
      "name":"web_search",
      "description":"Search the web.",
      "parameters":{
        "type":"object",
        "properties":{
          "query":{"role":"string"}
        },
        "required":["query"]
      }
    }
  },
  {
    "type":"function",
    "function":{
      "name":"get_weather",
      "description":"get weather of a city.",
      "parameters":{
        "type":"object",
        "properties":{
          "city":{"role":"string"}
        },
        "required":["city"]
      }
    }
  },
  {
    "type":"function",
    "function":{
      "name": "calculator",
      "description": "Solve math expression.",
      "parameters":{
        "type":"object",
        "properties":{
           "expression":{"role":"string"}
        },
        "required":["expression"]
      }
    }
  }
]

def planner_agent(query):
  prompt = f"""
Break the user query into steps.
For each step, specify which tool to use:
- get_weather for weather
- calculator for math
- web_search for general knowledge

User query: {query}
  """
  response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[{"role":"user", "content":prompt}]
  )
  return response.choices[0].message.content

# worker + router

SYSTEM ={
  "role":"system",
  "content":(
    "you are an autonomous agent. \n"
    "choose the correct tool for each step.\n"
    "Use calculator for math, get_weather for weather, web_search for general knowledge.\n"
    "call tools using valid JSON only."
  )
}

def call_model(messages):
  return client.chat.completions.create(
    model = "llama-3.3-70b-versatile",
    messages=messages,
    tools=tools
  )

def run_agent(query):
  print("\n PLANNER \n")
  plan = planner_agent(query)
  print(plan)

  messages = [SYSTEM, {"role":"user", "content":plan}]

  while True:
    response = call_model(messages)
    msg = response.choices[0].message

    if msg.content:
      print("\n Final Answer: ", msg.content)
      break

    if msg.tool_calls:
      for call in msg.tool_calls:
        args = json.loads(call.function.arguments)
        tool_name = call.function.name

        result = TOOL_FUNCTION[tool_name](**args)

        messages.append({"role":"assistant", "tool_calls":msg.tool_calls})
        messages.append({
          "role": "tool",
          "tool_call_id":call.id,
          "content": str(result)
        })

while True:
    q = input("\nAsk something (or exit): ")
    if q.lower() == "exit":
        break
    run_agent(q)



# Expected output:-

# Ask something (or exit): get weather on Kochi, calculate 88*99 and give info on mohanlal

#  PLANNER

# To address the user query, we can break it down into the following steps:

# 1. **Get the weather for Kochi**:
#    - Tool to use: get_weather

# 2. **Calculate 88*99**:
#    - Tool to use: calculator

# 3. **Provide information on Mohanlal**:
#    - Tool to use: web_search


# Each of these steps utilizes a specific tool to accurately respond to the different parts of the query.

#  Final Answer:  The weather in Kochi is sunny with a temperature of 30 degree celsius. The result of the calculation 88*99 is 8712. Mohanlal is an Indian actor, producer, and singer who is well known for his versatile and natural acting in Indian cinema. He has starred in mainstream blockbuster and art-house films. For more information, you can check out his profile on IMDb, Wikipedia, or his Facebook page.

# Ask something (or exit): exit

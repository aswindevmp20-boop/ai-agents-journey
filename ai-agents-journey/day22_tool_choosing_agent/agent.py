from groq import Groq
from tavily import TavilyClient
import os, json
from dotenv import load_dotenv

load_dotenv()

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

def web_search(query):
    try:
        result = tavily_client.search(
            query=query,
            max_results=5
        )
        return json.dumps(result)
    except Exception as e:
        return f"Web search failed: {str(e)}"

def calculator(expression):
  try:
    return str(eval(expression))
  except:
    return "Invalid calculation"


def get_weather(city):
  #Mock weather API
  return f"The weather in {city} in sunny with 30 degree celsius."

TOOL_FUNCTION = {
  "calculator": calculator,
  "get_weather": get_weather,
  "web_search": web_search
}

tools = [
  {
    "type":"function",
    "function":{
      "name": "calculator",
      "description": "Solve math expressions",
      "parameters":{
        "type":"object",
        "properties":{
          "expression": {"type":"string"}
        },
        "required":["expression"]
      }
    }
  },
  {
    "type":"function",
    "function":{
      "name":"get_weather",
      "description": "Get weather of a city.",
      "parameters":{
        "type":"object",
        "properties":{
          "city": {"type":"string"}
        },
        "required": ["city"]
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
  }
]

SYSTEM = {
    "role": "system",
    "content": (
        "You are a tool router agent.\n"
        "Choose the correct tool based on user request.\n"
        "Use calculator for math, get_weather for weather, web_search for general info.\n"
        "Call tools using valid JSON only."
    )
}

def call_model(messages):
  return groq_client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=messages,
    tools=tools
  )

def run_agent(query):
    messages = [SYSTEM, {"role": "user", "content": query}]

    while True:
        response = call_model(messages)
        msg = response.choices[0].message

        if msg.content:
            print("\nFinal Answer:", msg.content)
            break

        if msg.tool_calls:
            messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": msg.tool_calls
            })

            for call in msg.tool_calls:
                args = json.loads(call.function.arguments)
                tool_name = call.function.name

                result = TOOL_FUNCTION[tool_name](**args)

                messages.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": result   
                })


while True:
    q = input("\nAsk something (or exit): ")
    if q.lower() == "exit":
        break
    run_agent(q)


# Expected Output:-

# Ask something (or exit): Weather at kochi

# Final Answer: I see you're looking for the weather in Kochi. Since the function call has already been made, I'll provide you with the result of the function call. Please note that the actual weather may vary depending on the time and other factors. The function call result is: The weather in Kochi is sunny with 30 degree celsius.

# Ask something (or exit): Aritifical intelligence

# Final Answer: Artificial intelligence (AI) refers to the ability of a computer or computer-controlled robot to perform tasks that are commonly associated with human intelligence, such as reasoning, decision-making, and learning. It encompasses a range of disciplines, including computer science, data analytics, and statistics, and can be applied in various fields, including healthcare, finance, and transportation. AI can be categorized into different types, including artificial general intelligence, applied AI, and cognitive simulation. While AI has the potential to bring about significant benefits, it also raises concerns about its impact on society, including job displacement, bias, and privacy.   

# Ask something (or exit): wht is 65*88

# Final Answer: The answer to the math expression 65*88 is 5720.

# Ask something (or exit): exit
from groq import Groq
from tavily import TavilyClient
import os, json, requests, sqlite3
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

def web_search(query):
  try:
    result = tavily_client.search(
      query = query,
      max_results = 5
    )
    return json.dumps(result)
  except Exception as e:
    return f"Web search failed {str(e)}."


DB_FILE = "memory.db"

def init_db():
  conn = sqlite3.connect(DB_FILE)
  cursor = conn.cursor()
  cursor.execute("""
  CREATE TABLE IF NOT EXISTS memory(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  user TEXT,
  assistant TEXT
  )
  """)
  conn.commit()
  conn.close()

def save_memory(user, assistant):
  conn = sqlite3.connect(DB_FILE)
  cursor = conn.cursor()
  cursor.execute("INSERT INTO memory (user,assistant) VALUES (?,?)",
  (user, assistant))
  conn.commit()
  conn.close()

def load_memory(limit=5):
  conn = sqlite3.connect(DB_FILE)
  cursor = conn.cursor()
  cursor.execute("SELECT user, assistant FROM memory ORDER BY id DESC LIMIT ?", (limit,))
  rows = cursor.fetchall()
  conn.close()
  return rows[::-1]

init_db()

TOOL_FUNCTION = {
  "web_search": web_search
}

tools = [
  {
    "type":"function",
    "function" :{
      "name":"web_search",
      "description": "Search on Web.",
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
  "content":(
    "You are an assistant with database memory.\n"
    "use past memory when helpful.\n"
    "call tools using valid JSON."
  )
}


def call_model(messages):
  return client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages = messages,
    tools=tools
  )

def run_agent(query):
  messages = [SYSTEM]

  memory = load_memory()

  if memory:
    memory_text = "\n".join(
      f"User: {u} | Assistant: {a}" for u, a in memory
    )
    messages.append({
      "role":"assistant",
      "content":f"Past memory:\n{memory_text}"
    })

  messages.append({"role":"user", "content": query})

  while True:
    response = call_model(messages)
    msg = response.choices[0].message

    if msg.content:
      print("\n Final Answer: ", msg.content)
      save_memory(query, msg.content)
      return

    if msg.tool_calls:
      for call in msg.tool_calls:
        args = json.loads(call.function.arguments)
        result = TOOL_FUNCTION[call.function.name](**args)

        messages.append({"role":"assistant","tool_calls":msg.tool_calls})
        messages.append({
          "role":"tool",
          "tool_call_id": call.id,
          "content": str(result)
        })
      continue

while True:
  q = input("\n Ask something (or exit): ")
  if q.lower() == "exit":
    break
  run_agent(q)


# Expected Output:-

# python agent.py

#  Ask something (or exit): My name is Aswin.

#  Final Answer:  Hello Aswin, it's nice to meet you. Is there something I can help you with or would you like to chat?

#  Ask something (or exit): whts ur name ?

#  Final Answer:  I don't have a personal name, but I'm an AI assistant designed to help users like you. Earlier, you introduced yourself to me as Aswin. Is there anything else I can help you with?

#  Ask something (or exit): what is 56*99     

#  Final Answer:  To find the product of 56 and 99, we can multiply the two numbers:

#  56 * 99 = 5544

#  Ask something (or exit): whts my name ?

#  Final Answer:  Your name is Aswin.

#   Ask something (or exit): give info on BRICS

#  Final Answer:  BRICS is an intergovernmental organization comprising Brazil, Russia, India, China, and South Africa, aimed at greater economic and geopolitical integration among member states. The organization has recently expanded to include new members, including Saudi Arabia, Iran, the United Arab Emirates, Egypt, Ethiopia, and Argentina. The objectives of BRICS include strengthening economic, political, and social cooperation among its members, as well as increasing the influence of Global South countries in international governance.
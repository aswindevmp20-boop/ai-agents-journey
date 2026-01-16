import os,json, requests
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key = os.getenv("GROQ_API_KEY"))

CITY_COORDS = {
    "delhi": {"lat": 28.6, "lon": 77.2},
    "mumbai": {"lat": 19.07, "lon": 72.87},
    "new york": {"lat": 40.71, "lon": -74.01},
    "london": {"lat": 51.50, "lon": -0.12},
    "tokyo": {"lat": 35.68, "lon": 139.69},
}

def get_weather(city):
    city_key = city.lower()

    if city_key not in CITY_COORDS:
        return "City Not Supported."

    lat = CITY_COORDS[city_key]["lat"]
    lon = CITY_COORDS[city_key]["lon"]

    #REST API Endpoint
    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}&current_weather=true"
    )

    response = requests.get(url).json()
    return response.get("current_weather","Weather data not available.")

TOOL_FUNCTIONS = {
    "get_weather": get_weather
}

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                },
                "required": ["city"],
            },
        },
    }
]

def call_model(messages):
    return client.chat.completions.create(
        model = "llama-3.3-70b-versatile",
        messages = messages,
        tools = tools
    )

def run_agent(query):
    messages = [{"role":"user", "content": query}]

    while True:
        response = call_model(messages)
        msg = response.choices[0].message
    
        if msg.content:
            print("\n FINAL ANSWER\n", msg.content)
            break

        if msg.tool_calls:
            for call in msg.tool_calls:
                fn_name = call.function.name
                args = json.loads(call.function.arguments)

                print("Tool called:",fn_name, "args:", args)

                result = TOOL_FUNCTIONS[fn_name](**args)
                print("Tool Result:", result)

                messages.append({"role":"assistant","tool_calls":msg.tool_calls})
                messages.append({
                    "role":"tool",
                    "tool_call_id":call.id,
                    "content":json.dumps(result)
                })

query = "What is the weather in Delhi right now?"
run_agent(query)



# Expected Output:-

# Tool called: get_weather args: {'city': 'Delhi'}
# Tool Result: {'time': '2026-01-14T17:30', 'interval': 900, 'temperature': 8.3, 'windspeed': 3.0, 'winddirection': 346, 'is_day': 0, 'weathercode': 0}

#  FINAL ANSWER
#  The current weather in Delhi is 8.3 degrees Celsius with a wind speed of 3.0 meters per second and a wind direction of 346 degrees. The weather code is 0, which indicates clear sky.
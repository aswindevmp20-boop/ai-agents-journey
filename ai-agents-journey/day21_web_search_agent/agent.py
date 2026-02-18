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


tools = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for latest and relevant information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"}
                },
                "required": ["query"]
            }
        }
    }
]

SYSTEM = {
    "role": "system",
    "content": (
        "You are a web research assistant. "
        "Always use the web_search tool to fetch current information. "
        "After receiving tool results, summarize them clearly in bullet points."
    )
}

def call_model(messages):
    return groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        tools=tools,
        tool_choice={"type": "function", "function": {"name": "web_search"}},
        temperature=0,
        max_tokens=500
    )


def run_agent(query):
    messages = [SYSTEM, {"role": "user", "content": query}]

    response = call_model(messages)
    msg = response.choices[0].message

    if not msg.tool_calls:
        print("Model did not call tool.")
        return

    messages.append({"role": "assistant", "tool_calls": msg.tool_calls})

    for call in msg.tool_calls:
        args = json.loads(call.function.arguments)
        result = web_search(**args)

        messages.append({
            "role": "tool",
            "tool_call_id": call.id,
            "content": result
        })

    final_response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=0,
        max_tokens=500
    )

    print("\nFINAL ANSWER:\n")
    print(final_response.choices[0].message.content)


if __name__ == "__main__":
    while True:
        query = input("\nEnter your query (or type 'exit'): ")
        if query.lower() == "exit":
            break
        run_agent(query)


# Expected Output:

# Enter your query (or type 'exit'): Latest news on BRICS.

# FINAL ANSWER:

# Here are the latest news on BRICS:
# * The BRICS summit is seeking to expand Russia's clout, with Chinese President Xi Jinping and India Prime Minister Narendra Modi meeting on the sidelines.      
# * The US President has announced that BRICS nations will get a 10% tariff "pretty soon".
# * Laos is embracing China and Russia-led SCO amid frustration with Asean, becoming the third Southeast Asian nation to partner with the SCO.
# * The latest news and analysis on BRICS can be found on various news websites, including Geopolitical Monitor, The Independent, Reuters, and South China Morning Post.
# * BRICS countries are making efforts to expand their influence and cooperation, with the latest developments and updates available on TV BRICS and other news sources.

# Enter your query (or type 'exit'): Whts people's opinion on Modi.

# FINAL ANSWER:

# Here are the opinions on Modi based on the search results:

# * According to a survey by Morning Consult, PM Modi has a 75% approval rating, making him the most popular democratic leader globally.
# * A poll by India Today shows strong support for Modi's continuation beyond 2029, with 73.9% of NDA voters in favor.
# * A Facebook post by India Today mentions that despite a slight decline, the numbers reflect sustained public approval for PM Modi after 11 years in office.    
# * Statista reports that Indian Prime Minister Narendra Modi was the most popular of 24 democratically elected state leaders in 2025, with approval ratings at over 75%.
# * A YouTube video by India Today discusses a survey that shows strong support for Modi's continuation beyond 2029.

# Overall, the opinions on Modi are largely positive, with high approval ratings and strong support for his continuation as Prime Minister.

# Enter your query (or type 'exit'): exit
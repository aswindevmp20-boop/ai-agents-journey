from tavily import TavilyClient
import os, json
from dotenv import load_dotenv

load_dotenv()

client = TavilyClient(api_key = os.getenv("TAVILY_API_KEY"))

def web_search(query):
    try:
        result = client.search(
            query=query,
            max_results=5
        )
        return json.dumps(result)
    except Exception as e:
        return f"Web search failed: {str(e)}"

TOOLS = {
    "web_search": web_search
}
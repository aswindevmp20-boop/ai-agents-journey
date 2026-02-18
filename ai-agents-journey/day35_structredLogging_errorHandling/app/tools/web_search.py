from tavily import TavilyClient

def web_search(query):
    try:
        result = TavilyClient.search(
            query=query
            max_results=5
        )
    except Exception as e:
        return f"Web search error: {e}"
    

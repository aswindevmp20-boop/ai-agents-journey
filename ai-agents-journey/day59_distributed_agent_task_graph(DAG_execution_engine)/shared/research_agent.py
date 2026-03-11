from shared.llm_service import generate_response
from shared.tools import TOOLS

def generate_research_topics(query):
    system_prompt = """
You are an research agent.

Break the topic into 3-5 research subtopics.
Return them as Bullet points.
"""

    messages = [
        {"role": "system","content": system_prompt},
        {"role":"user", "content": query}
    ]

    response = generate_response(messages)

    topics = [
        line.strip("- ").strip()
        for line in response.split("\n")
        if line.strip()
    ]

    return topics

#Gathering information for each topic.
def research_topics(topics):

    research_results = []

    for topic in topics:
        search_result = TOOLS["web_search"](topic)

        research_results.append({
            "topic": topic,
            "content": search_result
        })

    return research_results

def generate_report(query, research_data):

        combined_text = "\n\n".join(
             f"{item['topic']}:\n{item['content']}"
             for item in research_data
        )

        system_prompt = """
You ae a research analyst.

Write a structured report using the research findings.
Include sections and a clear conclusion.
"""

        messages = [
             {"role": "system", "content": system_prompt},
             {"role": "user", "content": f"Research Topics: {query}\n Data: \n{combined_text}"}
        ]

        report = generate_response(messages)

        return report

def run_research_agent(query):
     
     topics = generate_research_topics(query)
     research_data = research_topics(topics)
     report = generate_report(query, research_data)
     return report
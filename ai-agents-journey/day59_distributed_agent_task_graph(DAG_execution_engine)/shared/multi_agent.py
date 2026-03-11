from shared.llm_service import generate_response
from shared.tools import TOOLS

def research_agent(query):

    #use tool to gather info
    data = TOOLS["web_search"](query)

    return data

def analyst_agent(query, research_data):

    system_prompt = """
You are an AI analyst.

Interpret the researcch data and extract key insights.
"""

    messages = [
        {"role":"system","content":system_prompt},
        {"role":"user","content": f"Question: {query}\n Research: {research_data}"}
    ]

    analysis = generate_response(messages)

    return analysis

def writer_agent(query, analysis):
    system_prompt = """
You are an AI writer.

Produce a clear final answer based on the analysis
"""

    messages = [
        {"role":"system", "content": system_prompt},
        {"role":"user", "content": f"Question: {query}\n Analysis: {analysis}"}
    ]

    final_answer = generate_response(messages)
    return final_answer


def run_multi_agent(query):

    research = research_agent(query)
    analysis = analyst_agent(query, research)
    final_answer = writer_agent(query, analysis)
    return final_answer
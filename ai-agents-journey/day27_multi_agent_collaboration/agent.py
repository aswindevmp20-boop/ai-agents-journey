from groq import Groq
from tavily import TavilyClient
import os, json, requests
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

def web_search(query):
  try:
    result = tavily_client.search(
      query=query,
      max_result = 5
    )
    return json.dumps(result)
  except Exception as e:
    return f"Web search failed {str(e)}."

def planner_agent(query):
  prompt = f"""
  You are a planner.
  Break the followin task into steps:

  task = {query}
  """

  response = client.chat.completions.create(
    model = "llama-3.3-70b-versatile",
    messages = [{"role":"user", "content": prompt}]
  )
  return response.choices[0].message.content

def research_agent(plan):
    research = []

    MAX_LINES = 5          
    MAX_CHARS_PER_RESULT = 800
    MAX_TOTAL_CHARS = 4000

    lines = [
        line.strip()
        for line in plan.split("\n")
        if line.strip()
    ][:MAX_LINES]

    total_chars = 0

    for line in lines:
        result = web_search(line)

        # hard trim each result
        result = result[:MAX_CHARS_PER_RESULT]

        if total_chars + len(result) > MAX_TOTAL_CHARS:
            break

        research.append(result)
        total_chars += len(result)

    return "\n".join(research)


def summarize_research_agent(raw_research):
    prompt = f"""
    You are a research summarizer.

    Summarize the following reqsearch into
    5–7 concise bullet points.
    Keep only key facts.
    Remove redundancy.

    Research:
    {raw_research}
    """

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content

def writer_agent(query, research):
  prompt = f"""
  Use the research below to answer the user's question..

  Question: {query}

  research: {research}

  """
  response = client.chat.completions.create(
    model = "llama-3.3-70b-versatile",
    messages = [{"role":"user", "content": prompt}]
  )
  return response.choices[0].message.content


def run_agent(query):

  print("\n----Planner-----")
  plan = planner_agent(query)
  print(plan)

  print("\n-----Research-----")
  research = research_agent(plan)
  summarize_research = summarize_research_agent(research)
  print(summarize_research)

  print("\n-----Writer-----")
  answer = writer_agent(query, research)
  print("\n Final Answer:\n", answer)


while True:
    q = input("\nAsk something (or exit): ")
    if q.lower() == "exit":
        break
    run_agent(q)


# Expected Output:-

# Ask something (or exit): importance of ocean protecton

# ----Planner-----
# As a planner, I'll break down the task of highlighting the importance of ocean protection into manageable steps. Here's a step-by-step approach:

# **Step 1: Research and Understanding (Gathering Information)**

# * Identify the key aspects of ocean protection, such as:
#         + Threats to ocean health (pollution, overfishing, climate change, etc.)
#         + Importance of marine ecosystems (biodiversity, food security, coastal protection, etc.)
#         + Current state of ocean conservation efforts 
# (existing laws, organizations, initiatives, etc.)     
# * Gather data and statistics on the impact of human activities on the ocean and its inhabitants
# * Review existing literature and research on ocean protection to gain a deeper understanding of the issue   

# **Step 2: Defining the Scope and Objectives (Clarifying the Goal)**

# * Determine the specific focus of the task, such as:  
#         + Raising awareness about ocean pollution     
#         + Promoting sustainable fishing practices     
#         + Supporting marine conservation efforts      
# * Set clear objectives, such as:
#         + Educating a certain number of people about ocean protection
#         + Reducing a specific amount of marine pollution
#         + Supporting a particular marine conservation 
# initiative

# **Step 3: Identifying the Target Audience (Understanding the Stakeholders)**

# * Determine the target audience for the message about 
# ocean protection, such as:
#         + General public
#         + Policy makers
#         + Businesses
#         + Educational institutions
# * Identify the key stakeholders who can influence or be impacted by ocean protection efforts, such as:      
#         + Governments
#         + Fishing industries
#         + Coastal communities
#         + Environmental organizations

# **Step 4: Developing Key Messages (Crafting the Message)**

# * Based on the research and objectives, develop clear 
# and concise key messages about the importance of ocean protection, such as:
#         + "The ocean provides 50-85% of the Earth's oxygen"
#         + "Overfishing and pollution are threatening marine ecosystems"
#         + "Individual actions can make a difference in protecting the ocean"
# * Create a compelling narrative that resonates with the target audience and inspires action

# **Step 5: Choosing Communication Channels (Selecting the Medium)**

# * Select the most effective communication channels to 
# reach the target audience, such as:
#         + Social media
#         + Traditional media (newspapers, TV, radio)   
#         + Educational programs
#         + Community events
#         + Policy briefs and reports
# * Consider the strengths and limitations of each channel and tailor the message accordingly

# **Step 6: Creating Engaging Content (Developing the Materials)**

# * Develop engaging and informative content, such as:  
#         + Infographics
#         + Videos
#         + Blog posts
#         + Social media posts
#         + Educational resources (curricula, lesson plans, etc.)
# * Use visual aids, storytelling, and emotional connections to make the message more relatable and memorable 

# **Step 7: Implementing the Plan (Executing the Strategy)**

# * Launch the communication campaign and engage with the target audience through the chosen channels
# * Monitor and evaluate the effectiveness of the campaign and adjust the strategy as needed
# * Collaborate with other stakeholders and organizations to amplify the message and increase impact

# **Step 8: Evaluating and Refining the Approach (Assessing Progress)**

# * Track key performance indicators (KPIs) to measure the success of the campaign, such as:
#         + Website traffic
#         + Social media engagement
#         + Number of people reached
#         + Policy changes or advocacy wins
# * Refine the approach based on lessons learned and feedback from the target audience and stakeholders       
# * Continuously update and improve the message and strategy to ensure the importance of ocean protection remains a priority.

# -----Research-----
# Here are 7 concise bullet points summarizing the research on protecting our oceans:

# * Reduce single-use plastics, which account for nearly 80% of marine debris.
# * Understand the key aspects of ocean protection, such as creating marine protected areas.
# * Identify the importance of marine ecosystems, including biodiversity, food security, and coastal protection.
# * Learn about the threats to ocean health, including pollution, overfishing, climate change, and ecosystem degradation.
# * Familiarize yourself with marine biodiversity and its role in maintaining healthy ecosystems.
# * Recognize the significance of ocean conservation in 
# ensuring the long-term sustainability of marine biological diversity.
# * Take part in everyday actions to help protect our oceans, such as supporting marine protected areas and reducing plastic use.

# -----Writer-----

#  Final Answer:
#  The importance of ocean protection cannot be overstated. Simple everyday actions, such as reducing single-use plastics, which account for nearly 80% of marine debris, can help protect our oceans. Additionally, creating marine protected areas (MPAs) can effectively protect and preserve the ocean.

# However, the ocean faces several threats, including pollution, overfishing, climate change, and ecosystem degradation. These threats pose significant challenges to ocean health and require measures to tackle each of 
# these issues.

# On the other hand, marine ecosystems play a crucial role in maintaining biodiversity, food security, coastal protection, and climate regulation. Coastal ecosystems, in particular, are hotspots of marine biodiversity 
# and provide essential ecosystem services like food security, coastal protection, and climate regulation.    

# Some key aspects of ocean protection include:

# 1. Creating marine protected areas (MPAs) to conserve 
# and sustainably use marine biological diversity.
# 2. Addressing threats to ocean health, such as pollution, overfishing, and climate change.
# 3. Preserving marine ecosystems, which provide vital services to humans, including oxygen production, food security, and climate regulation.

# Overall, ocean protection is essential for maintaining the health of our planet, and individual actions, combined with collective efforts, can make a significant difference in preserving the ocean and its ecosystems for future generations.
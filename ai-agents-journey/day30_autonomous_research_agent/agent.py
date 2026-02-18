from groq import Groq
from tavily import TavilyClient
import json, os, requests
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

def web_search(query):
    try:
        result = tavily_client.search(
            query=query,
            max_results= 5
        )
        return json.dumps(result)
    except Exception as e:
        print(f"Web search failed: ", e)

def research_agent(question):
    research_data = web_search(question)
    prompt = f"""
You are a research agent.
research the question in detail and provide the answer.

Question: {question}

Research: {research_data}
"""
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role":"user","content":prompt}]
    )
    return response.choices[0].message.content


def evaluation_agent(question,answer):
    prompt = f"""
You are a quality evaluation agent.
your job is to check whether the answer given to the given question is accurate or not.
Respond only with YES or NO.

Question: {question}
Answer:{answer}
"""
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role":"user","content":prompt}]
    )
    return response.choices[0].message.content.strip()

def run_agent(question):
    current_query = question
    max_iterations =3

    for i in range(max_iterations):
        print(f"\n-- ITERATION {i+1} --")

        answer = research_agent(current_query)
        print("\nResearch answer: ",answer)

        evaluation = evaluation_agent(question, answer)
        print("\n Evaluation: ", evaluation)

        if "YES" in evaluation.strip():
            print("\n Final Answer: \n", answer)
            return
        
        current_query = f"provide more detailed Explanation about {question}."
    
    print("\n Max iterations reached. Final answer: \n", answer)

while True:
    q = input("\nAsk research question (or exit): ")
    if q.lower() == "exit":
        break
    run_agent(q)


# Expected output:-

# Ask research question (or exit): How to become a president ?

# -- ITERATION 1 --

# Research answer:  To become the President of the United States, an individual must meet the eligibility requirements set forth in Article II, Section 1 of the U.S. Constitution. The qualifications are as follows:

# 1. **Age:** The candidate must be at least 35 years old.
# 2. **Residency:** The candidate must have been a resident of the United States for at least 14 years.
# 3. **Citizenship:** The candidate must be a natural-born citizen of the United States.

# After meeting these eligibility requirements, the candidate can proceed with the following steps to become President:

# **Step 1: Build a Career in Public Service**
# Start by building a career in public service, which can include roles in government, military, or non-profit organizations.

# **Step 2: Gain National Recognition**
# Increase your national recognition by taking on high-profile roles, building a strong network, and making notable achievements in your field.

# **Step 3: Declare Candidacy**
# Formally declare your candidacy for President by filing the necessary paperwork with the Federal Election Commission (FEC) and announcing your decision to the public.

# **Step 4: Win the Party Nomination**
# Secure the nomination of a major political party by winning the primary elections or caucuses.

# **Step 5: Participate in National Conventions**
# Attend the national convention of your party, where delegates will formally nominate you as the party's candidate for President.

# **Step 6: Campaign and Debate**
# Engage in a national campaign, debating your opponents and presenting your vision for the country to the American people.

# **Step 7: Win the General Election**
# Win the majority of the electoral votes by receiving the most votes in the general election.

# **Step 8: Inauguration**
# Take the oath of office and become the President of the United States during the inauguration ceremony.

# Some additional resources that can help with this process include:

# * The official website of the United States government (usa.gov)
# * The Federal Election Commission (fec.gov)
# * The U.S. Constitution (available online or through various educational resources)

# Remember, becoming the President of the United States is a challenging and rigorous process that requires dedication, hard work, and a deep understanding of the country's political system.

#  Evaluation:  YES

#  Final Answer:
#  To become the President of the United States, an individual must meet the eligibility requirements set forth in Article II, Section 1 of the U.S. Constitution. The qualifications are as follows:

# 1. **Age:** The candidate must be at least 35 years old.
# 2. **Residency:** The candidate must have been a resident of the United States for at least 14 years.
# 3. **Citizenship:** The candidate must be a natural-born citizen of the United States.

# After meeting these eligibility requirements, the candidate can proceed with the following steps to become President:

# **Step 1: Build a Career in Public Service**
# Start by building a career in public service, which can include roles in government, military, or non-profit organizations.

# **Step 2: Gain National Recognition**
# Increase your national recognition by taking on high-profile roles, building a strong network, and making notable achievements in your field.

# **Step 3: Declare Candidacy**
# Formally declare your candidacy for President by filing the necessary paperwork with the Federal Election Commission (FEC) and announcing your decision to the public.     

# **Step 4: Win the Party Nomination**
# Secure the nomination of a major political party by winning the primary elections or caucuses.

# **Step 5: Participate in National Conventions**
# Attend the national convention of your party, where delegates will formally nominate you as the party's candidate for President.

# **Step 6: Campaign and Debate**
# Engage in a national campaign, debating your opponents and presenting your vision for the country to the American people.

# **Step 7: Win the General Election**
# Win the majority of the electoral votes by receiving the most votes in the general election.

# **Step 8: Inauguration**
# Take the oath of office and become the President of the United States during the inauguration ceremony.

# Some additional resources that can help with this process include:

# * The official website of the United States government (usa.gov)
# * The Federal Election Commission (fec.gov)
# * The U.S. Constitution (available online or through various educational resources)

# Remember, becoming the President of the United States is a challenging and rigorous process that requires dedication, hard work, and a deep understanding of the country's political system.
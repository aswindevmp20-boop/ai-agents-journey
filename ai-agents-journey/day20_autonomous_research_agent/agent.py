from groq import Groq
from sentence_transformers import SentenceTransformer
import faiss
import os,json
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key = os.getenv("GROQ_API_KEY"))

DOCS_PATH = "./docs"
CHUNK_SIZE = 110
TOP_K = 3

embedder = SentenceTransformer("all-MiniLM-L6-v2")

shared_memory = [] 

def chunk_text(text):
    words = text.split()
    return [" ".join(words[i:i+CHUNK_SIZE]) for i in range(0, len(words), CHUNK_SIZE)]

def load_documents():
    chunks = []
    for file in os.listdir(DOCS_PATH):
        if file.endswith(".txt"):
            with open(os.path.join(DOCS_PATH, file), "r", encoding="utf-8") as f:
                text = f.read()
            for chunk in chunk_text(text):
                chunks.append({"file": file, "content": chunk})
    return chunks

DOCUMENT_CHUNKS = load_documents()

embeddings = embedder.encode([c["content"] for c in DOCUMENT_CHUNKS], convert_to_numpy=True)
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

def retrieve_chunks(query, top_k=TOP_K):
    q_vec = embedder.encode([query], convert_to_numpy=True)
    distances, indices = index.search(q_vec, top_k)
    return [DOCUMENT_CHUNKS[i] for i in indices[0]]


def planner_agent(topic):
    prompt = f"""
    You are a research planner.
    Create a step-by-step research plan for this topic:

    Topic: {topic}
    """
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


def worker_agent(plan, topic):
    global shared_memory

    chunks = retrieve_chunks(topic)
    context = "\n\n".join(
        f"[{c['file']}]\n{c['content']}" for c in chunks
    )

    prompt = f"""
    You are a research worker.
    Follow this plan:
    {plan}

    Context:
    {context}

    Extract key points and store them as research notes.
    """

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )

    notes = response.choices[0].message.content
    shared_memory.append(notes)
    return notes

def reporter_agent(topic):
    memory_text = "\n".join(shared_memory)

    prompt = f"""
    You are a research reporter.

    Topic: {topic}

    Research notes:
    {memory_text}

    Write a structured research report with:
    - Introduction
    - Key findings
    - Conclusion
    """

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content


def run_agent(topic):
    print("\n--- PLANNER ---")
    plan = planner_agent(topic)
    print(plan)

    print("\n--- WORKER ---")
    notes = worker_agent(plan, topic)
    print(notes)

    print("\n--- REPORTER ---")
    report = reporter_agent(topic)
    print("\nFINAL REPORT:\n", report)

~
topic = "Threats to ocean health"
run_agent(topic)


# Expected Output :-

# --- PLANNER ---
# As a research planner, I've developed a comprehensive step-by-step research plan to investigate the threats to ocean health. Here's the plan:

# **Research Title:** Investigating the Threats to Ocean Health: A Comprehensive Analysis

# **Research Objectives:**

# 1. To identify the primary threats to ocean health
# 2. To analyze the impact of these threats on marine ecosystems and biodiversity
# 3. To examine the effectiveness of current conservation efforts and policies
# 4. To provide recommendations for mitigating the threats to ocean health

# **Step 1: Literature Review (Weeks 1-4)**

# * Conduct a thorough review of existing literature on ocean health threats, including:
#         + Scientific articles and journals
#         + Government reports and policy documents
#         + Non-governmental organization (NGO) publications and reports
#         + Online databases and academic repositories
# * Identify key themes, trends, and gaps in the existing research
# * Develop a preliminary list of potential threats to ocean health, including:
#         + Pollution (plastic, chemical, noise)
#         + Overfishing and destructive fishing practices      
#         + Climate change (ocean acidification, sea-level rise)
#         + Coastal development and habitat destruction        
#         + Invasive species

# **Step 2: Expert Interviews and Surveys (Weeks 5-8)**        

# * Identify and recruit experts in the field of ocean health, 
# including:
#         + Marine biologists and ecologists
#         + Conservationists and policymakers
#         + Industry representatives (fishing, shipping, tourism)
#         + Community leaders and stakeholders
# * Conduct in-depth interviews with experts to gather insights on:
#         + The most significant threats to ocean health       
#         + The impact of these threats on marine ecosystems and biodiversity
#         + Effective conservation strategies and policies     
#         + Emerging issues and areas of concern
# * Develop and distribute surveys to a larger audience of stakeholders and experts to gather additional data and validate findings

# **Step 3: Data Collection and Analysis (Weeks 9-16)**        

# * Collect and analyze data on ocean health indicators, including:
#         + Water quality parameters (pH, temperature, nutrient levels)
#         + Marine species abundance and distribution
#         + Fishing effort and catch data
#         + Coastal development and habitat destruction        
#         + Climate change metrics (sea-level rise, ocean acidification)
# * Utilize existing datasets and databases, such as:
#         + NOAA's National Oceanic and Atmospheric Administration
#         + IPCC's Intergovernmental Panel on Climate Change   
#         + IUCN's International Union for Conservation of Nature
#         + FAO's Food and Agriculture Organization
# * Apply statistical and analytical techniques to identify patterns and trends in the data

# **Step 4: Case Studies and Examples (Weeks 17-20)**

# * Select 3-5 case studies of specific ocean health threats, such as:
#         + The Great Pacific Garbage Patch
#         + Coral bleaching in the Great Barrier Reef
#         + Overfishing in the Mediterranean Sea
#         + Coastal erosion in the Maldives
# * Conduct in-depth research on each case study, including:   
#         + Historical context and background information      
#         + Current status and impact of the threat
#         + Conservation efforts and policies in place
#         + Lessons learned and best practices for mitigation and management
# * Analyze and compare the case studies to identify common themes and trends

# **Step 5: Recommendations and Policy Analysis (Weeks 21-24)**
# * Based on the research findings, develop recommendations for:
#         + Mitigating the threats to ocean health
#         + Improving conservation efforts and policies        
#         + Enhancing international cooperation and governance 
#         + Supporting community-led initiatives and stakeholder engagement
# * Analyze existing policies and laws related to ocean health, including:
#         + International agreements and treaties (e.g., UNCLOS, Paris Agreement)
#         + National and regional laws and regulations
#         + Industry standards and certifications
# * Identify gaps and areas for improvement in current policies and laws

# **Step 6: Writing and Dissemination (Weeks 24-28)**

# * Write a comprehensive research report detailing the findings and recommendations
# * Develop a summary report and policy brief for stakeholders 
# and policymakers
# * Create visual aids and infographics to communicate the research findings
# * Disseminate the research report and summary through:       
#         + Academic journals and conferences
#         + Social media and online platforms
#         + Stakeholder workshops and events
#         + Policy briefings and government submissions        

# **Timeline:** The research project is expected to be completed within 28 weeks (approximately 7 months).

# **Resources:**

# * Access to academic databases and online repositories       
# * Funding for travel and stakeholder engagement
# * Research assistants and interns for data collection and analysis
# * Software and equipment for data analysis and visualization 

# This research plan provides a comprehensive framework for investigating the threats to ocean health. By following these steps, we can gather valuable insights and recommendations for 
# mitigating these threats and promoting the long-term health and sustainability of our oceans.

# --- WORKER ---
# **Research Notes:**

# **General Information:**

# 1. Oceans cover over 70% of the Earth's surface and are essential to life on the planet.
# 2. Oceans play a major role in regulating global climate by absorbing heat and carbon dioxide.
# 3. Ocean currents help distribute heat across different regions, influencing weather patterns worldwide.

# **Threats to Ocean Health:**

# 1. Climate change leads to coral bleaching and habitat loss. 
# 2. Overfishing has significantly reduced fish populations in 
# many regions.
# 3. Plastic pollution and chemical waste contaminate marine habitats and harm wildlife.
# 4. Rising sea temperatures caused by climate change pose a significant threat to ocean health.

# **Importance of Ocean Ecosystems:**

# 1. Marine ecosystems are incredibly diverse, ranging from shallow coral reefs to deep-sea trenches.
# 2. Millions of species live in the oceans, many of which are 
# still undiscovered.
# 3. Fish, plankton, corals, and marine mammals form complex food chains that maintain ecological balance.

# **Need for Global Cooperation:**

# 1. Protecting ocean ecosystems requires global cooperation.  
# 2. Sustainable fishing practices, reduced plastic use, and climate action are critical steps to mitigate the threats to ocean health.
# 3. Without intervention, the long-term health of the oceans — and life on Earth — is at risk.

# **Recommendations:**

# 1. Implement sustainable fishing practices to reduce overfishing.
# 2. Reduce plastic use and mitigate plastic pollution in the oceans.
# 3. Take climate action to reduce greenhouse gas emissions and mitigate the effects of climate change on ocean health.     
# 4. Promote global cooperation to protect ocean ecosystems and maintain ecological balance.

# These research notes will serve as a foundation for further investigation and analysis of the threats to ocean health, and will help inform the development of recommendations for mitigating these threats and promoting the long-term health and sustainability of our oceans.

# --- REPORTER ---

# FINAL REPORT:
#  **Research Report: Threats to Ocean Health**

# **Introduction**

# The world's oceans are a vital component of the Earth's ecosystem, covering over 70% of the planet's surface and playing a crucial role in regulating global climate, weather patterns, and supporting an incredible array of marine life. However, 
# the health of our oceans is under significant threat from various human activities, including climate change, overfishing, plastic pollution, and chemical waste. This report aims to summarize the key findings on the threats to ocean health and 
# provide recommendations for mitigating these threats to promote the long-term health and sustainability of our oceans.    

# **Key Findings**

# 1. **Climate Change**: Rising sea temperatures and ocean acidification caused by climate change are leading to coral bleaching, habitat loss, and changes in species distribution. This poses a significant threat to the delicate balance of marine ecosystems and the millions of species that depend on them. 
# 2. **Overfishing**: The overexploitation of fish populations 
# has significantly reduced the numbers of many species, disrupting the complex food chains that maintain ecological balance. Sustainable fishing practices are essential to mitigate this threat.
# 3. **Plastic Pollution**: The influx of plastic waste into the oceans is contaminating marine habitats, harming wildlife, 
# and entering the food chain. This highlights the need for reduced plastic use and effective waste management strategies.  
# 4. **Importance of Global Cooperation**: Protecting ocean ecosystems requires a collective effort from governments, industries, and individuals worldwide. Sustainable fishing practices, reduced plastic use, and climate action are critical steps to mitigate the threats to ocean health.
# 5. **Diverse and Complex Ecosystems**: Marine ecosystems are 
# incredibly diverse, ranging from shallow coral reefs to deep-sea trenches, and support a vast array of species. The preservation of these ecosystems is essential for maintaining ecological balance and ensuring the long-term health of the oceans.

# **Conclusion**

# The threats to ocean health are multifaceted and far-reaching, with significant implications for the health of our planet. The key findings of this report highlight the urgent need for collective action to address the impacts of climate change, overfishing, plastic pollution, and other human activities on the world's oceans. To mitigate these threats, we recommend:

# 1. **Implementing sustainable fishing practices** to reduce overfishing and protect marine ecosystems.
# verfishing and protect marine ecosystems.
# 2. **Reducing plastic use and mitigating plastic pollution** in the oceans through effective waste management and reduced plastic production.
# 3. **Taking climate action** to reduce greenhouse gas emissions and mitigate the effects of climate change on ocean health.
# 4. **Promoting global cooperation** to protect ocean ecosystems and maintain ecological balance.

# By working together to address these threats, we can help ensure the long-term health and sustainability of our oceans, which are essential for the well-being of our planet and all its inhabitants.


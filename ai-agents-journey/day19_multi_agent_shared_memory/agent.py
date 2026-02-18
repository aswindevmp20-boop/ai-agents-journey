from groq import Groq
from sentence_transformers import SentenceTransformer
import faiss
import os,json
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key = os.getenv("GROQ_API_KEY"))

DOCS_PATH = "./docs"
CHUNK_SIZE = 120
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

def planner_agent(user_query):
    prompt = f"""
    You are a planner agent.
    Break the task into steps.

    User question: {user_query}
    """
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

def worker_agent(plan, user_query):
    global shared_memory

    chunks = retrieve_chunks(user_query)

    context = "\n\n".join(
        f"[{c['file']}]\n{c['content']}" for c in chunks
    )

    memory_text = "\n".join(shared_memory)

    prompt = f"""
    You are a worker agent.

    Plan:
    {plan}

    Shared Memory:
    {memory_text}

    Context:
    {context}

    Execute the plan and produce a clear result.
    """

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )

    answer = response.choices[0].message.content

    shared_memory.append(answer)

    return answer

def run_agent(query):
    print("\n--- PLANNER ---")
    plan = planner_agent(query)
    print(plan)

    print("\n--- WORKER ---")
    answer = worker_agent(plan, query)
    print("\nFINAL ANSWER:\n", answer)

    print("\n--- SHARED MEMORY ---")
    for m in shared_memory:
        print("-", m)

query = "What are the threats to ocean health and why are they dangerous?"
run_agent(query)


# Expected Output:-

# -- PLANNER ---
# To address the question of threats to ocean health and why they are dangerous, I'll break down the task into steps:

# **Step 1: Identify Major Threats to Ocean Health**
# - Research and list the primary threats to ocean health, which may include:
#   1. Pollution (chemical, plastic, oil spills)
#   2. Overfishing and destructive fishing practices
#   3. Climate change (including ocean acidification and warming)
#   4. Coastal development and habitat destruction
#   5. Invasive species

# **Step 2: Analyze the Impact of Each Threat**
# - For each identified threat, analyze its specific impacts on ocean health, including:
#   1. Effects on marine biodiversity
#   2. Impact on marine ecosystems
#   3. Consequences for human health and livelihoods (e.g., fishing industries, tourism)
#   4. Contributions to climate change and its feedback loops

# **Step 3: Understand the Mechanisms Behind the Threats**
# - Delve into why these threats are occurring, including:
#   1. Human activities contributing to pollution (e.g., lack of waste management, industrial practices)
#   2. Economic and political drivers behind overfishing and destructive fishing
#   3. The science behind climate change and its acceleration
#   4. Urbanization and development patterns leading to coastal habitat destruction

# **Step 4: Evaluate the Consequences of Inaction**
# - Assess the potential long-term consequences if these threats are not addressed, such as:
#   1. Irreversible damage to marine ecosystems
#   2. Loss of biodiversity and extinction of species
#   3. Significant impacts on global food security and economies
#   4. Enhanced vulnerability to natural disasters and climate change effects

# **Step 5: Research Existing Solutions and Proposals**
# - Investigate current efforts, technologies, and policies aimed at mitigating these threats, including:
#   1. Renewable energy to reduce carbon emissions
#   2. Marine protected areas and sustainable fishing practices
#   3. Innovations in waste management and reduction
#   4. International agreements and legal frameworks for ocean protection

# **Step 6: Compile and Present the Information**
# - Organize the findings into a comprehensive overview that answers the question, including:
#   1. A summary of the primary threats to ocean health
#   2. An explanation of why these threats are dangerous
#   3. Examples of the impacts on marine life, ecosystems, and human societies
#   4. An overview of potential solutions and future directions for protection and conservation

# By following these steps, we can provide a detailed and informative response to the question about the threats to ocean health and why they are dangerous.      

# --- WORKER ---

# FINAL ANSWER:
#  **Threats to Ocean Health and Why They Are Dangerous**

# The oceans, which cover over 70% of the Earth's surface, are essential to life on the planet. However, they are facing numerous threats that can have severe consequences for marine ecosystems, human health, and the economy. In this response, we will identify the primary threats to ocean health, analyze their impacts, understand the mechanisms behind them, evaluate the consequences of inaction, and research existing solutions and proposals.

# **Step 1: Identify Major Threats to Ocean Health**

# The primary threats to ocean health are:

# 1. **Pollution**: Chemical, plastic, and oil spills contaminate marine habitats and harm wildlife.
# 2. **Overfishing and destructive fishing practices**: Overfishing reduces fish populations, while destructive fishing practices damage habitats and ecosystems. 
# 3. **Climate change**: Rising sea temperatures cause coral bleaching and habitat loss, while ocean acidification affects marine life.
# 4. **Coastal development and habitat destruction**: Urbanization and development lead to the destruction of coastal habitats and ecosystems.
# 5. **Invasive species**: Non-native species outcompete native species for resources and habitat, altering ecosystems.

# **Step 2: Analyze the Impact of Each Threat**

# Each threat has significant impacts on ocean health:

# 1. **Pollution**: Harms marine life, contaminates the food chain, and affects human health.
# 2. **Overfishing and destructive fishing practices**: Reduces fish populations, damages ecosystems, and impacts the livelihoods of people dependent on fishing. 
# 3. **Climate change**: Causes coral bleaching, habitat loss, and changes in species distribution, leading to ecosystem disruption.
# 4. **Coastal development and habitat destruction**: Destroys habitats, reduces biodiversity, and increases the risk of natural disasters.
# 5. **Invasive species**: Alters ecosystems, outcompetes native species, and can lead to the extinction of endemic species.

# **Step 3: Understand the Mechanisms Behind the Threats**

# The mechanisms behind these threats are:

# 1. **Human activities**: Lack of waste management, industrial practices, and consumer behavior contribute to pollution.
# 2. **Economic and political drivers**: Overfishing and destructive fishing practices are driven by economic gain and lack of regulation.
# 3. **Climate change**: Greenhouse gas emissions from human activities, such as burning fossil fuels, accelerate climate change.
# 4. **Urbanization and development patterns**: Coastal development and habitat destruction are driven by urbanization and economic growth.

# **Step 4: Evaluate the Consequences of Inaction**

# If these threats are not addressed, the consequences will be severe:

# 1. **Irreversible damage to marine ecosystems**: Loss of biodiversity, ecosystem disruption, and extinction of species.
# 2. **Loss of biodiversity and extinction of species**: Reduced ecosystem resilience, decreased fisheries, and negative impacts on human health.
# 3. **Significant impacts on global food security and economies**: Reduced fisheries, decreased tourism, and negative impacts on human livelihoods.
# 4. **Enhanced vulnerability to natural disasters and climate change effects**: Increased risk of coastal erosion, flooding, and storm damage.

# **Step 5: Research Existing Solutions and Proposals**

# Existing solutions and proposals include:

# 1. **Renewable energy**: Reduces carbon emissions and mitigates climate change.
# 2. **Marine protected areas and sustainable fishing practices**: Protects habitats, reduces overfishing, and promotes ecosystem resilience.
# 3. **Innovations in waste management and reduction**: Reduces pollution, promotes recycling, and minimizes waste.
# 4. **International agreements and legal frameworks**: Regulates human activities, protects marine ecosystems, and promotes global cooperation.

# **Step 6: Compile and Present the Information**

# In conclusion, the primary threats to ocean health are pollution, overfishing and destructive fishing practices, climate change, coastal development and habitat destruction, and invasive species. These threats have significant impacts on marine ecosystems, human health, and the economy. Understanding the mechanisms behind these threats and evaluating the consequences of inaction highlights the need for urgent action. Existing solutions and proposals, such as renewable energy, marine protected areas, and innovations in waste management, can mitigate these threats. It is essential to promote global cooperation, regulate human activities, and protect marine ecosystems to ensure the long-term health of the oceans and life on Earth.

# **Summary:**

# * Primary threats to ocean health: pollution, overfishing, climate change, coastal development, and invasive species.
# * Impacts: harm to marine life, contamination of the food chain, ecosystem disruption, and negative impacts on human health and livelihoods.
# * Mechanisms: human activities, economic and political drivers, climate change, and urbanization.
# * Consequences of inaction: irreversible damage to marine ecosystems, loss of biodiversity, and significant impacts on global food security and economies.      
# * Existing solutions and proposals: renewable energy, marine protected areas, innovations in waste management, and international agreements.

# --- SHARED MEMORY ---
# - **Threats to Ocean Health and Why They Are Dangerous**

# The oceans, which cover over 70% of the Earth's surface, are essential to life on the planet. However, they are facing numerous threats that can have severe consequences for marine ecosystems, human health, and the economy. In this response, we will identify the primary threats to ocean health, analyze their impacts, understand the mechanisms behind them, evaluate the consequences of inaction, and research existing solutions and proposals.

# **Step 1: Identify Major Threats to Ocean Health**

# The primary threats to ocean health are:

# 1. **Pollution**: Chemical, plastic, and oil spills contaminate marine habitats and harm wildlife.
# 2. **Overfishing and destructive fishing practices**: Overfishing reduces fish populations, while destructive fishing practices damage habitats and ecosystems. 
# 3. **Climate change**: Rising sea temperatures cause coral bleaching and habitat loss, while ocean acidification affects marine life.
# 4. **Coastal development and habitat destruction**: Urbanization and development lead to the destruction of coastal habitats and ecosystems.
# 5. **Invasive species**: Non-native species outcompete native species for resources and habitat, altering ecosystems.

# **Step 2: Analyze the Impact of Each Threat**

# Each threat has significant impacts on ocean health:

# 1. **Pollution**: Harms marine life, contaminates the food chain, and affects human health.
# 2. **Overfishing and destructive fishing practices**: Reduces fish populations, damages ecosystems, and impacts the livelihoods of people dependent on fishing. 
# 3. **Climate change**: Causes coral bleaching, habitat loss, and changes in species distribution, leading to ecosystem disruption.
# 4. **Coastal development and habitat destruction**: Destroys habitats, reduces biodiversity, and increases the risk of natural disasters.
# 5. **Invasive species**: Alters ecosystems, outcompetes native species, and can lead to the extinction of endemic species.

# **Step 3: Understand the Mechanisms Behind the Threats**

# The mechanisms behind these threats are:

# 1. **Human activities**: Lack of waste management, industrial practices, and consumer behavior contribute to pollution.
# 2. **Economic and political drivers**: Overfishing and destructive fishing practices are driven by economic gain and lack of regulation.
# 3. **Climate change**: Greenhouse gas emissions from human activities, such as burning fossil fuels, accelerate climate change.
# 4. **Urbanization and development patterns**: Coastal development and habitat destruction are driven by urbanization and economic growth.

# **Step 4: Evaluate the Consequences of Inaction**

# If these threats are not addressed, the consequences will be severe:

# 1. **Irreversible damage to marine ecosystems**: Loss of biodiversity, ecosystem disruption, and extinction of species.
# 2. **Loss of biodiversity and extinction of species**: Reduced ecosystem resilience, decreased fisheries, and negative impacts on human health.
# 3. **Significant impacts on global food security and economies**: Reduced fisheries, decreased tourism, and negative impacts on human livelihoods.
# 4. **Enhanced vulnerability to natural disasters and climate change effects**: Increased risk of coastal erosion, flooding, and storm damage.

# **Step 5: Research Existing Solutions and Proposals**

# Existing solutions and proposals include:

# 1. **Renewable energy**: Reduces carbon emissions and mitigates climate change.
# 2. **Marine protected areas and sustainable fishing practices**: Protects habitats, reduces overfishing, and promotes ecosystem resilience.
# 3. **Innovations in waste management and reduction**: Reduces pollution, promotes recycling, and minimizes waste.
# 4. **International agreements and legal frameworks**: Regulates human activities, protects marine ecosystems, and promotes global cooperation.

# **Step 6: Compile and Present the Information**

# In conclusion, the primary threats to ocean health are pollution, overfishing and destructive fishing practices, climate change, coastal development and habitat destruction, and invasive species. These threats have significant impacts on marine ecosystems, human health, and the economy. Understanding the mechanisms behind these threats and evaluating the consequences of inaction highlights the need for urgent action. Existing solutions and proposals, such as renewable energy, marine protected areas, and innovations in waste management, can mitigate these threats. It is essential to promote global cooperation, regulate human activities, and protect marine ecosystems to ensure the long-term health of the oceans and life on Earth.

# **Summary:**

# * Primary threats to ocean health: pollution, overfishing, climate change, coastal development, and invasive species.
# * Impacts: harm to marine life, contamination of the food chain, ecosystem disruption, and negative impacts on human health and livelihoods.
# * Mechanisms: human activities, economic and political drivers, climate change, and urbanization.
# * Consequences of inaction: irreversible damage to marine ecosystems, loss of biodiversity, and significant impacts on global food security and economies.      
# * Existing solutions and proposals: renewable energy, marine protected areas, innovations in waste management, and international agreements.
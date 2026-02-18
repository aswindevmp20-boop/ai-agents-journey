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

embedder = SentenceTransformer("all-MiniLm-L6-v2")

def chunk_text(text):
    words = text.split()
    return[
        " ".join(words[i:i + CHUNK_SIZE])
        for i in range(0,len(words),CHUNK_SIZE)
    ]

def load_documents():
    chunks =[]
    for file in os.listdir(DOCS_PATH):
        if file.endswith(".txt"):
            with open(os.path.join(DOCS_PATH,file),"r",encoding="utf-8") as f:
                text = f.read()
            for chunk in chunk_text(text):
                chunks.append({
                    "file":file,
                    "content": chunk
                })
    return chunks

DOCUMENT_CHUNKS = load_documents()

embeddings = embedder.encode([c['content'] for c in DOCUMENT_CHUNKS], convert_to_numpy=True)

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
    Break the user's question into clear steps.

    user question: {user_query}

    Return a numbered plan.
    """

    response = client.chat.completions.create(
        model = "llama-3.3-70b-versatile",
        messages = [{"role":"user","content":prompt}]
    )

    return response.choices[0].message.content


def worker_agent(plan, user_query):
    retrieved_chunks = retrieve_chunks(user_query)

    context = "\n\n".join(
        f"[{c['file']}]\n{c['content']}" for c in retrieved_chunks
    )

    prompt = f"""
    You are a worker agent.
    Excute this plan:
    {plan}

    User the following context to answer:
    {context}

    Answer the user clearly.
    """

    response = client.chat.completions.create(
        model = "llama-3.3-70b-versatile",
        messages = [{"role":"user", "content":prompt}]
    )
    return response.choices[0].message.content


def run_agent(query):
    print("\n....Planner Output....")
    plan = planner_agent(query)
    print(plan)

    print("\n....Worker Output....")
    answer = worker_agent(plan, query)
    print("\nFinal Answer:\n", answer)

query = "What are the threats to ocean health and why are they dangerous?"
run_agent(query)


# Expected output:-

# ....Planner Output....
# To address the user's question, "What are the threats to ocean health and why are they dangerous?", I've broken down the inquiry into a step-by-step plan. Here's how we can approach this:

# 1. **Identify Major Threats**: First, we need to categorize the primary threats to ocean health. These include pollution, overfishing, climate change, coastal development, and marine debris.

# 2. **Research Pollution Effects**: Delve into how different types of pollution (chemical, plastic, oil spills) affect marine life and ecosystems, including the impact on coral reefs, marine mammals, and fish populations.

# 3. **Examine Overfishing and Its Consequences**: Investigate how overfishing depletes fish populations, damages marine ecosystems, and affects the livelihoods of people dependent on fishing industries.

# 4. **Understand Climate Change Impacts**: Study how climate change leads to ocean acidification, warming, and sea-level rise, affecting marine biodiversity, habitats, and the global food chain.

# 5. **Investigate Coastal Development**: Look into how coastal development (such as construction, dredging, and port expansion) destroys habitats, disrupts ecosystems, and impacts local communities.

# 6. **Analyze Marine Debris**: Research the sources, effects, and solutions to marine debris, focusing on plastic waste, its ingestion by marine life, and the role of microplastics in the food chain.

# 7. **Evaluate the Interconnectedness of Threats**: Consider how these threats interact and exacerbate each other's impacts on ocean health, such as how pollution and climate change together affect marine ecosystems more severely than either would alone.

# 8. **Assess Human and Environmental Risks**: Discuss the dangers these threats pose to human health (through tainted seafood, decreased livelihoods), the economy (via damaged fisheries and tourism), and the environment (through loss of biodiversity and ecosystem resilience).

# 9. **Explore Existing Solutions and Policies**: Look into current initiatives, technologies, and policies aimed at mitigating these threats, such as marine protected areas, sustainable fishing practices, and international agreements on pollution and climate change.

# 10. **Propose Future Actions**: Based on the analysis, suggest future steps that individuals, communities, and governments can take to protect ocean health, emphasizing the importance of concerted global action to address these interconnected issues.

# By following these steps, we can gain a comprehensive understanding of the threats to ocean health and why they are dangerous, as well as explore potential solutions to mitigate these impacts.

# ....Worker Output....

# Final Answer:
#  The threats to ocean health are multifaceted and interconnected, posing significant dangers to both the environment and human societies. Based on the provided context and the step-by-step plan outlined, let's break down the major threats and their implications:

# 1. **Pollution**: Chemical, plastic, and oil spill pollution contaminate marine habitats, harming wildlife. For example, plastic pollution is ingested by marine life, entering the food chain and potentially affecting human health through tainted seafood.

# 2. **Overfishing**: Overfishing depletes fish populations, damages marine ecosystems, and impacts the livelihoods of people dependent on fishing industries. This not only affects the economy but also the food security of communities that rely on fish as a primary source of protein.

# 3. **Climate Change**: Climate change leads to ocean acidification, warming, and sea-level rise, affecting marine biodiversity, habitats, and the global food chain. Coral bleaching and habitat loss are direct consequences of rising sea temperatures, further threatening the resilience of marine ecosystems.

# 4. **Coastal Development**: Coastal development destroys habitats, disrupts ecosystems, and impacts local communities. The construction, dredging, and expansion of ports can lead to the destruction of natural barriers, increasing the vulnerability of coastal areas to storms and sea-level rise.

# 5. **Marine Debris**: Marine debris, particularly plastic waste, is a significant problem. It is ingested by marine life, contributing to their deaths, and also breaks down into microplastics that enter the food chain, posing a risk to human health.

# The interconnectedness of these threats exacerbates their impacts. For instance, pollution and climate change together can have a more severe effect on marine ecosystems than either would alone. This synergistic effect can lead to unpredictable and potentially catastrophic consequences for ocean health.

# The risks posed by these threats are not limited to the environment; they also have significant implications for human health and the economy. Consuming seafood contaminated with pollutants can lead to health issues, while the decline of fisheries and tourism due to degraded marine environments can have economic repercussions.

# Existing solutions and policies include the establishment of marine protected areas, the implementation of sustainable fishing practices, and international agreements aimed at reducing pollution and addressing climate change. However, more needs to be done to protect ocean health.

# Future actions should include a concerted global effort to address these interconnected issues. Individuals can make a difference by reducing their use of plastics, supporting sustainable seafood, and advocating for climate action. Communities and governments can implement policies that protect marine habitats, enforce sustainable fishing practices, and invest in technologies that help mitigate the effects of pollution and climate change.

# In summary, the health of our oceans is under serious threat from pollution, overfishing, climate change, coastal development, and marine debris. These threats are interconnected and can have severe impacts on marine ecosystems, human health, and the economy. It is crucial that we take immediate and collective action to protect ocean health, ensuring the long-term sustainability of our planet.
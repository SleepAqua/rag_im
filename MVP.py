import chromadb
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
import requests
from bs4 import BeautifulSoup

SOURCES = [
    # Australia
    ("https://immi.homeaffairs.gov.au/visas/getting-a-visa/visa-listing/skilled-work-regional-provisional-491", "au_491"),
    ("https://immi.homeaffairs.gov.au/visas/getting-a-visa/visa-listing/skilled-independent-189", "au_189"),
    # New Zealand
    ("https://www.immigration.govt.nz/new-zealand-visas/apply-for-a-visa/about-visa/accredited-employer-work-visa", "nz_aewv"),
    # Netherlands  
    ("https://ind.nl/en/residence-permits/work/highly-skilled-migrant", "nl_hsm"),
]

# --- 1. Scrape a source ---
def scrape_page(url, filename):
    html = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}).text
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(separator="\n", strip=True)
    with open(f"data/{filename}.txt", "w") as f:
        f.write(text)

import os; os.makedirs("data", exist_ok=True)
scrape_page("https://immi.homeaffairs.gov.au/visas/getting-a-visa/visa-listing/skilled-work-regional-provisional-491", "au_491")

# --- 2. Index ---
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
Settings.llm = OpenAI(model="gpt-4o-mini")

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection("emigration")
vector_store = ChromaVectorStore(chroma_collection=collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

docs = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(docs, storage_context=storage_context)

# --- 3. Query ---
engine = index.as_query_engine()
response = engine.query("What is the points requirement for the 491 visa?")
print(response)
print("\nSources:")
for node in response.source_nodes:
    print(f"  score: {node.score:.3f} | {node.text[:120]}...")
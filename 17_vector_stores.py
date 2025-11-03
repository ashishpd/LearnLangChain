"""
17_vector_stores.py - Vector Stores

This tutorial demonstrates:
- What vector stores are
- Using Chroma (local vector database)
- Storing and retrieving vectors
- Similarity search operations

Prerequisites:
- Understanding of embeddings
- chromadb package installed
"""

import os
from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
import shutil

# Load environment variables
load_dotenv()

# Initialize embeddings
embeddings = AzureOpenAIEmbeddings(
    azure_deployment=os.environ.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME", 
                                    os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"]),
)

print("=== What are Vector Stores? ===")
print("""
Vector stores are databases that:
- Store embeddings (vectors) of documents
- Enable fast similarity search
- Support retrieval for RAG applications
- Can store metadata with vectors

Common vector stores:
- Chroma: Lightweight, local, good for development
- FAISS: Facebook AI Similarity Search, efficient
- Pinecone: Managed service, scalable
- Weaviate: Open-source, feature-rich
""")

print("=== Setting Up Chroma Vector Store ===")
# Create sample documents
documents = [
    Document(
        page_content="LangChain is a framework for building applications with LLMs.",
        metadata={"topic": "framework", "source": "intro"}
    ),
    Document(
        page_content="Embeddings convert text into numerical vector representations.",
        metadata={"topic": "embeddings", "source": "concepts"}
    ),
    Document(
        page_content="Vector stores enable efficient similarity search over documents.",
        metadata={"topic": "storage", "source": "concepts"}
    ),
    Document(
        page_content="RAG combines retrieval and generation for better LLM responses.",
        metadata={"topic": "rag", "source": "advanced"}
    ),
    Document(
        page_content="Python is a popular programming language for data science.",
        metadata={"topic": "programming", "source": "general"}
    ),
]

print(f"Created {len(documents)} sample documents\n")

# Create Chroma vector store
# persist_directory specifies where to store the database
persist_directory = "./chroma_db"

# Remove existing database if it exists
if os.path.exists(persist_directory):
    shutil.rmtree(persist_directory)

vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    persist_directory=persist_directory
)

print(f"Vector store created with {len(documents)} documents")
print(f"Database stored in: {persist_directory}\n")

print("=== Similarity Search ===")
# Search for similar documents
query = "How do I build applications with language models?"
results = vectorstore.similarity_search(query, k=2)

print(f"Query: {query}")
print(f"\nTop {len(results)} results:")
for i, doc in enumerate(results, 1):
    print(f"\n{i}. {doc.page_content}")
    print(f"   Metadata: {doc.metadata}")

print()

print("=== Similarity Search with Scores ===")
# Get similarity scores along with results
results_with_scores = vectorstore.similarity_search_with_score(query, k=3)

print(f"Query: {query}")
print(f"\nResults with similarity scores:")
for i, (doc, score) in enumerate(results_with_scores, 1):
    print(f"\n{i}. Score: {score:.4f}")
    print(f"   Content: {doc.page_content}")
    print(f"   Metadata: {doc.metadata}")

print()

print("=== Search by Metadata ===")
# Chroma supports metadata filtering
print("Searching for documents with topic='rag':")
rag_docs = vectorstore.similarity_search(
    "retrieval augmented generation",
    k=2,
    filter={"topic": "rag"}  # Filter by metadata
)

for doc in rag_docs:
    print(f"  - {doc.page_content}")

print()

print("=== Adding More Documents ===")
# Add new documents to existing vector store
new_documents = [
    Document(
        page_content="Machine learning models learn patterns from data.",
        metadata={"topic": "machine_learning", "source": "ai"}
    ),
]

vectorstore.add_documents(new_documents)
print(f"Added {len(new_documents)} new document(s)")
print(f"Total documents in store: {vectorstore._collection.count()}\n")

print("=== Retrieving Specific Documents ===")
# Retrieve documents by ID (if you stored IDs)
all_docs = vectorstore.get()
print(f"Total documents in store: {len(all_docs.get('ids', []))}\n")

print("=== Loading Existing Vector Store ===")
# Load a persisted vector store
print("Loading vector store from disk...")
loaded_vectorstore = Chroma(
    persist_directory=persist_directory,
    embedding_function=embeddings
)

print(f"Loaded vector store with {loaded_vectorstore._collection.count()} documents\n")

# Test search on loaded store
test_query = "What is a vector store?"
results = loaded_vectorstore.similarity_search(test_query, k=2)
print(f"Query: {test_query}")
print("Results from loaded store:")
for i, doc in enumerate(results, 1):
    print(f"  {i}. {doc.page_content}")

print()

print("=== Maximum Marginal Relevance (MMR) Search ===")
# MMR finds diverse results, not just most similar
print("Using MMR for diverse results:")
mmr_results = vectorstore.max_marginal_relevance_search(
    query="machine learning and AI",
    k=3,
    lambda_mult=0.5  # Balance between similarity and diversity (0=diverse, 1=similar)
)

print(f"MMR results for 'machine learning and AI':")
for i, doc in enumerate(mmr_results, 1):
    print(f"  {i}. {doc.page_content}")

print()

print("=== Vector Store Operations ===")
print("""
Common operations:
1. add_documents(): Add new documents
2. similarity_search(): Find similar documents
3. similarity_search_with_score(): Get similarity scores
4. max_marginal_relevance_search(): Diverse results
5. delete(): Remove documents by ID
6. update(): Update existing documents

Best practices:
- Use appropriate chunk sizes when adding documents
- Store useful metadata for filtering
- Periodically update/refresh documents
- Use MMR when you need diverse results
""")

print("=== Cleanup ===")
# Clean up the vector store directory
if os.path.exists(persist_directory):
    shutil.rmtree(persist_directory)
    print(f"Cleaned up {persist_directory}\n")

print("Tutorial complete! You've learned how to use vector stores with LangChain.")


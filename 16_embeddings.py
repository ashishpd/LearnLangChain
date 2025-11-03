"""
16_embeddings.py - Understanding Embeddings

This tutorial demonstrates:
- What embeddings are
- Using OpenAI embeddings (Azure compatible)
- Embedding vectors
- Similarity calculations

Prerequisites:
- Azure OpenAI setup
- Understanding of documents
"""

import os
from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings
import numpy as np

# Load environment variables
load_dotenv()

print("=== What are Embeddings? ===")
print("""
Embeddings are:
- Vector representations of text
- Numerical arrays that capture semantic meaning
- Used for similarity search and retrieval
- The foundation of RAG (Retrieval Augmented Generation)
""")

print("=== Initializing Azure OpenAI Embeddings ===")
# Initialize embeddings model
# Note: You may need a separate embedding deployment in Azure OpenAI
embeddings = AzureOpenAIEmbeddings(
    azure_deployment=os.environ.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME", 
                                    os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"]),
)

print("Embeddings model initialized\n")

print("=== Basic Embedding Generation ===")
# Generate embeddings for text
text = "LangChain is a framework for building LLM applications."
embedding = embeddings.embed_query(text)

print(f"Text: {text}")
print(f"Embedding dimensions: {len(embedding)}")
print(f"First 10 values: {embedding[:10]}")
print(f"Embedding type: {type(embedding)}\n")

print("=== Embedding Multiple Texts ===")
# Generate embeddings for multiple texts at once
texts = [
    "Python is a programming language",
    "Machine learning uses algorithms",
    "LangChain helps build AI applications",
    "Embeddings represent text as vectors",
]

# Embed multiple texts (more efficient than individual calls)
text_embeddings = embeddings.embed_documents(texts)

print(f"Number of texts: {len(texts)}")
print(f"Number of embeddings: {len(text_embeddings)}")
print(f"Each embedding dimension: {len(text_embeddings[0])}")
print()

print("=== Similarity Between Embeddings ===")
# Calculate similarity using cosine similarity
def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors"""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot_product / (norm1 * norm2)

# Embed two similar texts
text1 = "Python programming language"
text2 = "Python is a coding language"
text3 = "Cooking recipes and ingredients"

embed1 = embeddings.embed_query(text1)
embed2 = embeddings.embed_query(text2)
embed3 = embeddings.embed_query(text3)

similarity_12 = cosine_similarity(embed1, embed2)
similarity_13 = cosine_similarity(embed1, embed3)

print(f"Text 1: {text1}")
print(f"Text 2: {text2}")
print(f"Similarity (1-2): {similarity_12:.4f}\n")

print(f"Text 1: {text1}")
print(f"Text 3: {text3}")
print(f"Similarity (1-3): {similarity_13:.4f}\n")

print("Note: Higher similarity means texts are more semantically similar\n")

print("=== Finding Most Similar Text ===")
# Find which text is most similar to a query
query = "programming with Python"
query_embedding = embeddings.embed_query(query)

print(f"Query: {query}")
print(f"\nComparing with texts:")

similarities = []
for i, text in enumerate(texts):
    similarity = cosine_similarity(query_embedding, text_embeddings[i])
    similarities.append((text, similarity))
    print(f"  {i+1}. '{text}': {similarity:.4f}")

# Sort by similarity
similarities.sort(key=lambda x: x[1], reverse=True)
print(f"\nMost similar: '{similarities[0][0]}' (similarity: {similarities[0][1]:.4f})\n")

print("=== Embeddings for Document Chunks ===")
# Embedding document chunks for RAG
document_chunks = [
    "LangChain provides tools for building LLM applications.",
    "Embeddings convert text into numerical vectors.",
    "Vector stores enable semantic search capabilities.",
    "RAG combines retrieval with generation for better answers.",
]

chunk_embeddings = embeddings.embed_documents(document_chunks)

print(f"Embedded {len(document_chunks)} document chunks")
print(f"Each chunk embedding dimension: {len(chunk_embeddings[0])}\n")

# Example: Find relevant chunks for a query
user_query = "How do I search through documents?"
query_embed = embeddings.embed_query(user_query)

chunk_similarities = []
for i, chunk in enumerate(document_chunks):
    similarity = cosine_similarity(query_embed, chunk_embeddings[i])
    chunk_similarities.append((chunk, similarity))

# Sort by similarity (most relevant first)
chunk_similarities.sort(key=lambda x: x[1], reverse=True)

print(f"Query: {user_query}")
print("\nRelevant chunks (sorted by similarity):")
for i, (chunk, sim) in enumerate(chunk_similarities, 1):
    print(f"  {i}. [{sim:.4f}] {chunk}")
print()

print("=== Embedding Characteristics ===")
print("""
Embedding properties:
- Fixed dimension (e.g., 1536 for OpenAI text-embedding-ada-002)
- Normalized vectors (typically)
- Semantic similarity = high cosine similarity
- Context-aware (same word has different embeddings in different contexts)

Use cases:
- Semantic search
- Document retrieval
- Similarity matching
- Clustering and classification
""")

print("=== Batch Embedding ===")
# Embeddings API supports batching for efficiency
large_text_list = [
    f"Document {i} about topic {i % 3}" for i in range(10)
]

print(f"Embedding {len(large_text_list)} texts in batch...")
batch_embeddings = embeddings.embed_documents(large_text_list)
print(f"Generated {len(batch_embeddings)} embeddings")
print(f"Each embedding has {len(batch_embeddings[0])} dimensions\n")

print("=== Embedding Best Practices ===")
print("""
1. Use batch embedding for multiple texts (more efficient)
2. Normalize embeddings if needed for your use case
3. Store embeddings with document metadata
4. Consider embedding dimension vs. model performance
5. Use appropriate embedding model for your domain
6. Cache embeddings when possible to save costs
""")

print("Tutorial complete! You've learned about embeddings and how to use them.")


"""
15_text_splitters.py - Text Splitters

This tutorial demonstrates:
- Why text splitting is important
- Different text splitting strategies
- Chunk sizes and overlaps
- Document chunking for vector stores

Prerequisites:
- Understanding of document loaders
"""

import os
from dotenv import load_dotenv
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    TokenTextSplitter,
)
from langchain.schema import Document

# Load environment variables
load_dotenv()

print("=== Why Split Text? ===")
print("""
Text splitting is important because:
1. LLMs have token limits
2. Vector stores work better with smaller chunks
3. Improves retrieval accuracy
4. Allows parallel processing
5. Better semantic search results
""")

print("=== Creating Sample Long Document ===")
# Create a longer document for splitting demonstration
long_text = """
LangChain is a powerful framework for building applications with large language models.
It provides comprehensive tools for developers to integrate LLM capabilities into their applications.

The framework supports various components:
1. Prompts and Prompt Templates: Manage and structure prompts efficiently
2. Chains: Connect multiple LLM calls together for complex workflows
3. Agents: Build intelligent agents that can use tools and make decisions
4. Memory: Maintain conversation state and context
5. Document Loaders: Load documents from various sources
6. Vector Stores: Store and retrieve embeddings for RAG applications
7. Embeddings: Convert text to vector representations

LangChain makes it easy to build production-ready LLM applications.
You can use it with various LLM providers like OpenAI, Azure OpenAI, Anthropic, and more.
The framework is designed to be modular and extensible.

Text splitting is crucial when working with long documents.
It helps break down large texts into manageable chunks that fit within token limits.
Proper chunking also improves the quality of semantic search in vector databases.
"""

# Create document
doc = Document(page_content=long_text, metadata={"source": "tutorial"})

print(f"Original document length: {len(long_text)} characters")
print(f"Original document word count: {len(long_text.split())} words\n")

print("=== RecursiveCharacterTextSplitter (Recommended) ===")
# RecursiveCharacterTextSplitter is the most commonly used splitter
# It tries to split on paragraphs, then sentences, then words

splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,  # Maximum characters per chunk
    chunk_overlap=50,  # Overlap between chunks (helps maintain context)
    length_function=len,  # Function to measure length
)

chunks = splitter.split_text(long_text)
print(f"Number of chunks: {len(chunks)}")
print("\nChunks:")
for i, chunk in enumerate(chunks, 1):
    print(f"  Chunk {i} ({len(chunk)} chars): {chunk[:80]}...")
print()

print("=== Splitting Documents (Preserving Metadata) ===")
# When splitting documents, metadata is preserved
splitter = RecursiveCharacterTextSplitter(
    chunk_size=150,
    chunk_overlap=30,
)

doc_chunks = splitter.split_documents([doc])
print(f"Number of document chunks: {len(doc_chunks)}")
print("\nDocument chunks with metadata:")
for i, chunk in enumerate(doc_chunks, 1):
    print(f"  Chunk {i}:")
    print(f"    Content: {chunk.page_content[:60]}...")
    print(f"    Metadata: {chunk.metadata}")
print()

print("=== CharacterTextSplitter (Simple) ===")
# CharacterTextSplitter splits on a specific separator
simple_splitter = CharacterTextSplitter(
    separator=".",
    chunk_size=100,
    chunk_overlap=20,
)

simple_chunks = simple_splitter.split_text(long_text)
print(f"Simple splitter chunks: {len(simple_chunks)}")
for i, chunk in enumerate(simple_chunks[:3], 1):
    print(f"  Chunk {i}: {chunk[:60]}...")
print()

print("=== Custom Separators ===")
# You can customize the separators for recursive splitting
custom_splitter = RecursiveCharacterTextSplitter(
    chunk_size=180,
    chunk_overlap=40,
    separators=["\n\n", "\n", ". ", " ", ""]  # Try these separators in order
)

custom_chunks = custom_splitter.split_text(long_text)
print(f"Custom splitter chunks: {len(custom_chunks)}")
print()

print("=== Chunk Size and Overlap Explained ===")
print("""
chunk_size: Maximum size of each chunk
- Too small: May lose context
- Too large: May exceed token limits
- Recommended: 1000-2000 characters for general use

chunk_overlap: Characters to overlap between chunks
- Helps maintain context across boundaries
- Prevents important information from being split
- Recommended: 10-20% of chunk_size
""")

print("=== Example: Optimal Chunking Strategy ===")
# For RAG applications, optimal chunking is important
optimal_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # Good size for embeddings
    chunk_overlap=200,  # 20% overlap
    length_function=len,
)

optimal_chunks = optimal_splitter.split_text(long_text)
print(f"Optimal chunks: {len(optimal_chunks)}")
print(f"Average chunk size: {sum(len(c) for c in optimal_chunks) / len(optimal_chunks):.0f} chars")
print(f"Chunk size range: {min(len(c) for c in optimal_chunks)} - {max(len(c) for c in optimal_chunks)} chars\n")

print("=== Token-Based Splitting (Advanced) ===")
print("""
TokenTextSplitter splits based on tokens rather than characters.
Useful when you need precise token limits.

Note: Requires tiktoken package for OpenAI models
""")

print("=== Chunking Best Practices ===")
print("""
1. Use RecursiveCharacterTextSplitter for most cases
2. Set chunk_size based on your model's context window
3. Use 10-20% overlap to maintain context
4. Consider document structure (paragraphs, sections)
5. Test different chunk sizes for your use case
6. Preserve metadata when splitting documents
7. Consider document type (code vs. prose may need different strategies)
""")

print("=== Multiple Document Splitting ===")
# Split multiple documents at once
documents = [
    Document(page_content="First document about machine learning.", metadata={"id": 1}),
    Document(page_content="Second document about deep learning and neural networks.", metadata={"id": 2}),
    Document(page_content="Third document about natural language processing.", metadata={"id": 3}),
]

multi_splitter = RecursiveCharacterTextSplitter(
    chunk_size=50,
    chunk_overlap=10,
)

all_chunks = multi_splitter.split_documents(documents)
print(f"Split {len(documents)} documents into {len(all_chunks)} chunks")
print("\nSample chunks:")
for i, chunk in enumerate(all_chunks[:3], 1):
    print(f"  Chunk {i} (from doc {chunk.metadata.get('id', 'unknown')}): {chunk.page_content}")
print()

print("Tutorial complete! You've learned how to split text for LangChain applications.")


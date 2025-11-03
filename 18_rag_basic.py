"""
18_rag_basic.py - Basic RAG Implementation

This tutorial demonstrates:
- What RAG (Retrieval Augmented Generation) is
- Building a basic RAG system
- Retrieval + Generation workflow
- Simple Q&A system

Prerequisites:
- Understanding of embeddings, vector stores, and LLMs
"""

import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document
import shutil

# Load environment variables
load_dotenv()

# Initialize LLM and embeddings
llm = AzureChatOpenAI(
    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
    temperature=0.7,
)

embeddings = AzureOpenAIEmbeddings(
    azure_deployment=os.environ.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME", 
                                    os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"]),
)

print("=== What is RAG? ===")
print("""
RAG (Retrieval Augmented Generation):
1. Retrieval: Find relevant documents from a knowledge base
2. Augmentation: Add retrieved context to the prompt
3. Generation: LLM generates answer using the context

Benefits:
- More accurate answers (based on your documents)
- Up-to-date information (without retraining)
- Traceable sources (can cite documents)
""")

print("=== Creating Knowledge Base ===")
# Create sample documents as knowledge base
documents = [
    Document(
        page_content="LangChain is a framework for building applications with large language models. "
        "It provides tools for prompt management, chains, agents, and document processing.",
        metadata={"source": "langchain_intro"}
    ),
    Document(
        page_content="Embeddings are vector representations of text that capture semantic meaning. "
        "They enable similarity search and are used in RAG applications.",
        metadata={"source": "embeddings_guide"}
    ),
    Document(
        page_content="Vector stores like Chroma store document embeddings and enable fast similarity search. "
        "They are essential for building RAG systems.",
        metadata={"source": "vector_stores"}
    ),
    Document(
        page_content="RAG combines retrieval of relevant documents with LLM generation. "
        "This improves answer quality by providing context to the model.",
        metadata={"source": "rag_explanation"}
    ),
    Document(
        page_content="Python is a high-level programming language known for its simplicity and readability. "
        "It's widely used in data science, web development, and AI applications.",
        metadata={"source": "python_basics"}
    ),
]

print(f"Created knowledge base with {len(documents)} documents\n")

print("=== Setting Up Vector Store ===")
# Create vector store from documents
persist_directory = "./rag_chroma_db"

# Clean up if exists
if os.path.exists(persist_directory):
    shutil.rmtree(persist_directory)

vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    persist_directory=persist_directory
)

print(f"Vector store created with {len(documents)} documents\n")

print("=== Basic RAG with RetrievalQA ===")
# RetrievalQA chain combines retrieval and QA
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",  # "stuff" means put all retrieved docs in prompt
    retriever=vectorstore.as_retriever(search_kwargs={"k": 2}),  # Retrieve top 2 docs
    return_source_documents=True,  # Return source docs for citation
    verbose=True,
)

print("=== Example Questions ===")
questions = [
    "What is LangChain?",
    "How do embeddings work?",
    "What is RAG?",
]

for question in questions:
    print(f"\n{'='*60}")
    print(f"Question: {question}")
    print('='*60)
    
    result = qa_chain({"query": question})
    
    print(f"\nAnswer: {result['result']}")
    print(f"\nSources ({len(result['source_documents'])}):")
    for i, doc in enumerate(result['source_documents'], 1):
        print(f"  {i}. {doc.page_content[:80]}...")
        print(f"     Source: {doc.metadata.get('source', 'unknown')}")

print()

print("=== Custom RAG Prompt ===")
# Customize the prompt for better answers
custom_prompt_template = """Use the following pieces of context to answer the question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}

Question: {question}

Provide a detailed answer based on the context above:"""

PROMPT = PromptTemplate(
    template=custom_prompt_template,
    input_variables=["context", "question"]
)

custom_qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    chain_type_kwargs={"prompt": PROMPT},
    return_source_documents=True,
)

print("Testing custom prompt:")
result = custom_qa_chain({"query": "Explain how RAG improves LLM responses"})
print(f"Answer: {result['result']}\n")

print("=== RAG Components Breakdown ===")
print("""
RAG System Components:

1. Document Loader: Load documents into the system
2. Text Splitter: Split documents into chunks
3. Embeddings: Convert chunks to vectors
4. Vector Store: Store and index vectors
5. Retriever: Find relevant chunks for a query
6. LLM: Generate answer using retrieved context

Flow:
Query → Embed Query → Search Vector Store → Retrieve Relevant Chunks 
→ Combine with Prompt → LLM Generation → Answer
""")

print("=== Retrieval Strategies ===")
print("""
Different retrieval strategies:

1. Similarity Search: Find most similar documents
   - Simple and effective
   - Good for most cases

2. MMR (Max Marginal Relevance): Diverse results
   - Prevents redundant information
   - Good when you need varied perspectives

3. Metadata Filtering: Filter by metadata
   - Narrow search scope
   - Good for domain-specific queries
""")

# Example with MMR
print("\n=== MMR Retrieval Example ===")
mmr_retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3, "lambda_mult": 0.5}
)

mmr_qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=mmr_retriever,
    return_source_documents=True,
)

result = mmr_qa({"query": "What are the key components of LangChain and RAG?"})
print(f"Question: What are the key components of LangChain and RAG?")
print(f"Answer: {result['result'][:200]}...\n")

print("=== RAG Best Practices ===")
print("""
1. Chunk size: Use 1000-2000 characters for good balance
2. Retrieval count: Retrieve 3-5 documents typically
3. Prompt engineering: Clearly instruct the model to use context
4. Source citation: Always return source documents
5. Handling no results: Instruct model what to do if no relevant docs found
6. Metadata: Store useful metadata for filtering
7. Update knowledge base: Keep documents up to date
""")

print("=== Cleanup ===")
if os.path.exists(persist_directory):
    shutil.rmtree(persist_directory)
    print(f"Cleaned up {persist_directory}\n")

print("Tutorial complete! You've learned how to build a basic RAG system.")


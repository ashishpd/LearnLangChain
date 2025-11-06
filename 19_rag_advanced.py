"""
19_rag_advanced.py - Advanced RAG Patterns

This tutorial demonstrates:
- Advanced RAG patterns
- Multi-document RAG
- RAG with citations
- Context compression
- Parent document retrieval

Prerequisites:
- Understanding of basic RAG
"""

import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain.schema import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
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

print("=== Advanced RAG Patterns ===")
print("""
Advanced techniques:
1. Context compression
2. Re-ranking results
3. Multi-hop retrieval
4. Parent document retrieval
5. Query expansion
6. Citation and source tracking
""")

print("=== Creating Rich Knowledge Base ===")
# Create detailed documents with structure
documents = [
    Document(
        page_content="LangChain Framework Overview: "
        "LangChain is a comprehensive framework for building LLM applications. "
        "It provides modular components including prompts, chains, agents, memory, "
        "and document processing. The framework supports multiple LLM providers and "
        "is designed for production use.",
        metadata={"source": "langchain_docs", "section": "overview", "page": 1}
    ),
    Document(
        page_content="LangChain Chains: "
        "Chains allow you to combine multiple LLM calls and tools. "
        "Types include LLMChain, SequentialChain, and RouterChain. "
        "Chains enable complex workflows and multi-step reasoning.",
        metadata={"source": "langchain_docs", "section": "chains", "page": 2}
    ),
    Document(
        page_content="LangChain Agents: "
        "Agents use LLMs to decide which tools to use and in what order. "
        "They can handle dynamic workflows and adapt to user needs. "
        "Common agent types include ReAct and conversational agents.",
        metadata={"source": "langchain_docs", "section": "agents", "page": 3}
    ),
    Document(
        page_content="RAG Implementation: "
        "RAG combines document retrieval with LLM generation. "
        "Steps include: document loading, chunking, embedding, "
        "vector storage, retrieval, and generation. "
        "This improves answer accuracy by providing relevant context.",
        metadata={"source": "rag_guide", "section": "implementation", "page": 1}
    ),
    Document(
        page_content="Embeddings and Vector Stores: "
        "Embeddings convert text to vectors capturing semantic meaning. "
        "Vector stores like Chroma enable fast similarity search. "
        "This is the foundation of retrieval in RAG systems.",
        metadata={"source": "rag_guide", "section": "embeddings", "page": 2}
    ),
]

print(f"Created knowledge base with {len(documents)} documents\n")

# Create vector store
persist_directory = "./advanced_rag_db"
if os.path.exists(persist_directory):
    shutil.rmtree(persist_directory)

vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    persist_directory=persist_directory
)

print("=== RAG with Detailed Citations ===")
# Custom prompt that emphasizes citations
citation_prompt = """Answer the question based on the provided context.
For each fact or claim you make, cite the source document.

Context from documents:
{context}

Question: {question}

Answer with citations (format: [Source: document_name, section: section_name]):"""

CITATION_PROMPT = PromptTemplate(
    template=citation_prompt,
    input_variables=["context", "question"]
)

# Helper function to format documents
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Create retriever and RAG chain
citation_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

qa_with_citations = (
    {
        "context": citation_retriever | format_docs,
        "question": RunnablePassthrough(),
    }
    | CITATION_PROMPT
    | llm
    | StrOutputParser()
)

print("Testing citation-based RAG:")
question = "How do chains and agents work in LangChain?"
answer = qa_with_citations.invoke(question)
source_documents = citation_retriever.invoke(question)

print(f"Question: {question}")
print(f"\nAnswer: {answer}")
print(f"\nSource documents:")
for i, doc in enumerate(source_documents, 1):
    print(f"  {i}. Section: {doc.metadata.get('section', 'N/A')}, "
          f"Page: {doc.metadata.get('page', 'N/A')}")
    print(f"     Content: {doc.page_content[:100]}...")
print()

print("=== Multi-Step Retrieval (Conceptual) ===")
# Simulate multi-hop retrieval
print("""
Multi-hop retrieval involves:
1. Initial query retrieval
2. Extract key entities/concepts from results
3. Use those to retrieve additional relevant documents
4. Combine all context for final answer
""")

# Example: First retrieve about LangChain, then about RAG
print("Step 1: Retrieve documents about LangChain")
langchain_docs = vectorstore.similarity_search("LangChain framework components", k=2)
print(f"Retrieved {len(langchain_docs)} documents about LangChain")

print("\nStep 2: Extract concepts and retrieve related RAG docs")
rag_docs = vectorstore.similarity_search("RAG implementation with embeddings", k=2)
print(f"Retrieved {len(rag_docs)} documents about RAG")

print("\nStep 3: Combine context")
combined_context = "\n\n".join([
    doc.page_content for doc in langchain_docs + rag_docs
])
print(f"Combined context length: {len(combined_context)} characters\n")

print("=== Context-Aware Prompting ===")
# Prompt that uses document structure
structured_prompt = """You are an expert assistant answering questions based on document sections.

Available document sections:
{context}

Question: {question}

Instructions:
1. Identify which sections are relevant
2. Synthesize information from multiple sections if needed
3. Be specific about which concepts come from which sections
4. If information is incomplete, say so

Answer:"""

STRUCTURED_PROMPT = PromptTemplate(
    template=structured_prompt,
    input_variables=["context", "question"]
)

structured_retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

structured_qa = (
    {
        "context": structured_retriever | format_docs,
        "question": RunnablePassthrough(),
    }
    | STRUCTURED_PROMPT
    | llm
    | StrOutputParser()
)

question = "Explain the relationship between LangChain components and RAG"
answer = structured_qa.invoke(question)
print(f"Question: {question}")
print(f"\nAnswer: {answer[:300]}...\n")

print("=== Metadata Filtering for Targeted Retrieval ===")
# Filter by metadata to narrow search
print("Retrieving only from 'langchain_docs' source:")
filtered_retriever = vectorstore.as_retriever(
    search_kwargs={"k": 2, "filter": {"source": "langchain_docs"}}
)

# Create prompt for filtered QA
filtered_prompt = PromptTemplate(
    template="""Answer the question based on the provided context.

Context: {context}

Question: {question}

Answer:""",
    input_variables=["context", "question"]
)

filtered_qa = (
    {
        "context": filtered_retriever | format_docs,
        "question": RunnablePassthrough(),
    }
    | filtered_prompt
    | llm
    | StrOutputParser()
)

answer = filtered_qa.invoke("What are the main components of LangChain?")
print(f"Answer: {answer}\n")

print("=== Advanced RAG Patterns Summary ===")
print("""
1. Citation Tracking:
   - Include source metadata in prompts
   - Return source documents with answers
   - Format citations clearly

2. Multi-Hop Retrieval:
   - Chain multiple retrieval steps
   - Use intermediate results to refine queries
   - Combine contexts from different sources

3. Context Compression:
   - Use only relevant parts of retrieved documents
   - Implement re-ranking of results
   - Filter by relevance scores

4. Query Expansion:
   - Generate multiple query variations
   - Retrieve for each variation
   - Combine results

5. Parent Document Retrieval:
   - Store small chunks for retrieval
   - Return larger parent documents for context
   - Balance between precision and context
""")

print("=== Best Practices for Advanced RAG ===")
print("""
1. Document Structure:
   - Organize documents with clear sections
   - Add meaningful metadata
   - Use consistent formatting

2. Retrieval Strategy:
   - Start with similarity search
   - Add metadata filters when appropriate
   - Consider MMR for diverse results

3. Prompt Engineering:
   - Clearly instruct model to use context
   - Specify citation format
   - Handle cases where answer isn't in context

4. Evaluation:
   - Test with various query types
   - Check citation accuracy
   - Monitor retrieval quality

5. Performance:
   - Balance retrieval count vs. context size
   - Consider async operations for scale
   - Cache embeddings when possible
""")

print("=== Cleanup ===")
if os.path.exists(persist_directory):
    shutil.rmtree(persist_directory)
    print(f"Cleaned up {persist_directory}\n")

print("Tutorial complete! You've learned advanced RAG patterns and techniques.")


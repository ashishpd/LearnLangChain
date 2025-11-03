"""
30_complete_app.py - Complete Application Example

This tutorial demonstrates:
- Combining multiple LangChain concepts
- Building an end-to-end application
- Best practices integration
- Production-ready patterns

Prerequisites:
- Understanding of all previous tutorials
"""

import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Dict
import shutil

# Load environment variables
load_dotenv()

print("=== Complete Application: Document Q&A System ===")
print("""
This example combines:
- Document loading and processing
- Text splitting
- Embeddings and vector stores
- RAG (Retrieval Augmented Generation)
- Error handling
- Production patterns
""")

# Initialize components
llm = AzureChatOpenAI(
    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
    temperature=0.7,
)

embeddings = AzureOpenAIEmbeddings(
    azure_deployment=os.environ.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME", 
                                    os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"]),
)

print("=== Step 1: Document Management ===")
# Create knowledge base documents
documents_data = [
    {
        "content": """
        LangChain is a framework for building applications with large language models.
        It provides comprehensive tools for developers including:
        - Prompt templates and management
        - Chains for connecting multiple operations
        - Agents that can use tools and make decisions
        - Memory for maintaining conversation state
        - Document loaders for various sources
        - Vector stores for semantic search
        """,
        "metadata": {"source": "langchain_intro", "topic": "framework"}
    },
    {
        "content": """
        Retrieval Augmented Generation (RAG) combines document retrieval with LLM generation.
        The process involves:
        1. Loading and chunking documents
        2. Creating embeddings for chunks
        3. Storing in a vector database
        4. Retrieving relevant chunks for queries
        5. Generating answers using retrieved context
        """,
        "metadata": {"source": "rag_guide", "topic": "rag"}
    },
    {
        "content": """
        Agents in LangChain use LLMs to decide which tools to use and in what order.
        Types of agents include:
        - Zero-shot agents: No memory, single-turn
        - Conversational agents: With memory for multi-turn
        - ReAct agents: Reasoning and acting agents
        Agents can use custom tools to interact with external systems.
        """,
        "metadata": {"source": "agents_guide", "topic": "agents"}
    },
    {
        "content": """
        Vector stores enable semantic search over documents.
        They store embeddings (vector representations) of text.
        When you query, the store finds similar embeddings.
        Popular vector stores include Chroma, FAISS, Pinecone, and Weaviate.
        """,
        "metadata": {"source": "vector_stores", "topic": "embeddings"}
    },
]

# Convert to Document objects
documents = [
    Document(page_content=doc["content"], metadata=doc["metadata"])
    for doc in documents_data
]

print(f"Created {len(documents)} documents\n")

print("=== Step 2: Text Splitting ===")
# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
)

split_documents = text_splitter.split_documents(documents)
print(f"Split into {len(split_documents)} chunks\n")

print("=== Step 3: Vector Store Setup ===")
# Create vector store
persist_directory = "./complete_app_db"

# Clean up if exists
if os.path.exists(persist_directory):
    shutil.rmtree(persist_directory)

vectorstore = Chroma.from_documents(
    documents=split_documents,
    embedding=embeddings,
    persist_directory=persist_directory,
)

print(f"Vector store created with {len(split_documents)} chunks\n")

print("=== Step 4: RAG Chain Setup ===")
# Create RAG chain with custom prompt
custom_prompt = """Use the following pieces of context to answer the question.
If you don't know the answer based on the context, say so.
Provide a clear, detailed answer.

Context: {context}

Question: {question}

Answer:"""

PROMPT = ChatPromptTemplate(
    template=custom_prompt,
    input_variables=["context", "question"]
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    chain_type_kwargs={"prompt": PROMPT},
    return_source_documents=True,
)

print("RAG chain created\n")

print("=== Step 5: Application Class ===")
# Create complete application class

class DocumentQAApp:
    """Complete document Q&A application"""
    
    def __init__(self, llm, embeddings, persist_directory: str = "./qa_app_db"):
        self.llm = llm
        self.embeddings = embeddings
        self.persist_directory = persist_directory
        self.vectorstore = None
        self.qa_chain = None
    
    def initialize(self, documents: List[Document]):
        """Initialize with documents"""
        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
        )
        chunks = text_splitter.split_documents(documents)
        
        # Create vector store
        if os.path.exists(self.persist_directory):
            shutil.rmtree(self.persist_directory)
        
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.persist_directory,
        )
        
        # Create QA chain
        prompt = ChatPromptTemplate(
            template="""Answer the question based on the context.
            
Context: {context}

Question: {question}

Provide a detailed answer:""",
            input_variables=["context", "question"]
        )
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 3}),
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True,
        )
        
        print(f"Application initialized with {len(chunks)} document chunks")
    
    def ask(self, question: str) -> Dict:
        """Ask a question"""
        if not self.qa_chain:
            return {"error": "Application not initialized"}
        
        try:
            result = self.qa_chain({"query": question})
            return {
                "question": question,
                "answer": result["result"],
                "sources": [
                    {
                        "content": doc.page_content[:100] + "...",
                        "metadata": doc.metadata
                    }
                    for doc in result["source_documents"]
                ]
            }
        except Exception as e:
            return {"error": str(e)}
    
    def add_documents(self, documents: List[Document]):
        """Add new documents"""
        if not self.vectorstore:
            self.initialize(documents)
            return
        
        # Split and add
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
        )
        chunks = text_splitter.split_documents(documents)
        self.vectorstore.add_documents(chunks)
        print(f"Added {len(chunks)} new chunks")

# Initialize application
app = DocumentQAApp(llm, embeddings)
app.initialize(documents)

print("\n=== Step 6: Using the Application ===")
# Test the application
questions = [
    "What is LangChain?",
    "How does RAG work?",
    "What are agents?",
]

print("Testing the application:")
for question in questions:
    print(f"\n{'='*60}")
    print(f"Question: {question}")
    print('='*60)
    
    result = app.ask(question)
    
    if "error" in result:
        print(f"Error: {result['error']}")
    else:
        print(f"\nAnswer: {result['answer']}")
        print(f"\nSources ({len(result['sources'])}):")
        for i, source in enumerate(result['sources'], 1):
            print(f"  {i}. {source['content']}")
            print(f"     Topic: {source['metadata'].get('topic', 'N/A')}")

print("\n=== Application Features Summary ===")
print("""
This application demonstrates:

1. Document Processing:
   - Loading and splitting documents
   - Managing document metadata

2. Vector Storage:
   - Creating embeddings
   - Storing in vector database
   - Efficient retrieval

3. RAG Implementation:
   - Retrieving relevant context
   - Generating answers with context
   - Source citation

4. Application Structure:
   - Clean class design
   - Error handling
   - Extensible architecture

5. Production Readiness:
   - Persistent storage
   - Structured outputs
   - Source tracking
""")

print("\n=== Extending the Application ===")
print("""
Possible enhancements:
- Add web interface
- Implement conversation memory
- Add user authentication
- Support multiple document types
- Add query history
- Implement caching
- Add monitoring and logging
- Support batch queries
""")

print("\n=== Cleanup ===")
if os.path.exists(persist_directory):
    shutil.rmtree(persist_directory)
    print(f"Cleaned up {persist_directory}\n")

print("=== Tutorial Series Complete! ===")
print("""
Congratulations! You've completed all 30 tutorials covering:
- Basics of LangChain
- Prompts and chains
- Memory and state
- Agents and tools
- Document processing
- RAG systems
- Advanced patterns
- Production practices
- Integration examples

You now have the knowledge to build sophisticated LLM applications with LangChain!
""")

print("Tutorial complete! You've built a complete LangChain application.")


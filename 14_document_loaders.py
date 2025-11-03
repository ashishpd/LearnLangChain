"""
14_document_loaders.py - Document Loaders

This tutorial demonstrates:
- Loading documents from various sources
- Different document loader types
- File processing (PDF, text, etc.)
- Web scraping for documents

Prerequisites:
- Basic LangChain understanding
- Documents to load (or we'll create sample ones)
"""

import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.schema import Document

# Load environment variables
load_dotenv()

print("=== What are Document Loaders? ===")
print("""
Document loaders help you:
- Load documents from various sources
- Convert documents into LangChain Document objects
- Extract text and metadata
- Handle different file formats
""")

print("=== Creating Sample Documents ===")
# Create sample text files for demonstration
sample_text = """
LangChain is a framework for building applications with large language models.
It provides tools for:
- Prompt management
- Chains and agents
- Memory and state management
- Document processing
- Vector stores and embeddings
"""

# Save sample text to file
with open("sample_document.txt", "w") as f:
    f.write(sample_text)

print("Created sample_document.txt\n")

print("=== Loading Text Files ===")
# Load text file using TextLoader
try:
    loader = TextLoader("sample_document.txt")
    documents = loader.load()
    
    print(f"Number of documents: {len(documents)}")
    print(f"Document content (first 200 chars): {documents[0].page_content[:200]}...")
    print(f"Document metadata: {documents[0].metadata}\n")
except Exception as e:
    print(f"Error loading text file: {e}\n")

print("=== Manual Document Creation ===")
# You can also create documents manually
manual_doc = Document(
    page_content="This is a manually created document.",
    metadata={"source": "manual", "author": "tutorial", "page": 1}
)

print(f"Manual document content: {manual_doc.page_content}")
print(f"Manual document metadata: {manual_doc.metadata}\n")

print("=== Multiple Documents ===")
# Create multiple documents
documents_list = [
    Document(page_content="First document about Python.", metadata={"topic": "Python"}),
    Document(page_content="Second document about JavaScript.", metadata={"topic": "JavaScript"}),
    Document(page_content="Third document about LangChain.", metadata={"topic": "LangChain"}),
]

print(f"Created {len(documents_list)} documents:")
for i, doc in enumerate(documents_list, 1):
    print(f"  {i}. {doc.page_content} (Topic: {doc.metadata['topic']})")
print()

print("=== Document Loader Types ===")
print("""
Common document loaders:
1. TextLoader: Plain text files
2. PyPDFLoader: PDF files
3. CSVLoader: CSV files
4. DirectoryLoader: Load all files in a directory
5. UnstructuredHTMLLoader: HTML files
6. WebBaseLoader: Web pages via URL
7. JSONLoader: JSON files

Note: Some loaders require additional dependencies
""")

print("=== Loading Multiple Text Files ===")
# Create multiple sample files
files_content = {
    "file1.txt": "Content about machine learning and neural networks.",
    "file2.txt": "Content about web development and frameworks.",
    "file3.txt": "Content about data science and analytics.",
}

# Create files
for filename, content in files_content.items():
    with open(filename, "w") as f:
        f.write(content)

# Load multiple files
all_documents = []
for filename in files_content.keys():
    try:
        loader = TextLoader(filename)
        docs = loader.load()
        all_documents.extend(docs)
        print(f"Loaded {filename}: {len(docs)} document(s)")
    except Exception as e:
        print(f"Error loading {filename}: {e}")

print(f"\nTotal documents loaded: {len(all_documents)}\n")

print("=== Document Metadata ===")
# Documents can have metadata for filtering and organization
for i, doc in enumerate(all_documents[:3], 1):
    print(f"Document {i}:")
    print(f"  Content: {doc.page_content[:50]}...")
    print(f"  Metadata: {doc.metadata}")
    print()

print("=== Working with PDF Files ===")
print("""
To load PDF files, use PyPDFLoader:

from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("path/to/file.pdf")
documents = loader.load()

Note: Requires pypdf package
""")

print("=== Loading from String ===")
# You can create documents from strings
string_content = "This content comes from a string variable, not a file."
string_doc = Document(
    page_content=string_content,
    metadata={"source": "string", "created": "programmatically"}
)

print(f"String document: {string_doc.page_content}")
print(f"Metadata: {string_doc.metadata}\n")

print("=== Document Properties ===")
# Documents have standard properties
sample_doc = Document(
    page_content="Sample document content here.",
    metadata={"key": "value"}
)

print("Document properties:")
print(f"  page_content: {sample_doc.page_content}")
print(f"  metadata: {sample_doc.metadata}")
print(f"  Type: {type(sample_doc)}\n")

print("=== Cleanup ===")
# Clean up sample files
import glob
sample_files = glob.glob("*.txt") + glob.glob("file*.txt")
for file in sample_files:
    try:
        if os.path.exists(file) and file.startswith(("sample_", "file")):
            os.remove(file)
            print(f"Removed {file}")
    except:
        pass

print("\nTutorial complete! You've learned how to load documents in LangChain.")


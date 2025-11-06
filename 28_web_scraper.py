"""
28_web_scraper.py - Web Scraping with LangChain

This tutorial demonstrates:
- Web scraping with LangChain
- Integration with web tools
- Processing web content
- Real-world scraping example

Prerequisites:
- Basic LangChain knowledge
- requests and beautifulsoup4 packages
"""

import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain.agents import initialize_agent, AgentType, Tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import requests
from bs4 import BeautifulSoup
from typing import Optional

# Load environment variables
load_dotenv()

# Initialize LLM
llm = AzureChatOpenAI(
    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
    temperature=0.7,
)

print("=== Web Scraping with LangChain ===")
print("""
This tutorial shows how to:
- Scrape web content
- Process scraped text
- Use LLMs to extract information
- Create a web scraping agent
""")

print("=== Simple Web Scraping Tool ===")
# Create a web scraping tool

def scrape_webpage(url: str) -> str:
    """Scrape text content from a webpage.
    
    Args:
        url: The URL to scrape
        
    Returns:
        Text content from the webpage
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Get text
        text = soup.get_text()
        
        # Clean up text
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        # Limit length for LLM processing
        return text[:5000] if len(text) > 5000 else text
    
    except Exception as e:
        return f"Error scraping {url}: {str(e)}"

# Create tool
scraping_tool = Tool(
    name="web_scraper",
    func=scrape_webpage,
    description="Scrapes text content from a webpage. Input should be a valid URL."
)

print("Testing web scraping tool:")
# Note: This is a demo - in practice, use real URLs
print("Tool created successfully\n")

print("=== Processing Scraped Content ===")
# Process scraped content with LLM

def extract_info_from_text(text: str, query: str) -> str:
    """Extract information from text using LLM"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert at extracting information from text."),
        ("human", """Text content:
{text}

Query: {query}

Extract relevant information based on the query:"""),
    ])
    
    chain = prompt | llm | StrOutputParser()
    
    # Truncate text if too long
    if len(text) > 3000:
        text = text[:3000] + "..."
    
    result = chain.invoke({"text": text, "query": query})
    return result

print("=== Web Scraping Agent ===")
# Create an agent that can scrape and process web content

# Mock scraping function for demo (since we don't want to scrape real sites)
def mock_scrape(url: str) -> str:
    """Mock scraper for demonstration"""
    mock_content = """
    LangChain is a framework for building applications with large language models.
    It provides tools for prompt management, chains, agents, and document processing.
    The framework supports multiple LLM providers including OpenAI, Azure OpenAI, and Anthropic.
    LangChain makes it easy to build production-ready LLM applications.
    """
    return mock_content

mock_tool = Tool(
    name="web_scraper",
    func=mock_scrape,
    description="Scrapes content from a webpage URL."
)

# Information extraction tool
def extract_information(text: str, topic: str) -> str:
    """Extract information about a topic from text"""
    prompt = f"""Extract information about '{topic}' from the following text:
    
{text}

Provide a concise summary:"""
    
    response = llm.invoke(prompt)
    return response.content

extract_tool = Tool(
    name="extract_info",
    func=lambda x: extract_information(x.split("|")[0], x.split("|")[1]) if "|" in x else "Error: Use format 'text|topic'",
    description="Extracts information about a topic from text. Input format: 'text|topic'"
)

# Create agent with web scraping capabilities
agent_tools = [mock_tool, extract_tool]

web_agent = initialize_agent(
    tools=agent_tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)

print("Testing web scraping agent:")
# Note: This uses mock data for demonstration
print("Agent created with web scraping capabilities\n")

print("=== Complete Web Scraping Workflow ===")
# End-to-end workflow

class WebScrapingPipeline:
    """Complete web scraping pipeline"""
    
    def __init__(self, llm):
        self.llm = llm
        self.scraping_tool = scraping_tool
    
    def scrape_and_analyze(self, url: str, analysis_query: str) -> dict:
        """Scrape webpage and analyze content"""
        # Step 1: Scrape
        print(f"Scraping: {url}")
        content = self.scraping_tool.func(url)
        
        if content.startswith("Error"):
            return {"error": content}
        
        # Step 2: Summarize
        print("Summarizing content...")
        summary_prompt = ChatPromptTemplate.from_messages([
            ("human", "Summarize this content in 2-3 sentences:\n\n{content}"),
        ])
        summary_chain = summary_prompt | self.llm | StrOutputParser()
        summary = summary_chain.invoke({"content": content[:2000]})  # Limit length
        
        # Step 3: Analyze
        print("Analyzing content...")
        analysis_prompt = ChatPromptTemplate.from_messages([
            ("human", "Based on this content:\n\n{content}\n\nAnswer: {query}"),
        ])
        analysis_chain = analysis_prompt | self.llm | StrOutputParser()
        analysis = analysis_chain.invoke({"content": content[:2000], "query": analysis_query})
        
        return {
            "url": url,
            "content_length": len(content),
            "summary": summary,
            "analysis": analysis,
        }

# Note: This is a demo - use real URLs in practice
print("Web scraping pipeline structure:")
print("1. Scrape webpage content")
print("2. Summarize content")
print("3. Analyze based on query")
print("4. Return structured results\n")

print("=== Best Practices ===")
print("""
1. Respect robots.txt
   - Check website's robots.txt
   - Follow crawling guidelines
   - Be respectful with rate limits

2. Error Handling
   - Handle network errors
   - Validate URLs
   - Handle timeouts

3. Content Processing
   - Clean HTML content
   - Handle encoding issues
   - Limit content size for LLM

4. Legal Considerations
   - Check terms of service
   - Respect copyright
   - Use public APIs when available

5. Performance
   - Cache results when possible
   - Use async for multiple requests
   - Implement rate limiting
""")

print("Tutorial complete! You've learned web scraping with LangChain.")


"""
29_sql_chain.py - Database Queries with LangChain

This tutorial demonstrates:
- SQL chains and agents
- Database integration
- Query generation and execution
- Data analysis examples

Prerequisites:
- Basic SQL knowledge
- Understanding of chains and agents
"""

import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import create_sql_agent
from langchain.sql_database import SQLDatabase
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain

# Load environment variables
load_dotenv()

# Initialize LLM
llm = AzureChatOpenAI(
    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
    temperature=0,
)

print("=== SQL Chains in LangChain ===")
print("""
SQL chains allow you to:
- Generate SQL queries from natural language
- Execute queries safely
- Analyze database results
- Build database-powered applications
""")

print("=== Concept: SQL Database Agent ===")
# Note: This is a conceptual example
# In practice, you'd need a real database connection

print("""
To use SQL agents, you need:
1. A database connection
2. SQLDatabase wrapper
3. SQL agent toolkit
4. Natural language queries
""")

print("=== SQL Query Generation ===")
# Generate SQL queries using LLM

def generate_sql_query(natural_language_query: str, table_schema: str) -> str:
    """Generate SQL query from natural language"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a SQL expert. Generate SQL queries based on natural language requests.
        
Table Schema:
{schema}

Rules:
- Only generate SELECT queries
- Use proper SQL syntax
- Be careful with injection attacks
- Return only the SQL query"""),
        ("human", "Generate a SQL query for: {query}"),
    ])
    
    chain = LLMChain(llm=llm, prompt=prompt)
    
    result = chain.run(schema=table_schema, query=natural_language_query)
    return result.strip()

# Example schema
example_schema = """
CREATE TABLE users (
    id INT PRIMARY KEY,
    name VARCHAR(100),
    email VARCHAR(100),
    age INT,
    created_at TIMESTAMP
);

CREATE TABLE orders (
    id INT PRIMARY KEY,
    user_id INT,
    product_name VARCHAR(100),
    amount DECIMAL(10,2),
    order_date TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id)
);
"""

print("Example SQL query generation:")
query = "Find all users older than 25"
sql = generate_sql_query(query, example_schema)
print(f"Natural language: {query}")
print(f"Generated SQL: {sql}\n")

print("=== SQL Query Analysis ===")
# Analyze SQL query results

def analyze_sql_results(query: str, results_summary: str) -> str:
    """Analyze SQL query results"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a data analyst. Analyze SQL query results and provide insights."),
        ("human", """SQL Query: {query}

Results Summary: {results}

Provide analysis and insights:"""),
    ])
    
    chain = LLMChain(llm=llm, prompt=prompt)
    analysis = chain.run(query=query, results=results_summary)
    return analysis

# Example
print("SQL result analysis example:")
sql_query = "SELECT COUNT(*) as total_users FROM users WHERE age > 25"
results = "Total users over 25: 150"
analysis = analyze_sql_results(sql_query, results)
print(f"Query: {sql_query}")
print(f"Results: {results}")
print(f"Analysis: {analysis}\n")

print("=== SQL Agent Concept ===")
print("""
A SQL agent would work like this:

1. User asks: "How many orders were placed last month?"
2. Agent generates SQL: "SELECT COUNT(*) FROM orders WHERE order_date >= DATE_SUB(NOW(), INTERVAL 1 MONTH)"
3. Agent executes query (with safety checks)
4. Agent interprets results: "150 orders were placed last month"
5. Agent formats response for user
""")

print("=== Database Schema Understanding ===")
# Help LLM understand database structure

def explain_schema(table_schema: str) -> str:
    """Explain database schema to LLM"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Explain the database schema and relationships."),
        ("human", "Schema:\n{schema}\n\nExplain the structure:"),
    ])
    
    chain = LLMChain(llm=llm, prompt=prompt)
    explanation = chain.run(schema=table_schema)
    return explanation

print("Schema explanation example:")
explanation = explain_schema(example_schema)
print(f"Explanation: {explanation[:200]}...\n")

print("=== SQL Query Safety ===")
print("""
Important safety considerations:

1. Input Validation:
   - Validate natural language queries
   - Sanitize user inputs
   - Prevent SQL injection

2. Query Restrictions:
   - Only allow SELECT queries
   - Restrict to specific tables
   - Limit result sizes

3. Error Handling:
   - Don't expose database errors
   - Handle syntax errors gracefully
   - Validate query results

4. Permissions:
   - Use read-only database user
   - Limit database access
   - Audit queries
""")

print("=== Example: Data Analysis Workflow ===")
# Complete workflow example

def data_analysis_workflow(query: str, schema: str) -> dict:
    """Complete data analysis workflow"""
    # Step 1: Generate SQL
    print("1. Generating SQL query...")
    sql = generate_sql_query(query, schema)
    print(f"   SQL: {sql}")
    
    # Step 2: Simulate execution (in practice, execute against DB)
    print("2. Executing query...")
    mock_results = "Query executed successfully. Returned 25 rows."
    print(f"   Results: {mock_results}")
    
    # Step 3: Analyze results
    print("3. Analyzing results...")
    analysis = analyze_sql_results(sql, mock_results)
    print(f"   Analysis: {analysis[:100]}...")
    
    return {
        "sql_query": sql,
        "results": mock_results,
        "analysis": analysis,
    }

print("\nComplete workflow example:")
result = data_analysis_workflow(
    "Find the average age of users",
    example_schema
)
print()

print("=== Integration Example Structure ===")
print("""
In practice, SQL integration would look like:

from langchain.sql_database import SQLDatabase
from langchain.agents import create_sql_agent

# Connect to database
db = SQLDatabase.from_uri("postgresql://user:pass@localhost/dbname")

# Create SQL agent
agent = create_sql_agent(
    llm=llm,
    db=db,
    verbose=True,
)

# Use agent
result = agent.run("How many users are there?")
""")

print("=== Best Practices ===")
print("""
1. Schema Documentation:
   - Provide clear schema descriptions
   - Include example data
   - Document relationships

2. Query Validation:
   - Check SQL syntax
   - Validate against schema
   - Test with safe queries first

3. Result Handling:
   - Limit result sets
   - Format results clearly
   - Handle empty results

4. Performance:
   - Add query timeouts
   - Use indexes effectively
   - Cache common queries

5. Security:
   - Use parameterized queries
   - Restrict database permissions
   - Log all queries
""")

print("Tutorial complete! You've learned about SQL chains and database integration.")


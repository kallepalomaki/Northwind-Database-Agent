from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain_openai import ChatOpenAI
import sqlite3
import os
import json
from langchain.tools import Tool
import argparse


class DatabaseTools:
    def __init__(self, cursor):
        self.cursor = cursor
    # Function to get schema information dynamically from the database
    def get_schema_info(self):

        # Get all table names
        self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = self.cursor.fetchall()

        schema_info = "Database Schema:\n"

        for table in tables:
            table_name = '"' + table[0] + '"'
            schema_info += f"\nTable: {table_name}\nColumns:\n"

            # Get columns for each table
            self.cursor.execute(f"PRAGMA table_info({table_name});")
            columns = self.cursor.fetchall()

            for column in columns:
                column_name = column[1]
                schema_info += f"  - {column_name}\n"

        return schema_info

    def get_few_shot_examples(self):

        few_shot_examples = [
            {
                "input": "Find the top 5 companies by revenue",
                "thought": "I need to retrieve the top N companies by revenue. I will use an ORDER BY clause with LIMIT.",
                "action": "sql_tool",
                "action_input": """
                        SELECT 
                            c.CompanyName AS Customer, 
                            SUM(od.UnitPrice * od.Quantity * (1 - od.Discount)) AS TotalRevenue
                        FROM "Order Details" od
                        JOIN Orders o ON od.OrderID = o.OrderID
                        JOIN Customers c ON o.CustomerID = c.CustomerID
                        GROUP BY c.CustomerID
                        ORDER BY TotalRevenue DESC
                        LIMIT 5;""",
                "observation": "Company A, Company B, Company C, Company D, Company E"
            },
            {
                "input": "Show companies with revenue between 4.5M and 5M",
                "thought": "I need to generate SELECT SQL query with given schema info for SQLite3. This is a tested query, follow it very closely. Repeat multiple times, rather than rewrite.",
                "action": "sql_tool",
                "action_input": """
                SELECT 
                    c.CompanyName AS Customer, 
                    SUM(od.UnitPrice * od.Quantity * (1 - od.Discount)) AS TotalRevenue
                FROM "Order Details" od
                JOIN Orders o ON od.OrderID = o.OrderID
                JOIN Customers c ON o.CustomerID = c.CustomerID
                GROUP BY c.CustomerID
                HAVING TotalRevenue BETWEEN 4500000 AND 5000000 
                ORDER BY TotalRevenue DESC;
                """,
                "observation": "Company X, Company Y, Company Z"
            },
            {
                "input": "Find all customers from Germany",
                "thought": "I need to retrieve customers from Germany using a WHERE clause on the Country column.",
                "action": "sql_tool",
                "action_input": "SELECT * FROM Customers WHERE Country = 'Germany';",
                "observation": "List of German customers"
            }
        ]
        return few_shot_examples

    # Function to execute free SQL query and return results
    def execute_sql_query(self, sql_query):
        sql_query = sql_query.strip("```sql").strip("```").strip()

        try:
            self.cursor.execute(sql_query)  # Execute the generated SQL query
            result = self.cursor.fetchall()
            return result
        except Exception as e:
            return f"Error executing query: {str(e)}"

    def tools(self):
        schema_info = self.get_schema_info()

        tools = [
            Tool(
                name="sql_tool",
                func=self.execute_sql_query,  # Single function for all cases
                description=(
                    "Use this tool to generate and execute SQL queries on the Northwind database. "
                    "Generated SQL must follow the exact structure of provided examples whenever applicable. Any deviations should be justified."
                    "Do not introduce new logic (e.g., CASE statements, additional aggregations, extra joins) unless explicitly present in the example."                    
                    "If initial generation deviates from the examples, refine and regenerate until the structure aligns closely."
                    "When applicable, reuse query fragments from examples verbatim instead of rewriting them."
                    "You must repeat query multiple times rather than construct a single query, if it helps keeping the query close to examples even if a single query would be more efficient"
                    "If the query is completely new, generate a reasonable SQL outside these examples."
                    "Use the database schema below in the query generation.\n\n"
                    f"{schema_info}"
                ),
            )
        ]


        return tools


def main():
    parser = argparse.ArgumentParser(description="Northwind database AI agent")
    parser.add_argument("--openai_key_path", type=str, nargs="?", default="../openai.key", help="Path to OpenAI api key.")
    parser.add_argument("--northwind_path", type=str, nargs="?", default="../northwind-SQLite3/dist/northwind.db", help="Path to Northwind database")

    args = parser.parse_args()
    # Connect to (or create) a database file
    conn = sqlite3.connect(args.northwind_path)
    # Create a cursor to execute SQL commands
    cursor = conn.cursor()

    with open(args.openai_key_path) as f:
        os.environ["OPENAI_API_KEY"] = f.read().strip().replace("OPENAI_API_KEY=", "")

    # Get column names
    database=DatabaseTools(cursor)
    tools=database.tools()


    llm = ChatOpenAI(
        model="gpt-4",
        temperature=0.5,
        max_tokens=2000,
        timeout=30,
        max_retries=2,
        api_key=os.getenv("OPENAI_API_KEY")  # Using the environment variable
    )


    agent = initialize_agent(
        tools=tools,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,  # Few-shot for guiding SQL generation
        llm=llm,
        verbose=True,
        agent_kwargs={
            "examples": database.get_few_shot_examples()  # Guides SQL generation
        }
    )

    # Three test queries:
    # query = "Please find companies with revenue between 4 M and 6 M in brackets of 200 k. List each bracket separately."
    # query = "Please find top 2 companies"
    # query = "Find the last name of all the employees and their date of birth."

    while (True):
        print()
        query=input("Please write your query here or quit by 'q': ")
        if query == 'q':
            break
        else:
            # Run the agent with a query question
            result = agent.invoke(query)
            print(result["input"])
            print()
            print(result["output"])
            print()
if __name__ == "__main__":
    main()

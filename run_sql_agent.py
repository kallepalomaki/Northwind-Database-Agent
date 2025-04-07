from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain_openai import ChatOpenAI
import sqlite3
import os
import json
from langchain.tools import Tool
import argparse
from SchemaInfo import SchemaInfo
class DatabaseTools(SchemaInfo):

#    def __init__(self, cursor):
#        self.cursor = cursor

    # Function to execute free SQL query and return results
    def execute_sql_query(self, sql_query):

        try:
            self.cursor.execute(sql_query)  # Execute the generated SQL query
            result = self.cursor.fetchall()
            return result
        except Exception as e:
            return f"Error executing query: {str(e)}"

    def n_largest_customers_by_revenue(self, N):
        try:
            self.cursor.execute("""
            SELECT 
                c.CompanyName AS Customer, 
                SUM(od.UnitPrice * od.Quantity * (1 - od.Discount)) AS TotalRevenue
            FROM "Order Details" od
            JOIN Orders o ON od.OrderID = o.OrderID
            JOIN Customers c ON o.CustomerID = c.CustomerID
            GROUP BY c.CustomerID
            ORDER BY TotalRevenue DESC
            LIMIT """ + str(N) + ";" )

            rows = self.cursor.fetchall()
            return rows
        except:
            print("failed n_largest_customers_by_revenue")

    def revenue_range(self, rang: str):
        try:
            print(rang)
            data=json.loads(rang)
            min_revenue=data["mini"]
            max_revenue=data["maxi"]

            query = """
                SELECT 
                    c.CompanyName AS Customer, 
                    SUM(od.UnitPrice * od.Quantity * (1 - od.Discount)) AS TotalRevenue
                FROM "Order Details" od
                JOIN Orders o ON od.OrderID = o.OrderID
                JOIN Customers c ON o.CustomerID = c.CustomerID
                GROUP BY c.CustomerID
                HAVING TotalRevenue BETWEEN """ + str(min_revenue) + " AND " + str(max_revenue) + """ 
                ORDER BY TotalRevenue DESC;
            """
            print(query)
            self.cursor.execute(query)
            rows = self.cursor.fetchall()
            return rows
        except:
            print("failed")
    def tools(self):
        schema_info = self.get_schema_info()

        tools = [
            Tool(
                name="n_largest_customers_by_revenue",
                func=self.n_largest_customers_by_revenue,
                description=(
                    "Use this to determine the largest companies by revenue. "
                    "Provide this number as a string with a single integer. "
                    "Do not include anything else in the response."
                ),
            ),
            Tool(
                name="revenue_range",
                func=self.revenue_range,
                description=(
                    "Use this to determine companies within a certain range of revenue. "
                    "Provide the range as JSON-formatted string with keys mini and maxi"
                    "and values as integers use the JSON to call function that includes pre-defined SQL template"
                    "that searches the database of interest"
                    "Do not include anything else in the response."
                ),
            ),

            Tool(
                name="sql_tool",
                func=self.execute_sql_query,  # Directly use the SQL execution function
                description=f"""
                    Use this as a fallback tool for free-form SQL queries after you have tried other agents multiple times.
                    Use this tool to generate and execute SQL queries on the Northwind database based on natural language input. 
                    The schema info is already embedded within the tool description:

                    {schema_info}

                    Your task is to generate SQL queries based on the natural language input, using the database schema information provided above.
                    """
            ),

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

    # Initialize the agent
    agent = initialize_agent(
        tools=tools,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        llm=llm,
        verbose=True
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

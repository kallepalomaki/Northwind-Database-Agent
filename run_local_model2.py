import subprocess
from langchain.llms import BaseLLM
from pydantic import BaseModel
from langchain.chat_models.base import BaseChatModel
from langchain.schema import LLMResult, Generation  # This is a placeholder for a wrapped result
#from langchain.chat_models import ChatGeneration, AIMessage
from langchain_core.outputs.chat_result import ChatResult, ChatGeneration
from langchain_core.messages.ai import AIMessage
from langchain_core.outputs import ChatResult
import json
import re

def load_chat_result(filename: str) -> ChatResult:
    """Load ChatResult from a JSON file."""
    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)
    return ChatResult.model_validate(data)

# Example Usage
chat_result_loaded = load_chat_result("chat_result_saved.json")

#class LocalModelLLM(BaseLLM, BaseModel):
class LocalModelLLM(BaseChatModel, BaseModel):
    exec_path: str  # Define as a Pydantic field
    model_file: str

    def __init__(self, **kwargs):
        super().__init__(**kwargs)  # Let Pydantic handle initialization

    def _generate(self, prompt: str, stop: list = None) -> str:
        # Verify the exec_path is a valid string
        if not isinstance(self.exec_path, str) or not self.exec_path:
            raise ValueError(f"Invalid exec_path: {self.exec_path}")
        if isinstance(prompt, list):
            #prompt = " ".join(prompt)
            prompt=prompt[0].content

            # Verify the model_file is a valid string
        if not isinstance(self.model_file, str) or not self.model_file:
            raise ValueError(f"Invalid model_file: {self.model_file}")

        # Construct the command with the correct parameters
        command = [
            self.exec_path,
            '-m', self.model_file,  # Model file
            '-p', prompt,           # The input prompt
            '--threads', '4',       # Number of threads
            '--n-gpu-layers', '0',   # Disabling GPU layers
            '--n-predict', '100',
            '-st'
        ]

        # Debug: Print out the command
        print("Running command:", command)

        # Running the subprocess with the constructed command
        try:
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            # Create a Generation object for the model output
            #generation = Generation(text=result.stdout.strip())
            # Create the message (equivalent to what AI might generate)
            assistant_text=result.stdout.strip().split("Assistant:")[1].split("Observation")[0].strip()

            if True:
                #assistant_text="dummy"
                message = AIMessage(content=assistant_text)
                # Create the ChatGeneration object (this represents a single response generation)
                generation = ChatGeneration(text=assistant_text, generation_info={'finish_reason': 'stop', 'logprobs': None},
                                            message=message)
                chat_result = ChatResult(generations=[generation], llm_output={
                    'token_usage': {'completion_tokens': 84, 'prompt_tokens': 879, 'total_tokens': 963},
                    'model_name': 'local-model',
                    'system_fingerprint': None,
                    'id': 'local-model-id'
                })
            #chat_result=load_chat_result('/Users/kpalomak/Northwind-Database-Agent/chat_result_saved.json')
            # Return the output wrapped in an LLMResult object (list of lists of Generation objects)
            return chat_result
        except:
            print("Error")
    @property
    def _llm_type(self) -> str:
        # Return the type of LLM, e.g., 'local' for local models
        return "local"

# Example usage
model_path = '../llama.cpp/build/bin/llama-cli'  # Path to llama-cli
#model_file = '../models/deepseek/deepseek-coder-6.7b-instruct.Q4_K_M.gguf'  # Path to the model file
model_file="../models/deepseek/DeepSeek-Coder-V2-Lite-Instruct-IQ4_XS.gguf"
#model_file = "../models/deepseek/deepseek-coder-1.3b-instruct-q4_k_m.gguf"
# Example usage
local_model = LocalModelLLM(exec_path=model_path, model_file=model_file)

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

    # Function to execute free SQL query and return results
    def execute_sql_query(self, sql_query):

        try:
            self.cursor.execute(sql_query)  # Execute the generated SQL query
            result = self.cursor.fetchall()
            return result
        except Exception as e:
            return f"Error executing query: {str(e)}"

    def n_largest_customers_by_revenue(self, N):
        N = str(re.findall(r'\d+', N)[0])
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
                    """
                    Use this to determine the N largest companies by revenue. "
                    You are a helpful AI agent. Given a question, respond in this format:
                    
                    Action:<Action Name>
                    Action Input:<Input>
                    
                    Example:
                    User: Who are the top 2 customers by revenue?
                    AI: I need to find the top 2 companies by revenue. To do this, I can use the `n_largest_customers_by_revenue` tool.
                    
                    Action:n_largest_customers_by_revenue
                    Action Input:"2"
                    """
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


    # Initialize the agent
    agent = initialize_agent(
        tools=tools,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        llm=local_model,
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


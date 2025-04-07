# Northwind-Database-Agent

This project implements an AI agent for controlled natural language database queries using Python's LangChain and the OpenAI API. The approach is demonstrated with the publicly available Northwind database.

Instead of allowing the AI to generate SQL queries freely, I use a two-stage approach. First, I provide query templates for anticipated complex queries, where the agent only determines the required parameters. Second, if a template-based approach does not answer the user’s question, free-form SQL generation is permitted.

To prioritize template-based queries, the agent is instructed to attempt template-based functions before resorting to free SQL generation. Corresponding instruction given to the agent is the following: "Use this as a fallback tool for free-form SQL queries after you have tried other agents multiple times."


# Installation

I'm have used python 3.11. for the implementation. Below is the install info using a virtual enviroment in Linux or Mac:

python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

For Northwind I'm using SQLite3 tools from this repo:
https://github.com/jpwhite3/northwind-SQLite3

Follow the instructions therein to install Northwind database. For this code I have done the all the three steps in the build instructions:
make build 
make populate
make report

For running the	code, you'll need OpenAI API key.

For local model you need to install: 
https://github.com/ggml-org/llama.cpp

and get a local model. I have used DeepSeek-Coder-V2-Lite-Instruct-IQ4_XS.gguf 
To be able to run it with CPU only with laptop using only 16 Gbytes memory.

# Usage

After you have installed and configured the agent there are three alternatives to run agent. First two apply priorization by a zeroshot approach.
1. python run_sql_agent.py 
2. python run_local_model.py

The third is a few shots version where templates are given as examples for LLM rather than realized as methods the DatabaseTools -class:
3. python run_sql_agent_few_shot.py 

Agent will prompt "Please write your query here or quit by 'q':"
Add your natural language query and press enter. 

I have tested the agent with following test queries.

1) Test of SQL template based revenue_range agent by a complex query for OpenAI models 1 & 3:
“Please find companies with revenue between 4 M and 6 M in brackets of 200 k. List each bracket separately.”
or
“Please find companies with revenue between 4 M and 6 M in brackets of 200 k. List names of companies in each bracket separately.”
and a simpler test for the local model 2:
"Find companies with revenue between 5.0M and 5.5M"

2) Test of SQL template based n_largest_customers_by_revenue agent:
“Please find top 2 companies”

3) Test of free SQL generation agent:
“Find the last name of all the employees and their date of birth.”

See the following demovideo (in Finnish): https://www.youtube.com/watch?v=qad31XJLfsU

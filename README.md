# Northwind-Database-Agent

This project is to make database queries using natural language in a controlled manner by AI agents using python language chain
and OpenAI api. I have used publicly available Northwind database to demonstrate the approach. 

Rather letting AI to generate SQL freely I have applied two stage approach. First, I provide query templates for anticipated complex 
queries for which agent finds only calling parameters. Second, if template based approaches do not answer users
question, then free query is allowed. The priority of using the template is achieved by instructing the agent to 
try template based functions before calling free SQL as follows. Here is the corresponding instruction to the agent.
"Use this as a fallback tool for free-form SQL queries after you have tried other agents multiple times."


# Dependencies

I'm using python 3.11. for the experiments. Install info using a virtual enviroment in Linux or Mac:

python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

For Northwind I'm using SQLite3 tools 
from this repo:
https://github.com/jpwhite3/northwind-SQLite3

Follow the instructions therein to install Northwind database. For this code I have done the all the three steps in the build instructions
make build 
make populate
make report

For running the	code, you'll need OpenAI API key.


# Usage

After you have installed and configured the agent run it with command:
python run_sql_agent.py

Agent will prompt:


1) Test SQL template based revenue_range agent by a complex query:

“Please find companies with revenue between 4 M and 6 M in brackets of 200 k. List each bracket separately.”

2) Test SQL template based  n_largest_customers_by_revenue agent.

“Please find top 2 companies”

3) Test free SQL generation agent

“Find the last name of all the employees and their date of birth.”
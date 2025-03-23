# Northwind-Database-Agent

This project is to make database queries using natural language in a controlled manner by AI agents using python language chain
and OpenAI api. I have used publicly available Northwind database to demonstrate the approach. 

Rather letting AI to generate SQL freely I have applied two stage approach. First, I provide query templates for anticipated complex 
queries for which agent finds only calling parameters. Second, if template based approaches do not answer users
question, then free query is allowed. The priority of using the template is achieved by instructing the agent to 
try template based functions before calling free SQL as follows. Here is the corresponding instruction to the agent.
"Use this as a fallback tool for free-form SQL queries after you have tried other agents multiple times."

# Dependencies

I'm using python 3.11. for the experiments

For Northwind I'm using SQLite3 tools 
from this repo:
https://github.com/jpwhite3/northwind-SQLite3

Follow the instructions therein to install Northwind db. For this code I have done the all the three steps in the build instructions
make build 
make populate
make report

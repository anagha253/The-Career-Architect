'''
Building a "Multi-Agent Research & Strategy System" that manages your job hunt autonomously but stops for your approval.

'''

import os
from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import ToolNode, tools_condition
from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START
from langchain_core.messages import SystemMessage, AIMessage, ToolMessage
from google import genai
from google.genai.types import GenerateContentConfig
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException
import chromadb

load_dotenv()
llm_key  = os.getenv("LLM_KEY")
tavily_key = os.getenv("TAVILY_KEY")

client = chromadb.PersistentClient(path="./data/rag/my_ai_memory")
collection = client.get(name="user_knowledge")

searcher = TavilySearchResults(max_results=3,api_key=tavily_key)
tools = [searcher]

manager = ChatGoogleGenerativeAI(model="gemini-2.5-flash",temperature=0,api_key=llm_key)

#researcher <-->  tavily searches
researcher = ChatGoogleGenerativeAI(model="gemini-2.5-flash",temperature=0,api_key=llm_key)
researcher_with_tools = researcher.bind_tools(tools)

#analyser <---> chromaDB to check on user data(CV,projects)
analyser = ChatGoogleGenerativeAI(model="gemini-2.5-flash",temperature=0,api_key=llm_key)
query = "List all projects and 1 one line for each of what it does, and academic and work experience  in  summarised into 5 points not details missed"
 
query_embed = genai.embed_content(
    model="models/text-embedding-004",
    content=query,
    task_type="retrieval_query"
)['embedding']

results = collection.query(query_embeddings=[query_embed], n_results=1)

retrieved_fact = results['documents'][0][0]
analyser_prompt = SystemMessage(
    content= '''


'''
)


res = analyser.invoke()

#writer
writer = ChatGoogleGenerativeAI(model="gemini-2.5-flash",temperature=0.6,api_key=llm_key)

#reviewer
reviewer = ChatGoogleGenerativeAI(model="gemini-2.5-flash",temperature=0.4,api_key=llm_key)




class JobOppurtunity(BaseModel):
    role_name: str = Field(...)
    

class State(TypedDict):
    message: Annotated[list, add_messages]
    location: str
    job_list: list



tool_node =ToolNode(tools)

builder = StateGraph(State)

builder.add_node(manager)
builder.add_node(researcher)
builder.add_node(analyser)
builder.add_node(writer)
builder.add_node(reviewer)

builder.add_edge(manager,researcher)
builder.add_edge(researcher,tool_node)
builder.add_edge(researcher,analyser)
builder.add_conditional_edges(

)



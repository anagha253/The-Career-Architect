'''
Building a "Multi-Agent Research & Strategy System" that manages your job hunt autonomously but stops for your approval.

'''

import os
from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import ToolNode, tools_condition
from typing import Annotated, TypedDict ,List
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START
from langchain_core.messages import SystemMessage, AIMessage, ToolMessage
from google import genai
from google.genai.types import GenerateContentConfig
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()
llm_key  = os.getenv("LLM_KEY")
tavily_key = os.getenv("TAVILY_KEY")

embedding = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001",api_key=llm_key)
vector_store = Chroma(
    collection_name="user_knowledge",
    embedding_function=embedding,
    persist_directory="./data/rag/my_ai_memory"
)

searcher = TavilySearchResults(max_results=3,api_key=tavily_key)
tools = [searcher]


tool_node =ToolNode(tools)

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash",temperature=0.6,api_key=llm_key)

class JobOppurtunity(BaseModel):
    title: str = Field(description="The job title")
    company: str = Field(description="Name of the company")
    link: str = Field(description="URL to the job posting")
    relevance_score: int = Field(description="Score from 1-10 based on user profile")


class State(TypedDict):
    message: Annotated[list, add_messages]
    location: str
    job_list: List[JobOppurtunity]
    interrupt: str

# class JobInfo(BaseModel):
#     job_name: str = Field(...)
#     job_link: str = Field(...)

# does Tavily searches
researcher_model = model.bind_tools(tools)
def researcher(state :State):
    print("I am researching for you....")
    prompt = SystemMessage(content=(
        "You are a job researcher. Your goal is to find 3 relevant job openings "
        "based on the user's location and interest. Use Tavily to search. "
        "Focus on 'Backend Engineer' or 'AI Engineer' roles."
    ))
    messages = [prompt] + state["message"]
    response = researcher_model.invoke(messages)
    return {"message":[response]}
'''
manager = ChatGoogleGenerativeAI(model="gemini-2.5-flash",temperature=0,api_key=llm_key)

researcher <-->  tavily searches
researcher = ChatGoogleGenerativeAI(model="gemini-2.5-flash",temperature=0,api_key=llm_key)
researcher_with_tools = researcher.bind_tools(tools)

analyser <---> chromaDB to check on user data(CV,projects)
analyser_model = ChatGoogleGenerativeAI(model="gemini-2.5-flash",temperature=0,api_key=llm_key)
'''

def analyser(state: State):
    print("Analysing all the info!!!")
    context = state["message"][-1].content
    query = "List all projects and 1 one line for each of what it does, and academic and work experience  in  summarised into 5 points not details missed"

    result = vector_store.as_retriever().invoke(query)

    analysis_prompt = f"Based on this resume data: {result}, " \
                      f"evaluate these jobs: {context}. " \
                      "Should we draft an email? Output Yes or No."

    response = model.invoke(analysis_prompt)

    return {"message" : [response], "interrupt": "Yes" in response.content}


#writer
writer = ChatGoogleGenerativeAI(model="gemini-2.5-flash",temperature=0.6,api_key=llm_key)

#reviewer
reviewer = ChatGoogleGenerativeAI(model="gemini-2.5-flash",temperature=0.4,api_key=llm_key)



builder = StateGraph(State)

builder.add_node("researcher",researcher)
builder.add_node("tools", tool_node)
builder.add_node("analyser",analyser)
builder.add_node(writer)
builder.add_node(reviewer)

builder.add_edge(START,researcher)
builder.add_edge(researcher,tool_node)
builder.add_edge(researcher,analyser)
builder.add_conditional_edges(

)

app = builder.compile(
    
)



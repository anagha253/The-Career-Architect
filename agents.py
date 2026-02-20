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
from langgraph.graph import StateGraph, START,END
from langchain_core.messages import SystemMessage
from pydantic import BaseModel, Field
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langgraph.types import interrupt 
from langgraph.checkpoint.sqlite import SqliteSaver

load_dotenv()
llm_key  = os.getenv("LLM_KEY")
tavily_key = os.getenv("TAVILY_KEY")

embedding = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001",api_key=llm_key)
vector_store = Chroma(
    collection_name="user_knowledge",
    embedding_function=embedding,
    persist_directory="./chroma_db"
)

searcher = TavilySearchResults(max_results=3,tavily_api_key=tavily_key)
tools = [searcher]


tool_node =ToolNode(tools)

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash",temperature=0.6,api_key=llm_key)

# class JobOppurtunity(BaseModel):
#     title: str = Field(description="The job title")
#     company: str = Field(description="Name of the company")
#     link: str = Field(description="URL to the job posting")
#     relevance_score: int = Field(description="Score from 1-10 based on user profile")


class State(TypedDict):
    message: Annotated[list, add_messages]
    location: str
    role: str
    generated_email: str
    decision: str

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

    return {"message" : [response]}


#human node to approve cold email generation
def human_decision(state: State):
    context = state["message"][-1].content
    return interrupt({
        "message" : f"This is the Analysis: {context} \n\n Want to generate cold email? (yes/no)"
    })

def route_after_human(state: State):
    if state.get("decision") == "yes":
        return "writer"
    return "__end__"

#writer
def writer(state:State):
    context = state["message"][-1].content
    prompt = f'''
        You are a professional cold email writer. You have 100% tendency to convert a  cold email to a call
        Thus, with the context of the candidate: {context}, generate a cold email for the same. Make sure its in
        the right format and grammatically perfect. The writing should be very crisp. Make sure you pitch the candidate 
        properly.
    '''
    response = model.invoke(prompt)
    return {"generated_email": response.content}

#reviewer
def reviewer(state: State):
    generated_email = state["generated_email"]
    prompt = f'''
        You are a professional cold email reviewer. You have a keen eye to mistakes in email and check if its professional
        enough or not. You have edit the {generated_email} and make sure it is devoid of any red-flags, mistakes in terms 
        grammar, professionalism, and flow of the matter in the email.
    '''
    response = model.invoke(prompt)
    return {"generated_email": response.content}

builder = StateGraph(State)

builder.add_node("researcher",researcher)
builder.add_node("tools", tool_node)
builder.add_node("analyser",analyser)
builder.add_node("human",human_decision)
builder.add_node("writer",writer)
builder.add_node("reviewer",reviewer)

builder.add_edge(START,"researcher")
builder.add_conditional_edges(
    "researcher",
    tools_condition,
    {
        "tools": "tools",
        "__end__": "analyser"
    }
)
builder.add_edge("tools","researcher")
builder.add_edge("analyser","human")
builder.add_conditional_edges(
    "human",
    route_after_human,
    {
        "writer": "writer" ,
        "__end__": END
    }
)
builder.add_edge("writer","reviewer")
builder.add_edge("reviewer",END)

sqlite_checkpointer = SqliteSaver.from_conn_string("sqlite:///memory.db")

app = builder.compile(checkpointer=sqlite_checkpointer)

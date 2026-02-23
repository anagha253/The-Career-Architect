"""
Cyclic Multi-Agent Career Strategist using LangGraph v0.2+, Gemini 2.5 Flash, and ChromaDB.
Researches jobs (Tavily), analyzes against RAG (ChromaDB), pauses for HITL approval,
then writes and reviews a cold email.
"""
import os
from typing import Annotated, TypedDict, List, Any, Optional
from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables import RunnableConfig

from backend import utility

load_dotenv()
LLM_KEY = os.getenv("LLM_KEY") or os.getenv("GOOGLE_API_KEY")
TAVILY_KEY = os.getenv("TAVILY_KEY")

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.6,
    api_key=LLM_KEY,
)


class State(TypedDict):
    """Graph state with message history, context, and outputs."""
    message: Annotated[List[BaseMessage], add_messages]
    location: str
    role: str
    job_list: Optional[List[dict]]
    generated_email: Optional[str]
    decision: Optional[str]


# --- Tools ---
searcher = TavilySearchResults(max_results=5, tavily_api_key=TAVILY_KEY)
tools = [searcher]
tool_node = ToolNode(tools, messages_key="message")
researcher_model = model.bind_tools(tools)


# --- Nodes ---
def researcher(state: State) -> dict:
    """Binds Tavily to Gemini to find relevant job openings."""
    try:
        prompt = SystemMessage(content=(
            f"You are a job researcher. Find 3–5 relevant job openings for the role "
            f"'{state['role']}' in '{state['location']}'. Use Tavily to search. "
            f"Focus on Backend Engineer, AI/ML Engineer, or Software Engineer roles. "
            f"Return structured job info: title, company, link, brief description."
        ))
        user_msg = state["message"][-1] if state["message"] else HumanMessage(
            content=f"Find {state['role']} jobs in {state['location']}"
        )
        messages = [prompt] + state["message"] + [user_msg]
        response = researcher_model.invoke(messages)
        return {"message": [response]}
    except Exception as e:
        return {"message": [AIMessage(content=f"Research error: {str(e)}")]}


def analyser(state: State, config: RunnableConfig) -> dict:
    """Queries ChromaDB (resume/projects), compares to found jobs, recommends yes/no."""
    try:
        thread_id = config.get("configurable", {}).get("thread_id", "default")
        vector_store = utility.get_vector_store(thread_id)
        retriever = vector_store.as_retriever(k=5)
        
        query = (
            "Summarize: projects with one-line descriptions, academic background, "
            "work experience in 5 bullet points. Be concise."
        )
        resume_context = retriever.invoke(query)
        resume_text = "\n".join(
            d.page_content if hasattr(d, "page_content") else str(d) for d in resume_context
        )
        
        jobs_context = ""
        for msg in reversed(state["message"]):
            if hasattr(msg, "content") and msg.content:
                jobs_context = str(msg.content)
                break
        
        analysis_prompt = (
            f"Resume/projects summary:\n{resume_text}\n\n"
            f"Found jobs:\n{jobs_context}\n\n"
            "Compare the candidate profile to these jobs. Recommend whether to draft a cold email. "
            "Output exactly: 'Yes' or 'No' with a 1–2 sentence reason."
        )
        response = model.invoke([HumanMessage(content=analysis_prompt)])
        analysis = response.content if hasattr(response, "content") else str(response)
        return {"message": [AIMessage(content=analysis)]}
    except Exception as e:
        return {"message": [AIMessage(content=f"Analysis error: {str(e)}")]}


def human_decision(state: State) -> dict:
    """Pauses for Human-in-the-Loop approval using interrupt()."""
    analysis = ""
    for msg in reversed(state["message"]):
        if hasattr(msg, "content") and msg.content:
            analysis = str(msg.content)
            break
    
    decision = interrupt({
        "message": f"Analysis:\n{analysis}\n\nGenerate cold email? Reply yes or no.",
        "analysis": analysis,
    })
    decision_str = str(decision).strip().lower() if decision else "no"
    return {"decision": "yes" if decision_str == "yes" else "no"}


def route_after_human(state: State) -> str:
    """Route to writer if approved, else end."""
    if state.get("decision") == "yes":
        return "writer"
    return "__end__"


def writer(state: State) -> dict:
    """Drafts the cold email from analysis context."""
    try:
        context = ""
        for msg in reversed(state["message"]):
            if hasattr(msg, "content") and msg.content:
                context = str(msg.content)
                break
        
        prompt = (
            f"You are a professional cold email writer. Using this candidate/job context:\n{context}\n\n"
            "Write a crisp, grammatically perfect cold email that pitches the candidate effectively. "
            "Keep it professional and conversion-oriented. Use proper email format."
        )
        response = model.invoke([HumanMessage(content=prompt)])
        content = response.content if hasattr(response, "content") else str(response)
        return {"generated_email": content}
    except Exception as e:
        return {"generated_email": f"Error drafting email: {str(e)}"}


def reviewer(state: State) -> dict:
    """Polishes the email for grammar, tone, and professionalism."""
    try:
        draft = state.get("generated_email", "")
        prompt = (
            f"You are a cold email reviewer. Edit this draft for grammar, professionalism, and flow:\n\n{draft}\n\n"
            "Remove red flags, fix mistakes, and ensure a professional tone. Return only the polished email."
        )
        response = model.invoke([HumanMessage(content=prompt)])
        content = response.content if hasattr(response, "content") else str(response)
        return {"generated_email": content}
    except Exception as e:
        return {"generated_email": state.get("generated_email", "")}


# --- Graph ---
builder = StateGraph(State)

builder.add_node("researcher", researcher)
builder.add_node("tools", tool_node)
builder.add_node("analyser", analyser)
builder.add_node("human_decision", human_decision)
builder.add_node("writer", writer)
builder.add_node("reviewer", reviewer)

builder.add_edge(START, "researcher")
builder.add_conditional_edges(
    "researcher",
    lambda s: tools_condition(s, messages_key="message"),
    {"tools": "tools", "__end__": "analyser"},  # No tool calls → go to analyser
)
builder.add_edge("tools", "researcher")
builder.add_edge("analyser", "human_decision")
builder.add_conditional_edges(
    "human_decision",
    route_after_human,
    {"writer": "writer", "__end__": END},
)
builder.add_edge("writer", "reviewer")
builder.add_edge("reviewer", END)

# MemorySaver for correct HITL interrupt/resume
memory = MemorySaver()
graph_app = builder.compile(checkpointer=memory)

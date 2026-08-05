from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.agents import AgentType, initialize_agent
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator

from src.tools.code_interpreter import run_python
from src.tools.document_loader import load_pdf, load_url
from src.mcp.client import MCPClient
from src.config.settings import settings

llm = ChatOpenAI(model="gpt-4o", temperature=0.3, api_key=settings.openai_api_key)

class ResearchState(TypedDict):
    query: str
    documents: Annotated[list, operator.add]
    summary: str
    citations: list

def plan_node(state: ResearchState):
    prompt = PromptTemplate.from_template(
        "Break the research query into 3-5 sub-questions.\nQuery: {query}\nReturn as JSON list."
    )
    chain = prompt | llm
    return {"documents": []}

def retrieve_node(state: ResearchState):
    search = DuckDuckGoSearchRun()
    docs = search.run(state["query"])
    return {"documents": [docs]}

def synthesize_node(state: ResearchState):
    prompt = PromptTemplate.from_template(
        "Synthesize a research brief from the following documents.\n"
        "Query: {query}\nDocuments: {documents}\nInclude citations."
    )
    chain = prompt | llm
    result = chain.invoke({"query": state["query"], "documents": state["documents"]})
    return {"summary": result.content, "citations": []}

workflow = StateGraph(ResearchState)
workflow.add_node("plan", plan_node)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("synthesize", synthesize_node)
workflow.set_entry_point("plan")
workflow.add_edge("plan", "retrieve")
workflow.add_edge("retrieve", "synthesize")
workflow.add_edge("synthesize", END)
research_graph = workflow.compile()

mcp_client = MCPClient(server_url=settings.mcp_server_url)

class ResearchAgent:
    def __init__(self):
        self.graph = research_graph
        self.mcp = mcp_client
        self.tools = [run_python, load_pdf, load_url, DuckDuckGoSearchRun()]

    def run(self, query: str) -> dict:
        state = self.graph.invoke({"query": query, "documents": [], "summary": "", "citations": []})
        extra = self.mcp.call_tool("academic_search", {"query": query, "limit": 10})
        state["documents"].extend(extra.get("results", []))
        return state

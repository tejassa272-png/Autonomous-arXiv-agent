from langgraph.graph import StateGraph, START, END

from core.state import AgentState
from nodes.retrieval import parse_query, retrieve_arxiv_papers
from nodes.selection import select_best_paper
from nodes.parsing import fetch_and_parse
from nodes.ingestion import chunk_and_embed
from nodes.summarizer import generate_briefing
from nodes.qa import answer_question

def route_entry(state: AgentState):

    """
    If a vector store path exists, the paper is already downloaded and indexed, 
    so this is a follow-up QA question -- Otherwise, starts the ingestion pipeline
    """

    if state.get("vector_store_path"):
        return "qa_node"
    return "parse_intent"

def route_or_error(next_node: str):

    """
    Returns a router function that runs END if an error exists,
    otherwise routes directly to the specified target node.
    """

    def router(state: AgentState) -> str:
        if state.get("error_message"):
            return END
        return next_node
    return router

def build_arxiv_agent():
    #Initializes the graph with the typeddict state
    workflow = StateGraph(AgentState)

    #Registers all the imported functions as nodes
    workflow.add_node("parse_intent", parse_query)
    workflow.add_node("fetch_metadata", retrieve_arxiv_papers)
    workflow.add_node("select_target", select_best_paper)
    workflow.add_node("fetch_and_parse", fetch_and_parse)
    workflow.add_node("chunk_and_embed", chunk_and_embed)
    workflow.add_node("generate_briefing", generate_briefing)
    workflow.add_node("qa_node", answer_question)

    #Dynamic Entry Point
    workflow.add_conditional_edges(START, route_entry)

    #Edges
    workflow.add_conditional_edges("parse_intent", route_or_error("fetch_metadata"))
    workflow.add_conditional_edges("fetch_metadata", route_or_error("select_target"))
    workflow.add_conditional_edges("select_target", route_or_error("fetch_and_parse"))
    workflow.add_conditional_edges("fetch_and_parse", route_or_error("chunk_and_embed"))
    workflow.add_conditional_edges("chunk_and_embed", route_or_error("generate_briefing"))
    
    #Exit Points
    workflow.add_edge("generate_briefing", END)
    workflow.add_edge("qa_node", END)

    #Compiles the graph into an executable Runnable format
    return workflow.compile()
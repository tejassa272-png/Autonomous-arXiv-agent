import operator
from typing import TypedDict, List, Optional, Dict, Any, Annotated

class AgentState(TypedDict):
    #input
    query: str
    is_direct_id: bool

    #arxiv retrival
    candidate_papers: List[Dict[str, Any]]
    selected_paper_id: Optional[str]
    pdf_url: Optional[str]

    #parsing and storage
    parsed_text: Optional[str]
    vector_stored_path: Optional[str]

    #output
    summary: Optional[Dict[str,Any]]
    chat_history: Annotated[List[Dict[str,str]], operator.add]

    #error handaling
    error_message: Optional[str]
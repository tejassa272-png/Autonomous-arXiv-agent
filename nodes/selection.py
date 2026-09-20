import os
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from core.state import AgentState

class PaperSelection(BaseModel):
    # schema definition for LLM
    selected_paper_id: str = Field(description="The exact arXiv ID of the most relevant paper.")
    reasoning: str = Field(description="Brief reason for selecting this paper based on the user's query.")

def select_best_paper(state: AgentState) -> dict:
    """
    This node performs the logic to fetch the best paper using the gemini LLM.
    (Only executed if the user searched by semantic topic, enforced by graph.py)
    """

    # checks for candidate_paper length
    candidate_papers = state.get("candidate_papers", [])
    if not candidate_papers:
        return {"error_message": "No papers available for selection"}

    # checks if exactly only one paper is present
    if len(candidate_papers) == 1:
        return {
            "selected_paper_id": candidate_papers[0]["id"],
            "pdf_url": candidate_papers[0]["pdf_url"]
        }

    # calls the gemini api to fetch the relevant paper
    query = state["query"]
    candidates_text = ""
    for i, paper in enumerate(candidate_papers):
        candidates_text += f"\n--- Paper {i+1} ---\nID: {paper['id']}\nTitle: {paper['title']}\nAbstract: {paper['abstract']}\n"
    
    prompt = f"""
    You are an expert AI research assistant. 
    A user is looking for a research paper on the following topic: "{query}"
    
    Below is a list of candidate papers from arXiv. Read their titles and abstracts, 
    and select the single most relevant paper that best matches the user's intent.
    
    Candidates:
    {candidates_text}
    """
    
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash", 
            temperature=0,
            api_key=os.getenv("GEMINI_API_KEY")
        )
        
        # binding the pydantic schema with the LLM
        structured_llm = llm.with_structured_output(PaperSelection)
        result = structured_llm.invoke(prompt)
        chosen_id = result.selected_paper_id
        
        pdf_url = None
        # id matching with the LLM result is done
        for paper in candidate_papers:
            if chosen_id in paper["id"] or paper["id"] in chosen_id:
                chosen_id = paper["id"] 
                pdf_url = paper["pdf_url"]
                break
                
        if not pdf_url:
            chosen_id = candidate_papers[0]["id"]
            pdf_url = candidate_papers[0]["pdf_url"]
            
        return {"selected_paper_id": chosen_id, "pdf_url": pdf_url}
    
    except Exception as e:
        print("Exception block triggered in selection.py") # for debugging
        return {
            "selected_paper_id": candidate_papers[0]["id"],
            "pdf_url": candidate_papers[0]["pdf_url"]
        }
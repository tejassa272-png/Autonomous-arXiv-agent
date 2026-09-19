import re
import arxiv
from core.state import AgentState

def parse_query(state: AgentState):
    """
    Checks if the query is an general topic search or the arXiv ID
    """

    query = state["query"].strip()

    id_pattern = r"(\d{4}\.\d{4,5}(?:v\d+)?)"
    match = re.search(id_pattern, query)

    if match:
        return {
            "is_direct_id" : True,
            "selected_paper_id" : match.group(1)
        }
    return {
        "is_direct_id" : False,
        "selected_paper_id" : None
    }

def retrieve_arxiv_papers(state: AgentState) -> dict:
    """
    Calls the arXiv API to fetch candidate papers based on the query or ID.
    """
    query = state["query"]
    is_direct_id = state.get("is_direct_id", False)
    selected_id = state.get("selected_paper_id")
    
    client = arxiv.Client()
    candidate_papers = []
    
    try:
        if is_direct_id and selected_id:
            # Fetches the specific paper
            search = arxiv.Search(id_list=[selected_id])
        else:
            # Fetches the top 5 papers for a topic
            search = arxiv.Search(
                query=query,
                max_results=5,
                sort_by=arxiv.SortCriterion.Relevance
            )

        for result in client.results(search):
            paper_data = {
                # Extracts metadata required for the executive briefing
                "id": result.get_short_id(), 
                "title": result.title,
                "authors": [author.name for author in result.authors],
                "abstract": result.summary,
                "publish_date": result.published.strftime("%Y-%m-%d"),
                "pdf_url": result.pdf_url,
                "categories": result.categories
            }
            candidate_papers.append(paper_data)
            
        if not candidate_papers:
            # Handles the zero-result case
            return {"error_message": "No papers found for the given query."}
            
        # If it is a direct Id , automatically lock in the PDF url
        if is_direct_id:
            return {
                "candidate_papers": candidate_papers,
                "pdf_url": candidate_papers[0]["pdf_url"]
            }
            
        return {"candidate_papers": candidate_papers}
        
    except Exception as e:
        return {"error_message": f"arXiv API error: {str(e)}"}
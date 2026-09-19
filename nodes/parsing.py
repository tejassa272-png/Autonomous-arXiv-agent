import os
import requests
import pymupdf4llm
from core.state import AgentState

def fetch_and_parse(state: AgentState):
    """
    This node fetches the pdf from arXiv and parses it
    """

    #returns empty if any error is found in the upper nodes
    if state.get("error_message"):
        return {}

    pdf_url = state.get("pdf_url")
    paper_id = state.get("selected_paper_id")
    
    if not pdf_url or not paper_id:
        return {"error_message": "No PDF URL or Paper ID provided to the parser."}

    #makes a directory if not exists to store the chunks
    os.makedirs("data", exist_ok=True)
    
    #safe paper id for the legacy arxiv ids
    safe_paper_id = paper_id.replace("/", "_")
    #pdf path
    pdf_path = f"data/{safe_paper_id}.pdf"
    
    try:
        #A custom user agent with 30s timeout 
        headers = {"User-Agent": "ArxivAgent/1.0 (Contact: hr@8byte.ai)"}
        response = requests.get(pdf_url, stream=True, headers=headers, timeout=30)
        response.raise_for_status()
        
        #chunks are written into the pdf sequentially
        with open(pdf_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        markdown_text = pymupdf4llm.to_markdown(pdf_path)
        
        if not markdown_text or len(markdown_text.strip()) < 100:
            return {"error_message": "PDF parsing failed: Document appears empty or unreadable."}
            
        # cuts the References/Bibliography section to save tokens and reduce latency
        split_match = re.search(r'\n#+\s*(References|Bibliography)\s*\n', markdown_text, re.IGNORECASE)
        if split_match:
            markdown_text = markdown_text[:split_match.start()]

        #updates the state with parsed text            
        return {"parsed_text": markdown_text}

    except requests.exceptions.RequestException as e:
        return {"error_message": f"Failed to download the PDF: {str(e)}"}
    except Exception as e:
        return {"error_message": f"An error occurred during parsing: {str(e)}"}
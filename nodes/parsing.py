import os
import re
import requests
import pymupdf4llm
from core.state import AgentState

def fetch_and_parse(state: AgentState):
    """
    Fetches the PDF from arXiv and parses it. Falls back to abstract-only 
    if the PDF is a scanned image or fails to download.
    """
    pdf_url = state.get("pdf_url")
    paper_id = state.get("selected_paper_id")
    candidate_papers = state.get("candidate_papers", [])
    
    # Extract the abstract from metadata to use as a fallback
    fallback_text = "No abstract available."
    for paper in candidate_papers:
        if paper_id and (paper_id in paper["id"] or paper["id"] in paper_id):
            fallback_text = f"# Abstract\n{paper['abstract']}"
            break

    if not pdf_url or not paper_id:
        print("\n[!] Warning: Missing PDF URL. Falling back to abstract-only mode.")
        return {"parsed_text": fallback_text}

    # makes a directory if not exists to store the chunks
    os.makedirs("data", exist_ok=True)
    
    # safe paper id for the legacy arxiv ids
    safe_paper_id = paper_id.replace("/", "_")
    # pdf path
    pdf_path = f"data/{safe_paper_id}.pdf"
    
    try:
        # A custom user agent with 30s timeout 
        headers = {"User-Agent": "ArxivAgent/1.0 (Contact: hr@8byte.ai)"}
        response = requests.get(pdf_url, stream=True, headers=headers, timeout=30)
        response.raise_for_status()
        
        # chunks are written into the pdf sequentially
        with open(pdf_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        markdown_text = pymupdf4llm.to_markdown(pdf_path)
        
        # If the PDF is scanned (empty text), trigger the fallback gracefully
        if not markdown_text or len(markdown_text.strip()) < 100:
            print("\n[!] Scanned or unreadable PDF detected. Falling back to abstract-only mode.")
            return {"parsed_text": fallback_text}
            
        # cuts the References/Bibliography section to save tokens and reduce latency
        split_match = re.search(r'\n#+\s*(References|Bibliography)\s*\n', markdown_text, re.IGNORECASE)
        if split_match:
            markdown_text = markdown_text[:split_match.start()]

        # updates the state with parsed text            
        return {"parsed_text": markdown_text}

    except Exception as e:
        # Catches download failures, PyMuPDF crashes, or RequestExceptions
        print(f"\n[!] PDF fetch/parse failed ({str(e)}). Falling back to abstract-only mode.")
        return {"parsed_text": fallback_text}
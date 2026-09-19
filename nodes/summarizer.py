import os
import json
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from core.state import AgentState

#schema for LLM
class ExecutiveBriefing(BaseModel):
    title: str
    authors: list[str]
    arxiv_id: str
    publish_date: str
    link: str
    summary: str = Field(description="1-paragraph plain-English summary ('why this paper matters')")
    problem_statement: str
    methodology: list[str] = Field(description="Method/approach structured as bullet points")
    key_results: list[str] = Field(description="Key results or claims made by the authors")
    limitations: list[str] = Field(description="Explicit limitations of the study. Do not skip this.")
    follow_up_questions: list[str] = Field(description="3 to 5 suggested follow-up questions a reader might ask")

def generate_briefing(state: AgentState):
    """
    Consumes the parsed Markdown and generates a JSON executive briefing
    """
    #checks for error message
    if state.get("error_message"):
        return {}

    #extracts the parsed text and selected paper id
    parsed_text = state.get("parsed_text")
    selected_id = state.get("selected_paper_id")
    
    if not parsed_text:
        return {"error_message": "No parsed text available for summarization"}

    # Extracts metadata for the selected paper
    candidate_papers = state.get("candidate_papers", [])
    paper_meta = next((p for p in candidate_papers if p["id"] == selected_id), {})

    prompt = f"""
    You are an expert AI research analyst. Read the following parsed academic paper and 
    generate a structured executive briefing.
    
    Paper Metadata:
    Title: {paper_meta.get('title', 'Unknown')}
    Authors: {', '.join(paper_meta.get('authors', []))}
    ID: {selected_id}
    Date: {paper_meta.get('publish_date', 'Unknown')}
    Link: {paper_meta.get('pdf_url', 'Unknown')}

    Paper Content:
    {parsed_text}
    """

    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.2, # Low temperature to prevent hallucinations
            api_key=os.getenv("GEMINI_API_KEY")
        )
        
        # Force the LLM to output the exact JSON structure as of in the Pydantic model
        structured_llm = llm.with_structured_output(ExecutiveBriefing)
        briefing = structured_llm.invoke(prompt)

        # Converts the validated Pydantic object to a standard Python dictionary
        briefing_dict = briefing.model_dump()
        
        # Saves the artifacts to the persistent data folder
        os.makedirs("data", exist_ok=True)
        safe_id = selected_id.replace("/", "_") if selected_id else "unknown"
        output_path = f"data/{safe_id}_briefing.json"
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(briefing_dict, f, indent=4)

        #updates the state
        return {"summary": briefing_dict}

    except Exception as e:
        return {"error_message": f"Failed to generate summary: {str(e)}"}
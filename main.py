import os
import sys
from dotenv import load_dotenv
from graph import build_arxiv_agent

def main():
    # Loads environment variables from .env
    load_dotenv()
    
    # Validates the required API keys to prevent execution crashes
    missing_keys = []
    if not os.getenv("GEMINI_API_KEY"):
        missing_keys.append("GEMINI_API_KEY")
    if not os.getenv("GROQ_API_KEY"):
        missing_keys.append("GROQ_API_KEY")
        
    if missing_keys:
        print(f"Error: Missing API keys in .env file: {', '.join(missing_keys)}")
        print("Please add them to run the pipeline.")
        sys.exit(1)
        
    # Handles the Docker CLI args or fallbacks to the manual input
    if len(sys.argv) > 2 and sys.argv[1] == "--query":
        query = " ".join(sys.argv[2:])
    else:
        query = input("Enter an arXiv ID or research topic: ").strip()
        if not query:
            print("Query cannot be empty.")
            sys.exit(1)

    print(f"\n[1] Starting Agent Pipeline for: '{query}'")
    agent = build_arxiv_agent()
    
    # Initializes the AgentState matching TypedDict
    state = {
        "query": query,
        "is_direct_id": False,
        "candidate_papers": [],
        "selected_paper_id": None,
        "pdf_url": None,
        "parsed_text": None,
        "vector_store_path": None,
        "summary": None,
        "chat_history": [],
        "error_message": None
    }
    
    # Executes the Ingestion Graph and stream the progress
    for step in agent.stream(state):
        node_name = list(step.keys())[0]
        print(f" -> Completed node: {node_name}")
        
        if "error_message" in step[node_name] and step[node_name]["error_message"]:
            print(f"\n[!] Pipeline Error: {step[node_name]['error_message']}")
            sys.exit(1)
            
        # Updates the local state tracker
        state.update(step[node_name])
        
    # Ensures the summary was successfully generated before printing
    if not state.get("summary"):
        print("\n[!] Pipeline failed to generate summary.")
        sys.exit(1)
        
    # Outputs the structured Executive Briefing
    print("\n" + "="*60)
    print("EXECUTIVE BRIEFING")
    print("="*60)
    summary = state["summary"]
    print(f"Title: {summary['title']}")
    print(f"Authors: {', '.join(summary['authors'])}")
    print(f"ID: {summary['arxiv_id']} | Date: {summary['publish_date']}\n")
    print(f"Summary:\n{summary['summary']}\n")
    
    print("Methodology:")
    for m in summary['methodology']: 
        print(f" - {m}")
    
    print("\nLimitations:")
    for l in summary['limitations']: 
        print(f" - {l}")
    
    print("\nFollow-up Questions:")
    for f in summary['follow_up_questions']: 
        print(f" - {f}")
    print("="*60)
    
    # Launches the RAG qa loop
    print("\n[?] Entering QA Mode. Ask questions about the paper (type 'exit' to quit).")
    while True:
        try:
            question = input("\nYou: ").strip()
            if question.lower() in ['exit', 'quit']:
                print("Exiting...")
                break
            if not question:
                continue
                
            # Overwrites the state query with the users new question
            state["query"] = question  
            
            # Invoking the agent again automatically routes to qa_node because vector_store_path exists
            result = agent.invoke(state)
            state.update(result)
            
            # Extracts and prints the latest AI response from the appended history
            ai_response = state["chat_history"][-1]["content"]
            print(f"\nAgent: {ai_response}")
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\n[!] Error during QA loop: {str(e)}")

if __name__ == "__main__":
    main()
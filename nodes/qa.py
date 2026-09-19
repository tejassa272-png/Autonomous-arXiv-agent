import os
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank
from langchain_groq import ChatGroq
from core.state import AgentState

def answer_question(state: AgentState) -> dict:
    """
    Retrieves relevant chunks from Qdrant, reranks them using FlashRank, 
    and generates a grounded answer using Groq (Llama 3).
    """
        
    query = state["query"]
    qdrant_path = state.get("vector_store_path")
    paper_id = state.get("selected_paper_id")
    
    if not qdrant_path or not paper_id:
        return {"error_message": "Vector store not initialized for QA."}
        
    try:
        # Connects to the local embedded Qdrant
        client = QdrantClient(path=qdrant_path)
        safe_paper_id = paper_id.replace("/", "_")
        collection_name = f"arxiv_{safe_paper_id}"
        
        # Initializes the Embeddings and Vector Store
        embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
        vector_store = QdrantVectorStore(
            client=client,
            collection_name=collection_name,
            embedding=embeddings
        )
        
        # Fetches the top 10 chunks
        retriever = vector_store.as_retriever(search_kwargs={"k": 10})
        
        # Flashrank selects the top 3 chunks
        compressor = FlashrankRerank(top_n=3)
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=retriever
        )
        
        # Retrieves docs
        retrieved_docs = compression_retriever.invoke(query)
        
        # Empty retrieval safety check
        if not retrieved_docs:
            return {
                "chat_history": [
                    {"role": "user", "content": query},
                    {"role": "assistant", "content": "The answer is not found in the provided paper context."}
                ]
            }
            
        # Extracts text only once
        context = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])
        
        # History truncation -- limits to last 6 messages ONLY once
        history_text = ""
        recent_history = state.get("chat_history", [])[-6:]
        for msg in recent_history:
            role = "User" if msg["role"] == "user" else "Assistant"
            history_text += f"{role}: {msg['content']}\n"

        # RAG Prompt
        prompt = f"""
        You are a strict, expert AI research assistant answering questions about a specific paper.
        Answer the user's question using ONLY the provided extracted context chunks from the paper.
        If the answer is not explicitly contained within the context below, explicitly say "The answer is not found in the provided paper context" rather than hallucinating.
        
        Context from paper:
        {context}
        
        Conversation History:
        {history_text}
        
        User Question: {query}
        """
        
        # Used Groq for low latency generation
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0, 
            api_key=os.getenv("GROQ_API_KEY")
        )
        
        response = llm.invoke(prompt)
        
        # Automatically adds to the previous history
        new_history = [
            {"role": "user", "content": query},
            {"role": "assistant", "content": response.content}
        ]
        
        # Updates the state
        return {"chat_history": new_history}
        
    except Exception as e:
        return {"error_message": f"QA engine failed: {str(e)}"}
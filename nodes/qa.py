import os
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from core.state import AgentState

def answer_question(state: AgentState) -> dict:
    """
    Retrieves relevant chunks from Qdrant, reranks them using FlashRank, 
    and generates a grounded answer using Groq.
    """
        
    query = state.get("query", "")
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
        
        # Fetches the top 10 chunks manually
        retriever = vector_store.as_retriever(search_kwargs={"k": 10})
        initial_docs = retriever.invoke(query)
        
        # Reranks chunks
        compressor = FlashrankRerank(top_n=3)
        retrieved_docs = compressor.compress_documents(documents=initial_docs, query=query)
        
        # Empty retrieval safety check
        if not retrieved_docs:
            return {
                "chat_history": [
                    {"role": "user", "content": query},
                    {"role": "assistant", "content": "The answer is not found in the provided paper context."}
                ]
            }
            
        # Extracts text
        context = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])
        
        # History truncation -- limits to last 6 messages ONLY once
        history_text = ""
        recent_history = state.get("chat_history", [])[-6:]
        for msg in recent_history:
            role = "User" if msg["role"] == "user" else "Assistant"
            history_text += f"{role}: {msg['content']}\n"

        # Formally separate System and Human messages to prevent Groq 400 errors
        messages = [
            SystemMessage(content="You are a strict, expert AI research assistant. Answer the user's question using ONLY the provided extracted context chunks from the paper. If the answer is not explicitly contained within the context, explicitly say 'The answer is not found in the provided paper context' rather than hallucinating."),
            HumanMessage(content=f"Context from paper:\n{context}\n\nConversation History:\n{history_text}\n\nUser Question: {query}")
        ]
        
        # Switch back to the active, supported Groq model
        llm = ChatGroq(
            model="openai/gpt-oss-120b",
            temperature=0, 
            api_key=os.getenv("GROQ_API_KEY")
        )
        
        response = llm.invoke(messages)
        
        new_history = [
            {"role": "user", "content": query},
            {"role": "assistant", "content": response.content}
        ]
        
        return {"chat_history": new_history}
        
    except Exception as e:
        # Fallback history injection: Safely passes the error back to the terminal chat loop
        fallback_history = [
            {"role": "user", "content": query},
            {"role": "assistant", "content": f"[API Error: {str(e)}] - Please try asking again."}
        ]
        return {"error_message": f"QA engine failed: {str(e)}", "chat_history": fallback_history}
    
    finally:
            #releases the file lock so consecutive queries don't flake
            client.close()
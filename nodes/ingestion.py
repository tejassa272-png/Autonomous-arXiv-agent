import os
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from core.state import AgentState

def chunk_and_embed(state: AgentState) -> dict:
    """
    Chunks the parsed Markdown using header-aware splitting, 
    embeds chunks locally with FastEmbed, and indexes them in embedded Qdrant.
    """

    parsed_text = state.get("parsed_text")
    paper_id = state.get("selected_paper_id")
    
    if not parsed_text or not paper_id:
        return {"error_message": "Missing parsed text or paper ID for vector ingestion."}

    try:
        # Header-Aware Splitting (preserves section structure)
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]
        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on, 
            strip_headers=False
        )
        header_splits = markdown_splitter.split_text(parsed_text)

        # Secondary Recursive Splitter (ensures chunks fit embedding context)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150
        )
        splits = text_splitter.split_documents(header_splits)

        if not splits:
            return {"error_message": "Document chunking produced zero splits"}

        # Local Qdrant Directory
        safe_paper_id = paper_id.replace("/", "_")
        qdrant_path = f"data/qdrant_{safe_paper_id}"
        os.makedirs(qdrant_path, exist_ok=True)
        collection_name = f"arxiv_{safe_paper_id}"

        # Local FastEmbed Embeddings -- BAAI/bge-small-en-v1.5
        embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")

        # Ingests documents into Qdrant (Let LangChain handle the client creation via path)
        QdrantVectorStore.from_documents(
            documents=splits,
            embedding=embeddings,
            path=qdrant_path, 
            collection_name=collection_name
        )
        
        # updates the state
        return {"vector_store_path": qdrant_path}

    except Exception as e:
        return {"error_message": f"Failed to chunk and embed document: {str(e)}"}
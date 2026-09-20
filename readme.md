# Autonomous arXiv Paper Digest & QA Agent

An autonomous, containerized AI research agent built using **LangGraph**, **Gemini 2.5 Flash**, **Groq (LLaMA 3)**, **Qdrant (Local Mode)**, and **FlashRank**. This agent fetches academic papers from arXiv via natural language topics or direct IDs, parses them with robust fallback handling, generates structured executive briefings, and provides a grounded RAG-based conversational Q&A loop.

---

## 1. Architecture & State Graph

The agent is designed as an explicit stateful graph using LangGraph. The pipeline passes a shared `AgentState` dictionary across modular, single-responsibility nodes.

```text
                  [ User Input (Topic or ID) ]
                               │
                               ▼
                    ┌─────────────────────┐
                    │     parse_intent    │  <-- Detects if input is a direct ID or topic
                    └─────────────────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   fetch_metadata    │  <-- Queries official arXiv API (Atom feed)
                    └─────────────────────┘
                               │
                       [ Is Direct ID? ]
                      /                 \
                   (Yes)                (No)
                    /                     \
                   ▼                       ▼
    ┌────────────────────────┐  ┌─────────────────────┐
    │  (Skip LLM Selection)  │  │    select_target    │ <-- Gemini Flash chooses best paper
    └────────────────────────┘  └─────────────────────┘
                    \                     /
                     \                   /
                      ▼                 ▼
                    ┌─────────────────────┐
                    │   fetch_and_parse   │  <-- Downloads PDF, parses via PyMuPDF4LLM
                    └─────────────────────>      (Graceful fallback to abstract if unreadable)
                               │
                               ▼
                    ┌─────────────────────┐
                    │   chunk_and_embed   │  <-- Header-aware splitting + FastEmbed (Local)
                    └─────────────────────>      Stored in Embedded Qdrant
                               │
                               ▼
                    ┌─────────────────────┐
                    │  generate_briefing  │  <-- Gemini Flash generates structured JSON briefing
                    └─────────────────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │      qa_node        │  <-- Interactive RAG loop with FlashRank + Groq LLaMA 3
                    └─────────────────────┘
```

### State Schema (`AgentState`)
* **`query`** (str): Initial user prompt or arXiv ID.
* **`is_direct_id`** (bool): Parsed flag indicating direct ID lookup vs. semantic search.
* **`selected_paper_id`** (str): Locked arXiv ID for processing.
* **`pdf_url`** (str): Direct link to the source PDF.
* **`candidate_papers`** (list): Metadata payload from arXiv.
* **`parsed_text`** (str): Cleaned markdown text of the paper (or abstract fallback).
* **`vector_store_path`** (str): Filesystem path to the local Qdrant collection.
* **`summary`** (dict): Structured Pydantic-validated executive briefing JSON.
* **`chat_history`** (list): Bounded message history (`[-6:]`) for multi-turn QA memory.
* **`error_message`** (str): Error containment flag for graceful exception management.

---

## 2. Setup & Run Instructions

### Prerequisites
* Docker installed and running locally.
* API Keys:
  * **Google AI Studio API Key** (Free tier for Gemini 2.5 Flash).
  * **Groq API Key** (Free tier for LLaMA 3 Q&A).

### Step 1: Configure Environment Variables
Create a file named `.env` in the root directory and add your keys without quotation marks:
```env
GEMINI_API_KEY=your_google_api_key_here
GROQ_API_KEY=your_groq_api_key_here
```

### Step 2: Build the Docker Image
```bash
docker build -t arxiv-agent .
```

### Step 3: Run the Agent
Mount a local `data/` directory to persist downloaded PDFs, generated briefing JSONs, and local Qdrant vector databases across runs:
```bash
docker run -it -v $(pwd)/data:/app/data --env-file .env arxiv-agent
```

---

## 3. Example Run

### Input
```text
Enter an arXiv ID or research topic: 2106.09685
```

### Pipeline Execution Logs
```text
[1] Starting Agent Pipeline for: '2106.09685'
 -> Completed node: parse_intent
 -> Completed node: fetch_metadata
 -> Completed node: select_target
 -> Completed node: fetch_and_parse
 -> Completed node: chunk_and_embed
 -> Completed node: generate_briefing
```

### Executive Briefing Output (Excerpt)
```text
============================================================
EXECUTIVE BRIEFING
============================================================
Title: LORA: LOW-RANK ADAPTATION OF LARGE LANGUAGE MODELS
Authors: Edward Hu, Yelong Shen, Phillip Wallis, et al.
ID: 2106.09685 | Date: 2021-06-17

Summary:
This paper introduces Low-Rank Adaptation (LoRA), a novel method to efficiently adapt large language models (LLMs) to downstream tasks by freezing pre-trained weights and injecting trainable rank decomposition matrices. This reduces trainable parameters by up to 10,000 times and GPU memory requirements by 3 times without adding inference latency.
...
```

### Sample QA Exchanges
```text
[?] Entering QA Mode. Ask questions about the paper (type 'exit' to quit).

You: Compared to GPT-3 175B fine-tuned with Adam, by what factor does LoRA reduce the number of trainable parameters?
Agent: LoRA reduces the number of trainable parameters by a factor of 10,000.

You: And by what factor does it reduce the GPU memory requirement?
Agent: The paper reports that VRAM consumption during training for GPT-3 175B drops from 1.2 TB to 350 GB, which is a reduction factor of roughly 3.4× (or ~2/3 reduction).

You: What does the paper say about training autonomous driving vision models?
Agent: The answer is not found in the provided paper context.
```

---

## 4. Design Decisions & Tradeoffs

* **Dual-LLM Architecture (Gemini + Groq):** We leverage **Gemini 2.5 Flash** for its massive context window during the heavy summarization node (passing up to 40,000 characters of parsed markdown), and **Groq (LLaMA 3)** for the interactive Q&A loop to achieve sub-second response latency via specialized LPUs.
* **Local Embedded Qdrant & FastEmbed:** By utilizing local file-based Qdrant and FastEmbed (`BAAI/bge-small-en-v1.5`), the system runs entirely offline without requiring cloud-hosted vector databases or paid vector services.
* **FlashRank Document Compression:** Rather than blindly dumping all retrieved chunks into the LLM context window, the system uses FlashRank to re-rank and filter down to the top 5 most relevant chunks, eliminating noise and token bloat.
* **Graceful Failure Degradation:** If an arXiv PDF is scanned, encrypted, or fails to download cleanly, the pipeline catches the exception and automatically falls back to an abstract-only briefing instead of crashing the run.
* **Context Truncation & State Protection:** Chat history is capped at the last 6 messages (`[-6:]`) to prevent context window overflow during extended Q&A sessions.
* **Resource Management & File Locking:** Embedded Qdrant clients are explicitly closed via a `try...finally` block after each query to prevent flat-file SQLite/RocksDB locking conflicts.

### Known Limitations & Edge Cases
* **Legacy arXiv IDs:** The intent parser regex (`\d{4}\.\d{4,5}(?:v\d+)?`) is optimized for post-2007 identifiers. Legacy IDs (e.g., `hep-th/9901001`) will silently fall through to topic-search mode.
* **Free-Tier Rate Limits:** 
  * *Gemini 2.5 Flash:* 15 Requests Per Minute (RPM), 1 million Tokens Per Minute (TPM). Large parsed papers can approach token limits if executed in rapid succession.
  * *Groq LLaMA 3:* 30 RPM and 14,400 TPM. Mitigated effectively by FlashRank chunk compression.
* **Artifact & Cache Management:** Currently, all downloaded PDFs, JSON briefings, and Qdrant vector directories accumulate flatly inside a local `data/` folder without automatic TTL expiration.

---

## 5. What I Would Do With More Time
1. **Multi-Paper Comparative Synthesis:** Extend the selection node to ingest and cross-compare 2 or 3 competing papers on a given topic simultaneously.
2. **Vision-Language OCR Fallback:** Integrate Tesseract OCR directly into `parsing.py` for scanned math-heavy PDFs instead of falling back strictly to abstracts.
3. **Automated Cache Management:** Implement a lightweight cache manager with a CLI cleanup command (e.g., `clean-cache`) to purge old Qdrant vector indices and temporary PDFs from the `data/` directory.
4. **Persistent Chat Sessions:** Implement a lightweight SQLite session layer to persist chat histories across container restarts.
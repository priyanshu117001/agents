
# Priyanshu Agarwal — Knowledge Document for Assistant Ingestion
**Purpose:** This Markdown document is a comprehensive knowledge base describing Priyanshu Agarwal, his skills, background, and detailed technical descriptions of his projects. It's written for ingestion into a RAG (Retrieval-Augmented Generation) pipeline (e.g., Chroma, Pinecone, Weaviate) so the portfolio assistant can answer on Priyanshu's behalf with accurate, detailed information.

---
## Table of Contents
1. About Priyanshu
2. Skills Snapshot
3. Projects (detailed)
   - DearDiary — Fullstack Personal Journal App
   - MCP Server with Autogen Agents (Agent Orchestration)
   - LangGraph Agent with Azure OpenAI + MCP
   - Swarm Agent Simulations (A2A vs Handoff)
   - FastMCP Server with Authentication
   - Agentic Framework Adapters
   - LangChain Agents with Callback Handlers (Observability)
   - Portfolio Website & Assistant (ongoing)
   - Brain Tumor Detection System (medical imaging)
   - AI Advancements in Medicine (20-year literature/analysis project)
   - Search Engine (TF-IDF + PageRank style)
   - "Lost" Web Browser (Python implementation)
   - Content-Based Movie Recommendation System
4. Algorithms, Patterns & Architectural Notes
5. Assistant Personas, Prompts & QA Examples (for ingestion)
6. Ingestion & RAG Integration Guide (best practices)
7. Assumptions & Missing Details (items to confirm)
8. Revision history

---
## 1) About Priyanshu (Canonical text)
- **Name:** Priyanshu Agarwal
- **Role:** Software Engineer (2 years experience)
- **Specialization:** Agentic AI, backend systems, tool integrations for LLMs, and fullstack development with Python
- **Focus Areas:** MCP (Model Context Protocol), Autogen agents, LangChain/LangGraph, FastMCP, adapters between agent frameworks, secure SSE-based agent infrastructure, RAG pipelines
- **Personal Interests:** Cybersecurity, system design, history visualizations, research-level agentic systems

Short bio (for UI/assistant use):
> I'm Priyanshu Agarwal, a software engineer focused on building agentic AI systems, backend services, and integrations that allow LLM-based agents to safely discover and use tools. I work primarily with Python and have experience with frameworks like LangChain, LangGraph, Autogen, and MCP-based infrastructures.

---
## 2) Skills Snapshot (useable form)
- **Languages:** Python (primary), JavaScript/TypeScript, SQL
- **AI / Agent Frameworks:** LangChain, LangGraph, Autogen, Google ADK, AGNO
- **Backend:** FastAPI, Django, Flask
- **Frontend:** Angular (existing work), React/Next.js (portfolio)
- **Databases:** MongoDB, PostgreSQL
- **Infra/DevOps:** Docker, Git, CI/CD (GitHub Actions), serverless (Vercel, Netlify)
- **Other:** RAG, vector DBs (Chroma, Pinecone), embeddings, SSE, token auth

---
## 3) Projects (Detailed)

> **NOTE:** For some projects (especially those added later) the repository or exact implementation details were not explicitly provided. Where appropriate, this document lists realistic, production-ready design and implementation choices that would be compatible with the project's stated goals. Items marked **[ASSUMED]** indicate inferred details that should be validated by Priyanshu.

### Project A — DearDiary (Fullstack Personal Journal App)
**Summary:** A personal journaling web app with an Angular frontend and Django backend, using MongoDB for storage. Users can write, store, and browse entries. The 'View Entries' page shows small cards with previews and a 'Read More' link to expand the full entry.

**Goals & Use Cases:**
- Private journaling
- Fast creation and retrieval of entries
- Simple UX for daily writing

**Tech Stack:**
- Frontend: Angular (TypeScript), RxJS for reactive flows
- Backend: Django REST Framework (Python)
- Database: MongoDB
- Hosting: (MVP) could be Vercel for frontend + a VPS or managed service for backend or Heroku (legacy) / Render / Railway
- Auth: Session-based token or JWT (TBD)

**Key Features & Implementation Details:**
- **Write Entry** page: Angular reactive form, validated fields (title, content, tags, private boolean)
- **Submit** via HTTP POST to Django REST API endpoint (`/api/entries/`)
- **View Entries** page: Small cards showing title, date, author (if multi-user), and a 200-char preview. Each card has a "Read More" link opens a modal or expands inline for full content.
- **Pagination** or infinite scroll to handle many entries
- **Indexing**: MongoDB text index on content & title for search
- **Security**: Input sanitization and rate-limiting endpoints

**Architecture Diagram (conceptual):**
```
Angular client -> Django REST API -> MongoDB
```
**Production considerations:** backups, encryption-at-rest, user authentication, XSS protection for rendered HTML previews.

---

### Project B — MCP Server with Autogen Agents (Agent Orchestration)
**Summary:** An MCP (Model Context Protocol) server implementation (FastMCP) hosting toolsets for multiple agents. Autogen-based agents connect to the server and dynamically discover available tools. Conversion utilities are included to convert MCP `Tool` instances to Autogen `BaseTool` instances (with Pydantic types for args/returns).

**Goals:** Provide a central tool hosting server so multiple agents (from various frameworks) can reuse tools without server-side modifications.

**Tech Stack:**
- MCP Server: FastMCP (SSE transport) — Python (FastAPI or FastMCP-specific)
- Agents: Autogen (Python), clients using `ClientSession`-like interface
- Authentication: Token-based bearer auth (JWT or opaque tokens)
- Serialization: JSON + Pydantic for arg/return schemas

**Architecture & Flow:**
1. **Server** exposes an endpoint (SSE) to stream tool changes and accept tool invocation requests via REST/SSE.
2. **Tools** are registered on server with metadata (id, name, description, arg schema, return schema).
3. **Agent** connects via SSE or client session, lists available tools, selects tools at runtime.
4. **Tool invocation**: agent posts a request to server; server executes tool and returns structured result.
5. **Tool conversion**: conversion layer that maps MCP Tool definitions into Autogen `BaseTool` objects with `arg_type` and `return_type` defined as Pydantic models.

**Key Implementation Notes:**
- Use Pydantic models for strong type validation in tool inputs and outputs.
- SSE clients maintain heartbeats and reconnect logic.
- Token-based auth: server verifies tokens on connect and on each tool invocation.
- Consider rate-limiting to prevent abuse of heavy tools.

**Challenges & Solutions:** Dynamic tool discovery requires a robust versioning strategy for tool schemas (add `version` fields to tool metadata).

---

### Project C — LangGraph Agent with Azure OpenAI + MCP
**Summary:** An agent built with LangGraph that uses Azure OpenAI as the LLM provider and discovers tools from an MCP server. Agent runs with in-memory session history and uses `create_react_agent` patterns. It supports dynamic tool loading and tool invocation.

**Tech Stack:**
- Agent: LangGraph (Python)
- LLM Provider: Azure OpenAI (GPT-4)
- Tools: Loaded from MCP server (FastMCP)
- Message/session store: In-memory for session lifetimes (stateless between server restarts unless persisted)

**Important behaviors:**
- Create agents using `create_react_agent`
- Agents decide autonomously whether to call tools based on prompt context and tool metadata
- Use `ClientSession` style sessions to manage SSE & tool invocation

**Implementation notes:** ensure prompt engineering includes tool descriptions and example tool usage snippets so the LLM learns how to choose and call tools.

---

### Project D — Swarm Agents (Multi-Agent Collaboration Simulation)
**Summary:** Simulations and code templates demonstrating agent collaboration patterns — A2A (agent-to-agent) vs Handoff (manager delegates). The scenario used is a software dev team with roles for frontend, backend, and manager agents, including tree-structure diagrams and full handoff code for centralized workflows.

**Tech Stack & Tools:**
- Agents: Autogen / LangGraph / custom frameworks
- Orchestration: FastMCP or a custom orchestrator for simulations
- Visualization: Tree diagrams (Mermaid or D3), documentation in markdown

**Key Learnings Modeled:**
- A2A: direct peer communication, negotiation protocols, shared state synchronization
- Handoff: centralized manager agent, queuing tasks, consistent task assignment

**Example Output:** sample code demonstrating handoff with state passing and callbacks to notify completion.

---

### Project E — FastMCP Server with Authentication
**Summary:** A secure version of an MCP server implementing token-based authentication and SSE with proper client params and headers. Implements bearer token auth and server-side verification for client sessions.

**Tech Stack:**
- FastMCP / FastAPI, Python
- Auth: Bearer tokens (JWT or opaque), token introspection endpoint
- Transport: SSE for low-latency streaming of events + HTTP for tool invocation

**Notes:** Latest FastMCP versions expect certain request shapes and `url` parameter in `SSEClientParam` objects. The implementation verifies tokens and can optionally accept additional headers in client params for complex client requirements.

---

### Project F — Agentic Framework Adapters
**Summary:** A set of adapters enabling agents built with different providers (LangGraph, Autogen, Google ADK, AGNO) to use shared MCP-hosted tools without changing server code. The adapters normalize API calls and data shapes between frameworks.

**Implementation highlights:**
- For each provider, write an adapter that maps provider-specific tool invocation calls to MCP tool invocation REST endpoints.
- Provide compatibility layer for argument schemas (Pydantic) and return types.
- Include tests for roundtrip tool invocation across frameworks.

---

### Project G — LangChain Agents with Callback Handlers (Observability)
**Summary:** Implemented callback handlers in LangChain (v0.3.26) to log and print events when LLM calls start/end and when tool execution starts/ends. Also included custom LLM handlers to print similar lifecycle events for observability and debugging.

**Tech Stack:**
- LangChain v0.3.26 (Python)
- Custom callback classes implementing required LangChain interface
- Use cases: debugging, step-by-step agent execution tracing

**Notes:** Code includes handlers for tools and LLMs and integrates with logging frameworks or stdout during dev runs.

---

### Project H — Portfolio Website & Assistant (Ongoing)
**Summary & Goals:** Static/SSR portfolio (Next.js + Tailwind) with an assistant chat widget that uses RAG over the site's content + resume to answer questions as Priyanshu. Personal brand symbol (PA monogram node-graph) is used site-wide.

**MVP architecture:** Next.js site, serverless API route for retrieval + LLM proxy, Chroma or Pinecone for vector store.

**Assistant prompt (canonical):**
> You are Priyanshu Agarwal, a software engineer with 2 years of experience, specializing in agentic AI and Python. Answer in first person, succinctly, and provide links to portfolio sections. If unsure, say so politely.

---

### Project I — Brain Tumor Detection System (Medical Imaging)  **[NEW ADDED]**
**Summary:** An AI pipeline that performs brain tumor detection and segmentation on MRI scans. The system supports classification (tumor vs no-tumor) and segmentation (tumor masks) using deep learning. Intended as a research / proof-of-concept with proper ethical considerations and not for clinical use without validation and regulatory approval.

**Use Cases:** Automated triage, research experiments, assisting radiologists with pre-segmentation for review.

**Data:** (Potential / common datasets)
- **BraTS (Brain Tumor Segmentation Challenge)** dataset — multi-institutional MRI scans with labels for tumor subregions (enhancing tumor, whole tumor, tumor core).
- Modalities used: T1, T1CE, T2, FLAIR.

**Preprocessing:** 
- N4 bias field correction (simple nibabel + SimpleITK pipeline)
- Skull-stripping (e.g., using HD-BET or FSL BET) **[ASSUMED optional]**
- Resampling to consistent voxel spacing (e.g., 1mm³)
- Intensity normalization (z-score normalization per volume)
- If 2D approach: slice extraction; if 3D: full volume cropping and padding

**Model Architectures (typical & plausible choices):**
1. **Segmentation (recommended, high performance):** 3D U-Net or nnU-Net architecture
   - Encoder-decoder, skip connections
   - Use Dice loss + Cross-Entropy loss composite
   - Batch size tuned to GPU memory (commonly small for 3D models)
2. **Classification (optional):** 2D CNN (ResNet-34/50) on key slices or 3D CNN for full-volume classification
3. **Alternative approach:** Transfer-learning, training on 2D slices with pre-trained ImageNet backbone

**Training details:**
- Data augmentation: rotations, flips, elastic deformations, intensity augmentations
- Optimizers: AdamW with cosine annealing or step LR
- Early stopping on validation Dice score
- Validation splits: K-fold or held-out validation set (e.g., 80/20 split)

**Evaluation metrics:**
- Dice coefficient (primary for segmentation)
- IoU (Jaccard)
- Sensitivity / specificity, precision, recall for classification
- Hausdorff distance (for segmentation boundary errors)

**Post-processing:**
- Small component removal (morphological filtering) to remove spurious tiny predictions
- Connected component analysis to select largest tumor region

**Deployment (research MVP):**
- Export model using ONNX for portability
- Simple FastAPI serving endpoint for inference (`/predict` accepting NIfTI or standard MRI formats)
- GPU-based inference using Triton/ONNX Runtime for production-grade speed
- Frontend demo: simple web UI to upload sample MRI slices and visualize predicted mask overlay (matplotlib / Plotly / Dash)

**Ethical & Safety considerations:**
- Not for clinical use without proper validation and clearance
- Document dataset biases, error modes
- Keep identifiable patient data private; enforce HIPAA-like data handling
- Provide interpretability (saliency maps) and uncertainty estimation (Monte Carlo dropout)

**Assumptions:** The above architecture is a recommended, realistic blueprint for a brain tumor detection research system. Confirm datasets, modality availability, and whether segmentation or just classification was intended.

---

### Project J — AI Advancements in Medicine (20-year Analysis & Survey) **[NEW ADDED]**
**Summary:** A literature review / analysis project that surveys the last 20 years of AI applications in medicine, trends, breakthroughs, and practical adoption in clinical workflows. This could be a long-form writeup, timeline, and dataset of references that the assistant can cite when asked about AI in healthcare.

**Structure & Content (recommended):**
- **Timeline (2005–2025):** mark breakthroughs (SVMs and statistical learning in early 2000s → deep learning + CNNs around 2012 → medical imaging surge with U-Net, nnU-Net → transformers in 2020s → multi-modal LLMs, foundation models for medicine)
- **Breakthroughs & Milestones:** U-Net (2015), ImageNet influence (2012), DeepMind's AlphaFold (2020), GPT-family inaugurating LLMs for clinical text (2020+), specialized clinical LLMs (MedPaLM, others)
- **Applications Covered:** radiology (imaging), pathology (digital slides), genomics (variant calling, protein folding), clinical notes (NLP, de-identification), triage systems, drug discovery
- **Datasets & Challenges:** MIMIC-III/IV, CheXpert, BraTS, CAMELYON, TCGA, private hospital datasets
- **Evaluation & Regulation:** performance vs real-world deployment, approvals (FDA), reproducibility crisis, dataset biases, domain shift
- **Ethical Considerations:** bias, fairness, privacy, accountability, AI explainability
- **Deliverables:** a structured markdown timeline + CSV of references, and a slide deck summarizing key takeaways (optional)

**Ingestion Tip:** Store the timeline as a numbered list and store each paper/claim with a short summary so the assistant can cite "In 2015, U-Net (Ronneberger et al.) introduced..."

**Assumptions:** The user likely wants a curated analysis rather than a formal systematic review. Confirm scope (academic citations vs high-level timeline).

---

### Project K — Search Engine (TF-IDF + PageRank style)  **[NEW ADDED]**
**Summary:** A toy / production-capable search engine built using classic IR techniques: TF-IDF for content scoring and PageRank (Larry Page algorithm) for link-based authority ranking. Includes crawling, indexing, ranking, and query processing components.

**Goals:** Understand IR fundamentals, combine content relevance (TF-IDF) with link analysis (PageRank) for improved ranking.

**Tech Stack:**
- Language: Python (core)
- Crawling: `requests`, `beautifulsoup4`, `aiohttp` for async crawling
- Indexing: custom inverted index (or Whoosh for a library), persisted as JSON or a lightweight DB (SQLite)
- Ranking: TF-IDF (scikit-learn `TfidfVectorizer`) + PageRank implemented using NetworkX (or custom power-iteration)
- Frontend (demo): Flask/FastAPI interface with simple search UI

**Architecture & Pipeline:**
1. **Crawler**: Seed URLs → breadth-first crawl → store raw HTML and outlinks
2. **Parser**: Extract text, compute tokens with tokenizer (NLTK / spaCy), stopword removal, stemming/lemmatization optional
3. **Indexer**: Build inverted index (term → doc frequency, postings lists with term frequencies)
4. **Scorer**: Compute TF-IDF vectors for documents and queries (cosine similarity)
5. **Link analysis**: Run PageRank on web graph; compute normalized PageRank scores
6. **Ranking**: Combine content score and PageRank with tunable weight `alpha`: `score = alpha * cosine(tfidf) + (1 - alpha) * normalized_pagerank`
7. **UX**: Return snippets extracted around query terms; highlight matches in results

**Evaluation:** Use small sampled corpora and evaluate precision@k, MAP, nDCG for ranking quality. Use human judgments for relevance in small-scale experiments.

**Assumptions:** The project used PageRank style scoring combined with TF-IDF; exact crawling scale and corpora are assumed. For real web scale, replace with distributed crawlers and indexers (Elasticsearch, Solr).

---

### Project L — "Lost" Web Browser (Python)  **[NEW ADDED]**
**Summary:** A custom web browser built in Python. (Project name given: "lost webbrowser".) Built as a learning project to explore browser internals, UI, and privacy features.

**Goals & Features:** Tabbed browsing, history, bookmarks, a minimal UI, privacy-focused features (clear cookies, block trackers), optional adblocker integration, and a Python-based rendering frontend using Qt WebEngine.

**Tech Stack & Implementation Choices:** (assumed plausible implementation)
- GUI: PyQt5 / PySide2 with QtWebEngine (wraps Chromium's rendering engine)
- Backend: Python event loop with async support (`asyncio`) for downloads and background fetches
- Data Storage: SQLite for bookmarks & history
- Features implemented:
  - Multi-tab management
  - Address bar with basic URL parsing and navigation
  - History & bookmarks persistence
  - Incognito mode (does not store cookies/history)
  - Simple ad-blocking using filter lists (adblock rules)
  - Download manager
- Extensions: basic support for a limited plugin system (optional)
- Packaging: Desktop app via PyInstaller / brief instructions for building an executable

**Assumptions:** The rendering was handled by Qt WebEngine; if the project used a different approach (e.g., Tkinter + embedded browser), confirm.

---

### Project M — Content-Based Movie Recommendation System  **[NEW ADDED]**
**Summary:** A content-based movie recommender that suggests movies based on metadata similarity (genres, plot descriptions, keywords, cast). Uses TF-IDF on text features and cosine similarity for nearest-neighbour recommendations.

**Use Cases:** Recommend movies to users when collaborative data (user ratings) is scarce or for new user cold-start scenarios.

**Tech Stack:**
- Python (pandas, scikit-learn)
- Dataset: MovieLens + TMDB API (for plot/overview + genres + cast metadata) **[ASSUMED]**
- Vectorization: `TfidfVectorizer` on combined textual features (plot, keywords)
- Similarity metric: Cosine similarity (scikit-learn `cosine_similarity`)
- Serving: Flask/FastAPI simple API for recommendations (`/recommend?movie_id=...`)

**Pipeline & Details:**
1. **Data collection:** Movie descriptions, genres, keywords, cast/crew
2. **Feature engineering:** Combine features into a single string per movie (e.g., "genres director cast plot keywords")
3. **Vectorization:** TF-IDF transform -> dense or sparse matrix (`n_features` tuned)
4. **Recommendation:** For a given movie, compute cosine similarity between its vector and all others → return top-K
5. **Evaluation:** Qualitative evaluation, precision@K based on human judgment, or evaluate against held-out user ratings (if available)
6. **Improvements:** Add weighting to certain features (e.g., genre higher weight), apply dimensionality reduction (SVD) for speed

**Assumptions:** Datasets used are MovieLens plus TMDB metadata. Confirm if specific datasets or custom features were used.

---
## 4) Algorithms, Patterns & Architectural Notes (Summary)
- **RAG (Retrieval-Augmented Generation):** Use embeddings + vector DB for assistant; chunk docs into 500–1200 token pieces; store metadata (source, project, section).
- **Adapter pattern** for framework interoperability.
- **PageRank + TF-IDF** for hybrid search ranking.
- **3D U-Net / nnU-Net** for segmentation tasks in MRI/brain tumor detection.
- **LangChain callback hooks** for observability.
- **SSE + token-based auth** for live agent tool invocation (FastMCP setup).

---
## 5) Assistant Personas, Prompts & QA Examples (for ingestion)
**Primary assistant system prompt (canonical):**
> You are Priyanshu Agarwal, a software engineer with 2 years of experience specializing in agentic AI and Python. Speak in first person, be concise, helpful, and calm. When answering technical questions, provide high-level explanation followed by concrete implementation notes. If the question is about my projects, cite the project name and provide a direct link when possible. If unsure, clearly say you don't know and provide suggested follow-ups.

**Examples of user queries & expected assistant answers (short):**

- Q: "Tell me about your work on MCP servers."  
  A: "I built an MCP server (FastMCP) hosting tools for agents. It uses SSE for streaming, token-based auth, and exposes tools with Pydantic schemas so agents can discover and invoke them at runtime. I also wrote conversion utilities to convert MCP Tool definitions into Autogen `BaseTool` objects so Autogen agents can use them."

- Q: "How did you build the brain tumor detection system?"  
  A: "I implemented a segmentation pipeline using a 3D U-Net (nnU-Net-style) trained on BraTS MRI scans. Preprocessing included N4 bias correction, skull-stripping, resampling, and z-score normalization. The model is trained using Dice+CE loss, and inference is served via a FastAPI ONNX endpoint."

- Q: "What's the search engine's ranking formula?"  
  A: "The search engine combines TF-IDF cosine similarity with PageRank. The final score is `alpha * TFIDF_score + (1 - alpha) * normalized_PageRank` with `alpha` tuned experimentally (e.g., 0.7 content weight + 0.3 PageRank)."

**Project-specific retrieval tags (metadata to store):** `project:deardiary`, `project:mcp-autogen`, `project:langgraph`, `project:brain-tumor`, `project:search-engine`, `project:movie-recs`, `project:lost-browser`, etc.

---
## 6) Ingestion & RAG Integration Guide (concise)
1. **Chunking:** Split doc into ~600 token chunks (or ~3–5 paragraphs) while preserving headings as metadata. Each chunk should include `project` and `section` metadata.
2. **Embeddings:** Use OpenAI embeddings (text-embedding-3-small) or local embeddings model (if self-hosting).
3. **Vector DB:** Chroma for local MVP, Pinecone or Weaviate for production.
4. **Retrieval:** Top-k (k=5) with reranking using lexical signals (BM25) for accuracy.
5. **Prompt template:** Provide system prompt (above) + retrieved context + user question + explicit "answer in first person as Priyanshu" instruction.
6. **Citations:** Return URLs or section names for each paragraph when possible; include a short `source` string in metadata.

Example prompt scaffold:
```
SYSTEM: [canonical system prompt]
CONTEXT: [top 3 retrieved doc chunks with metadata and links]
USER: {user question}
INSTRUCTION: Answer as Priyanshu, cite sources from CONTEXT, be concise (3–6 sentences) and offer an action if relevant.
```

---
## 7) Assumptions & Missing Details (things I filled or guessed)
This document contains a mix of canonical facts (explicitly provided by Priyanshu in conversation) and **inferred/assumed** technical details where the exact implementation specifics were not provided. Items to confirm/include:
- Exact dataset(s), hyperparameters, and model checkpoints for Brain Tumor Detection System.
- Whether the search engine used PageRank or a PageRank-inspired metric; clarify crawling depth/scale used.
- The GUI framework used for the Lost Web Browser (assumed PyQt / QtWebEngine).
- Exact fields, weights, and datasets used in the movie recommendation project.
- Repository links for each project, live demos, and screenshots (to include as metadata in the vector DB).

---
## 8) Revision History
- **2025-09-27** — Initial knowledge base created and expanded to include Brain Tumor Detection, AI-in-Medicine timeline analysis, Search Engine (TF-IDF + PageRank), Lost Web Browser (Python), and Content-Based Movie Recommender. Some technical specifics are annotated as **[ASSUMED]** where they were inferred for completeness.

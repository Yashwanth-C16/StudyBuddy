# 📚 StudyBuddy

An AI-powered study assistant that lets you chat with your own PDFs — textbooks, notes, or question papers — and get answers grounded in the actual document, with page-level source citations so you can verify every answer.

**Live demo:** [Add your Streamlit Cloud link here]

---

## What it does

- Upload a PDF (textbook chapter, notes, PYQs) from the sidebar.
- Ask questions about it in a chat interface.
- Answers are generated using Retrieval-Augmented Generation (RAG) — the app retrieves the most relevant sections of your PDF and passes them to the LLM as context, instead of relying on the model's general knowledge.
- Every answer shows which page(s) it was sourced from, so you can go verify it directly in your material.
- Supports natural follow-up questions (e.g. "explain that more simply") using short-term conversation memory.

## Why I built it

Most "chat with PDF" demos are shallow wrappers around an LLM call. I wanted something I'd actually use before exams — and to build a project that forced me to deal with real engineering problems: deployment constraints, caching expensive operations, error handling, and making answers verifiable instead of just trusting the model.

## Tech stack

| Layer | Tool |
|---|---|
| UI | Streamlit (chat interface) |
| LLM | Groq (`openai/gpt-oss-120b`) |
| Embeddings | Google Gemini (`gemini-embedding-001`) |
| Vector store | Chroma |
| Orchestration | LangChain (LCEL) |
| PDF parsing | PyMuPDF |

## Architecture

```
PDF Upload (sidebar)
   → PyMuPDFLoader (extract text + page metadata)
   → RecursiveCharacterTextSplitter (chunk_size=500, overlap=50)
   → Gemini Embeddings
   → Chroma vector store (cached in session state)

User Question (chat input)
   → similarity_search() on Chroma
   → top-k relevant chunks + recent Q&A history → prompt
   → Groq LLM
   → Answer + source citations (page numbers) displayed in chat
```

## Key engineering decisions

- **Session-scoped caching** — the vector index is built once per uploaded file and cached in `st.session_state`, so follow-up questions don't re-embed the document on every query (each embedding call is a network request, so this matters for both speed and cost).
- **Deployability over local-only tooling** — originally used Ollama for embeddings during development, but switched to a hosted embedding model (Gemini) since Ollama requires a locally running server and can't run on free hosting platforms like Streamlit Cloud.
- **Source citation** — chunk-level metadata (page number) is preserved from ingestion through to the UI, so answers are traceable back to the original document rather than being opaque LLM output.
- **Scoped conversation memory** — only the last 3 Q&A turns are included in the prompt, rather than full history, to keep the model focused on the current thread and avoid excessive token usage.
- **Error handling** — empty queries, unparseable/scanned PDFs, and LLM API failures are all handled explicitly with user-facing messages instead of raw stack traces.

## Known limitations

- No handling for scanned/image-only PDFs (no OCR).
- Single-document search only — can't currently query across multiple uploaded files at once.
- Retrieval is pure vector similarity — no reranking or hybrid (keyword + vector) search yet.
- No automated evaluation of retrieval/answer accuracy.

## Running locally

```bash
git clone https://github.com/Yashwanth-C16/StudyBuddy.git
cd StudyBuddy
pip install -r requirements.txt
```

Create a `.env` file in the project root:
```
GROQ_API_KEY=your_groq_key
GEMINI_API_KEY=your_gemini_key
```

Then run:
```bash
streamlit run app.py
```

## Future improvements

- Multi-document search
- Hybrid (keyword + vector) retrieval
- OCR support for scanned PDFs
- Retrieval accuracy evaluation on a test question set


## Evaluation

Built a small retrieval evaluation script (`eval.py`) that tests whether the vector search 
retrieves the correct source page for a set of known questions. On a 5-question test set 
covering each section of a sample document, retrieval achieved 5/5 (100%) accuracy at k=3.
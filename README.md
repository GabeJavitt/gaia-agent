# GAIA Benchmark Agent

A LangGraph-based tool-calling agent built for the [GAIA benchmark](https://huggingface.co/spaces/gaia-benchmark/leaderboard) — a challenging evaluation suite for general AI assistants. Designed to run on HuggingFace Spaces with a Gradio interface.

## Architecture

The agent uses a **StateGraph** (LangGraph) with a multi-model fallback chain:

```
Gemini 2.0 Flash → Gemini 2.5 Flash → Groq llama-3.3-70b → Groq qwen-3-32b → Claude
```

If the primary model hits a rate limit or quota error, the agent automatically falls through to the next available model — no restarts needed.

### Tools

The agent has access to 20 tools for solving GAIA tasks:

| Category | Tools |
|---|---|
| **Reasoning** | Think, Plan, Reflect, Validate Answer |
| **Search & Browse** | Wikipedia, DuckDuckGo Search, Web Browse, Web Scrape |
| **Code & Math** | Python Code Execution, Calculator, Data Analysis (pandas) |
| **Files** | Read File, Write File, List Directory |
| **Media** | Image Analysis (Gemini vision), Audio Transcription (AssemblyAI), YouTube Transcript, Video Analysis |
| **Specialized** | Chess Analysis (Stockfish + Gemini) |
| **Control** | Final Answer |

### Key Features

- **Automatic rate-limit handling** — detects `RESOURCE_EXHAUSTED`, daily quota limits, and 429s; falls through to the next model instantly
- **Groq API key rotation** — supports multiple Groq keys, rotating on daily limit exhaustion
- **Context pruning** — trims conversation history to stay within token limits
- **Retry with exponential backoff** — retries transient failures automatically
- **Answer normalization** — cleans and formats answers to match GAIA's expected output
- **Reflection loop** — periodically reflects on progress every N turns
- **RAG for large files** — chunks large documents and uses FAISS + HuggingFace embeddings for retrieval

## Setup

### Environment Variables

| Variable | Required | Description |
|---|---|---|
| `GOOGLE_API_KEY` | Yes | Google AI Studio API key (for Gemini models) |
| `GROQ_API_KEY` | No | Groq API key(s), comma-separated for rotation |
| `ANTHROPIC_API_KEY` | No | Anthropic API key (Claude fallback) |
| `ASSEMBLYAI_API_KEY` | No | AssemblyAI key for audio transcription |

### Local

```bash
pip install -r requirements.txt
python app.py
```

### HuggingFace Spaces

1. Create a new Space with **Gradio** SDK
2. Upload `app.py` and `requirements.txt`
3. Set the environment variables above as Space secrets
4. The Gradio interface will launch automatically

## Usage

The Gradio UI provides:
- **Run Agent** — processes all GAIA questions, submits answers, and reports the score
- **Test Random Question** — runs a single random question for debugging

Results are submitted to the GAIA scoring API at `https://agents-course-unit4-scoring.hf.space`.

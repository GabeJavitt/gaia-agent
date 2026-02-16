"""
GAIA Benchmark Agent - Refactored Version
Improvements:
- Better error handling with retry logic
- Caching for expensive operations
- Telemetry and progress tracking
- Modular architecture
- Parallel processing support
- Memory management
"""

import gc
import os
import io
import subprocess
import json
import re
import traceback
import contextlib
import uuid
import time
import ast
from typing import List, Optional, TypedDict, Annotated, Dict, Tuple
from pathlib import Path
from collections import Counter, defaultdict
from functools import wraps, lru_cache
import gradio as gr

# Workaround: Gradio 5.x/6.x bug where Queue.pending_message_lock stays None if the
# ASGI lifespan startup events don't fire (Python 3.13 asyncio compatibility issue).
# Must be patched AFTER gradio is fully imported.
try:
    import asyncio as _asyncio
    from gradio.queueing import Queue as _GradioQueue
    _orig_queue_init = _GradioQueue.__init__
    def _patched_queue_init(self, *args, **kwargs):
        _orig_queue_init(self, *args, **kwargs)
        for _attr, _val in list(vars(self).items()):
            if _attr.endswith("_lock") and _val is None:
                setattr(self, _attr, _asyncio.Lock())
    _GradioQueue.__init__ = _patched_queue_init
    print("✅ Applied Gradio queue lock workaround")
except Exception as _patch_err:
    print(f"ℹ️ Gradio queue patch skipped: {_patch_err}")

import pandas as pd
import numpy as np
import torch
from pydantic import BaseModel, Field

# Multimodal & Web Tools
import chess
import chess.engine
from transformers import pipeline
from youtube_transcript_api import YouTubeTranscriptApi
from bs4 import BeautifulSoup
import requests
from PIL import Image
import base64
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import assemblyai as aai

# LangChain & LangGraph
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage, AnyMessage, ToolCall
from langchain_core.tools import tool, ToolException
from langgraph.prebuilt import ToolNode
from langgraph.graph import START, END, StateGraph
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI


# RAG
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.documents import Document

# =============================================================================
# CONFIGURATION
# =============================================================================
class Config:
    DEFAULT_API_URL = "https://agents-course-unit4-scoring.hf.space"
    MAX_TURNS = 25
    MAX_MESSAGE_LENGTH = 8000
    REFLECT_EVERY_N_TURNS = 5
    MAX_RETRIES = 3
    BASE_RETRY_DELAY = 1
    CACHE_SIZE = 100
    MAX_PARALLEL_WORKERS = 3
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50

config = Config()

# =============================================================================
# UTILITIES: RETRY & CACHING
# =============================================================================
def retry_with_backoff(max_retries=None, base_delay=None):
    """Decorator for automatic retry with exponential backoff"""
    max_retries = max_retries or Config.MAX_RETRIES
    base_delay = base_delay or config.BASE_RETRY_DELAY
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    delay = base_delay * (2 ** attempt)
                    print(f"⚠️ {func.__name__} retry {attempt+1}/{max_retries} after {delay}s: {e}")
                    time.sleep(delay)
        return wrapper
    return decorator

def normalize_answer(answer: str, question: str = "") -> str:
    """
    Normalize answer to match expected format.
    
    Args:
        answer: The answer to normalize
        question: Optional question text to determine if order matters
    """
    if not answer:
        return answer
    
    original = answer
    answer = answer.strip()
    
    # Remove common prefixes
    prefixes_to_remove = [
        "the answer is:",
        "the answer is",
        "answer:",
        "final answer:",
        "result:",
    ]
    for prefix in prefixes_to_remove:
        if answer.lower().startswith(prefix):
            answer = answer[len(prefix):].strip()
    
    # Handle lists
    if "," in answer:
        items = [item.strip() for item in answer.split(",")]
        items = [item for item in items if item]
        
        # Determine if order matters based on question
        order_matters_keywords = [
            "first", "last", "before", "after", "sequence", 
            "order", "chronological", "oldest", "newest",
            "in the form", "format"
        ]
        
        order_matters = any(kw in question.lower() for kw in order_matters_keywords)
        
        if not order_matters:
            # Sort alphabetically for consistency
            items.sort()
            print(f"   📋 Sorted list alphabetically (order doesn't seem to matter)")
        else:
            print(f"   📋 Kept original order (question specifies order)")
        
        # Normalize each item
        items = [item.strip().rstrip('.') for item in items]
        
        # Consistent spacing
        answer = ", ".join(items)
    
    # Single word capitalization
    if len(answer.split()) == 1:
        if answer.lower() in ['right', 'left', 'yes', 'no', 'true', 'false']:
            answer = answer.capitalize()
    
    # Handle "St." vs "Saint"
    if "without abbreviations" in question.lower():
        answer = answer.replace("St.", "Saint")
        answer = answer.replace("Dr.", "Doctor")
        answer = answer.replace("Mt.", "Mount")
    
    # Remove trailing period (unless decimal)
    if answer.endswith('.') and not (len(answer) > 1 and answer[-2].isdigit()):
        answer = answer[:-1]
    
    # Remove wrapping quotes
    if (answer.startswith('"') and answer.endswith('"')) or \
       (answer.startswith("'") and answer.endswith("'")):
        answer = answer[1:-1]
    
    return answer

class SearchCache:
    """LRU cache for search results"""
    def __init__(self, maxsize=None):
        self.maxsize = maxsize or config.CACHE_SIZE
        self._cache = {}
        self._access_order = []
    
    def get(self, key: str) -> Optional[str]:
        if key in self._cache:
            # Move to end (most recently used)
            self._access_order.remove(key)
            self._access_order.append(key)
            return self._cache[key]
        return None
    
    def put(self, key: str, value: str):
        if key in self._cache:
            self._access_order.remove(key)
        elif len(self._cache) >= self.maxsize:
            # Remove least recently used
            oldest = self._access_order.pop(0)
            del self._cache[oldest]
        
        self._cache[key] = value
        self._access_order.append(key)
    
    def clear(self):
        self._cache.clear()
        self._access_order.clear()

search_cache = SearchCache()

# =============================================================================
# TELEMETRY
# =============================================================================
class Telemetry:
    """Track tool usage, timing, and errors"""
    def __init__(self):
        self.tool_times = defaultdict(list)
        self.tool_errors = defaultdict(int)
        self.tool_calls = defaultdict(int)
        self.start_time = time.time()
    
    def record_call(self, tool_name: str, duration: float, success: bool):
        self.tool_calls[tool_name] += 1
        self.tool_times[tool_name].append(duration)
        if not success:
            self.tool_errors[tool_name] += 1
    
    def report(self):
        total_time = time.time() - self.start_time
        print(f"\n{'='*70}")
        print(f"📊 TELEMETRY REPORT")
        print(f"{'='*70}")
        print(f"Total runtime: {total_time:.2f}s")
        print(f"\nTool Usage:")
        
        for tool in sorted(self.tool_calls.keys()):
            calls = self.tool_calls[tool]
            times = self.tool_times[tool]
            errors = self.tool_errors[tool]
            avg_time = sum(times) / len(times) if times else 0
            
            print(f"  {tool}:")
            print(f"    Calls: {calls}")
            print(f"    Avg time: {avg_time:.2f}s")
            print(f"    Errors: {errors}")
        
        print(f"{'='*70}\n")
    
    def reset(self):
        self.tool_times.clear()
        self.tool_errors.clear()
        self.tool_calls.clear()
        self.start_time = time.time()

telemetry = Telemetry()

# =============================================================================
# PROGRESS TRACKER
# =============================================================================
class ProgressTracker:
    """Track question processing progress"""
    def __init__(self, total: int):
        self.total = total
        self.current = 0
        self.correct = 0
        self.start_time = time.time()
    
    def update(self, is_correct: bool):
        self.current += 1
        if is_correct:
            self.correct += 1
        
        accuracy = (self.correct / self.current) * 100 if self.current > 0 else 0
        elapsed = time.time() - self.start_time
        avg_time = elapsed / self.current if self.current > 0 else 0
        eta = avg_time * (self.total - self.current)
        
        print(f"📊 Progress: {self.current}/{self.total} ({self.current/self.total*100:.1f}%)")
        print(f"   Accuracy: {accuracy:.1f}% ({self.correct} correct)")
        print(f"   Avg time: {avg_time:.1f}s per question")
        print(f"   ETA: {eta/60:.1f} minutes")

# =============================================================================
# CUSTOM EXCEPTIONS
# =============================================================================
class ToolError(ToolException):
    """Custom exception with context"""
    def __init__(self, tool_name: str, error: Exception, suggestion: str = ""):
        self.tool_name = tool_name
        self.original_error = error
        self.suggestion = suggestion
        message = f"Tool '{tool_name}' failed: {error}"
        if suggestion:
            message += f"\n💡 Suggestion: {suggestion}"
        super().__init__(message)

# =============================================================================
# GLOBAL RAG COMPONENTS
# =============================================================================
class RAGManager:
    """Manage RAG components with lazy initialization"""
    def __init__(self):
        self.embeddings = None
        self.text_splitter = None
        self._initialized = False
    
    def initialize(self):
        if self._initialized:
            return True
        
        print("Initializing RAG components...")
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'}
            )
            
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=config.CHUNK_SIZE,
                chunk_overlap=config.CHUNK_OVERLAP,
                length_function=len,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            
            self._initialized = True
            print("✅ RAG components initialized")
            return True
            
        except Exception as e:
            print(f"❌ RAG initialization failed: {e}")
            return False
    
    def is_ready(self):
        return self._initialized

rag_manager = RAGManager()

# =============================================================================
# ASR INITIALIZATION
# =============================================================================
class ASRManager:
    """Manage ASR pipeline"""
    def __init__(self):
        self.pipeline = None
        self._initialized = False
    
    def initialize(self):
        if self._initialized:
            return True
        
        try:
            print("Loading ASR (Whisper) pipeline...")
            device = 0 if torch.cuda.is_available() else -1
            device_name = "cuda:0" if device == 0 else "cpu"
            print(f"Using device: {device_name}")
            
            self.pipeline = pipeline(
                "automatic-speech-recognition",
                model="openai/whisper-base",
                torch_dtype=torch.float16 if device == 0 else torch.float32,
                device=device
            )
            
            self._initialized = True
            print("✅ ASR pipeline loaded")
            return True
            
        except Exception as e:
            print(f"⚠️ ASR pipeline failed to load: {e}")
            return False
    
    def is_ready(self):
        return self._initialized

asr_manager = ASRManager()

# =============================================================================
# ANSWER VALIDATION
# =============================================================================
class AnswerValidator:
    """Validate and check answers"""
    
    @staticmethod
    def load_answer_sheet(filepath: str = "answer_sheet_json.json") -> Dict[str, str]:
        """Load answer sheet"""
        try:
            if os.path.exists(filepath):
                with open(filepath, 'r', encoding='utf-8') as f:
                    answers = json.load(f)
                print(f"✅ Loaded {len(answers)} answers from {filepath}")
                return answers
            else:
                print(f"⚠️ Answer sheet not found: {filepath}")
                return {}
        except Exception as e:
            print(f"❌ Error loading answer sheet: {e}")
            return {}
    
    @staticmethod
    def check_correctness(submitted: str, correct: str) -> Tuple[bool, str]:
        """Check if answer is correct with fuzzy matching"""
        import string
        
        submitted_norm = submitted.strip().lower()
        correct_norm = correct.strip().lower()
        
        # Exact match
        if submitted_norm == correct_norm:
            return True, "✅ EXACT MATCH"
        
        # Remove punctuation
        trans = str.maketrans('', '', string.punctuation)
        submitted_clean = submitted_norm.translate(trans)
        correct_clean = correct_norm.translate(trans)
        
        if submitted_clean == correct_clean:
            return True, "✅ MATCH (punctuation)"
        
        # Numeric comparison
        try:
            submitted_num = float(submitted_clean.replace(',', '').replace('$', ''))
            correct_num = float(correct_clean.replace(',', '').replace('$', ''))
            if abs(submitted_num - correct_num) < 0.01:
                return True, "✅ MATCH (numeric)"
        except (ValueError, AttributeError):
            pass
        
        # List comparison
        if ',' in correct_norm:
            correct_items = set(item.strip() for item in correct_norm.split(','))
            submitted_items = set(item.strip() for item in submitted_norm.split(','))
            
            if correct_items == submitted_items:
                return True, "✅ MATCH (order)"
            
            missing = correct_items - submitted_items
            extra = submitted_items - correct_items
            
            if missing or extra:
                msg = []
                if missing:
                    msg.append(f"MISSING: {', '.join(missing)}")
                if extra:
                    msg.append(f"EXTRA: {', '.join(extra)}")
                return False, f"❌ {' | '.join(msg)}"
        
        # Partial match
        if submitted_norm in correct_norm or correct_norm in submitted_norm:
            return False, f"❌ PARTIAL ('{submitted}' vs '{correct}')"
        
        return False, f"❌ WRONG ('{submitted}' vs '{correct}')"

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================
def remove_fences_simple(text: str) -> str:
    """Remove code fences"""
    text = text.strip()
    if text.startswith("```") and text.endswith("```"):
        text = text[3:-3].strip()
        if '\n' in text:
            first_line, rest = text.split('\n', 1)
            if first_line.strip().replace('_','').isalnum() and len(first_line.strip()) < 15:
                text = rest.strip()
    return text

def truncate_if_needed(content: str, max_length: int = None) -> str:
    """Truncate long content"""
    max_length = max_length or config.MAX_MESSAGE_LENGTH
    if len(content) > max_length:
        return content[:max_length] + f"\n...[truncated, {len(content)} chars total]"
    return content

def find_file(path: str) -> Optional[Path]:
    """Find file with multiple path attempts"""
    script_dir = Path.cwd()
    safe_path = Path(path).as_posix()
    
    paths = [
        script_dir / safe_path,
        Path(safe_path),
        script_dir / Path(path).name,
        Path("files") / Path(path).name
    ]
    
    for p in paths:
        if p.exists():
            return p
    
    return None

# =============================================================================
# TOOL INPUT VALIDATION
# =============================================================================
def validate_tool_inputs(tool_name: str, inputs: dict) -> Tuple[bool, str]:
    """Validate tool inputs before execution"""
    validators = {
        "scrape_and_retrieve": lambda i: i.get("url", "").startswith(("http://", "https://")),
        "calculator": lambda i: bool(re.match(r'^[\d\+\-\*/\(\)\s\.,a-z]+$', i.get("expression", ""), re.I)),
        "read_file": lambda i: len(i.get("path", "")) > 0 and ".." not in i.get("path", ""),
        "search_tool": lambda i: len(i.get("query", "").strip()) > 0,
        "code_interpreter": lambda i: "import os" not in i.get("code", "").lower(),
    }
    
    if tool_name in validators:
        try:
            if not validators[tool_name](inputs):
                return False, f"Invalid input format for {tool_name}"
        except Exception as e:
            return False, f"Validation error: {e}"
    
    return True, ""

# =============================================================================
# PLANNING & REFLECTION TOOLS
# =============================================================================
class ThinkInput(BaseModel):
    reasoning: str = Field(description="Brief reasoning (under 150 chars)")

@tool(args_schema=ThinkInput)
def think_through_logic(reasoning: str) -> str:
    """
    Use ONLY for logic puzzles and riddles (NOT research questions).
    
    Use for:
    - Brain teasers, logic puzzles, riddles
    
    DON'T use for:
    - Research questions → use search_tool or wikipedia_search
    - Math → use calculator
    - File analysis → use file tools
    """
    print(f"🧠 Thinking: {reasoning[:100]}...")
    
    return f"""✅ Reasoning: {reasoning[:100]}
⚠️ DO NOT CALL think_through_logic AGAIN!
For research → use search_tool() or wikipedia_search()
For math → use calculator()
If you have answer → use final_answer_tool()
TAKE ACTION NOW!"""


class PlanInput(BaseModel):
    task_summary: str = Field(description="Brief task summary (under 80 chars)")

@tool(args_schema=PlanInput)
def create_plan(task_summary: str) -> str:
    """Create plan for complex tasks"""
    start_time = time.time()
    try:
        print(f"📋 Planning: {task_summary[:80]}...")
        result = f"""✅ Plan: {task_summary}
Framework:
1. What info needed?
2. Which tools?
3. What order?
Execute step 1 now."""
        telemetry.record_call("create_plan", time.time() - start_time, True)
        return result
    except Exception as e:
        telemetry.record_call("create_plan", time.time() - start_time, False)
        raise


class ReflectInput(BaseModel):
    situation: str = Field(description="Brief situation (under 80 chars)")

@tool(args_schema=ReflectInput)
def reflect_on_progress(situation: str) -> str:
    """Reflect when stuck"""
    start_time = time.time()
    try:
        print(f"🤔 Reflecting: {situation[:80]}...")
        result = f"""🔍 Reflection: {situation}
Questions:
1. Right approach?
2. Try different tool?
3. Have answer already?
Try DIFFERENT approach now."""
        telemetry.record_call("reflect_on_progress", time.time() - start_time, True)
        return result
    except Exception as e:
        telemetry.record_call("reflect_on_progress", time.time() - start_time, False)
        raise


class ValidateAnswerInput(BaseModel):
    proposed_answer: str = Field(description="Answer to validate")
    original_question: str = Field(description="Original question (first 100 chars)")

@tool(args_schema=ValidateAnswerInput)
def validate_answer(proposed_answer: str, original_question: str = "") -> str:

    """
    Validate answer format and provide warnings.
    Returns validation result with normalization suggestions.
    """
    start_time = time.time()
    
    try:
        print(f"✓ Validating: '{proposed_answer[:50]}...'")

        warnings = []
        errors = []
        normalization_needed = []

        # Normalize for validation
        normalized = normalize_answer(proposed_answer)

        if normalized != proposed_answer:
            normalization_needed.append(f"Consider using normalized form: '{normalized}'")

        # Check 1: Empty answer
        if not proposed_answer or not proposed_answer.strip():
            errors.append("Answer is empty")

        # Check 2: Too long (probably explaining instead of answering)
        if len(proposed_answer) > 200:
            warnings.append("Answer is very long (>200 chars). Consider if question asks for brief response.")

        # Check 3: Contains question words
        question_words = ['what', 'who', 'when', 'where', 'why', 'how', 'which']
        if any(word in proposed_answer.lower() for word in question_words):
            warnings.append("Answer contains question words. Make sure you're providing the answer, not rephrasing the question.")

        # Check 4: List ordering
        if "," in proposed_answer:
            items = [item.strip() for item in proposed_answer.split(",")]
            if len(items) > 1:
                warnings.append(f"List detected with {len(items)} items. Verify order matches question requirements.")

        # Check 5: Capitalization consistency
        if proposed_answer.lower() in ['right', 'left', 'yes', 'no', 'true', 'false']:
            if not proposed_answer[0].isupper():
                normalization_needed.append(f"Consider capitalizing: '{proposed_answer.capitalize()}'")

        # Check 6: Abbreviations
        if any(abbrev in proposed_answer.lower() for abbrev in ['st.', 'dr.', 'mt.']):
            if "without abbreviations" in str(proposed_answer).lower() or "full" in str(proposed_answer).lower():
                warnings.append("Question may ask for full form without abbreviations")

        # Check 7: Spacing in lists
        if "," in proposed_answer:
            # Check for inconsistent spacing
            if ", " in proposed_answer and "," in proposed_answer.replace(", ", ""):
                normalization_needed.append("Inconsistent spacing in list. Use consistent ', ' format")
        
        # Build result
        result_parts = []
        
        if errors:
            result_parts.append("🚫 VALIDATION FAILED:")
            for error in errors:
                result_parts.append(f"❌ {error}")
            result_parts.append("Fix issues then retry validation.")
        else:
            result_parts.append("✅ VALIDATION PASSED!")
            
            if normalization_needed:
                result_parts.append("\n💡 NORMALIZATION SUGGESTIONS:")
                for suggestion in normalization_needed:
                    result_parts.append(f"   • {suggestion}")
            
            if warnings:
                result_parts.append("\n⚠️ WARNINGS:")
                for warning in warnings:
                    result_parts.append(f"⚠️ {warning}")
                result_parts.append("Proceed if confident, or refine answer.")
            else:
                result_parts.append("Call final_answer_tool() now.")
        
        result = "\n".join(result_parts)
        
        telemetry.record_call("validate_answer", time.time() - start_time, True)
        return result
        
    except Exception as e:
        telemetry.record_call("validate_answer", time.time() - start_time, False)
        raise ToolError("validate_answer", e)

# =============================================================================
# CORE TOOLS
# =============================================================================
class WikipediaInput(BaseModel):
    query: str = Field(description="Topic to search (just the subject name)")

@tool(args_schema=WikipediaInput)
def wikipedia_search(query: str) -> str:
    """
    Search Wikipedia directly. Keep query SHORT!
    ✅ GOOD: "Mercedes Sosa"
    ❌ BAD: "Mercedes Sosa discography 2022 Wikipedia version"
    """
    # AGGRESSIVE query cleaning
    original = query
    query = query.lower().strip()
    
    # Remove these phrases (order matters - longest first!)
    remove_list = [
        "2022 english wikipedia version",
        "english wikipedia version",
        "2022 version",
        "wikipedia version",
        "latest version",
        "wikipedia",
        "wiki",
        "discography",
        "site:",
        " the ",
        " a ",
        " an "
    ]
    
    for phrase in remove_list:
        query = query.replace(phrase, "")
    
    # Clean whitespace
    query = " ".join(query.split()).strip()
    
    # Fallback if query too short
    if len(query) < 2:
        words = original.split()
        query = words[0] if words else original
    
    print(f"📚 Wikipedia: '{original}' → '{query}'")
    
    # Try direct page
    page_name = query.title().replace(" ", "_")
    page_url = f"https://en.wikipedia.org/wiki/{page_name}"
    
    print(f"   Trying: {page_url}")
    
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(page_url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            
            title_tag = soup.find('h1', class_='firstHeading')
            title = title_tag.get_text() if title_tag else page_name
            
            content_div = soup.find('div', class_='mw-parser-output')
            preview = ""
            if content_div:
                paragraphs = content_div.find_all('p', limit=3)
                for p in paragraphs:
                    text = p.get_text().strip()
                    if len(text) > 50:
                        preview = text[:300]
                        break
            
            result = f"""✅ Found: {title}
URL: {page_url}
Preview: {preview}...
NEXT: Use scrape_and_retrieve(url="{page_url}", query="specific info")"""
            
            print(f"✓ Success: {title}")
            return result
            
        else:
            # Try search
            print(f"   404, trying search")
            search_url = f"https://en.wikipedia.org/w/index.php?search={query.replace(' ', '+')}"
            
            try:
                search_resp = requests.get(search_url, headers=headers, timeout=10)
                
                if "wikipedia.org/wiki/" in search_resp.url and search_resp.url != search_url:
                    return f"✅ Redirected to: {search_resp.url}\n\nUse scrape_and_retrieve() for details."
                
                soup = BeautifulSoup(search_resp.text, 'html.parser')
                results = soup.find_all('div', class_='mw-search-result-heading', limit=3)
                
                if results:
                    formatted = []
                    for i, result in enumerate(results, 1):
                        link = result.find('a')
                        if link:
                            title = link.get_text()
                            href = link.get('href')
                            full_url = f"https://en.wikipedia.org{href}"
                            formatted.append(f"{i}. {title}\n   {full_url}")
                    
                    return "Wikipedia results:\n\n" + "\n\n".join(formatted) + "\n\nUse scrape_and_retrieve() with relevant URL."
                
                return f"""No Wikipedia page found for '{query}'.
Try:
1. search_tool("{query}")
2. Different search term
3. Check spelling"""
                
            except Exception as search_err:
                return f"Wikipedia search failed. Try search_tool('{query}') instead."
    
    except requests.Timeout:
        return f"Wikipedia timed out. Try search_tool('{query}') instead."
    
    except Exception as e:
        print(f"⚠️ Wikipedia error: {str(e)[:100]}")
        return f"Wikipedia error. Try search_tool('{query}') instead."


class SearchInput(BaseModel):
    query: str = Field(description="Search query (concise)")

@tool(args_schema=SearchInput)
@retry_with_backoff(max_retries=3)
def search_tool(query: str) -> str:
    """Web search with caching and language filtering"""
    start_time = time.time()
    
    try:
        # Input validation
        is_valid, msg = validate_tool_inputs("search_tool", {"query": query})
        if not is_valid:
            raise ValueError(msg)
        
        # Check cache
        cached = search_cache.get(query)
        if cached:
            print(f"🔍 Search (cached): {query}")
            telemetry.record_call("search_tool", time.time() - start_time, True)
            return cached
        
        # Auto-add Wikipedia filter
        if 'wikipedia' in query.lower() and 'site:' not in query:
            query = f"{query} site:wikipedia.org"
        
        print(f"🔍 Searching: {query}")
        
        # DuckDuckGo doesn't support these params directly, 
        # but we can filter by adding language hints
        # For English results, add hint to query
        search = DuckDuckGoSearchRun()
        
        # Add language hint to force English results
        if not any(keyword in query.lower() for keyword in ['lang:', 'region:']):
            query = f"{query} lang:en"
        
        result = search.run(query)
        
        if not result or len(result) < 50:
            result = "No results found. Try different terms."
        
        result = truncate_if_needed(result)
        
        # Cache result
        search_cache.put(query, result)
        
        telemetry.record_call("search_tool", time.time() - start_time, True)
        return result
        
    except Exception as e:
        telemetry.record_call("search_tool", time.time() - start_time, False)
        raise ToolError("search_tool", e, "Try rephrasing query")


class CalcInput(BaseModel):
    expression: str = Field(description="Math expression")

@tool(args_schema=CalcInput)
def calculator(expression: str) -> str:
    """Evaluate math expressions"""
    start_time = time.time()
    
    try:
        # Input validation
        is_valid, msg = validate_tool_inputs("calculator", {"expression": expression})
        if not is_valid:
            raise ValueError(msg)
        
        print(f"🧮 Calculating: {expression}")
        
        import math
        safe_dict = {
            'sqrt': math.sqrt, 'sin': math.sin, 'cos': math.cos, 'tan': math.tan,
            'log': math.log, 'log10': math.log10, 'exp': math.exp,
            'pi': math.pi, 'e': math.e, 'abs': abs, 'round': round,
            'pow': pow, 'sum': sum, 'min': min, 'max': max
        }
        
        result = eval(expression, {"__builtins__": {}}, safe_dict)
        
        telemetry.record_call("calculator", time.time() - start_time, True)
        return str(result)
        
    except Exception as e:
        telemetry.record_call("calculator", time.time() - start_time, False)
        raise ToolError("calculator", e, f"Check expression: {expression}")


class CodeInput(BaseModel):
    code: str = Field(description="Python code (MUST use print())")

@tool(args_schema=CodeInput)
def code_interpreter(code: str) -> str:
    """Execute Python code"""
    start_time = time.time()
    
    try:
        # Safety checks
        dangerous = ['__import__', 'eval(', 'compile(', 'subprocess', 'os.system', 'exec(']
        if any(d in code.lower() for d in dangerous):
            raise ValueError("Dangerous operation not allowed")
        
        if 'open(' in code.lower() and any(m in code for m in ["'w'", '"w"', "'a'", '"a"']):
            raise ValueError("File writing not allowed, use write_file tool")
        
        print(f"💻 Executing code ({len(code)} chars)...")
        
        output_stream = io.StringIO()
        error_stream = io.StringIO()
        
        with contextlib.redirect_stdout(output_stream), contextlib.redirect_stderr(error_stream):
            safe_globals = {
                "pd": pd,
                "np": np,
                "json": json,
                "re": re,
                "__builtins__": __builtins__
            }
            exec(code, safe_globals, {})
        
        stdout = output_stream.getvalue()
        stderr = error_stream.getvalue()
        
        if stderr:
            result = f"Error:\n{stderr}\n\nOutput:\n{stdout}"
        elif stdout:
            result = truncate_if_needed(stdout)
        else:
            result = "Code executed but no output. Use print()!"
        
        telemetry.record_call("code_interpreter", time.time() - start_time, True)
        return result
        
    except Exception as e:
        telemetry.record_call("code_interpreter", time.time() - start_time, False)
        raise ToolError("code_interpreter", e, "Check code syntax")


class AnalyzeDataInput(BaseModel):
    file_path: str = Field(description="Path to CSV or Excel file")
    question: str = Field(description="What to find (e.g., 'count rows where year > 2000')")

@tool(args_schema=AnalyzeDataInput)
def analyze_data_file(file_path: str, question: str) -> str:
    """
    Analyze CSV/Excel files with automatic data profiling.
    
    Generates Python code to answer questions about data files.
    Better than code_interpreter alone because it:
    1. Profiles the data first (columns, types, sample)
    2. Generates appropriate pandas code
    3. Handles common data issues (encoding, missing values)
    
    Use for questions like:
    - "How many rows have X?"
    - "What's the sum/average of column Y?"
    - "Count items grouped by Z"
    """
    start_time = time.time()
    
    try:
        print(f"📊 Analyzing data file: {file_path}")
        print(f"   Question: {question[:100]}...")
        
        # Find file
        data_file = find_file(file_path)
        if not data_file:
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        file_ext = data_file.suffix.lower()
        
        if file_ext not in ['.csv', '.xlsx', '.xls', '.tsv']:
            raise ValueError(f"Unsupported file type: {file_ext}. Use .csv, .xlsx, .xls, or .tsv")
        
        print(f"   File type: {file_ext}")
        
        # Generate profiling code
        profiling_code = f"""
import pandas as pd
import numpy as np
# Load file
file_path = r"{data_file}"
"""
        
        if file_ext == '.csv':
            profiling_code += """
# Try different encodings
for encoding in ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']:
    try:
        df = pd.read_csv(file_path, encoding=encoding)
        break
    except:
        continue
"""
        elif file_ext == '.tsv':
            profiling_code += """
df = pd.read_csv(file_path, sep='\\t', encoding='utf-8')
"""
        else:  # Excel
            profiling_code += """
df = pd.read_excel(file_path)
"""
        
        profiling_code += """
# Profile data
print("=" * 60)
print("DATA PROFILE")
print("=" * 60)
print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"\\nColumns: {', '.join(df.columns.tolist())}")
print(f"\\nData types:")
print(df.dtypes)
print(f"\\nFirst 3 rows:")
print(df.head(3))
print(f"\\nMissing values:")
print(df.isnull().sum())
"""
        
        # Execute profiling
        print(f"   Profiling data...")
        output_stream = io.StringIO()
        error_stream = io.StringIO()
        
        with contextlib.redirect_stdout(output_stream), contextlib.redirect_stderr(error_stream):
            exec(profiling_code, {"pd": pd, "np": np, "__builtins__": __builtins__})
        
        profile_output = output_stream.getvalue()
        
        if error_stream.getvalue():
            raise RuntimeError(f"Profiling failed: {error_stream.getvalue()}")
        
        print(f"   Profiling complete")
        print(profile_output[:500] + "..." if len(profile_output) > 500 else profile_output)
        
        # Now generate analysis code based on question
        analysis_code = profiling_code + f"""
# Analysis for: {question}
print("\\n" + "=" * 60)
print("ANALYSIS RESULT")
print("=" * 60)
"""
        
        # Add intelligent code based on question keywords
        q_lower = question.lower()
        
        if 'count' in q_lower or 'how many' in q_lower:
            if 'where' in q_lower or 'with' in q_lower:
                analysis_code += """
# Count rows matching condition
# NOTE: Adjust the filter condition based on your needs
result = len(df)  # Total count
print(f"Total rows: {result}")
# Example filters (uncomment and modify as needed):
# result = len(df[df['column'] > value])
# result = len(df[df['column'].str.contains('text', na=False)])
"""
            else:
                analysis_code += """
result = len(df)
print(f"Total rows: {result}")
"""
        
        elif 'sum' in q_lower or 'total' in q_lower:
            analysis_code += """
# Sum a numeric column
# NOTE: Replace 'column_name' with actual column
# result = df['column_name'].sum()
# print(f"Sum: {result}")
"""
        
        elif 'average' in q_lower or 'mean' in q_lower:
            analysis_code += """
# Average of a column
# result = df['column_name'].mean()
# print(f"Average: {result}")
"""
        
        elif 'group' in q_lower or 'by' in q_lower:
            analysis_code += """
# Group by and count
# result = df.groupby('column_name').size()
# print(result)
"""
        
        else:
            # Generic: show summary
            analysis_code += """
# Summary statistics
print(df.describe())
"""
        
        result = f"""Data Profile:
{profile_output}
Generated Analysis Code:
```python
{analysis_code}
```
**IMPORTANT**: The code above needs column names adjusted. 
Use code_interpreter() with the corrected code to get the answer.
Columns available: {", ".join((pd.read_csv(data_file) if file_ext == '.csv' else pd.read_excel(data_file)).columns.tolist())}
"""
        
        telemetry.record_call("analyze_data_file", time.time() - start_time, True)
        return truncate_if_needed(result)
        
    except Exception as e:
        telemetry.record_call("analyze_data_file", time.time() - start_time, False)
        raise ToolError("analyze_data_file", e, "Check file path and format")


class ReadFileInput(BaseModel):
    path: str = Field(description="File path")

@tool(args_schema=ReadFileInput)
def read_file(path: str) -> str:
    """Read file content"""
    start_time = time.time()
    
    try:
        # Input validation
        is_valid, msg = validate_tool_inputs("read_file", {"path": path})
        if not is_valid:
            raise ValueError(msg)
        
        print(f"📄 Reading: {path}")
        
        file_path = find_file(path)
        if not file_path:
            raise FileNotFoundError(f"File not found: {path}")
        
        content = file_path.read_text(encoding='utf-8')
        
        telemetry.record_call("read_file", time.time() - start_time, True)
        return truncate_if_needed(content)
        
    except UnicodeDecodeError:
        telemetry.record_call("read_file", time.time() - start_time, False)
        return f"Binary file. Try audio_transcription_tool."
    except Exception as e:
        telemetry.record_call("read_file", time.time() - start_time, False)
        raise ToolError("read_file", e, f"Check file path: {path}")


class WriteFileInput(BaseModel):
    path: str = Field(description="File path")
    content: str = Field(description="Content to write")

@tool(args_schema=WriteFileInput)
def write_file(path: str, content: str) -> str:
    """Write content to file"""
    start_time = time.time()
    
    try:
        print(f"✍️ Writing: {path}")
        
        file_path = Path.cwd() / path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding='utf-8')
        
        telemetry.record_call("write_file", time.time() - start_time, True)
        return f"Wrote {len(content)} chars to '{path}'"
        
    except Exception as e:
        telemetry.record_call("write_file", time.time() - start_time, False)
        raise ToolError("write_file", e)


class ListDirInput(BaseModel):
    path: str = Field(description="Directory path", default=".")

@tool(args_schema=ListDirInput)
def list_directory(path: str = ".") -> str:
    """List directory contents"""
    start_time = time.time()
    
    try:
        print(f"📁 Listing: {path}")
        
        dir_path = Path.cwd() / path if path != "." else Path.cwd()
        
        if not dir_path.is_dir():
            raise NotADirectoryError(f"'{path}' not a directory")
        
        items = sorted(dir_path.iterdir())
        
        if not items:
            return f"Directory '{path}' is empty"
        
        files, dirs = [], []
        
        for item in items:
            if item.is_dir():
                dirs.append(f"📁 {item.name}/")
            else:
                files.append(f"📄 {item.name} ({item.stat().st_size} bytes)")
        
        result = f"Contents of '{path}':\n\n"
        if dirs:
            result += "Directories:\n" + "\n".join(dirs) + "\n\n"
        if files:
            result += "Files:\n" + "\n".join(files)
        
        telemetry.record_call("list_directory", time.time() - start_time, True)
        return result
        
    except Exception as e:
        telemetry.record_call("list_directory", time.time() - start_time, False)
        raise ToolError("list_directory", e)


class AudioInput(BaseModel):
    file_path: str = Field(description="Audio file path")

@tool(args_schema=AudioInput)
def audio_transcription_tool(file_path: str) -> str:
    """Transcribe audio using Whisper"""
    start_time = time.time()
    
    try:
        print(f"🎤 Transcribing: {file_path}")
        
        if not asr_manager.is_ready():
            asr_manager.initialize()
        
        if not asr_manager.is_ready():
            raise RuntimeError("ASR not available")
        
        audio_path = find_file(file_path)
        if not audio_path:
            raise FileNotFoundError(f"Audio file not found: {file_path}")
        
        transcription = asr_manager.pipeline(
            str(audio_path),
            return_timestamps=True,
            chunk_length_s=30,
            stride_length_s=5
        )
        
        result_text = transcription.get("text", "")
        
        if not result_text:
            raise ValueError("Transcription empty")
        
        telemetry.record_call("audio_transcription_tool", time.time() - start_time, True)
        return f"Transcription:\n{truncate_if_needed(result_text)}"
        
    except Exception as e:
        telemetry.record_call("audio_transcription_tool", time.time() - start_time, False)
        raise ToolError("audio_transcription_tool", e)

class ChessAnalysisInput(BaseModel):
    image_path: str = Field(description="Path to chess board image")
    description: str = Field(description="Context about position", default="")

@tool(args_schema=ChessAnalysisInput)
def analyze_chess_position(image_path: str, description: str = "") -> str:
    """
    Analyze chess position from image using Gemini Vision + Stockfish.
    Extracts FEN, analyzes best move.
    """
    start_time = time.time()

    try:
        print(f"♟️ Analyzing chess: {image_path}")

        # Find file
        image_path_obj = find_file(image_path)
        if not image_path_obj and os.path.exists(image_path):
            image_path_obj = Path(image_path)

        if not image_path_obj or not image_path_obj.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
        if not GOOGLE_API_KEY:
            raise ValueError("GEMINI_API_KEY not set")
        
        # Read image as base64
        with open(image_path_obj, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")
        
        # Use Gemini to extract FEN
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=GOOGLE_API_KEY,
            temperature=0
        )
        
        message = HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": """Analyze this chess position and provide the FEN notation.
                    
CRITICAL: The FEN string MUST include whose turn it is:
- If White to move: end with "w - - 0 1"
- If Black to move: end with "b - - 0 1"
Look at the board carefully to determine whose turn it is based on:
1. Any text in the image indicating whose turn
2. The position context
3. If unclear, look at piece positions
Respond with ONLY the FEN string, nothing else."""
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image_data}"
                    }
                }
            ]
        )
        
        response = llm.invoke([message])
        fen = response.content.strip()
        
        print(f"✓ FEN: {fen}")
        
        # ===== FIX: Parse whose turn it is from FEN =====
        # FEN format: position w/b castling en-passant halfmove fullmove
        fen_parts = fen.split()
        
        # Ensure we have the turn indicator
        if len(fen_parts) < 2:
            # Default to white if not specified
            fen = f"{fen} w - - 0 1"
            fen_parts = fen.split()
        
        # Get whose turn it is
        turn = fen_parts[1] if len(fen_parts) > 1 else 'w'
        print(f"✓ Turn: {'Black' if turn == 'b' else 'White'}")
        
        # ===== END FIX =====
        
        # Analyze with Stockfish
        try:
            board = chess.Board(fen)
        except ValueError as e:
            raise ValueError(f"Invalid FEN from Gemini: {fen}. Error: {e}")
        
        # Configure Stockfish
        stockfish_path = "/usr/games/stockfish"
        if not os.path.exists(stockfish_path):
            raise FileNotFoundError("Stockfish not found at /usr/games/stockfish")
        
        engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
        
        # ===== FIX: Analyze with appropriate depth =====
        # For tactical positions (like mate puzzles), need deeper analysis
        result = engine.analyse(board, chess.engine.Limit(depth=20))
        # ===== END FIX =====
        
        best_move = result["pv"][0]  # Principal variation (best line)
        engine.quit()
        
        # Convert to algebraic notation
        move_san = board.san(best_move)
        
        print(f"✓ Best move: {move_san}")
        
        telemetry.record_call("analyze_chess_position", time.time() - start_time, True)
        
        # ===== FIX: Include turn info in response =====
        turn_text = "Black" if turn == 'b' else "White"
        return f"{move_san} ({turn_text} to move, from FEN: {fen})"
        # ===== END FIX =====
        
    except Exception as e:
        telemetry.record_call("analyze_chess_position", time.time() - start_time, False)
        raise ToolError("analyze_chess_position", e, "Check image quality and Stockfish installation")
        
class ImageAnalysisInput(BaseModel):
    file_path: str = Field(description="Image file path")
    query: str = Field(description="What to analyze")

@tool(args_schema=ImageAnalysisInput)
def analyze_image(file_path: str, query: str) -> str:
    """Analyze images using Gemini Vision"""
    start_time = time.time()
    
    try:
        print(f"🖼️ Analyzing: {file_path}")
        print(f"   Query: {query[:100]}...")
        
        image_path = find_file(file_path)
        if not image_path and os.path.exists(file_path):
            image_path = Path(file_path)
        
        if not image_path or not image_path.exists():
            raise FileNotFoundError(f"Image not found: {file_path}")
        
        GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
        if not GOOGLE_API_KEY:
            raise ValueError("GEMINI_API_KEY not set")
        
        # Load and encode
        img = Image.open(image_path)
        if img.mode not in ['RGB', 'RGBA']:
            img = img.convert('RGB')
        
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        
        # Use FLASH model for cost efficiency
        vision_llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=GOOGLE_API_KEY,
            temperature=0
        )
        
        message = HumanMessage(
            content=[
                {"type": "text", "text": query},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}}
            ]
        )
        
        response = vision_llm.invoke([message])
        
        telemetry.record_call("analyze_image", time.time() - start_time, True)
        return f"Image Analysis:\n{truncate_if_needed(response.content)}"
        
    except Exception as e:
        telemetry.record_call("analyze_image", time.time() - start_time, False)
        raise ToolError("analyze_image", e)


class YoutubeInput(BaseModel):
    video_url: str = Field(description="YouTube URL")

@tool(args_schema=YoutubeInput)
def get_youtube_transcript(video_url: str) -> str:
    """Get YouTube transcript using AssemblyAI with proper status handling"""
    start_time = time.time()
    
    try:
        aai.settings.api_key = os.getenv("ASSEMBLYAI_API_KEY")
        if not aai.settings.api_key:
            raise ValueError("ASSEMBLYAI_API_KEY not set in Space secrets")
        
        print(f"📺 Transcribing YouTube: {video_url}")
        
        # Validate URL
        if not ("youtube.com" in video_url or "youtu.be" in video_url):
            raise ValueError(f"Invalid YouTube URL: {video_url}")
        
        # Submit transcription request
        transcriber = aai.Transcriber()
        print(f"   Submitting to AssemblyAI...")
        
        config_obj = aai.TranscriptionConfig(
            speech_model=aai.SpeechModel.best,
        )
        
        transcript = transcriber.transcribe(video_url, config=config_obj)
        
        # Wait for completion
        print(f"   Initial status: {transcript.status}")
        
        # Poll for completion (max 5 minutes)
        max_wait = 300
        poll_interval = 5
        elapsed = 0
        
        while transcript.status == aai.TranscriptStatus.queued or transcript.status == aai.TranscriptStatus.processing:
            if elapsed >= max_wait:
                raise TimeoutError(f"Transcription timed out after {max_wait}s. Video may be too long.")
            
            time.sleep(poll_interval)
            elapsed += poll_interval
            
            # Refresh transcript object
            try:
                transcript = transcriber.get_transcript(transcript.id)
                print(f"   Status after {elapsed}s: {transcript.status}")
            except Exception as refresh_err:
                print(f"   Warning: Could not refresh status: {refresh_err}")
                break
        
        # Check final status
        if transcript.status == aai.TranscriptStatus.error:
            error_msg = getattr(transcript, 'error', 'Unknown error')
            
            # ===== NEW: Check for network block =====
            if "text/html" in error_msg or "HTML document" in error_msg:
                raise RuntimeError(
                    "YouTube access blocked. "
                    "If a local video file was provided, use analyze_image or audio_transcription_tool instead. "
                    "Or try downloading the video first."
                )
            # ===== END NEW =====
            
            raise RuntimeError(f"AssemblyAI transcription failed: {error_msg}")
        
        if transcript.status != aai.TranscriptStatus.completed:
            raise RuntimeError(f"Unexpected status: {transcript.status}")
        
        # Extract text
        if not hasattr(transcript, 'text'):
            raise AttributeError("Transcript object has no 'text' attribute")
        
        result_text = transcript.text
        
        if not result_text or not isinstance(result_text, str):
            raise ValueError(f"Transcript text is invalid: {type(result_text)}")
        
        result_text = result_text.strip()
        
        if len(result_text) < 10:
            raise ValueError(f"Transcript too short ({len(result_text)} chars). Video may have no audio.")
        
        print(f"✓ Transcribed {len(result_text)} chars")
        
        telemetry.record_call("get_youtube_transcript", time.time() - start_time, True)
        return f"YouTube Transcript:\n{truncate_if_needed(result_text)}"
        
    except Exception as e:
        telemetry.record_call("get_youtube_transcript", time.time() - start_time, False)
        error_msg = str(e)
        
        suggestions = []
        if "text/html" in error_msg.lower() or "html document" in error_msg.lower():
            suggestions.append("YouTube blocked on HuggingFace. Use the local .mp4 file instead with audio_transcription_tool or analyze_image")
        elif "not found" in error_msg.lower():
            suggestions.append("Video may be private or deleted")
        elif "quota" in error_msg.lower() or "limit" in error_msg.lower():
            suggestions.append("AssemblyAI quota exceeded")
        elif "timeout" in error_msg.lower():
            suggestions.append("Video may be too long (try shorter video)")
        
        suggestion_text = " | ".join(suggestions) if suggestions else "Check video URL is valid and public"
        
        raise ToolError("get_youtube_transcript", e, suggestion_text)


class BrowseInput(BaseModel):
    start_url: str = Field(description="Starting URL (http:// or https://)")
    goal: str = Field(description="What you're trying to find (e.g., 'Mercedes Sosa albums 2000-2009')")
    max_steps: int = Field(description="Max pages to visit (1-5)", default=3)

@tool(args_schema=BrowseInput)
@retry_with_backoff(max_retries=2)
def iterative_web_browser(start_url: str, goal: str, max_steps: int = 3) -> str:
    """
    Multi-turn web browsing - follows links iteratively to find information.
    
    Use when:
    - Information requires navigating through multiple pages
    - Need to follow "Read more" or "Details" links
    - Example: "Find Mercedes Sosa's discography, then count 2000-2009 albums"
    
    This tool:
    1. Visits start_url
    2. Searches content for goal-related info
    3. Extracts relevant links
    4. Follows most promising link
    5. Repeats until info found or max_steps reached
    
    Better than scrape_and_retrieve when single page doesn't have complete info.
    """
    start_time = time.time()
    
    try:
        if not rag_manager.is_ready():
            rag_manager.initialize()
        
        print(f"🌐 Iterative browsing starting at: {start_url}")
        print(f"   Goal: {goal[:100]}...")
        print(f"   Max steps: {max_steps}")
        
        visited_urls = set()
        current_url = start_url
        all_findings = []
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        for step in range(max_steps):
            if current_url in visited_urls:
                print(f"   Step {step+1}: Already visited, stopping")
                break
            
            visited_urls.add(current_url)
            print(f"   Step {step+1}: Visiting {current_url}")
            
            try:
                response = requests.get(current_url, headers=headers, timeout=15)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Remove noise
                for tag in soup(["script", "style", "nav", "footer", "aside", "header", "iframe"]):
                    tag.extract()
                
                # Extract main content
                main = soup.find('main') or soup.find('article') or soup.find('div', class_='mw-parser-output') or soup.body
                
                if not main:
                    print(f"      No main content found")
                    continue
                
                text = main.get_text(separator='\n', strip=True)
                lines = [l.strip() for l in text.splitlines() if l.strip()]
                text = '\n'.join(lines)
                
                print(f"      Extracted {len(text)} chars")
                
                # Search for goal-related content
                chunks = rag_manager.text_splitter.split_text(text)
                docs = [Document(page_content=c, metadata={"source": current_url, "step": step+1}) for c in chunks]
                
                db = FAISS.from_documents(docs, rag_manager.embeddings)
                retriever = db.as_retriever(search_kwargs={"k": 3})
                retrieved = retriever.invoke(goal)
                
                # Clean up
                del db
                del retriever
                import gc
                gc.collect()
                
                if retrieved:
                    print(f"      Found {len(retrieved)} relevant chunks")
                    for i, doc in enumerate(retrieved):
                        all_findings.append({
                            'step': step + 1,
                            'url': current_url,
                            'content': doc.page_content
                        })
                
                # Extract links for next step
                if step < max_steps - 1:
                    links = []
                    for a in main.find_all('a', href=True):
                        href = a.get('href')
                        text = a.get_text(strip=True).lower()
                        
                        # Make absolute URL
                        if href.startswith('/'):
                            from urllib.parse import urljoin
                            href = urljoin(current_url, href)
                        
                        # Filter relevant links
                        goal_keywords = goal.lower().split()
                        if any(keyword in href.lower() or keyword in text for keyword in goal_keywords):
                            if href.startswith('http') and href not in visited_urls:
                                links.append((href, text))
                    
                    if links:
                        # Pick most relevant link
                        current_url = links[0][0]
                        print(f"      Found {len(links)} potential links, following: {links[0][1][:50]}")
                    else:
                        print(f"      No more relevant links found")
                        break
                else:
                    print(f"      Max steps reached")
                    break
                    
            except Exception as e:
                print(f"      Error on step {step+1}: {e}")
                break
        
        # Compile findings
        if not all_findings:
            result = f"Browsed {len(visited_urls)} pages but found no relevant information for: '{goal}'"
        else:
            result = f"Information gathered from {len(visited_urls)} pages:\n\n"
            for finding in all_findings:
                result += f"[Step {finding['step']} - {finding['url']}]\n{finding['content']}\n\n---\n\n"
            result = truncate_if_needed(result)
        
        telemetry.record_call("iterative_web_browser", time.time() - start_time, True)
        return result
        
    except Exception as e:
        telemetry.record_call("iterative_web_browser", time.time() - start_time, False)
        raise ToolError("iterative_web_browser", e, "Try starting from a more specific URL")


class ScrapeInput(BaseModel):
    url: str = Field(description="URL (http:// or https://)")
    query: str = Field(description="Specific info to find")

@tool(args_schema=ScrapeInput)
@retry_with_backoff(max_retries=3)
def scrape_and_retrieve(url: str, query: str) -> str:
    """
    Scrape webpage and retrieve relevant sections using RAG with smart fallbacks.
    """
    start_time = time.time()
    
    try:
        is_valid, msg = validate_tool_inputs("scrape_and_retrieve", {"url": url, "query": query})
        if not is_valid:
            raise ValueError(msg)
        
        print(f"🌐 Scraping: {url}")
        print(f"   Looking for: {query[:50]}...")
        
        # ===== TRY PRIMARY URL =====
        try:
            response = requests.get(url, timeout=15, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })
            response.raise_for_status()
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                print(f"   ❌ 404 error, trying fallbacks...")
                
                # ===== FALLBACK 1: Try alternative URL formats =====
                if "wikipedia.org" in url:
                    fallback_urls = []
                    
                    # Example: Wikipedia:Featured_articles/2016_November
                    # Try: Wikipedia:Featured_articles#2016
                    if "/20" in url and "_" in url:
                        # Extract year
                        import re
                        year_match = re.search(r'/(\d{4})', url)
                        if year_match:
                            year = year_match.group(1)
                            # Try anchor link format
                            base_url = url.split('/20')[0]
                            fallback_urls.append(f"{base_url}#{year}")
                            # Try without year suffix
                            fallback_urls.append(base_url)
                    
                    # Try with underscores replaced by spaces (URL encoded)
                    if "_" in url:
                        fallback_urls.append(url.replace("_", "%20"))
                    
                    # Try each fallback
                    for fallback_url in fallback_urls:
                        try:
                            print(f"   Trying fallback: {fallback_url}")
                            response = requests.get(fallback_url, timeout=15, headers={
                                'User-Agent': 'Mozilla/5.0'
                            })
                            response.raise_for_status()
                            url = fallback_url  # Update URL for later
                            print(f"   ✓ Fallback succeeded!")
                            break
                        except:
                            continue
                    else:
                        # All fallbacks failed
                        # ===== FALLBACK 2: Use Wikipedia search =====
                        print(f"   All URL fallbacks failed, trying Wikipedia search...")
                        
                        # Extract search terms from URL
                        search_terms = url.split('/')[-1].replace('_', ' ').replace('%20', ' ')
                        
                        # Search Wikipedia
                        search_url = f"https://en.wikipedia.org/w/api.php?action=opensearch&search={search_terms}&limit=1&format=json"
                        search_response = requests.get(search_url, timeout=10)
                        search_data = search_response.json()
                        
                        if len(search_data) > 3 and search_data[3]:
                            # Found a result
                            wiki_url = search_data[3][0]
                            print(f"   ✓ Found via search: {wiki_url}")
                            response = requests.get(wiki_url, timeout=15, headers={
                                'User-Agent': 'Mozilla/5.0'
                            })
                            response.raise_for_status()
                            url = wiki_url
                        else:
                            raise ToolError(
                                "scrape_and_retrieve",
                                Exception(f"404 and all fallbacks failed for {url}"),
                                "Try using wikipedia_search tool to find the correct article first"
                            )
                
                else:
                    # Non-Wikipedia 404
                    raise
            else:
                # Other HTTP error
                raise
        
        # ===== END FALLBACKS =====
        
        # Parse content
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'header', 'footer']):
            element.decompose()
        
        text = soup.get_text(separator='\n', strip=True)
        
        if len(text) < 100:
            raise ValueError(f"Insufficient content extracted from {url}")
        
        print(f"✓ Extracted {len(text)} characters")
        
        # RAG retrieval
        docs = [Document(page_content=text, metadata={"source": url})]
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP
        )
        chunks = text_splitter.split_documents(docs)
        
        print(f"✓ Created {len(chunks)} chunks")
        
        # Search for relevant chunks
        vectorstore = FAISS.from_documents(chunks, rag_manager.embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        relevant_docs = retriever.invoke(query)
        
        print(f"✓ Found {len(relevant_docs)} relevant chunks")
        
        # Format results
        results = []
        for i, doc in enumerate(relevant_docs, 1):
            content = doc.page_content.strip()
            results.append(f"[Section {i}]\n{content}")
        
        result = f"From {url}:\n\n" + "\n\n".join(results)
        
        # Cleanup
        del vectorstore
        gc.collect()
        
        telemetry.record_call("scrape_and_retrieve", time.time() - start_time, True)
        return truncate_if_needed(result)
        
    except Exception as e:
        telemetry.record_call("scrape_and_retrieve", time.time() - start_time, False)
        raise ToolError("scrape_and_retrieve", e)

class VideoAnalysisInput(BaseModel):
    file_path: str = Field(description="Path to video file (.mp4, .mov, etc.)")
    query: str = Field(description="What to find in the video")

@tool(args_schema=VideoAnalysisInput)
def analyze_video(file_path: str, query: str) -> str:
    """
    Analyze video using Gemini Vision (supports video).
    
    Use for:
    - Counting objects/people/animals in video
    - Describing what happens
    - Finding specific moments
    - Visual Q&A about video content
    """
    start_time = time.time()
    
    try:
        print(f"🎥 Analyzing video: {file_path}")
        print(f"   Query: {query[:100]}...")
        
        video_path = find_file(file_path)
        if not video_path and os.path.exists(file_path):
            video_path = Path(file_path)
        
        if not video_path or not video_path.exists():
            raise FileNotFoundError(f"Video not found: {file_path}")
        
        GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
        if not GOOGLE_API_KEY:
            raise ValueError("GEMINI_API_KEY not set")

        # Use Google GenAI SDK directly — LangChain wrapper doesn't support video
        # Try new SDK (google-genai) first, fall back to old SDK (google-generativeai)
        import time as _time
        try:
            from google import genai as _genai
            client = _genai.Client(api_key=GOOGLE_API_KEY)

            print(f"   Uploading video to Gemini Files API (new SDK)...")
            video_file = client.files.upload(file=str(video_path))

            while video_file.state.name == "PROCESSING":
                _time.sleep(2)
                video_file = client.files.get(name=video_file.name)

            if video_file.state.name == "FAILED":
                raise RuntimeError(f"Gemini file processing failed: {video_file.state}")

            print(f"   Analyzing with Gemini...")
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[query, video_file]
            )
            result = response.text

            try:
                client.files.delete(name=video_file.name)
            except Exception:
                pass

        except ImportError:
            import google.generativeai as genai_old
            genai_old.configure(api_key=GOOGLE_API_KEY)

            print(f"   Uploading video to Gemini Files API (old SDK)...")
            video_file = genai_old.upload_file(str(video_path))

            while video_file.state.name == "PROCESSING":
                _time.sleep(2)
                video_file = genai_old.get_file(video_file.name)

            if video_file.state.name == "FAILED":
                raise RuntimeError(f"Gemini file processing failed: {video_file.state}")

            print(f"   Analyzing with Gemini...")
            model = genai_old.GenerativeModel("gemini-2.5-flash")
            response = model.generate_content([query, video_file])
            result = response.text

            try:
                genai_old.delete_file(video_file.name)
            except Exception:
                pass
        
        print(f"✓ Analysis complete: {len(result)} chars")
        
        telemetry.record_call("analyze_video", time.time() - start_time, True)
        return f"Video Analysis:\n{truncate_if_needed(result)}"
        
    except Exception as e:
        telemetry.record_call("analyze_video", time.time() - start_time, False)
        raise ToolError("analyze_video", e, "Check video file path and Gemini API")
        
class FinalAnswerInput(BaseModel):
    answer: str = Field(description="Final answer - exact, no fluff")

@tool(args_schema=FinalAnswerInput)
def final_answer_tool(answer: str) -> str:
    """Submit final answer with normalization"""
    start_time = time.time()
    
    try:
        # Get question from state (you'll need to pass this through)
        # For now, normalize without question context
        original_answer = answer
        answer = normalize_answer(answer)
        
        if answer != original_answer:
            print(f"📝 Normalized answer:")
            print(f"   Before: '{original_answer}'")
            print(f"   After:  '{answer}'")
        
        print(f"\n✅ FINAL: '{answer}'\n")
        
        telemetry.record_call("final_answer_tool", time.time() - start_time, True)
        return f"FINAL_ANSWER: {answer}"
        
    except Exception as e:
        telemetry.record_call("final_answer_tool", time.time() - start_time, False)
        raise ToolError("final_answer_tool", e)


# =============================================================================
# TOOLS LIST
# =============================================================================
defined_tools = [
    think_through_logic,
    validate_answer,
    analyze_data_file,
    search_tool,
    wikipedia_search,
    calculator,
    analyze_video,
    code_interpreter,
    read_file,
    list_directory,
    audio_transcription_tool,
    analyze_image,
    scrape_and_retrieve,
    analyze_chess_position,
    final_answer_tool
]

# =============================================================================
# AGENT STATE
# =============================================================================
class AgentState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
    turn: int
    has_plan: bool
    consecutive_errors: int
    tool_history: List[str]
    last_tool_was_thinking: bool

# =============================================================================
# TOOL CALL PARSER
# =============================================================================
def parse_tool_call_from_string(content: str, tools: List) -> List[ToolCall]:
    """Enhanced fallback parser"""
    print(f"🔧 Parsing tool call from: {content[:300]}...")
    
    tool_name = None
    tool_input = None
    
    # Strategy 1: Groq format
    groq_match = re.search(r"<function=(\w+)\s*(\{.*?\})\s*(?:>|</function>)", content, re.DOTALL)
    if groq_match:
        try:
            tool_name = groq_match.group(1).strip()
            json_str = groq_match.group(2).strip()
            json_str = json_str.encode().decode('unicode_escape')
            tool_input = json.loads(json_str)
            print(f"✓ Parsed Groq format: {tool_name}")
        except:
            tool_name = None
    
    # Strategy 2: Standard format
    if not tool_name:
        func_match = re.search(r"<function[(=]\s*([^)]+)\s*[)>](.*)", content, re.DOTALL | re.IGNORECASE)
        if func_match:
            try:
                tool_name = func_match.group(1).strip().replace("'", "").replace('"', '')
                remaining = func_match.group(2)
                json_start = remaining.find('{')
                if json_start != -1:
                    json_str = remaining[json_start:].strip().rstrip(',')
                    tool_input = json.loads(json_str)
                    print(f"✓ Parsed standard format: {tool_name}")
            except:
                tool_name = None
    
    # Strategy 3: Code block → code_interpreter
    if not tool_name and "```python" in content:
        try:
            code_match = re.search(r"```python\n(.*?)```", content, re.DOTALL)
            if code_match:
                code = code_match.group(1).strip()
                tool_name = "code_interpreter"
                tool_input = {"code": code}
                print(f"✓ Extracted Python code")
        except:
            pass
    
    # Strategy 4: Tool mention
    if not tool_name:
        for tool in tools:
            if tool.name.lower() in content.lower():
                tool_name = tool.name
                tool_input = {}
                
                if tool.args_schema:
                    schema = tool.args_schema.model_json_schema()
                    for prop in schema.get('properties', {}).keys():
                        if prop in schema.get('required', []):
                            tool_input[prop] = "auto_extracted"
                
                print(f"✓ Found mention: {tool_name}")
                break
    
    # Strategy 5: Force thinking
    if not tool_name:
        if len(content) > 50:
            tool_name = "think_through_logic"
            tool_input = {"reasoning": content[:150]}
            print(f"⚠️ Forcing think_through_logic")
    
    if tool_name and tool_input is not None:
        matching = [t for t in tools if t.name == tool_name]
        if matching:
            return [ToolCall(name=tool_name, args=tool_input, id=str(uuid.uuid4()))]
    
    print("❌ All parsing failed")
    return []

# =============================================================================
# CONDITIONAL EDGE
# =============================================================================
def should_continue(state: AgentState):
    """Decide next step"""
    messages = state.get('messages', [])
    if not messages:
        return "agent"
    
    last_message = messages[-1]
    current_turn = state.get('turn', 0)
    
    print(f"📍 Turn {current_turn}, Last: {type(last_message).__name__}")
    
    if current_turn >= config.MAX_TURNS:
        print(f"🛑 Max turns reached")
        return END
    
    if isinstance(last_message, ToolMessage):
        print(f"📨 Tool result → agent")
        return "agent"
    
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        first_tool = last_message.tool_calls[0]
        if first_tool.get("name") == "final_answer_tool":
            return END
        return "tools"
    
    if isinstance(last_message, AIMessage) and not last_message.tool_calls:
        if len(messages) >= 2 and isinstance(messages[-2], AIMessage) and not messages[-2].tool_calls:
            print(f"⚠️ Loop detected")
            return END
        print(f"💭 AI without tool → agent")
        return "agent"
    
    return "agent"

# =============================================================================
# MAIN AGENT CLASS
# =============================================================================
class PlanningReflectionAgent:
    def __init__(self):
        print("🧠 Initializing PlanningReflectionAgent...")
        
        # Load all Groq API keys (GROQ_API_KEY, GROQ_API_KEY_2, GROQ_API_KEY_3, ...)
        groq_keys = []
        primary = os.getenv("GROQ_API_KEY")
        if primary:
            groq_keys.append(primary)
        for i in range(2, 10):
            k = os.getenv(f"GROQ_API_KEY_{i}")
            if k:
                groq_keys.append(k)
        self._groq_keys = groq_keys
        self._groq_key_index = 0
        GROQ_API_KEY = groq_keys[0] if groq_keys else None
        if groq_keys:
            print(f"✅ Loaded {len(groq_keys)} Groq API key(s)")
        else:
            print("ℹ️ No Groq API keys found (Groq fallback unavailable)")
        
        GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
        if not GOOGLE_API_KEY:
            raise ValueError("GEMINI_API_KEY not set")
        
        self.tools = defined_tools
        
        # Initialize RAG
        rag_manager.initialize()
        
        self.system_prompt = """You are a GAIA benchmark agent. Provide the EXACT answer requested — no explanations.
RULES:
1. Every turn must call exactly one tool. Never output text without a tool call.
2. Final answer must be exactly what was asked (a number, a name, a word — not a sentence).
3. Always call validate_answer() before final_answer_tool().
4. Never call the same tool 3 times in a row.
5. think_through_logic is ONLY for logic puzzles, not research.
TOOL ROUTING:
- Logic puzzle → think_through_logic → validate → final_answer
- Factual/person/place → wikipedia_search OR search_tool → scrape_and_retrieve → validate → final_answer
- Count from web → wikipedia_search → code_interpreter → validate → final_answer
- CSV/Excel file → list_directory → analyze_data_file → code_interpreter → validate → final_answer
- Chess position image → analyze_chess_position (NOT analyze_image) → validate → final_answer
- Other image → analyze_image → validate → final_answer
- Audio → audio_transcription_tool → validate → final_answer
- Math/table/logic with data → code_interpreter → validate → final_answer
SPECIAL:
- wikipedia_search takes just the SUBJECT NAME, not the full question.
  RIGHT: wikipedia_search("Mercedes Sosa")  WRONG: wikipedia_search("How many albums Mercedes Sosa")
- YouTube URLs are BLOCKED. Use analyze_video() on the local .mp4 file instead.
- When transcribing audio, preserve EXACT wording (e.g. "freshly squeezed lemon juice" not "lemon juice").
- For math operation tables, use code_interpreter to compute the answer — do not guess.
EXAMPLE:
Q: "How many studio albums did Mercedes Sosa release 2000-2009?"
T1: wikipedia_search("Mercedes Sosa") → discography
T2: code_interpreter("count 2000-2009 albums") → 3
T3: validate_answer("3", question) → PASSED
T4: final_answer_tool("3")
"""
        
        # Initialize LLMs
        print("Initializing LLMs...")

        # Primary: Gemini 2.0 Flash (15 RPM, unlimited TPM, 1500 RPD)
        self.gemini_pro_llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=GOOGLE_API_KEY,
            temperature=0,
            max_tokens=1024
        ).bind_tools(self.tools, tool_choice="auto")
        print("✅ Gemini 2.0 Flash primary initialized")

        # Fallback 1: Gemini 2.5 Flash (5 RPM, 250K TPM, 20 RPD)
        self.gemini_flash_llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=GOOGLE_API_KEY,
            temperature=0,
            max_tokens=1024
        ).bind_tools(self.tools, tool_choice="auto")
        print("✅ Gemini 2.5 Flash fallback initialized")

        # Fallback 2: Groq (if keys provided)
        def _make_groq_llm(model_name, api_key):
            return ChatGroq(
                temperature=0,
                groq_api_key=api_key,
                model_name=model_name,
                max_tokens=1024,
                timeout=60
            ).bind_tools(self.tools, tool_choice="auto")

        if GROQ_API_KEY:
            self.groq_llm = _make_groq_llm("llama-3.3-70b-versatile", GROQ_API_KEY)
            self.groq_qwen_llm = _make_groq_llm("qwen/qwen3-32b", GROQ_API_KEY)
            print("✅ Groq llama + qwen fallback initialized")
        else:
            self.groq_llm = None
            self.groq_qwen_llm = None

        def _rotate_groq_key():
            """Switch to next Groq API key on daily limit exhaustion. Returns True if rotated."""
            next_index = self._groq_key_index + 1
            if next_index >= len(self._groq_keys):
                return False
            self._groq_key_index = next_index
            new_key = self._groq_keys[next_index]
            print(f"🔄 Rotating to Groq API key #{next_index + 1}")
            self.groq_llm = _make_groq_llm("llama-3.3-70b-versatile", new_key)
            self.groq_qwen_llm = _make_groq_llm("qwen/qwen3-32b", new_key)
            self.llm_with_tools = self.groq_llm
            self.current_llm = "groq"
            return True

        self._rotate_groq_key = _rotate_groq_key

        # Fallback 3: Claude (if key provided)
        ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
        if ANTHROPIC_API_KEY:
            try:
                from langchain_anthropic import ChatAnthropic
                self.claude_llm = ChatAnthropic(
                    model="claude-sonnet-4-20250514",
                    anthropic_api_key=ANTHROPIC_API_KEY,
                    temperature=0,
                    max_tokens=4096
                ).bind_tools(self.tools, tool_choice="auto")
                print("✅ Claude fallback initialized")
            except ImportError:
                self.claude_llm = None
                print("⚠️ langchain_anthropic not installed; Claude fallback unavailable")
        else:
            self.claude_llm = None
            print("ℹ️ Claude fallback unavailable (no ANTHROPIC_API_KEY)")

        chain = "Gemini 2.0 Flash → Gemini 2.5 Flash"
        if self.groq_llm:
            chain += " → Groq llama-3.3-70b → Groq qwen3-32b"
        if self.claude_llm:
            chain += " → Claude"
        print(f"✅ LLM chain: {chain}")

        # Start with Gemini 3 Pro
        self.llm_with_tools = self.gemini_pro_llm
        self.current_llm = "gemini_pro"

        def prune_context_if_needed(state: AgentState) -> AgentState:
            """
            Prune conversation history if it's getting too long.
            Keeps system message + recent history to stay under token limits.
            """
            messages = state.get("messages", [])

            # Keep first message (system prompt) + last N messages
            MAX_MESSAGES = 20
            # ~6000 token limit on Groq; system msg ~3000 chars leaves ~18000 for the rest
            MAX_TOOL_CONTENT = 1500

            # Prune by count
            if len(messages) > MAX_MESSAGES:
                print(f"⚠️ Context pruning: {len(messages)} messages → {MAX_MESSAGES}")

                system_msg = None
                if messages and isinstance(messages[0], SystemMessage):
                    system_msg = messages[0]
                    messages = messages[1:]

                recent_messages = messages[-(MAX_MESSAGES-1):]

                if system_msg:
                    messages = [system_msg] + recent_messages
                else:
                    messages = recent_messages

            # Truncate oversized tool outputs to prevent 413 errors
            pruned = []
            for msg in messages:
                if isinstance(msg, ToolMessage) and len(msg.content) > MAX_TOOL_CONTENT:
                    msg = ToolMessage(
                        content=msg.content[:MAX_TOOL_CONTENT] + "...[truncated]",
                        tool_call_id=msg.tool_call_id,
                        name=msg.name
                    )
                pruned.append(msg)

            state["messages"] = pruned
            return state
        
        # Build agent graph
        def agent_node(state: AgentState):
            current_turn = state.get('turn', 0) + 1
            max_retries = config.MAX_RETRIES
            print(f"\n{'='*70}")
            print(f"🤖 AGENT TURN {current_turn}/{config.MAX_TURNS}")
            print('='*70)

            state = prune_context_if_needed(state)
            
            if current_turn > config.MAX_TURNS:
                return {
                    "messages": [SystemMessage(content="Max turns reached.")],
                    "turn": current_turn
                }
            tool_history = state.get('tool_history', [])
    
            # Check for loops (same tool called 3+ times)
            if len(tool_history) >= 3:
                last_3 = tool_history[-3:]
                
                # If same tool 3 times in a row, FORCE change
                if len(set(last_3)) == 1:
                    problem_tool = last_3[0]
                    print(f"🚨 LOOP DETECTED: {problem_tool} called 3x - FORCING CHANGE")
                    
                    force_msg = HumanMessage(
                        content=f"""⚠️ EMERGENCY: You called {problem_tool}() 3 times in a row!
THIS IS A LOOP. You MUST use a DIFFERENT tool now.
BANNED this turn: {problem_tool}
Pick ANY other tool and call it NOW."""
                    )
                    
                    messages_to_send = state["messages"].copy()
                    messages_to_send.append(force_msg)
                else:
                    messages_to_send = state["messages"].copy()
            else:
                messages_to_send = state["messages"].copy()
            # ===== END LOOP DETECTION =====
        
            # Check if we should force reflection
            consecutive_errors = state.get('consecutive_errors', 0)
            should_reflect = (current_turn > 5 and current_turn % Config.REFLECT_EVERY_N_TURNS == 0) or consecutive_errors >= 3         
            
            # Force tool usage
            if len(messages_to_send) >= 2:
                last_msg = messages_to_send[-1]
                if isinstance(last_msg, AIMessage) and not last_msg.tool_calls:
                    force_msg = HumanMessage(
                        content="⚠️ CRITICAL: MUST call a tool. NO reasoning text."
                    )
                    messages_to_send.append(force_msg)
                    print("🚨 Forcing tool usage")

            if should_reflect:
                hint = HumanMessage(
                    content="⚠️ HINT: No progress. Try reflect_on_progress() or different approach."
                )
                messages_to_send.append(hint)
                print("🤔 Reflection hint")
            
            # Invoke LLM with retries and fallback
            ai_message = None
            
            for attempt in range(max_retries):
                try:
                    ai_message = self.llm_with_tools.invoke(messages_to_send)
                    
                    if ai_message.tool_calls:
                        break
                        
                except Exception as e:
                    error_str = str(e)
                    print(f"⚠️ Groq error (attempt {attempt+1}): {error_str[:200]}")
                    
                    # ===== IMPROVED RATE LIMIT HANDLING =====
                    # Non-recoverable errors: bail out immediately, no retries
                    _hard_fail_keywords = [
                        "non-consecutive system messages",
                        "credit balance", "too low",
                        "Function calling is not enabled",
                        "context_length_exceeded",
                    ]
                    if any(kw in error_str for kw in _hard_fail_keywords):
                        print(f"🚨 Unrecoverable LLM error, giving up on question: {error_str[:120]}")
                        ai_message = AIMessage(
                            content="",
                            tool_calls=[ToolCall(
                                name="final_answer_tool",
                                args={"answer": "AGENT FAILED"},
                                id=str(uuid.uuid4())
                            )]
                        )
                        break

                    # Context too large — truncate aggressively and retry immediately
                    if "413" in error_str or "request too large" in error_str.lower():
                        print("❌ Request too large (413) - aggressively pruning context")
                        # Keep system message + last 4 messages, truncate tool content to 1000 chars
                        pruned = []
                        for msg in messages_to_send:
                            if isinstance(msg, SystemMessage):
                                pruned.append(msg)
                                break
                        pruned += messages_to_send[-4:]
                        for msg in pruned:
                            if isinstance(msg, ToolMessage) and len(msg.content) > 1000:
                                msg = ToolMessage(
                                    content=msg.content[:1000] + "...[truncated]",
                                    tool_call_id=msg.tool_call_id,
                                    name=msg.name
                                )
                        messages_to_send = pruned
                        print(f"   Pruned to {len(messages_to_send)} messages, retrying...")
                        continue

                    # Check for rate limit
                    if "429" in error_str or "rate limit" in error_str.lower():
                        print(f"❌ Rate limit hit on {self.current_llm}!")

                        # Detect quota exhaustion (not just temporary rate limit)
                        is_daily_limit = (
                            "per day" in error_str.lower()
                            or "TPD" in error_str
                            or "RESOURCE_EXHAUSTED" in error_str
                            or "exceeded your current quota" in error_str.lower()
                        )
                        if is_daily_limit and self.current_llm in ("groq", "groq_qwen"):
                            if self._rotate_groq_key():
                                print("   Retrying with new Groq API key...")
                                continue
                            else:
                                print("   No more Groq API keys available.")

                        if attempt < max_retries - 1 and not is_daily_limit:
                            wait = 10 * (2 ** attempt)  # 10s, 20s, 40s
                            print(f"   Waiting {wait}s before retry...")
                            time.sleep(wait)
                            continue

                        # Fallback chain: Gemini 3 Pro → Gemini Flash → Groq llama → Groq qwen → Claude → search
                        if self.current_llm != "gemini_flash":
                            print("🔄 Switching to Gemini 2.5 Flash fallback")
                            self.llm_with_tools = self.gemini_flash_llm
                            self.current_llm = "gemini_flash"
                            try:
                                ai_message = self.gemini_flash_llm.invoke(messages_to_send)
                                break
                            except Exception as flash_err:
                                print(f"❌ Gemini Flash fallback also failed: {flash_err}")

                        if self.groq_llm and self.current_llm not in ("groq", "groq_qwen"):
                            print("🔄 Switching to Groq llama fallback")
                            self.llm_with_tools = self.groq_llm
                            self.current_llm = "groq"
                            try:
                                ai_message = self.groq_llm.invoke(messages_to_send)
                                break
                            except Exception as groq_err:
                                print(f"❌ Groq llama fallback also failed: {groq_err}")

                        if self.groq_qwen_llm and self.current_llm != "groq_qwen":
                            print("🔄 Switching to Groq qwen3-32b fallback")
                            self.llm_with_tools = self.groq_qwen_llm
                            self.current_llm = "groq_qwen"
                            try:
                                ai_message = self.groq_qwen_llm.invoke(messages_to_send)
                                break
                            except Exception as qwen_err:
                                print(f"❌ Groq qwen fallback also failed: {qwen_err}")

                        if self.claude_llm and self.current_llm != "claude":
                            print("🔄 Switching to Claude fallback")
                            self.llm_with_tools = self.claude_llm
                            self.current_llm = "claude"
                            try:
                                ai_message = self.claude_llm.invoke(messages_to_send)
                                break
                            except Exception as claude_err:
                                print(f"❌ Claude fallback also failed: {claude_err}")

                        # No LLM available — extract question and do one targeted search
                        print("🔄 No LLM available - attempting targeted search fallback")
                        question_text = ""
                        for msg in state["messages"]:
                            if isinstance(msg, HumanMessage) and msg.content:
                                question_text = str(msg.content)[:200].strip()
                                break
                        ai_message = AIMessage(
                            content="",
                            tool_calls=[ToolCall(
                                name="search_tool",
                                args={"query": question_text or "unknown question"},
                                id=str(uuid.uuid4())
                            )]
                        )
                        break
                    # ===== END RATE LIMIT HANDLING =====

                    # Tool use failed error — try next model before wasting a turn
                    if any(kw in error_str for kw in ["tool_use_failed", "tool call validation"]):
                        if self.current_llm == "gemini_pro" and self.gemini_flash_llm:
                            print("🔄 Tool call failed on Gemini Pro - trying Gemini Flash")
                            self.llm_with_tools = self.gemini_flash_llm
                            self.current_llm = "gemini_flash"
                            try:
                                ai_message = self.gemini_flash_llm.invoke(messages_to_send)
                                break
                            except Exception as flash_err:
                                print(f"❌ Gemini Flash also failed on tool call: {flash_err}")
                        elif self.groq_qwen_llm and self.current_llm != "groq_qwen":
                            print("🔄 Tool call failed - switching to Groq qwen3-32b")
                            self.llm_with_tools = self.groq_qwen_llm
                            self.current_llm = "groq_qwen"
                            try:
                                ai_message = self.groq_qwen_llm.invoke(messages_to_send)
                                break
                            except Exception as qwen_err:
                                print(f"❌ Qwen also failed on tool call: {qwen_err}")
                        print("🚨 Tool error - forcing think_through_logic")
                        ai_message = AIMessage(
                            content="",
                            tool_calls=[ToolCall(
                                name="think_through_logic",
                                args={"reasoning": "Processing..."},
                                id=str(uuid.uuid4())
                            )]
                        )
                        break
                    
                    # Final retry
                    if attempt == max_retries - 1:
                        # Detect permanent (non-retryable) failures and bail immediately
                        _permanent_keywords = [
                            "credit balance", "too low", "Function calling is not enabled",
                            "non-consecutive system messages", "context_length_exceeded",
                        ]
                        if any(kw in error_str for kw in _permanent_keywords):
                            print(f"🚨 Permanent LLM failure, stopping question: {error_str[:100]}")
                            ai_message = AIMessage(
                                content="",
                                tool_calls=[ToolCall(
                                    name="final_answer_tool",
                                    args={"answer": "AGENT FAILED"},
                                    id=str(uuid.uuid4())
                                )]
                            )
                        else:
                            print("🚨 All attempts failed - forcing think_through_logic")
                            ai_message = AIMessage(
                                content="",
                                tool_calls=[ToolCall(
                                    name="think_through_logic",
                                    args={"reasoning": "Processing"},
                                    id=str(uuid.uuid4())
                                )]
                            )
                    else:
                        time.sleep(2 ** attempt)
            
            # Ensure tool calls exist
            if not ai_message.tool_calls:
                if ai_message.content:
                    parsed = parse_tool_call_from_string(ai_message.content, self.tools)
                    if parsed:
                        ai_message.tool_calls = parsed
                        ai_message.content = ""
                    else:
                        ai_message.tool_calls = [ToolCall(
                            name="think_through_logic",
                            args={"reasoning": "analyzing"},
                            id=str(uuid.uuid4())
                        )]
                        ai_message.content = ""
            
            # Track usage
            tool_history = state.get('tool_history', [])
            has_plan = state.get('has_plan', False)
            
            if ai_message.tool_calls:
                tool_name = ai_message.tool_calls[0]['name']
                print(f"🔧 Tool: {tool_name}")
                tool_history.append(tool_name)
                
                if tool_name == "create_plan":
                    has_plan = True
            
            return {
                "messages": [ai_message],
                "turn": current_turn,
                "has_plan": has_plan,
                "tool_history": tool_history,
                "last_tool_was_thinking": ai_message.tool_calls and ai_message.tool_calls[0]['name'] == 'think_through_logic'
            }
        
        def tool_node_wrapper(state: AgentState):
            """Execute tools with error tracking"""
            print(f"🔧 Executing tools...")
            
            tool_executor = ToolNode(self.tools)
            result = tool_executor.invoke(state)
            
            consecutive_errors = state.get('consecutive_errors', 0)
            
            if result.get('messages'):
                last_msg = result['messages'][-1]
                if isinstance(last_msg, ToolMessage):
                    if "Error" in last_msg.content or "error" in last_msg.content.lower():
                        consecutive_errors += 1
                        print(f"⚠️ Tool error (consecutive: {consecutive_errors})")
                    else:
                        consecutive_errors = 0
            
            result['consecutive_errors'] = consecutive_errors
            return result
        
        # Build graph
        print("Building graph...")
        graph_builder = StateGraph(AgentState)
        
        graph_builder.add_node("agent", agent_node)
        graph_builder.add_node("tools", tool_node_wrapper)
        
        graph_builder.add_edge(START, "agent")
        
        graph_builder.add_conditional_edges(
            "agent",
            should_continue,
            {
                "tools": "tools",
                "agent": "agent",
                END: END
            }
        )
        
        graph_builder.add_edge("tools", "agent")
        
        self.graph = graph_builder.compile()
        print("✅ Graph compiled")
    
    def __call__(self, question: str, file_path: str = None) -> str:
        """Execute agent"""
        print(f"\n{'='*70}")
        print(f"🎯 NEW QUESTION")
        print(f"{'='*70}")
        print(f"Q: {question[:200]}...")
        if file_path:
            print(f"📎 File: {file_path}")
        print(f"{'='*70}\n")
        
        # Build question context
        question_text = question
        if file_path:
            file_ext = Path(file_path).suffix.lower()
            file_type = "unknown"
            
            if file_ext in ['.jpg', '.jpeg', '.png', '.gif']:
                file_type = "image"
            elif file_ext in ['.mp3', '.wav', '.m4a']:
                file_type = "audio"
            elif file_ext in ['.csv', '.xlsx']:
                file_type = "data"
            elif file_ext in ['.txt', '.pdf', '.doc']:
                file_type = "document"
            
            question_text += f"\n\n[FILE: {file_path}]"
            question_text += f"\n[TYPE: {file_type}]"
            question_text += f"\nUse appropriate tool first!"
        
        graph_input = {
            "messages": [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=question_text)
            ],
            "file_path": file_path,
            "turn": 0,
            "has_plan": False,
            "consecutive_errors": 0,
            "tool_history": [],
            "last_tool_was_thinking": False
        }
        
        # Reset to Gemini 3 Pro for each question
        self.llm_with_tools = self.gemini_pro_llm
        self.current_llm = "gemini_pro"
        
        final_answer = "AGENT FAILED"
        all_messages = []
        
        try:
            config_dict = {"recursion_limit": config.MAX_TURNS * 2 + 10}
            
            for event in self.graph.stream(graph_input, stream_mode="values", config=config_dict):
                if not event.get('messages'):
                    continue
                
                all_messages = event["messages"]
                last_message = all_messages[-1]
                
                # Check for final answer
                if isinstance(last_message, AIMessage) and last_message.tool_calls:
                    for tool_call in last_message.tool_calls:
                        if tool_call.get("name") == "final_answer_tool":
                            args = tool_call.get('args', {})
                            if 'answer' in args:
                                final_answer = normalize_answer(args['answer'])
                                print(f"\n✅ FINAL: '{final_answer}'\n")
                                break
                
                elif isinstance(last_message, ToolMessage):
                    preview = last_message.content[:200].replace('\n', ' ')
                    print(f"📊 Tool '{last_message.name}': {preview}...")
            
            # Fallback: extract from tool results
            if final_answer == "AGENT FAILED":
                print("⚠️ No final_answer_tool. Checking tools...")
                
                for msg in reversed(all_messages):
                    if isinstance(msg, ToolMessage):
                        if msg.name in ["calculator", "think_through_logic", "code_interpreter"]:
                            content = msg.content.strip()
                            if content and len(content) < 200 and not content.startswith("Error"):
                                lines = content.split('\n')
                                for line in reversed(lines):
                                    if line.strip() and not line.startswith(('✅', '⚠️', 'Next', 'Remember')):
                                        final_answer = line.strip()
                                        print(f"📝 Extracted: '{final_answer}'")
                                        break
                                break
            
            # Clean answer more aggressively
            cleaned = str(final_answer).strip()
            
            # Remove common prefixes (case-insensitive)
            prefixes = [
                "the answer is:", "here is the answer:", "based on",
                "final answer:", "answer:", "the final answer is:",
                "my answer is:", "according to", "i found that",
                "the result is:", "result:", "here's the answer:",
                "after analysis:", "the correct answer is:",
                "from the data:", "from the search:",
            ]
            for prefix in prefixes:
                if cleaned.lower().startswith(prefix.lower()):
                    potential = cleaned[len(prefix):].strip()
                    if potential:
                        cleaned = potential
                        break
            
            # Remove code fences
            cleaned = remove_fences_simple(cleaned)
            
            # Remove backticks
            while cleaned.startswith("`") and cleaned.endswith("`"):
                cleaned = cleaned[1:-1].strip()
            
            # Remove quotes (but only if they wrap entire answer)
            if (cleaned.startswith('"') and cleaned.endswith('"')) or \
               (cleaned.startswith("'") and cleaned.endswith("'")):
                cleaned = cleaned[1:-1].strip()
            
            # Remove trailing period for short answers
            if cleaned.endswith('.') and len(cleaned.split()) < 10:
                cleaned = cleaned[:-1]
            
            # Remove markdown bold/italic
            cleaned = cleaned.replace('**', '').replace('__', '').replace('*', '').replace('_', '')
            
            # Remove bullet points
            if cleaned.startswith(('- ', '* ', '• ')):
                cleaned = cleaned[2:].strip()
            
            # Remove numbered list prefix
            import re
            cleaned = re.sub(r'^\d+\.\s+', '', cleaned)
            
            # Final whitespace cleanup
            cleaned = ' '.join(cleaned.split())
            
            print(f"\n🎉 RETURNING: {cleaned}\n")
            
            return cleaned
            
        except Exception as e:
            print(f"❌ Graph error: {e}")
            print(traceback.format_exc())
            return f"ERROR: {e}"

# =============================================================================
# GLOBAL AGENT
# =============================================================================
agent = None

try:
    rag_manager.initialize()
    agent = PlanningReflectionAgent()
    print("✅ Global agent ready")
    
    if not callable(agent):
        print("❌ Agent not callable")
        agent = None
    else:
        print("✅ Agent is callable")
    
except Exception as e:
    print(f"❌ FATAL: {e}")
    traceback.print_exc()
    agent = None

# =============================================================================
# RUN AND SUBMIT
# =============================================================================
def run_and_submit_all(profile: gr.OAuthProfile | None):
    """Run evaluation and submit"""
    space_id = os.getenv("SPACE_ID")
    
    if profile:
        username = profile.username
        print(f"User: {username}")
    else:
        print("Not logged in")
        return "Please login to HuggingFace", None
    
    global agent
    
    if agent is None:
        return "FATAL: Agent failed to initialize", None
    
    print("✅ Using global agent")
    
    api_url = config.DEFAULT_API_URL
    questions_url = f"{api_url}/questions"
    submit_url = f"{api_url}/submit"
    agent_code = f"https://huggingface.co/spaces/{space_id}/tree/main"
    
    # Fetch questions
    print(f"\n{'='*70}")
    print(f"📥 FETCHING QUESTIONS")
    print(f"{'='*70}\n")
    
    try:
        response = requests.get(questions_url, timeout=15)
        response.raise_for_status()
        questions_data = response.json()
        
        if not questions_data:
            return "No questions fetched", None
        
        print(f"✅ Fetched {len(questions_data)} questions\n")
        
    except Exception as e:
        print(f"❌ Fetch error: {e}")
        return f"Error fetching questions: {e}", None
    
    # Load answer sheet
    validator = AnswerValidator()
    answer_sheet = validator.load_answer_sheet("answer_sheet_json.json")
    
    # Initialize tracking
    progress = ProgressTracker(len(questions_data))
    telemetry.reset()
    
    results_log = []
    answers_payload = []
    
    # Process questions
    print(f"\n{'='*70}")
    print(f"🚀 STARTING EVALUATION")
    print(f"{'='*70}\n")
    
    for idx, item in enumerate(questions_data, 1):
        print(f"\n{'='*70}")
        print(f"📝 QUESTION {idx}/{len(questions_data)}")
        print(f"{'='*70}\n")
        
        task_id = item.get("task_id")
        question_text = item.get("question")
        correct_answer = answer_sheet.get(task_id, "")
        
        # Find file
        local_file_path = None
        files_dir = "files"
        
        try:
            if os.path.exists(files_dir):
                matching_files = [f for f in os.listdir(files_dir) if f.startswith(task_id)]
                
                if matching_files:
                    local_file_path = os.path.join(files_dir, matching_files[0])
                    print(f"✅ Found file: {matching_files[0]}")
                else:
                    print(f"ℹ️ No file for {task_id}")
            else:
                print(f"⚠️ '{files_dir}' not found")
        except Exception as e:
            print(f"❌ File search error: {e}")
        
        try:
            # Run agent
            submitted_answer = agent(question_text, local_file_path)
            answers_payload.append({"task_id": task_id, "submitted_answer": submitted_answer})
            
            # Check correctness
            is_correct, feedback = validator.check_correctness(submitted_answer, correct_answer)
            
            print(f"\n{feedback} - Task {task_id}")
            print(f"   Submitted: '{submitted_answer}'")
            print(f"   Expected:  '{correct_answer}'")
            
            results_log.append({
                "Task ID": task_id,
                "Question": question_text[:100] + "..." if len(question_text) > 100 else question_text,
                "Submitted": submitted_answer,
                "Correct": correct_answer,
                "Status": "✅" if is_correct else "❌"
            })
            
            progress.update(is_correct)
            print(f"\n✅ Question {idx} completed")
            
        except Exception as e:
            print(f"❌ Error on {task_id}: {e}")
            print(traceback.format_exc())
            
            results_log.append({
                "Task ID": task_id,
                "Question": question_text[:100] + "...",
                "Submitted": f"ERROR: {e}",
                "Correct": correct_answer,
                "Status": "❌"
            })
            
            answers_payload.append({"task_id": task_id, "submitted_answer": f"ERROR: {str(e)[:100]}"})
            progress.update(False)
    
    # Print telemetry
    telemetry.report()
    
    # Summary
    correct_count = sum(1 for log in results_log if log.get("Status") == "✅")
    total_count = len(results_log)
    accuracy = (correct_count / total_count * 100) if total_count > 0 else 0
    
    print(f"\n{'='*70}")
    print(f"📊 PRE-SUBMISSION SUMMARY")
    print(f"{'='*70}")
    print(f"Correct: {correct_count}/{total_count} ({accuracy:.1f}%)")
    print(f"{'='*70}\n")
    
    if not answers_payload:
        return "No answers produced", pd.DataFrame(results_log)
    
    # Submit
    submission_data = {
        "username": username.strip(),
        "agent_code": agent_code,
        "answers": answers_payload
    }

    # Save so we can resubmit without re-running the agent
    import json as _json
    os.makedirs("results", exist_ok=True)
    with open("results/latest_submission.json", "w") as f:
        _json.dump(submission_data, f, indent=2)
    print("💾 Saved submission data to results/latest_submission.json")

    print(f"\n{'='*70}")
    print(f"📤 SUBMITTING")
    print(f"{'='*70}\n")
    
    try:
        response = requests.post(submit_url, json=submission_data, timeout=60)
        response.raise_for_status()
        result_data = response.json()

        final_status = (
            f"Submission Successful!\n"
            f"User: {result_data.get('username')}\n"
            f"Score: {result_data.get('score', 'N/A')}% "
            f"({result_data.get('correct_count', '?')}/{result_data.get('total_attempted', '?')})\n"
            f"Message: {result_data.get('message', 'No message')}"
        )

        print(final_status)
        results_df = pd.DataFrame(results_log)
        return final_status, results_df

    except requests.exceptions.HTTPError as e:
        error_detail = f"Server responded with status {e.response.status_code}."
        try:
            error_json = e.response.json()
            error_detail += f" Detail: {error_json.get('detail', e.response.text)}"
        except Exception:
            error_detail += f" Response: {e.response.text[:500]}"
        print(f"❌ Submission failed: {error_detail}")
        results_df = pd.DataFrame(results_log)
        return f"Submission Failed: {error_detail}", results_df

    except Exception as e:
        print(f"❌ Submission failed: {e}")
        results_df = pd.DataFrame(results_log)
        return f"Submission failed: {e}", results_df

def resubmit_saved(profile: gr.OAuthProfile | None):
    """Resubmit the last saved answers without re-running the agent."""
    import json as _json

    if not profile:
        return "Please login to HuggingFace first.", None

    saved_file = "results/latest_submission.json"
    if not os.path.exists(saved_file):
        return "No saved submission found. Run the full evaluation first.", None

    with open(saved_file) as f:
        submission_data = _json.load(f)

    # Override username with current logged-in user
    submission_data["username"] = profile.username.strip()

    submit_url = f"{config.DEFAULT_API_URL}/submit"
    print(f"Resubmitting {len(submission_data['answers'])} answers for {profile.username}...")

    try:
        response = requests.post(submit_url, json=submission_data, timeout=60)
        response.raise_for_status()
        result_data = response.json()
        return (
            f"Submission Successful!\n"
            f"User: {result_data.get('username')}\n"
            f"Score: {result_data.get('score', 'N/A')}% "
            f"({result_data.get('correct_count', '?')}/{result_data.get('total_attempted', '?')})\n"
            f"Message: {result_data.get('message', 'No message')}"
        ), None
    except requests.exceptions.HTTPError as e:
        error_detail = f"Server responded with status {e.response.status_code}."
        try:
            error_json = e.response.json()
            error_detail += f" Detail: {error_json.get('detail', e.response.text)}"
        except Exception:
            error_detail += f" Response: {e.response.text[:500]}"
        return f"Submission Failed: {error_detail}", None
    except Exception as e:
        return f"Submission failed: {e}", None


# =============================================================================
# GRADIO INTERFACE
# =============================================================================
with gr.Blocks() as demo:
    gr.Markdown("# GAIA Agent Evaluation - Refactored")
    gr.Markdown("""
    **Improvements:**
    - Better error handling with retry logic
    - Caching for search results
    - Telemetry and progress tracking
    - Memory management
    - Modular architecture
    **Instructions:**
    1. Clone this space and modify as needed
    2. Login with HuggingFace account
    3. Click 'Run Evaluation & Submit'
    """)

    gr.LoginButton()

    run_button = gr.Button("Run Evaluation & Submit All")
    resubmit_button = gr.Button("Resubmit Last Results (no re-run)")
    status_output = gr.Textbox(label="Status", lines=5, interactive=False)
    results_table = gr.DataFrame(label="Results", wrap=True)

    run_button.click(
        fn=run_and_submit_all,
        outputs=[status_output, results_table],
        queue=False
    )
    resubmit_button.click(
        fn=resubmit_saved,
        outputs=[status_output, results_table],
        queue=False
    )

if __name__ == "__main__":
    print("\n" + "-"*70)
    print("Starting Refactored GAIA Agent")
    print("-"*70 + "\n")
    
    space_host = os.getenv("SPACE_HOST")
    space_id = os.getenv("SPACE_ID")
    
    if space_host:
        print(f"✅ SPACE_HOST: {space_host}")
        print(f"   URL: https://{space_host}.hf.space")
    
    if space_id:
        print(f"✅ SPACE_ID: {space_id}")
        print(f"   Repo: https://huggingface.co/spaces/{space_id}")
    
    print("\n" + "-"*70 + "\n")
    
    demo.launch(debug=True, share=False, ssr_mode=False)
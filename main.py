# fastmcp_ai_tools.py

import base64
import io
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.prompts.base import UserMessage
from transformers import pipeline

# ── 1. Initialize FastMCP ─────────────────────────────────────────────────────
mcp = FastMCP(name="Local AI Tools Wrapper")

# ── 2. Initialize model variables (lazy loading) ──────────────────────────────
summarizer = None
sentiment_analyzer = None
asr = None

def get_summarizer():
    global summarizer
    if summarizer is None:
        print("Loading summarization model...")
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        print("✓ Summarization model loaded")
    return summarizer

def get_sentiment_analyzer():
    global sentiment_analyzer
    if sentiment_analyzer is None:
        print("Loading sentiment analysis model...")
        sentiment_analyzer = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
        print("✓ Sentiment analysis model loaded")
    return sentiment_analyzer

def get_asr():
    global asr
    if asr is None:
        print("Loading speech recognition model...")
        asr = pipeline("automatic-speech-recognition", model="openai/whisper-small")
        print("✓ Speech recognition model loaded")
    return asr

# ── 3. Define your tools ─────────────────────────────────────────────────────

@mcp.tool()
def summarize(text: str) -> str:
    """Summarize the provided text."""
    # Hugging Face summarization returns a list of dicts
    model = get_summarizer()
    result = model(text, truncation=True, max_length=150, min_length=30)
    return result[0]["summary_text"]


@mcp.tool()
def transcribe(audio_base64: str) -> str:
    """Transcribe the provided base64-encoded audio."""
    # Decode base64 string into bytes
    audio_bytes = base64.b64decode(audio_base64)
    # Wrap in an in-memory file
    audio_file = io.BytesIO(audio_bytes)
    # Whisper ASR pipeline accepts file-like objects
    model = get_asr()
    result = model(audio_file)
    return result["text"]


@mcp.tool()
def classify_sentiment_tool(text: str) -> str:
    """Classify the sentiment of the provided text."""
    model = get_sentiment_analyzer()
    result = model(text)[0]
    label = result["label"]
    score = result["score"]
    return f"{label} (confidence: {score:.2f})"


# ── 4. Define a dynamic prompt ────────────────────────────────────────────────

@mcp.prompt()
def ask_about_topic(topic: str) -> UserMessage:
    """
    Generates a user message asking for an explanation of a topic.
    """
    prompt_text = f"Can you please explain the concept of \"{topic}\" in simple terms?"
    return UserMessage(prompt_text)


# ── 5. Run the MCP server ────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Starting MCP server with lazy-loaded models...")
    mcp.run(transport="stdio")

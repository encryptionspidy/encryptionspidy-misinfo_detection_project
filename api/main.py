# api/main.py
import os
import json
import requests
import re
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from typing import Optional, Dict, Any, List
import logging
import hashlib
import yaml
from .utils import is_valid_url
# Import helper functions
from .groq_utils import analyze_misinformation, ask_groq, extract_intent
from .vt_utils import check_url_safety
# Import RAG functionality
from .langchain_utils import RealTimeDataProcessor, load_data_from_web
# Import RSS data loader
from rss_feed_ingestor import get_rss_news  # Corrected import statement

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("api.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="Misinformation Detection API",
              description="A three-way pipeline for detecting misinformation, analyzing URLs, and answering factual questions")

# Add CORS middleware to allow cross-origin requests from frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load configuration
def load_config(config_path="config/config.yaml"):
    """Load configuration from the YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

config = load_config()

# API Keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
VIRUSTOTAL_API_KEY = os.getenv("VIRUSTOTAL_API_KEY")
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH") #RAG VECTOR PATH

# RAG Setup
rag_processor = RealTimeDataProcessor(api_key=GROQ_API_KEY, index_path=FAISS_INDEX_PATH) # Initialize the vector store at the API Level , its ready to process incoming real time query

# Data Ingestion Setup

async def scheduled_data_ingestion():

    rss_feeds = config['data_sources'] #loaded dictionary

    all_articles = [] #collect all here
    for source, feed_url in rss_feeds.items(): #now the web scraper get news one by one for a clean load. if problem please refer to the async operation.
        try:
            if "{query}" in feed_url:
                query = "technology+news"  # Default query - customize or expand as needed
                feed_url = feed_url.format(query=query)
            articles = get_rss_news(feed_url)
            if articles:
                all_articles.extend(articles)
                logger.info(f"Successfully ingested {len(articles)} from {source}")
            else:
                logger.warning(f"No articles ingested from {source} - check URL or network.")
        except Exception as e:
            logger.error(f"Failed to ingest from {source}: {e}", exc_info=True)

    if all_articles:
        try:
            await rag_processor.update_index(all_articles)
            logger.info("RAG index updated with latest news.")
        except Exception as e:
            logger.error(f"Error updating RAG index after ingestion: {e}")
    else:
        logger.warning("No articles to update RAG index.")

# scheduled_data_ingestion() #call it later or write some function call

from fastapi import BackgroundTasks
@app.on_event("startup") #on load API background tasks
async def startup_event():
    """
    Startup event that runs when the FastAPI application starts.
    """
    logger.info("Running startup event...")

    # Run your scheduled tasks here
    await scheduled_data_ingestion()
    logger.info("Startup event completed.")


# Define request models
class QueryModel(BaseModel):
    question: str

class AnalysisResponse(BaseModel):
    result: Dict[str, Any]
    analysis_type: str
    status: str
    message: Optional[str] = None

# Cache for storing recent results to avoid repeated API calls
result_cache = {}

# Home route
@app.get("/")
def home():
    api_status = {
        "groq_api": "Available" if GROQ_API_KEY else "MISSING",
        "virustotal_api": "Available" if VIRUSTOTAL_API_KEY else "MISSING"
    }
    return {
        "message": "Misinformation Detection API is running!",
        "api_status": api_status,
        "endpoints": {
            "/analyze": "POST - Analyze input based on analysis_type",
            "/health": "GET - Health check endpoint"
        }
    }

# Health check endpoint
@app.get("/health")
def health_check():
    return {"status": "healthy"}

# Main analysis endpoint
@app.post("/analyze/", response_model=AnalysisResponse)
async def analyze_input(query: QueryModel):

    try:
        # Check for required API keys
        if not GROQ_API_KEY or not VIRUSTOTAL_API_KEY:
            missing_keys = []
            if not GROQ_API_KEY:
                missing_keys.append("GROQ_API_KEY")
            if not VIRUSTOTAL_API_KEY:
                missing_keys.append("VIRUSTOTAL_API_KEY")
            raise HTTPException(
                status_code=500,
                detail=f"Missing API keys: {', '.join(missing_keys)}. Please check your .env file."
            )

        # 1. Classify the query type
        analysis_type = await classify_query(query.question)
        logger.info(f"Detected input type: {analysis_type} for query: {query.question[:50]}...")

        # Generate cache key
        cache_key = hashlib.md5(f"{analysis_type}:{query.question}".encode()).hexdigest()

        # Check cache for existing result
        if cache_key in result_cache:
            logger.info(f"Cache hit for query: {query.question[:30]}...")
            return AnalysisResponse(
                result=result_cache[cache_key],
                analysis_type=analysis_type,
                status="success (cached)"
            )

        # Process based on detected type
        if analysis_type == "url":
            if not is_valid_url(query.question): # validate url first
                raise HTTPException(status_code=400, detail="Invalid URL format.")
            result = check_url_safety(query.question, VIRUSTOTAL_API_KEY)

        elif analysis_type == "misinfo":
            result = await analyze_misinformation(query.question, GROQ_API_KEY, config['groq']['model']) # using api and passing the model params

        elif analysis_type == "factual":
            # Use RAG to answer factual questions
            result = await rag_processor.query_rag(query.question, GROQ_API_KEY, config['groq']['model'])
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid analysis type: {analysis_type}. Must be one of: url, misinfo, factual"
            )

        # Cache the result
        result_cache[cache_key] = result

        # Limit cache size to prevent memory issues
        if len(result_cache) > 100:  # Adjust as needed
            # Remove oldest items
            keys_to_remove = list(result_cache.keys())[:-100]
            for k in keys_to_remove:
                del result_cache[k]

        return AnalysisResponse(
            result=result,
            analysis_type=analysis_type,
            status="success"
        )
    except HTTPException as e:
        logger.error(f"HTTP Exception: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        return AnalysisResponse(
            result={},
            analysis_type="unknown",
            status="error",
            message=str(e)
        )

# 10. Automated Query Classification Function (RAG for query classification)
async def classify_query(query: str) -> str:
    """
    Classify the type of query: "url", "misinfo", or "factual".  Leverages RAG to aid in Classification.

    Args:
        query (str): The query string.

    Returns:
        str: The classification result ("url", "misinfo", or "factual").
    """
    try:
        classification_prompt = f"""Classify the following query into one of these categories: 'url', 'misinfo', or 'factual'.
        - 'url': If the query is a URL or asks about the safety of a URL.
        - 'misinfo': If the query is a statement that needs to be checked for misinformation.
        - 'factual': If the query is a question seeking a factual answer.

        Query: {query}
        Classification:"""

        # First attempt without RAG

        # Enhance Classification with RAG
        rag_result = await rag_processor.query_rag(query=classification_prompt,api_key=GROQ_API_KEY,model=config['groq']['model'])

        classification = rag_result['answer'].strip().lower() #get model direct classification


        if "url" in classification:
            return "url"
        elif "misinfo" in classification:
            return "misinfo"
        elif "factual" in classification:
            return "factual"

        # Additional heuristic checks (after RAG if necessary)
        if is_valid_url(query):
            return "url"
        elif query.strip().endswith('?'):
            return "factual"

        return "misinfo"  # Default to misinformation if uncertain

    except Exception as e:
        logger.error(f"Query classification error: {e}", exc_info=True)
        return "misinfo"  # Default to misinformation on error


# Run the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

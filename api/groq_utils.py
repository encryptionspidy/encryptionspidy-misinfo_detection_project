# groq_utils.py (modified)

from api.factcheck_api_util import query_fact_check_api  # Corrected import path

import requests
import json
import logging
import yaml
import re
from typing import Dict, Any

logger = logging.getLogger(__name__)

def load_config(config_path="config/config.yaml"):
    """Load configuration from the YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

config = load_config()

async def query_groq(content: str, api_key: str, model: str, temperature: float = None) -> Dict[str, Any]:
    temperature = temperature if temperature is not None else config['groq']['temperature']

    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        data = {
            "model": model,
            "messages": [{"role": "user", "content": content}],
            "temperature": temperature
        }

        response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=data)
        response.raise_for_status()

        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Groq API error: {str(e)}", exc_info=True)
        return {"error": f"Groq API error: {str(e)}"}

async def analyze_misinformation(text: str, api_key:str , model:str) -> Dict[str, Any]:
    # try method for a smooth performance load ( error handling before loading heavy methods . good case study too!)

    #if error fact_check_util this give log trace but method doesnt even star if there . it run super
    fact_check_result = await query_fact_check_api(text)  # Fetch fact check result

    if fact_check_result and fact_check_result.get("verdict"):
        return fact_check_result

    # Enhanced prompt with more context and examples ( after method good . can ensure all things happen
    prompt = f"""
    ## Task: Analyze the following statement for misinformation
    Statement: "{text}"
    ## Analysis Instructions:
    1. Carefully analyze the factual claims in the statement.
    2. Consider the veracity, context, completeness, and intent of the statement.
    3. Check for logical fallacies, emotional manipulation, or misleading framing.
    4. Consider whether the statement contains objectively verifiable claims or subjective opinions.
    5. Be fair and balanced in your assessment.
    ## Classification System:
    - **"hoax"**: Detected misinformation or dubious claims without factual basis, often circulated as truth.
    - **"truth"**: Verified accurate information that aligns with established facts.
    - **"opinion"**: Subjective statements that express personal feelings or judgments without claiming factual status.
    - **"uncertain"**: Statements with insufficient evidence to classify definitively as true or false.
    - **"verified"**: Claims strongly supported by multiple credible sources and evidence.
    - **"fake"**: Deliberately fabricated information intended to deceive.
    - **"satire"**: Content intended for humor or parody but which may be mistaken for factual claims.
    - **"biased"**: Information that may contain factual elements but is presented with a strong slant or selective framing.
    - **"misleading"**: Statements that are technically true but framed to create a false impression or conclusion.
    - **"spam"**: Unsolicited or irrelevant information without meaningful content.
    - **"incomplete"**: Statements that lack important context needed for accurate interpretation.
    ## Response Format:
    Return a JSON object with exactly the following structure and no additional text:
    {{
        "category": "[SELECTED CATEGORY]",
        "confidence": [CONFIDENCE SCORE BETWEEN 0.0 AND 1.0],
        "explanation": "[DETAILED EXPLANATION OF CLASSIFICATION]",
        "recommendation": "[ACTIONABLE ADVICE FOR THE USER]",
        "key_issues": ["ISSUE 1", "ISSUE 2", "..."],
        "verifiable_claims": ["CLAIM 1", "CLAIM 2", "..."]
    }}
    """
    response = await query_groq(prompt, api_key, model,0.1)

    # Extract the assistant's message content
    message_content = response.get("choices", [{}])[0].get("message", {}).get("content", "")

    # Try to parse JSON from the response
    try:
        # Look for JSON in the response (the model might wrap it in markdown code blocks)
        # First, try to extract from code blocks
        json_match = re.search(r'```(?:json)?\s*({[\s\S]*?})\s*```', message_content)
        if json_match:
            json_str = json_match.group(1)
            result = json.loads(json_str)
        else:
            # Look for JSON object in the text
            json_start = message_content.find("{")
            json_end = message_content.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                json_str = message_content[json_start:json_end]
                result = json.loads(json_str)
            else:
                # If no JSON found, create a structured response
                logger.warning("No JSON found in model response, using raw text.")
                result = {
                    "category": "uncertain",
                    "confidence": 0.5,
                    "explanation": message_content[:500],  # First 500 chars as explanation
                    "recommendation": "The AI did not return a properly formatted response. Consider rephrasing your input.",
                    "raw_response": message_content
                }
    except json.JSONDecodeError as e:
        logger.error(f"JSON parse error: {str(e)}")
        result = {
            "category": "uncertain",
            "confidence": 0.5,
            "explanation": "The AI response could not be parsed correctly.",
            "recommendation": "Try rephrasing your input for better results.",
            "raw_response": message_content,
            "error": f"Failed to parse JSON: {str(e)}"
        }

    # Ensure we have all expected fields
    required_fields = ["category", "confidence", "explanation", "recommendation"]
    for field in required_fields:
        if field not in result:
            result[field] = "Not provided"

    logger.info(f"Misinformation analysis complete. Category: {result.get('category', 'unknown')}")
    return result

# Factual Question Answering using Groq
async def ask_groq(question: str,api_key:str , model:str) -> Dict[str, Any]:

    try:
        # Enhanced prompt for better factual answers
        prompt = f"""
        ## Factual Question Answering

        Question: "{question}"

        Please provide a factual, accurate answer to this question. Follow these guidelines:

        1. Answer concisely but completely, focusing on verified information
        2. If you're uncertain about any aspect, clearly indicate what is known vs. uncertain
        3. Support your answer with evidence or reasoning where appropriate
        4. If the question contains false premises, note this in your answer
        5. If the question asks for an opinion on a debated topic, present major viewpoints fairly
        6. If asked about recent events beyond your knowledge cutoff, acknowledge the limitation

        Your response should be helpful, accurate, and avoid any misleading information.
        """

        response = await query_groq(prompt,api_key, model,0.1)

        # Extract the assistant's message content
        message_content = response.get("choices", [{}])[0].get("message", {}).get("content", "")

        # Generate confidence assessment based on language patterns
        confidence_level = "high"
        uncertainty_phrases = ["I'm not sure", "I don't know", "uncertain", "unclear", "possibly", "might be", "could be"]
        for phrase in uncertainty_phrases:
            if phrase.lower() in message_content.lower():
                confidence_level = "medium"
                break

        high_uncertainty_phrases = ["highly uncertain", "impossible to determine", "cannot answer", "no reliable information"]
        for phrase in high_uncertainty_phrases:
            if phrase.lower() in message_content.lower():
                confidence_level = "low"
                break

        result = {
            "question": question,
            "answer": message_content,
            "confidence_level": confidence_level,
            "model": "llama3-70b-8192"
        }

        logger.info(f"Factual question answered. Confidence: {confidence_level}")
        return result
    except Exception as e:
        logger.error(f"Factual question answering error: {str(e)}", exc_info=True)
        return {"error": f"Factual question answering error: {str(e)}"}
# Optional method of groq which may need in future or dependend on your call you add function which use groq for other
async def extract_intent(query: str, api_key: str, model: str) -> str:

    prompt = f"""
    Determine the primary intent of the following query.  Choose ONLY ONE of these categories:
    - check_fact: The user wants to know if a statement is true or false.
    - url_safety: The user is asking if a URL is safe.
    - general_question:  The user is asking a factual question.
    - other:  The intent is unclear or does not fit the categories.

    Here are some examples:
    Query: "Is climate change a hoax?"  Intent: check_fact
    Query: "Is this website a scam?  [insert url] " Intent: url_safety
    Query: "Who is the president of France?"  Intent: general_question

    Query: "{query}" Intent:
    """
    response = await query_groq(prompt, api_key, model, 0.2)
    intent = response.get("choices", [{}])[0].get("message", {}).get("content", "").strip()

    return intent

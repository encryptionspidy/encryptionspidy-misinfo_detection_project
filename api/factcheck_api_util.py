# factcheck_api_util.py

import requests
import json
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

async def query_fact_check_api(statement: str) -> Dict[str, Any]:
    try:
        url = "https://factcheckapi.org/check"  # Replace with actual API endpoint
        payload = json.dumps({"statement": statement})
        headers = {'Content-Type': 'application/json'}

        response = requests.post(url, headers=headers, data=payload)
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        return response.json()

    except requests.exceptions.RequestException as e:
        logger.error(f"Fact Check API request failed: {e}", exc_info=True)
        return {"error": str(e)}
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse Fact Check API response: {e}", exc_info=True)
        return {"error": "Failed to parse API response"}

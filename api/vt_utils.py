# api/vt_utils.py
import requests
import base64
import logging
import yaml
from typing import Dict, Any

logger = logging.getLogger(__name__)

def load_config(config_path="config/config.yaml"):
    """Load configuration from the YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

config = load_config()
VIRUSTOTAL_API_KEY = ""  # Should be fetched from env variables. Not to be committed directly to code.

def check_url_safety(url: str, api_key: str) -> Dict[str, Any]:
    try:
        headers = {
            "x-apikey": api_key,
            "Content-Type": "application/x-www-form-urlencoded"
        }

        # Generate a URL identifier acceptable to VirusTotal
        url_id = base64.urlsafe_b64encode(url.encode()).decode().strip("=")
        direct_url = f"https://www.virustotal.com/api/v3/urls/{url_id}"
        direct_response = requests.get(direct_url, headers=headers)
        direct_response.raise_for_status()

        result = direct_response.json()

        attributes = result.get("data", {}).get("attributes", {})
        last_analysis_stats = attributes.get("last_analysis_stats", {})

        total_engines = sum(last_analysis_stats.values()) or 1
        harmless_count = last_analysis_stats.get("harmless", 0)
        malicious_count = last_analysis_stats.get("malicious", 0)
        suspicious_count = last_analysis_stats.get("suspicious", 0)

        safety_score = (harmless_count / total_engines) * 100
        danger_score = ((malicious_count + suspicious_count) / total_engines) * 100

        # Get assessment and safety category from configuration
        default_assessment = config['virustotal']['default_assessment']
        assessment = default_assessment

        if malicious_count > 0:
            assessment = "Malicious"
        elif suspicious_count > 0:
            assessment = "Suspicious"

        simplified_result = {
            "url": url,
            "scan_stats": last_analysis_stats,
            "last_analysis_date": attributes.get("last_analysis_date"),
            "reputation": attributes.get("reputation", 0),
            "title": attributes.get("title", ""),
            "assessment": assessment,
            "safety_score": round(safety_score, 1),
            "danger_score": round(danger_score, 1),
            "categories": attributes.get("categories", {}),
            "total_votes": attributes.get("total_votes", {})
        }

        logger.info(f"URL analysis complete. Assessment: {assessment}")
        return simplified_result

    except requests.exceptions.RequestException as e:
        logger.error(f"VirusTotal API connection error: {str(e)}", exc_info=True)
        return {"error": f"VirusTotal API connection error: {str(e)}"}
    except Exception as e:
        logger.error(f"Unexpected error processing URL analysis: {str(e)}", exc_info=True)
        return {"error": f"Unexpected error processing URL analysis: {str(e)}"}

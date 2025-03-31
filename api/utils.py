# api/utils.py
import re
from urllib.parse import urlparse
import logging

logger = logging.getLogger(__name__)

def is_valid_url(url: str) -> bool:
    """
    Validates if the given string is a URL.

    Args:
        url (str): The string to validate.

    Returns:
        bool: True if the string is a valid URL, False otherwise.
    """
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])  # check scheme and domain
    except Exception as e:
        logger.warning(f"URL validation failed: {e}")
        return False

# Optional additional utility functions can be added here as needed

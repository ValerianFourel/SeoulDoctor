"""
Cookie Consent Management for GDPR/CCPA Compliance
"""

from pydantic import BaseModel
from typing import Optional, Any
import json
import logging

logger = logging.getLogger(__name__)


class CookieConsent(BaseModel):
    """
    Cookie consent model for GDPR/CCPA compliance.
    
    Attributes:
        necessary: Always true - required for basic functionality
        analytics: User consent for analytics tracking
        advertising: User consent for advertising/marketing
        timestamp: ISO timestamp of when consent was given
    """
    necessary: bool = True  # Always required
    analytics: bool = False
    advertising: bool = False
    timestamp: Optional[str] = None


def get_consent_from_cookie(cookie_value: Optional[str]) -> CookieConsent:
    """
    Parse consent from cookie string and return CookieConsent object.
    Always returns a CookieConsent object (never None or bool).
    
    Args:
        cookie_value: Raw cookie string from request
        
    Returns:
        CookieConsent object (default with all False if parsing fails)
    """
    if not cookie_value:
        logger.debug("No cookie consent found, using defaults")
        return CookieConsent()
    
    try:
        consent_data = json.loads(cookie_value)
        
        # Handle if consent_data is a boolean (legacy format)
        if isinstance(consent_data, bool):
            logger.warning("Legacy boolean cookie format detected, converting to CookieConsent")
            return CookieConsent(analytics=consent_data, advertising=consent_data)
        
        # Handle if consent_data is a dict
        if isinstance(consent_data, dict):
            return CookieConsent(**consent_data)
        
        # Unknown format
        logger.warning(f"Unknown cookie format: {type(consent_data)}, using defaults")
        return CookieConsent()
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse cookie consent: {e}")
        return CookieConsent()
    except Exception as e:
        logger.error(f"Unexpected error parsing cookie consent: {e}", exc_info=True)
        return CookieConsent()


def should_log_analytics(consent: CookieConsent) -> bool:
    """
    Check if analytics logging is allowed based on user consent.
    
    Args:
        consent: CookieConsent object
        
    Returns:
        True if analytics consent is given, False otherwise
    """
    if not isinstance(consent, CookieConsent):
        logger.error(f"Invalid consent object type: {type(consent)}, defaulting to False")
        return False
    
    return consent.analytics


def should_use_advertising(consent: CookieConsent) -> bool:
    """
    Check if advertising features are allowed based on user consent.
    
    Args:
        consent: CookieConsent object
        
    Returns:
        True if advertising consent is given, False otherwise
    """
    if not isinstance(consent, CookieConsent):
        logger.error(f"Invalid consent object type: {type(consent)}, defaulting to False")
        return False
    
    return consent.advertising


def privacy_safe_log(consent: CookieConsent, message: str, level: str = "info"):
    """
    Log a message only if analytics consent is given.
    
    Args:
        consent: CookieConsent object
        message: Log message
        level: Log level (info, debug, warning, error)
    """
    if not isinstance(consent, CookieConsent):
        # If consent is invalid, don't log anything except the error
        logger.error(f"Invalid consent object in privacy_safe_log: {type(consent)}")
        return
    
    if should_log_analytics(consent):
        log_func = getattr(logger, level, logger.info)
        log_func(message)


def get_consent_summary(consent: CookieConsent) -> dict:
    """
    Get a summary of consent status.
    
    Args:
        consent: CookieConsent object
        
    Returns:
        Dictionary with consent status
    """
    if not isinstance(consent, CookieConsent):
        logger.error(f"Invalid consent object type: {type(consent)}")
        return {
            "necessary": True,
            "analytics": False,
            "advertising": False,
            "error": "Invalid consent object"
        }
    
    return {
        "necessary": consent.necessary,
        "analytics": consent.analytics,
        "advertising": consent.advertising,
        "timestamp": consent.timestamp
    }

def ensure_consent_object(consent: Any) -> CookieConsent:
    """
    Ensure we have a valid CookieConsent object.
    Converts legacy formats or returns default if invalid.
    """
    if isinstance(consent, CookieConsent):
        return consent
    
    if isinstance(consent, bool):
        # Legacy format
        logger.warning("Converting legacy boolean consent to CookieConsent object")
        return CookieConsent(analytics=consent, advertising=consent)
    
    if isinstance(consent, dict):
        try:
            return CookieConsent(**consent)
        except Exception as e:
            logger.error(f"Failed to create CookieConsent from dict: {e}")
            return CookieConsent()
    
    # Unknown type
    logger.error(f"Invalid consent type: {type(consent)}, using defaults")
    return CookieConsent()
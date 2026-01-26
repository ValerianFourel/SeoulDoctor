from fastapi import FastAPI, HTTPException, Cookie, Response, Request
from typing import List, Optional, Dict, Any, Tuple
from pydantic import BaseModel
import logging
import sys
import json
# ==========================================
# LOGGING SETUP
# ==========================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ==========================================
# COOKIE CONSENT MODEL
# ==========================================

class CookieConsent(BaseModel):
    """Model for cookie consent settings - GDPR/CCPA compliant"""
    necessary: bool = True
    analytics: bool = False
    advertising: bool = False
    timestamp: Optional[str] = None



# ==========================================
# COOKIE CONSENT HELPERS
# ==========================================

def get_consent_from_cookie(cookie_value: Optional[str]) -> CookieConsent:
    """
    Parse cookie consent from request.
    Returns default (all denied except necessary) if not present.
    """
    if not cookie_value:
        return CookieConsent()
    
    try:
        consent_data = json.loads(cookie_value)
        return CookieConsent(**consent_data)
    except (json.JSONDecodeError, TypeError, ValueError) as e:
        logger.warning(f"Invalid cookieConsent format: {e}")
        return CookieConsent()


def should_log_analytics(consent: CookieConsent) -> bool:
    """Check if we can log analytics data"""
    return consent.analytics


def should_use_advertising(consent: CookieConsent) -> bool:
    """Check if we can use advertising features"""
    return consent.advertising


def privacy_safe_log(consent: CookieConsent, message: str, level: str = "info"):
    """
    Privacy-aware logging that respects user consent.
    Only logs detailed information if analytics consent is given.
    """
    if should_log_analytics(consent):
        if level == "info":
            logger.info(message)
        elif level == "debug":
            logger.debug(message)
        elif level == "warning":
            logger.warning(message)
        elif level == "error":
            logger.error(message)
    else:
        # Minimal logging without user data
        if level in ["warning", "error"]:
            logger.log(logging.WARNING if level == "warning" else logging.ERROR, 
                      "Operation logged (details hidden - no analytics consent)")


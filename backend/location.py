
from typing import List, Optional, Dict, Any
import logging
import re
from dotenv import load_dotenv
import os
from cookies import CookieConsent, should_log_analytics
logger = logging.getLogger(__name__)
import requests 
load_dotenv()

GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")
KAKAO_REST_API_KEY = os.getenv("KAKAO_REST_API_KEY")

# Defaults
DEFAULT_LAT = 37.5219  # Yeouido
DEFAULT_LON = 126.9243


def _log_location_detail(
    consent: Optional[CookieConsent],
    level: str,
    message: str,
    *args: Any,
) -> None:
    """Log raw location data only when the caller granted analytics consent."""
    if consent is not None and should_log_analytics(consent):
        getattr(logger, level)(message, *args)


# --- 4. GOOGLE MAPS API FUNCTIONS ---

def _log_google_request_error(operation: str, error: BaseException) -> None:
    """Log Google transport failures without serializing the key-bearing URL."""
    response = getattr(error, "response", None)
    status_code = getattr(response, "status_code", None)
    status = f", HTTP {status_code}" if status_code is not None else ""
    logger.error(
        "Google Maps %s request failed (%s%s)",
        operation,
        type(error).__name__,
        status,
    )


def google_maps_geocode(
    address: str,
    add_seoul: bool = False,
    consent: Optional[CookieConsent] = None,
) -> Optional[Dict[str, Any]]:
    """
    Convert address to coordinates using Google Maps Geocoding API.
    Returns dict with {lat, lon, address_korean, district, dong, formatted_address} or None.
    
    Args:
        address: Address or place name to geocode
        add_seoul: If True, appends ", Seoul" to the search query
    """
    if not GOOGLE_MAPS_API_KEY:
        logger.warning("GOOGLE_MAPS_API_KEY not set - skipping geocoding")
        return None
    
    # Add Seoul to query if requested
    search_query = f"{address}, Seoul" if add_seoul else address
    
    url = "https://maps.googleapis.com/maps/api/geocode/json"
    params = {
        "address": search_query,
        "key": GOOGLE_MAPS_API_KEY,
        "language": "ko"  # Request Korean results
    }
    
    try:
        response = requests.get(url, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()
        
        if data.get('status') == 'OK' and data.get('results'):
            result_data = data['results'][0]
            location = result_data['geometry']['location']
            
            result = {
                'lat': location['lat'],
                'lon': location['lng'],
                'formatted_address': result_data['formatted_address']
            }
            
            # Parse address components for district and dong
            for component in result_data.get('address_components', []):
                types = component.get('types', [])
                
                # District (구) - sublocality_level_1
                if 'sublocality_level_1' in types:
                    result['district'] = component['long_name']
                
                # Neighborhood (동) - sublocality_level_2
                elif 'sublocality_level_2' in types:
                    result['dong'] = component['long_name']
            
            # Use formatted address as Korean address
            result['address_korean'] = result['formatted_address']
            
            # Set defaults if not found
            result.setdefault('district', '')
            result.setdefault('dong', '')
            
            _log_location_detail(
                consent,
                "info",
                "Google geocoded: %s... → %s",
                address[:30],
                result.get("district", "Unknown"),
            )
            return result
        else:
            _log_location_detail(
                consent,
                "warning",
                "Google geocoding failed for %s: %s",
                address[:30],
                data.get("status"),
            )
            return None
            
    except requests.exceptions.RequestException as e:
        _log_google_request_error("geocoding", e)
        return None
    except (KeyError, ValueError, IndexError) as e:
        logger.error(f"Google Maps geocoding parse error: {e}")
        return None


def google_maps_reverse_geocode(
    lat: float,
    lon: float,
    consent: Optional[CookieConsent] = None,
) -> Optional[Dict[str, Any]]:
    """
    Convert coordinates to address using Google Maps Reverse Geocoding API.
    Returns dict with {address_korean, district, dong, formatted_address} or None.
    """
    if not GOOGLE_MAPS_API_KEY:
        logger.warning("GOOGLE_MAPS_API_KEY not set - skipping reverse geocoding")
        return None
    
    url = "https://maps.googleapis.com/maps/api/geocode/json"
    params = {
        "latlng": f"{lat},{lon}",
        "key": GOOGLE_MAPS_API_KEY,
        "language": "ko"
    }
    
    try:
        response = requests.get(url, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()
        
        if data.get('status') == 'OK' and data.get('results'):
            result_data = data['results'][0]
            
            result = {
                'formatted_address': result_data['formatted_address'],
                'address_korean': result_data['formatted_address']
            }
            
            # Parse address components for district and dong
            for component in result_data.get('address_components', []):
                types = component.get('types', [])
                
                # District (구)
                if 'sublocality_level_1' in types:
                    result['district'] = component['long_name']
                
                # Neighborhood (동)
                elif 'sublocality_level_2' in types:
                    result['dong'] = component['long_name']
            
            # Set defaults if not found
            result.setdefault('district', '')
            result.setdefault('dong', '')
            
            _log_location_detail(
                consent,
                "info",
                "Google reverse geocoded: (%.4f, %.4f) → %s",
                lat,
                lon,
                result.get("district", "Unknown"),
            )
            return result
        else:
            return None
            
    except requests.exceptions.RequestException as e:
        _log_google_request_error("reverse geocoding", e)
        return None
    except (KeyError, ValueError, IndexError) as e:
        logger.error(f"Google Maps reverse geocoding parse error: {e}")
        return None


def google_maps_place_search(
    query: str,
    consent: Optional[CookieConsent] = None,
) -> Optional[Dict[str, Any]]:
    """
    Search for a place using Google Maps Places API (Text Search).
    Returns dict with {lat, lon, address_korean, district, dong, place_name} or None.
    
    This is useful for landmarks, business names, etc.
    """
    if not GOOGLE_MAPS_API_KEY:
        logger.warning("GOOGLE_MAPS_API_KEY not set - skipping place search")
        return None
    
    url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
    params = {
        "query": f"{query}, Seoul",
        "key": GOOGLE_MAPS_API_KEY,
        "language": "ko"
    }
    
    try:
        response = requests.get(url, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()
        
        if data.get('status') == 'OK' and data.get('results'):
            result_data = data['results'][0]
            location = result_data['geometry']['location']
            
            result = {
                'lat': location['lat'],
                'lon': location['lng'],
                'place_name': result_data.get('name', ''),
                'formatted_address': result_data.get('formatted_address', '')
            }
            
            # Get detailed info using place_id
            place_id = result_data.get('place_id')
            if place_id:
                details = google_maps_place_details(place_id)
                if details:
                    result.update(details)
            
            result['address_korean'] = result.get('formatted_address', '')
            result.setdefault('district', '')
            result.setdefault('dong', '')
            
            _log_location_detail(
                consent,
                "info",
                "Google place search: %s... → %s",
                query[:30],
                result.get("place_name", "Unknown"),
            )
            return result
        else:
            return None
            
    except requests.exceptions.RequestException as e:
        _log_google_request_error("place search", e)
        return None
    except (KeyError, ValueError, IndexError) as e:
        logger.error(f"Google Maps place search parse error: {e}")
        return None


def google_maps_place_details(place_id: str) -> Optional[Dict[str, Any]]:
    """
    Get detailed information about a place using Google Maps Place Details API.
    """
    if not GOOGLE_MAPS_API_KEY:
        return None
    
    url = "https://maps.googleapis.com/maps/api/place/details/json"
    params = {
        "place_id": place_id,
        "key": GOOGLE_MAPS_API_KEY,
        "language": "ko",
        "fields": "address_components,formatted_address"
    }
    
    try:
        response = requests.get(url, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()
        
        if data.get('status') == 'OK' and data.get('result'):
            result_data = data['result']
            result = {}
            
            # Parse address components
            for component in result_data.get('address_components', []):
                types = component.get('types', [])
                
                if 'sublocality_level_1' in types:
                    result['district'] = component['long_name']
                elif 'sublocality_level_2' in types:
                    result['dong'] = component['long_name']
            
            return result
        else:
            return None
            
    except requests.exceptions.RequestException as e:
        _log_google_request_error("place details", e)
        return None
    except (KeyError, ValueError, IndexError, TypeError) as e:
        logger.error(f"Google Maps place details parse error: {e}")
        return None


# --- 5. ENHANCED KAKAO API FUNCTIONS (kept as fallback) ---

def kakao_geocode(
    address: str,
    consent: Optional[CookieConsent] = None,
) -> Optional[Dict[str, Any]]:
    """
    Convert address to coordinates using Kakao API with full address standardization.
    Returns dict with {lat, lon, address_korean, district, dong, address_type} or None.
    """
    if not KAKAO_REST_API_KEY:
        logger.warning("KAKAO_REST_API_KEY not set - skipping geocoding")
        return None
    
    url = "https://dapi.kakao.com/v2/local/search/address.json"
    headers = {"Authorization": f"KakaoAK {KAKAO_REST_API_KEY}"}
    params = {"query": address}
    
    try:
        response = requests.get(url, headers=headers, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()
        
        if data.get('documents') and len(data['documents']) > 0:
            doc = data['documents'][0]
            result = {}
            
            # Try road address first, then jibun address
            if doc.get('road_address'):
                addr = doc['road_address']
                result['lat'] = float(addr['y'])
                result['lon'] = float(addr['x'])
                result['address_korean'] = addr['address_name']
                result['address_type'] = 'road'
                result['district'] = addr.get('region_2depth_name', '')  # 구
                result['dong'] = addr.get('region_3depth_name', '')      # 동
            elif doc.get('address'):
                addr = doc['address']
                result['lat'] = float(addr['y'])
                result['lon'] = float(addr['x'])
                result['address_korean'] = addr['address_name']
                result['address_type'] = 'jibun'
                result['district'] = addr.get('region_2depth_name', '')  # 구
                result['dong'] = addr.get('region_3depth_name', '')      # 동
            else:
                return None
            
            _log_location_detail(
                consent,
                "info",
                "Kakao geocoded: %s... → %s",
                address[:30],
                result["district"],
            )
            return result
        else:
            return None
            
    except requests.exceptions.RequestException as e:
        logger.error("Kakao geocoding request failed (%s)", type(e).__name__)
        return None
    except (KeyError, ValueError, IndexError) as e:
        logger.error("Kakao geocoding parse error (%s)", type(e).__name__)
        return None


def kakao_keyword_search(
    query: str,
    consent: Optional[CookieConsent] = None,
) -> Optional[Dict[str, Any]]:
    """Resolve a Seoul landmark or place name through Kakao local search."""
    if not KAKAO_REST_API_KEY:
        logger.warning("KAKAO_REST_API_KEY not set - skipping keyword search")
        return None

    url = "https://dapi.kakao.com/v2/local/search/keyword.json"
    headers = {"Authorization": f"KakaoAK {KAKAO_REST_API_KEY}"}
    try:
        response = requests.get(
            url,
            headers=headers,
            params={"query": query, "size": 15},
            timeout=5,
        )
        response.raise_for_status()
        documents = response.json().get("documents", [])
        document = next(
            (
                item
                for item in documents
                if str(
                    item.get("road_address_name")
                    or item.get("address_name")
                    or ""
                ).startswith("서울")
            ),
            None,
        )
        if not document:
            return None

        latitude = float(document["y"])
        longitude = float(document["x"])
        address = (
            document.get("road_address_name")
            or document.get("address_name")
            or ""
        )
        result = {
            "lat": latitude,
            "lon": longitude,
            "place_name": document.get("place_name", ""),
            "address_korean": address,
            "formatted_address": address,
            "address_type": "place_keyword",
            "district": "",
            "dong": "",
        }
        reverse = kakao_reverse_geocode(latitude, longitude, consent=consent)
        if reverse:
            result.update(reverse)
        _log_location_detail(
            consent,
            "info",
            "Kakao keyword search: %s... → %s",
            query[:30],
            result.get("place_name", "Unknown"),
        )
        return result
    except requests.exceptions.RequestException as error:
        logger.error(
            "Kakao keyword search request failed (%s)",
            type(error).__name__,
        )
        return None
    except (KeyError, TypeError, ValueError, IndexError) as error:
        logger.error("Kakao keyword search parse error (%s)", type(error).__name__)
        return None


def kakao_reverse_geocode(
    lat: float,
    lon: float,
    consent: Optional[CookieConsent] = None,
) -> Optional[Dict[str, Any]]:
    """
    Convert coordinates to address using Kakao API.
    Returns dict with {address_korean, district, dong} or None.
    """
    if not KAKAO_REST_API_KEY:
        logger.warning("KAKAO_REST_API_KEY not set - skipping reverse geocoding")
        return None
    
    url = "https://dapi.kakao.com/v2/local/geo/coord2address.json"
    headers = {"Authorization": f"KakaoAK {KAKAO_REST_API_KEY}"}
    params = {"x": lon, "y": lat}  # Note: Kakao uses x=lon, y=lat
    
    try:
        response = requests.get(url, headers=headers, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()
        
        if data.get('documents') and len(data['documents']) > 0:
            doc = data['documents'][0]
            result = {}
            
            # Try road address first, then jibun address
            if doc.get('road_address'):
                addr = doc['road_address']
                result['address_korean'] = addr['address_name']
                result['district'] = addr.get('region_2depth_name', '')
                result['dong'] = addr.get('region_3depth_name', '')
            elif doc.get('address'):
                addr = doc['address']
                result['address_korean'] = addr['address_name']
                result['district'] = addr.get('region_2depth_name', '')
                result['dong'] = addr.get('region_3depth_name', '')
            else:
                return None
            
            _log_location_detail(
                consent,
                "info",
                "Kakao reverse geocoded: (%.4f, %.4f) → %s",
                lat,
                lon,
                result["district"],
            )
            return result
        else:
            return None
            
    except requests.exceptions.RequestException as e:
        logger.error("Kakao reverse geocoding request failed (%s)", type(e).__name__)
        return None
    except (KeyError, ValueError, IndexError) as e:
        logger.error("Kakao reverse geocoding parse error (%s)", type(e).__name__)
        return None


def verify_and_standardize_address(
    location_text: str,
    consent: Optional[CookieConsent] = None,
) -> Optional[Dict[str, Any]]:
    """
    Verify and standardize any location input using Google Maps API with fallbacks.
    Returns standardized address info with coordinates and zone details.
    
    FALLBACK STRATEGY:
    1. Try Google Maps geocoding directly
    2. Try Google Maps with ", Seoul" added
    3. Try Google Maps place search (for landmarks, business names)
    4. Try Kakao address and landmark search as final fallbacks
    
    This is the main entry point for address verification.
    """
    if not location_text:
        return None
    
    _log_location_detail(consent, "debug", "Verifying location: %s", location_text)
    
    alias = re.fullmatch(
        r"(?:(?:seoul|서울(?:특별시)?)\s*)?(?:중구\s*)?"
        r"(?:myeon(?:g)?[ -]?dong|명동)\s*(station|역)?"
        r"(?:\s*,?\s*(?:seoul|서울))?",
        location_text.strip(),
        flags=re.IGNORECASE,
    )
    if alias:
        location_text = "서울 중구 명동" + ("역" if alias.group(1) else "")

    def matches_requested_place(result: Optional[Dict[str, Any]]) -> bool:
        if not result:
            return False
        if not alias:
            return True
        identity = " ".join(str(result.get(field, "")) for field in (
            "district", "dong", "place_name", "formatted_address", "address_korean"
        ))
        return "서울" in identity and "중구" in identity and "명동" in identity

    # ===== STRATEGY 1: Direct Google Maps Geocoding =====
    if GOOGLE_MAPS_API_KEY:
        logger.debug("Trying Google Maps direct geocoding...")
        result = google_maps_geocode(
            location_text, add_seoul=False, consent=consent
        )
        if matches_requested_place(result):
            _log_location_detail(
                consent,
                "info",
                "Google Maps direct geocoding succeeded: %s → %s",
                location_text,
                result.get("district", "Unknown"),
            )
            return result
    
    # ===== STRATEGY 2: Google Maps with ", Seoul" =====
    if GOOGLE_MAPS_API_KEY:
        logger.debug("Trying Google Maps with ', Seoul' appended...")
        result = google_maps_geocode(
            location_text, add_seoul=True, consent=consent
        )
        if matches_requested_place(result):
            _log_location_detail(
                consent,
                "info",
                "Google Maps Seoul-suffixed geocoding succeeded: %s → %s",
                location_text,
                result.get("district", "Unknown"),
            )
            return result
    
    # ===== STRATEGY 3: Google Maps Place Search =====
    if GOOGLE_MAPS_API_KEY:
        logger.debug("Trying Google Maps place search...")
        result = google_maps_place_search(location_text, consent=consent)
        if matches_requested_place(result):
            _log_location_detail(
                consent,
                "info",
                "Google Maps place search succeeded: %s → %s",
                location_text,
                result.get("place_name", "Unknown"),
            )
            return result
    
    # ===== STRATEGY 4: Kakao Maps Fallback =====
    if KAKAO_REST_API_KEY:
        logger.debug("Trying Kakao Maps as fallback...")
        
        # Try direct Kakao geocoding
        result = kakao_geocode(location_text, consent=consent)
        if matches_requested_place(result):
            _log_location_detail(
                consent,
                "info",
                "Kakao geocoding succeeded: %s → %s",
                location_text,
                result.get("district", "Unknown"),
            )
            return result

        result = kakao_keyword_search(location_text, consent=consent)
        if matches_requested_place(result):
            _log_location_detail(
                consent,
                "info",
                "Kakao keyword search succeeded: %s → %s",
                location_text,
                result.get("place_name", "Unknown"),
            )
            return result
        
        # Try adding "서울" prefix for Kakao
        if not location_text.startswith("서울") and not location_text.lower().startswith("seoul"):
            result = kakao_geocode(f"서울 {location_text}", consent=consent)
            if matches_requested_place(result):
                _log_location_detail(
                    consent,
                    "info",
                    "Kakao Seoul-suffixed geocoding succeeded: %s → %s",
                    location_text,
                    result.get("district", "Unknown"),
                )
                return result

            result = kakao_keyword_search(
                f"서울 {location_text}", consent=consent
            )
            if matches_requested_place(result):
                _log_location_detail(
                    consent,
                    "info",
                    "Kakao Seoul-suffixed keyword search succeeded: %s → %s",
                    location_text,
                    result.get("place_name", "Unknown"),
                )
                return result
        
        # Try adding "구" suffix for Kakao
        if "구" not in location_text and "동" not in location_text:
            result = kakao_geocode(
                f"서울 {location_text}구", consent=consent
            )
            if matches_requested_place(result):
                _log_location_detail(
                    consent,
                    "info",
                    "Kakao district-suffixed geocoding succeeded: %s → %s",
                    location_text,
                    result.get("district", "Unknown"),
                )
                return result
    
    # ===== ALL STRATEGIES FAILED =====
    logger.warning("All geocoding attempts failed")
    logger.warning("   → Google Maps API: " + ("Available" if GOOGLE_MAPS_API_KEY else "NOT CONFIGURED"))
    logger.warning("   → Kakao Maps API: " + ("Available" if KAKAO_REST_API_KEY else "NOT CONFIGURED"))
    
    return None

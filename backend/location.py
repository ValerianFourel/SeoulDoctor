
from typing import List, Optional, Dict, Any
import logging
from dotenv import load_dotenv
import os
logger = logging.getLogger(__name__)
import requests 
load_dotenv()

GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")
KAKAO_REST_API_KEY = os.getenv("KAKAO_REST_API_KEY")

# Defaults
DEFAULT_LAT = 37.5219  # Yeouido
DEFAULT_LON = 126.9243


# --- 4. GOOGLE MAPS API FUNCTIONS ---

def google_maps_geocode(address: str, add_seoul: bool = False) -> Optional[Dict[str, Any]]:
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
            
            logger.info(f"✓ Google geocoded: {address[:30]}... → {result.get('district', 'Unknown')}")
            return result
        else:
            logger.warning(f"Google geocoding failed: {data.get('status')} for '{address}'")
            return None
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Google Maps geocoding error: {e}")
        return None
    except (KeyError, ValueError, IndexError) as e:
        logger.error(f"Google Maps geocoding parse error: {e}")
        return None


def google_maps_reverse_geocode(lat: float, lon: float) -> Optional[Dict[str, Any]]:
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
            
            logger.info(f"✓ Google reverse geocoded: ({lat:.4f}, {lon:.4f}) → {result.get('district', 'Unknown')}")
            return result
        else:
            return None
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Google Maps reverse geocoding error: {e}")
        return None
    except (KeyError, ValueError, IndexError) as e:
        logger.error(f"Google Maps reverse geocoding parse error: {e}")
        return None


def google_maps_place_search(query: str) -> Optional[Dict[str, Any]]:
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
            
            logger.info(f"✓ Google place search: {query[:30]}... → {result.get('place_name', 'Unknown')}")
            return result
        else:
            return None
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Google Maps place search error: {e}")
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
            
    except Exception as e:
        logger.error(f"Google Maps place details error: {e}")
        return None


# --- 5. ENHANCED KAKAO API FUNCTIONS (kept as fallback) ---

def kakao_geocode(address: str) -> Optional[Dict[str, Any]]:
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
            
            logger.info(f"✓ Geocoded: {address[:30]}... → {result['district']}")
            return result
        else:
            return None
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Kakao geocoding error: {e}")
        return None
    except (KeyError, ValueError, IndexError) as e:
        logger.error(f"Kakao geocoding parse error: {e}")
        return None


def kakao_reverse_geocode(lat: float, lon: float) -> Optional[Dict[str, Any]]:
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
            
            logger.info(f"✓ Reverse geocoded: ({lat:.4f}, {lon:.4f}) → {result['district']}")
            return result
        else:
            return None
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Kakao reverse geocoding error: {e}")
        return None
    except (KeyError, ValueError, IndexError) as e:
        logger.error(f"Kakao reverse geocoding parse error: {e}")
        return None


def verify_and_standardize_address(location_text: str) -> Optional[Dict[str, Any]]:
    """
    Verify and standardize any location input using Google Maps API with fallbacks.
    Returns standardized address info with coordinates and zone details.
    
    FALLBACK STRATEGY:
    1. Try Google Maps geocoding directly
    2. Try Google Maps with ", Seoul" added
    3. Try Google Maps place search (for landmarks, business names)
    4. Try Kakao Maps as final fallback (if available)
    
    This is the main entry point for address verification.
    """
    if not location_text:
        return None
    
    logger.debug(f"🔍 Verifying location: '{location_text}'")
    
    # ===== STRATEGY 1: Direct Google Maps Geocoding =====
    if GOOGLE_MAPS_API_KEY:
        logger.debug("Trying Google Maps direct geocoding...")
        result = google_maps_geocode(location_text, add_seoul=False)
        if result:
            logger.info(f"✅ Google Maps (direct): '{location_text}' → {result.get('district', 'Unknown')}")
            return result
    
    # ===== STRATEGY 2: Google Maps with ", Seoul" =====
    if GOOGLE_MAPS_API_KEY:
        logger.debug("Trying Google Maps with ', Seoul' appended...")
        result = google_maps_geocode(location_text, add_seoul=True)
        if result:
            logger.info(f"✅ Google Maps (+ Seoul): '{location_text}' → {result.get('district', 'Unknown')}")
            return result
    
    # ===== STRATEGY 3: Google Maps Place Search =====
    if GOOGLE_MAPS_API_KEY:
        logger.debug("Trying Google Maps place search...")
        result = google_maps_place_search(location_text)
        if result:
            logger.info(f"✅ Google Maps (place search): '{location_text}' → {result.get('place_name', 'Unknown')}")
            return result
    
    # ===== STRATEGY 4: Kakao Maps Fallback =====
    if KAKAO_REST_API_KEY:
        logger.debug("Trying Kakao Maps as fallback...")
        
        # Try direct Kakao geocoding
        result = kakao_geocode(location_text)
        if result:
            logger.info(f"✅ Kakao Maps (fallback): '{location_text}' → {result.get('district', 'Unknown')}")
            return result
        
        # Try adding "서울" prefix for Kakao
        if not location_text.startswith("서울") and not location_text.lower().startswith("seoul"):
            result = kakao_geocode(f"서울 {location_text}")
            if result:
                logger.info(f"✅ Kakao Maps (+ 서울): '{location_text}' → {result.get('district', 'Unknown')}")
                return result
        
        # Try adding "구" suffix for Kakao
        if "구" not in location_text and "동" not in location_text:
            result = kakao_geocode(f"서울 {location_text}구")
            if result:
                logger.info(f"✅ Kakao Maps (+ 구): '{location_text}' → {result.get('district', 'Unknown')}")
                return result
    
    # ===== ALL STRATEGIES FAILED =====
    logger.warning(f"❌ All geocoding attempts failed for: '{location_text}'")
    logger.warning("   → Google Maps API: " + ("Available" if GOOGLE_MAPS_API_KEY else "NOT CONFIGURED"))
    logger.warning("   → Kakao Maps API: " + ("Available" if KAKAO_REST_API_KEY else "NOT CONFIGURED"))
    
    return None

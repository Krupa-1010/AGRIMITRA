"""
AgriMitra Agentic Prototype - Agent implementations
"""
import json
import logging
import os
from typing import Dict, Any, List, Optional
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
import requests
import re
from config import (
    REASONER_SYSTEM_PROMPT, DISEASE_AGENT_SYSTEM_PROMPT, 
    COORDINATOR_SYSTEM_PROMPT, MOCK_PRICE_DATA, REMEDIES_FILE
)
try:
    from cnn_model import PlantDiseaseCNN
    CNN_AVAILABLE = True
except ImportError:
    CNN_AVAILABLE = False
    logging.warning("CNN model module not available. Image-based detection disabled.")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AgriMitraLLM:
    """Base LLM class for all agents"""
    def __init__(self, model_name: str = "openai/gpt-oss-20b"):
        from config import GROQ_API_KEY, GROQ_API_BASE
        
        self.llm = None
        self.api_available = False
        
        if GROQ_API_KEY:
            try:
                self.llm = ChatOpenAI(
                    model=model_name,
                    temperature=0.1,
                    api_key=GROQ_API_KEY,
                    base_url=GROQ_API_BASE
                )
                self.api_available = True
                logger.info("Groq API configured successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize Groq API: {e}")
                self.api_available = False
        else:
            logger.info("No Groq API key provided, using fallback logic")
    
    def chat(self, system_prompt: str, user_input: str) -> str:
        """Generic chat method for all agents"""
        if not self.api_available or not self.llm:
            logger.info("Using fallback logic - no API available")
            return self._fallback_response(system_prompt, user_input)
        
        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_input)
            ]
            response = self.llm.invoke(messages)
            return response.content
        except Exception as e:
            logger.error(f"LLM Error: {e}")
            return self._fallback_response(system_prompt, user_input)
    
    def _fallback_response(self, system_prompt: str, user_input: str) -> str:
        """Fallback response when API is not available"""
        # Simple keyword-based fallback logic
        user_lower = user_input.lower()
        
        # Check if this is the reasoner system prompt
        if "AI coordinator" in system_prompt and "intent" in system_prompt:
            # Fallback reasoning logic
            disease_keywords = ['disease', 'sick', 'infected', 'spots', 'mold', 'wilting', 'yellow', 'blight', 'mildew', 'rust', 'leaf', 'stem', 'root']
            price_keywords = ['price', 'market', 'sell', 'cost', 'value', 'rate', 'mandi']
            
            has_disease = any(keyword in user_lower for keyword in disease_keywords)
            has_price = any(keyword in user_lower for keyword in price_keywords)
            
            crop = None
            for crop_name in MOCK_PRICE_DATA.keys():
                if crop_name in user_lower:
                    crop = crop_name
                    break
            
            intent = []
            agents = []
            
            if has_disease:
                intent.append("disease")
                agents.append("disease_agent")
            
            if has_price:
                intent.append("market")
                agents.append("price_agent")
            
            # If neither disease nor price signals appear and no known crop found, treat as out_of_scope
            if not intent and crop is None:
                return json.dumps({
                    "intent": ["out_of_scope"],
                    "crop": None,
                    "agents_to_trigger": []
                })
            
            if not intent:
                intent = ["disease"]
                agents = ["disease_agent"]
            
            return json.dumps({
                "intent": intent,
                "crop": crop,
                "agents_to_trigger": agents
            })
        
        # Check if this is the disease agent system prompt
        elif "plant disease diagnosis expert" in system_prompt:
            # Fallback disease diagnosis
            return json.dumps({
                "disease": "Unknown disease (demo mode)",
                "confidence": "low",
                "needs_remedy": True,
                "explanation": "Running in demo mode - please provide more specific symptoms for accurate diagnosis"
            })
        
        # Check if this is the coordinator system prompt
        elif "synthesizes information from multiple agricultural agents" in system_prompt:
            # Fallback coordinator response
            return "I'm running in demo mode without API access. For full functionality, please configure your Groq API key. The system can still provide basic disease and price information using local data."
        
        else:
            return "Demo mode response - API not configured"

# Tools
@tool
def remedy_tool(disease_name: str) -> str:
    """Tool to get remedy information for a specific disease"""
    try:
        with open(REMEDIES_FILE, 'r') as f:
            remedies = json.load(f)
        
        for remedy in remedies:
            if remedy['disease_name'].lower() == disease_name.lower():
                return json.dumps(remedy, indent=2)
        
        return json.dumps({
            "error": f"No remedy found for disease: {disease_name}",
            "suggestion": "Please consult with a local agricultural expert"
        })
    except Exception as e:
        logger.error(f"Remedy tool error: {e}")
        return json.dumps({"error": f"Failed to load remedies: {e}"})

@tool
def price_tool(crop_name: str) -> str:
    """Tool to get current market price information for a crop"""
    try:
        crop_lower = crop_name.lower()
        if crop_lower in MOCK_PRICE_DATA:
            price_info = MOCK_PRICE_DATA[crop_lower]
            return json.dumps(price_info, indent=2)
        else:
            return json.dumps({
                "error": f"No price data available for: {crop_name}",
                "available_crops": list(MOCK_PRICE_DATA.keys())
            })
    except Exception as e:
        logger.error(f"Price tool error: {e}")
        return json.dumps({"error": f"Failed to get price data: {e}"})

@tool
def get_current_location() -> str:
    """Automatically get current location (latitude/longitude) using IP-based geolocation."""
    try:
        # Use ip-api.com free service for IP-based geolocation
        resp = requests.get(
            "http://ip-api.com/json/",
            params={"fields": "status,message,lat,lon,city,region,country"},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        
        if data.get("status") == "success":
            return json.dumps({
                "lat": float(data.get("lat", 0)),
                "lon": float(data.get("lon", 0)),
                "display_name": f"{data.get('city', '')}, {data.get('region', '')}, {data.get('country', '')}".strip(", "),
                "city": data.get("city"),
                "region": data.get("region"),
                "country": data.get("country")
            })
        else:
            return json.dumps({"error": f"IP geolocation failed: {data.get('message', 'Unknown error')}"})
    except Exception as e:
        logger.error(f"Current location error: {e}")
        return json.dumps({"error": f"Failed to get current location: {e}"})

@tool
def geocode_location(query: str) -> str:
    """Geocode a free-form location string to latitude/longitude using OpenStreetMap Nominatim."""
    try:
        resp = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": query, "format": "json", "limit": 1},
            headers={"User-Agent": "AgriMitra/1.0 (educational)"},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        if not data:
            return json.dumps({"error": f"No results for location: {query}"})
        top = data[0]
        return json.dumps({
            "lat": float(top.get("lat")),
            "lon": float(top.get("lon")),
            "display_name": top.get("display_name")
        })
    except Exception as e:
        logger.error(f"Geocode error: {e}")
        return json.dumps({"error": f"Geocoding failed: {e}"})

def _find_shops_overpass(lat: float, lon: float, radius_m: int = 20000) -> List[Dict[str, Any]]:
    """Find shops using Overpass API (OpenStreetMap) - Method 1"""
    try:
        overpass_url = "https://overpass-api.de/api/interpreter"
        query = f"""
        [out:json][timeout:25];
        (
          node["shop"="agrarian"](around:{radius_m},{lat},{lon});
          node["shop"="farm"](around:{radius_m},{lat},{lon});
          node["name"~"fertilizer|fertiliser|agro|agricultural|pesticide|seed", i](around:{radius_m},{lat},{lon});
          way["shop"="agrarian"](around:{radius_m},{lat},{lon});
          way["shop"="farm"](around:{radius_m},{lat},{lon});
          way["name"~"fertilizer|fertiliser|agro|agricultural|pesticide|seed", i](around:{radius_m},{lat},{lon});
        );
        out center 20;
        """
        resp = requests.post(overpass_url, data={"data": query}, headers={"User-Agent": "AgriMitra/1.0 (educational)"}, timeout=25)
        resp.raise_for_status()
        data = resp.json()
        elements = data.get("elements", [])
        results = []
        for el in elements:
            tags = el.get("tags", {})
            name = tags.get("name") or tags.get("brand") or "Unknown"
            shop = tags.get("shop")
            addr = ", ".join(filter(None, [
                tags.get("addr:street"), tags.get("addr:city"), tags.get("addr:state"), tags.get("addr:postcode")
            ])) or None
            center = el.get("center") or {"lat": el.get("lat"), "lon": el.get("lon")}
            results.append({
                "name": name,
                "shop": shop,
                "address": addr,
                "lat": center.get("lat"),
                "lon": center.get("lon"),
                "source": "overpass"
            })
        return results
    except Exception as e:
        logger.error(f"Overpass error: {e}")
        return []

def _find_shops_nominatim(lat: float, lon: float, radius_m: int = 20000) -> List[Dict[str, Any]]:
    """Find shops using Nominatim place search - Method 2"""
    try:
        # Convert radius from meters to approximate degrees (rough approximation)
        # 1 degree latitude ≈ 111 km, so radius_m/111000 gives approximate degree radius
        radius_deg = radius_m / 111000
        
        # Search terms for fertilizer shops
        search_terms = [
            "fertilizer shop",
            "agro shop",
            "agricultural input store",
            "pesticide shop",
            "seed store"
        ]
        
        results = []
        seen_names = set()
        
        for term in search_terms:
            try:
                # Use reverse geocoding area and search nearby
                resp = requests.get(
                    "https://nominatim.openstreetmap.org/search",
                    params={
                        "q": term,
                        "format": "json",
                        "limit": 10,
                        "bounded": 1,
                        "viewbox": f"{lon-radius_deg},{lat+radius_deg},{lon+radius_deg},{lat-radius_deg}",
                        "addressdetails": 1
                    },
                    headers={"User-Agent": "AgriMitra/1.0 (educational)"},
                    timeout=10,
                )
                resp.raise_for_status()
                data = resp.json()
                
                for item in data:
                    name = item.get("display_name", "").split(",")[0]  # Get first part as name
                    if name.lower() not in seen_names and any(keyword in name.lower() for keyword in ["fertilizer", "agro", "agricultural", "pesticide", "seed"]):
                        seen_names.add(name.lower())
                        results.append({
                            "name": name,
                            "shop": "agricultural",
                            "address": item.get("display_name"),
                            "lat": float(item.get("lat", 0)),
                            "lon": float(item.get("lon", 0)),
                            "source": "nominatim"
                        })
            except Exception as e:
                logger.warning(f"Nominatim search error for '{term}': {e}")
                continue
        
        return results
    except Exception as e:
        logger.error(f"Nominatim search error: {e}")
        return []

def _generate_google_maps_search_urls(lat: float, lon: float, location_name: str = None) -> List[Dict[str, str]]:
    """Generate Google Maps search URLs for fertilizer shops (NO API KEY REQUIRED - FREE)
    
    Returns a list of search URLs that users can open directly in their browser.
    """
    search_queries = [
        "fertilizer shop",
        "agricultural input store",
        "agro shop",
        "pesticide shop",
        "seed store",
        "fertilizer store"
    ]
    
    urls = []
    base_location = location_name if location_name else f"{lat},{lon}"
    
    for query in search_queries:
        # Google Maps search URL format (NO API KEY REQUIRED - FREE)
        # Format 1: Direct search with location
        query_encoded = query.replace(' ', '+')
        search_url = f"https://www.google.com/maps/search/{query_encoded}/@{lat},{lon},15z"
        
        # Alternative format using query parameters (more reliable)
        alt_url = f"https://www.google.com/maps/search/?api=1&query={query_encoded}+near+{lat},{lon}"
        
        urls.append({
            "query": query,
            "url": search_url,
            "alt_url": alt_url,
            "description": f"Search for {query} near this location"
        })
    
    return urls

def _find_shops_photon(lat: float, lon: float, radius_m: int = 20000) -> List[Dict[str, Any]]:
    """Find shops using Photon geocoding API - Method 3"""
    try:
        # Photon API for place search
        search_terms = ["fertilizer", "agro", "agricultural", "pesticide", "seed"]
        results = []
        seen_names = set()
        
        for term in search_terms:
            try:
                resp = requests.get(
                    "https://photon.komoot.io/api/",
                    params={
                        "q": f"{term} shop",
                        "lat": lat,
                        "lon": lon,
                        "limit": 10
                    },
                    timeout=10,
                )
                resp.raise_for_status()
                data = resp.json()
                
                features = data.get("features", [])
                for feature in features:
                    props = feature.get("properties", {})
                    name = props.get("name", "Unknown")
                    geometry = feature.get("geometry", {})
                    coords = geometry.get("coordinates", [])
                    
                    if coords and len(coords) >= 2:
                        shop_lon, shop_lat = coords[0], coords[1]
                        # Calculate distance (rough check)
                        distance_km = ((shop_lat - lat) ** 2 + (shop_lon - lon) ** 2) ** 0.5 * 111
                        
                        if distance_km <= (radius_m / 1000) and name.lower() not in seen_names:
                            seen_names.add(name.lower())
                            results.append({
                                "name": name,
                                "shop": "agricultural",
                                "address": props.get("street", "") + ", " + props.get("city", "") if props.get("street") else props.get("city", ""),
                                "lat": shop_lat,
                                "lon": shop_lon,
                                "source": "photon"
                            })
            except Exception as e:
                logger.warning(f"Photon search error for '{term}': {e}")
                continue
        
        return results
    except Exception as e:
        logger.error(f"Photon search error: {e}")
        return []

@tool
def find_fertilizer_shops(lat: float, lon: float, radius_m: int = 20000) -> str:
    """Find nearby fertilizer/agro input shops using multiple data sources with fallback.
    
    Tries multiple methods:
    1. Google Maps Search URLs (FREE, NO API KEY) - Direct links to Google Maps
    2. Overpass API (OpenStreetMap) - Primary data method
    3. Nominatim place search - Fallback method
    4. Photon geocoding API - Additional fallback
    
    Returns combined results from all successful sources plus Google Maps search URLs.
    """
    all_results = []
    seen_locations = set()  # To avoid duplicates based on lat/lon
    
    # Method 1: Generate Google Maps search URLs (FREE, NO API KEY REQUIRED)
    logger.info("Generating Google Maps search URLs (free, no API key)...")
    google_maps_urls = _generate_google_maps_search_urls(lat, lon)
    
    # Method 2: Try Overpass API (most reliable for OSM data)
    logger.info("Searching for shops using Overpass API...")
    overpass_results = _find_shops_overpass(lat, lon, radius_m)
    for shop in overpass_results:
        key = (round(shop["lat"], 4), round(shop["lon"], 4))
        if key not in seen_locations:
            seen_locations.add(key)
            all_results.append(shop)
    
    # Method 3: Try Nominatim if Overpass didn't return enough results
    if len(all_results) < 5:
        logger.info("Searching for shops using Nominatim API...")
        nominatim_results = _find_shops_nominatim(lat, lon, radius_m)
        for shop in nominatim_results:
            key = (round(shop["lat"], 4), round(shop["lon"], 4))
            if key not in seen_locations:
                seen_locations.add(key)
                all_results.append(shop)
    
    # Method 4: Try Photon API as additional source
    if len(all_results) < 10:
        logger.info("Searching for shops using Photon API...")
        photon_results = _find_shops_photon(lat, lon, radius_m)
        for shop in photon_results:
            key = (round(shop["lat"], 4), round(shop["lon"], 4))
            if key not in seen_locations:
                seen_locations.add(key)
                all_results.append(shop)
    
    # Sort by distance (closest first) - simple distance calculation
    def distance_from_center(shop):
        shop_lat, shop_lon = shop.get("lat", 0), shop.get("lon", 0)
        return ((shop_lat - lat) ** 2 + (shop_lon - lon) ** 2) ** 0.5
    
    all_results.sort(key=distance_from_center)
    
    # Prepare response with both shop data and Google Maps URLs
    response_data = {
        "count": len(all_results),
        "shops": all_results[:20],
        "sources_used": list(set(shop.get("source", "unknown") for shop in all_results)),
        "google_maps_search_urls": google_maps_urls,
        "note": "Click the Google Maps URLs above to see shops directly on Google Maps (free, no API key required)"
    }
    
    if all_results:
        return json.dumps(response_data, indent=2)
    else:
        # Even if no shops found via APIs, still provide Google Maps URLs
        return json.dumps({
            "error": "No fertilizer shops found in the specified radius via OpenStreetMap",
            "count": 0,
            "shops": [],
            "google_maps_search_urls": google_maps_urls,
            "note": "Use the Google Maps search URLs above to find shops directly on Google Maps (free, no API key required)"
        }, indent=2)

class ReasonerNode:
    """Reasoner Node - Analyzes user input and determines which agents to trigger"""
    
    def __init__(self):
        self.llm = AgriMitraLLM()
    
    def process(self, user_input: str) -> Dict[str, Any]:
        """Process user input and determine agent routing"""
        logger.info(f"Reasoner processing: {user_input}")
        
        try:
            response = self.llm.chat(REASONER_SYSTEM_PROMPT, user_input)
            
            # Try to parse JSON response
            try:
                parsed_response = json.loads(response)
            except json.JSONDecodeError:
                # Fallback if JSON parsing fails
                logger.warning("Failed to parse JSON, using fallback logic")
                parsed_response = self._fallback_reasoning(user_input)
            
            logger.info(f"Reasoner output: {parsed_response}")
            # If out_of_scope, ensure no next nodes
            next_nodes = parsed_response.get("agents_to_trigger", [])
            if parsed_response.get("intent") and "out_of_scope" in parsed_response.get("intent"):
                next_nodes = []
            return {
                "reasoner_output": parsed_response,
                "user_input": user_input,
                "next_nodes": next_nodes
            }
            
        except Exception as e:
            logger.error(f"Reasoner error: {e}")
            return {
                "reasoner_output": {"error": str(e)},
                "user_input": user_input,
                "next_nodes": []
            }
    
    def _fallback_reasoning(self, user_input: str) -> Dict[str, Any]:
        """Fallback reasoning when JSON parsing fails"""
        user_lower = user_input.lower()
        
        # Simple keyword-based reasoning
        disease_keywords = ['disease', 'sick', 'infected', 'spots', 'mold', 'wilting', 'yellow']
        price_keywords = ['price', 'market', 'sell', 'cost', 'value']
        
        has_disease = any(keyword in user_lower for keyword in disease_keywords)
        has_price = any(keyword in user_lower for keyword in price_keywords)
        
        crop = None
        for crop_name in MOCK_PRICE_DATA.keys():
            if crop_name in user_lower:
                crop = crop_name
                break
        
        intent = []
        agents = []
        
        if has_disease:
            intent.append("disease")
            agents.append("disease_agent")
        
        if has_price:
            intent.append("market")
            agents.append("price_agent")
        
        if not intent:  # Default to disease if unclear
            intent = ["disease"]
            agents = ["disease_agent"]
        
        return {
            "intent": intent,
            "crop": crop,
            "agents_to_trigger": agents
        }

class DiseaseAgentNode:
    """Disease Agent - Diagnoses plant diseases and provides remedies"""
    
    def __init__(self):
        self.llm = AgriMitraLLM()
        # Initialize CNN model if available
        self.cnn_model = None
        if CNN_AVAILABLE:
            try:
                self.cnn_model = PlantDiseaseCNN()
                if self.cnn_model.is_available():
                    logger.info("CNN model loaded successfully for image-based disease detection")
                else:
                    logger.warning("CNN model file not found or could not be loaded")
            except Exception as e:
                logger.warning(f"Could not initialize CNN model: {e}")
    
    def process(self, user_input: str, crop: Optional[str] = None, image_path: Optional[str] = None) -> Dict[str, Any]:
        """Process disease diagnosis request - handles both image and text inputs"""
        logger.info(f"DiseaseAgent processing: {user_input}, image_path: {image_path}")
        
        # Check if both image and text are provided
        has_image = image_path and os.path.exists(image_path) and self.cnn_model and self.cnn_model.is_available()
        has_text = user_input and user_input.strip() and user_input.strip() != "Analyze this plant image for disease detection"
        
        if has_image and has_text:
            # Process both and compare confidence scores
            logger.info("Processing both image and text inputs, will compare confidence scores")
            image_result = self._process_image(image_path, user_input)
            text_result = self._process_text(user_input, crop)
            
            # Extract confidence scores
            image_confidence = self._get_confidence_score(image_result)
            text_confidence = self._get_confidence_score(text_result)
            
            logger.info(f"Image confidence: {image_confidence}, Text confidence: {text_confidence}")
            
            # Return the result with higher confidence
            if image_confidence >= text_confidence:
                logger.info("Selecting image-based result (higher confidence)")
                return image_result
            else:
                logger.info("Selecting text-based result (higher confidence)")
                return text_result
        elif has_image:
            logger.info("Processing image-based disease detection")
            return self._process_image(image_path, user_input)
        else:
            logger.info("Processing text-based disease detection")
            return self._process_text(user_input, crop)
    
    def _get_confidence_score(self, result: Dict[str, Any]) -> float:
        """Extract confidence score from result (numeric value between 0.0 and 1.0)"""
        diagnosis = result.get("disease_diagnosis", {})
        
        # Check if confidence_score exists (from image processing)
        if "confidence_score" in diagnosis:
            return float(diagnosis["confidence_score"])
        
        # Convert text confidence string to numeric score
        confidence_str = diagnosis.get("confidence", "low").lower()
        if confidence_str == "high":
            return 0.8
        elif confidence_str == "medium":
            return 0.5
        else:  # low or unknown
            return 0.3
    
    def _process_image(self, image_path: str, user_input: str) -> Dict[str, Any]:
        """Process disease diagnosis from image using CNN model"""
        try:
            # Get CNN prediction
            cnn_result = self.cnn_model.predict(image_path)
            
            if "error" in cnn_result:
                logger.error(f"CNN prediction error: {cnn_result['error']}")
                # Fallback to text-based if CNN fails
                return self._process_text(user_input, cnn_result.get("crop"))
            
            disease = cnn_result.get("disease")
            crop = cnn_result.get("crop")
            confidence = cnn_result.get("confidence", 0.0)
            full_class_name = cnn_result.get("full_class_name", "")
            is_healthy = cnn_result.get("is_healthy", False)
            
            # Create diagnosis structure
            diagnosis = {
                "disease": disease if not is_healthy else "Healthy",
                "crop": crop,
                "confidence": "high" if confidence > 0.8 else "medium" if confidence > 0.5 else "low",
                "confidence_score": confidence,
                "needs_remedy": not is_healthy and disease and disease != "Healthy",
                "explanation": f"CNN model detected: {full_class_name} with {confidence*100:.1f}% confidence",
                "is_healthy": is_healthy,
                "detection_method": "CNN",
                "top_3_predictions": cnn_result.get("top_3_predictions", [])
            }
            
            # Get remedy if disease detected
            remedy_info = None
            if diagnosis.get("needs_remedy", False) and disease:
                # Try multiple disease name variations
                remedy_result = remedy_tool.invoke({"disease_name": disease})
                remedy_data = json.loads(remedy_result)
                
                # If remedy not found, try with crop name prefix
                if "error" in remedy_data:
                    # Try alternative disease names
                    alternative_names = [
                        f"{crop} {disease}" if crop else disease,
                        disease.replace(" ", "_"),
                        disease.lower()
                    ]
                    for alt_name in alternative_names:
                        remedy_result = remedy_tool.invoke({"disease_name": alt_name})
                        remedy_data = json.loads(remedy_result)
                        if "error" not in remedy_data:
                            break
                
                remedy_info = remedy_data if "error" not in remedy_data else None
            
            # Automatically search for nearby fertilizer shops when disease is detected
            shops_info = None
            if diagnosis.get("needs_remedy", False) and disease and not is_healthy:
                logger.info("Disease detected - automatically searching for nearby fertilizer shops")
                shops_info = self._search_shops(user_input, auto_location=True)
            
            # Also check if user explicitly requested shop search
            if not shops_info:
                text_lower = user_input.lower() if user_input else ""
                ask_shop_keywords = [
                    "fertilizer shop", "fertilizer shops", "fertiliser shop", "fertiliser shops",
                    "agro shop", "agro shops", "agri input", "agri store", "agri stores",
                    "buy fertilizer", "where to buy", "shop near", "shops near", "store near"
                ]
                if any(k in text_lower for k in ask_shop_keywords):
                    shops_info = self._search_shops(user_input, auto_location=False)
            
            result = {
                "disease_diagnosis": diagnosis,
                "remedy_info": remedy_info,
                "shops_info": shops_info,
                "cnn_result": cnn_result,
                "agent": "disease_agent"
            }
            
            logger.info(f"DiseaseAgent (CNN) output: {result}")
            return result
            
        except Exception as e:
            logger.error(f"DiseaseAgent (CNN) error: {e}")
            # Fallback to text-based processing
            return self._process_text(user_input, None)
    
    def _process_text(self, user_input: str, crop: Optional[str] = None) -> Dict[str, Any]:
        """Process disease diagnosis from text using LLM"""
        # Prepare context with crop information
        context = user_input
        if crop:
            context = f"Crop: {crop}. Symptoms: {user_input}"
        
        try:
            response = self.llm.chat(DISEASE_AGENT_SYSTEM_PROMPT, context)
            
            # Parse response
            try:
                diagnosis = json.loads(response)
            except json.JSONDecodeError:
                # Fallback diagnosis
                diagnosis = {
                    "disease": "Unknown disease",
                    "confidence": "low",
                    "needs_remedy": False,
                    "explanation": "Unable to parse diagnosis, please provide more specific symptoms"
                }
            
            # Add confidence_score for consistency with image processing
            confidence_str = diagnosis.get("confidence", "low").lower()
            if confidence_str == "high":
                diagnosis["confidence_score"] = 0.8
            elif confidence_str == "medium":
                diagnosis["confidence_score"] = 0.5
            else:  # low or unknown
                diagnosis["confidence_score"] = 0.3
            
            remedy_info = None
            if diagnosis.get("needs_remedy", False):
                remedy_result = remedy_tool.invoke({"disease_name": diagnosis["disease"]})
                remedy_info = json.loads(remedy_result)

            # Automatically search for nearby fertilizer shops when disease is detected
            shops_info = None
            if diagnosis.get("needs_remedy", False) and diagnosis.get("disease") and diagnosis.get("disease") != "Healthy":
                logger.info("Disease detected - automatically searching for nearby fertilizer shops")
                shops_info = self._search_shops(user_input, auto_location=True)
            
            # Also check if user explicitly requested shop search
            if not shops_info:
                shops_info = self._search_shops(user_input, auto_location=False)
            
            result = {
                "disease_diagnosis": diagnosis,
                "remedy_info": remedy_info,
                "shops_info": shops_info,
                "detection_method": "LLM",
                "agent": "disease_agent"
            }
            
            logger.info(f"DiseaseAgent (Text) output: {result}")
            return result
            
        except Exception as e:
            logger.error(f"DiseaseAgent error: {e}")
            return {
                "disease_diagnosis": {"error": str(e)},
                "remedy_info": None,
                "agent": "disease_agent"
            }
    
    def _search_shops(self, user_input: str = None, auto_location: bool = False) -> Optional[Dict[str, Any]]:
        """Helper method to search for fertilizer shops.
        
        Args:
            user_input: User query text (optional if auto_location is True)
            auto_location: If True, automatically fetch current location instead of parsing from user_input
        """
        shops_info = None
        
        # If auto_location is True, automatically get current location
        if auto_location:
            logger.info("Automatically fetching current location for shop search")
            geo_raw = get_current_location.invoke({})
            geo = json.loads(geo_raw)
            if not geo.get("error"):
                logger.info(f"Current location detected: {geo.get('display_name', 'Unknown')}")
                shops_raw = find_fertilizer_shops.invoke({
                    "lat": geo["lat"],
                    "lon": geo["lon"],
                    "radius_m": 20000
                })
                shops_info = {
                    "location": geo,
                    "results": json.loads(shops_raw)
                }
            else:
                logger.warning(f"Failed to get current location: {geo.get('error')}")
            return shops_info
        
        # Otherwise, check if user explicitly requested shop search
        if not user_input:
            return None
            
        text_lower = user_input.lower()
        ask_shop_keywords = [
            "fertilizer shop", "fertilizer shops", "fertiliser shop", "fertiliser shops",
            "agro shop", "agro shops", "agri input", "agri store", "agri stores",
            "buy fertilizer", "where to buy", "shop near", "shops near", "store near"
        ]
        if any(k in text_lower for k in ask_shop_keywords):
            logger.info("Shop search requested by user query")
            # Extract location after 'in|at|near <location>' up to a stop token
            loc = None
            m = re.search(r"\b(?:in|at|near)\s+([a-zA-Z\s,]+?)\b(?:to|for|\.|,|$)", user_input, re.IGNORECASE)
            if m:
                loc = m.group(1).strip()
            # If still no loc, try shorter cleanup by trimming trailing purpose phrases
            if not loc:
                maybe_loc = re.sub(r"\b(to|for)\b.*$", "", user_input, flags=re.IGNORECASE).strip()
                # Heuristic: if contains at least one space and not too long, attempt
                if 2 <= len(maybe_loc.split()) <= 6:
                    loc = maybe_loc
            # Final fallback: try just the last proper noun token if present
            if not loc:
                toks = re.findall(r"[A-Z][a-z]+", user_input)
                if toks:
                    loc = toks[-1]
            if loc:
                logger.info(f"Geocoding location for shop search: {loc}")
                geo_raw = geocode_location.invoke({"query": loc})
                geo = json.loads(geo_raw)
                if not geo.get("error"):
                    shops_raw = find_fertilizer_shops.invoke({
                        "lat": geo["lat"],
                        "lon": geo["lon"],
                        "radius_m": 20000
                    })
                    shops_info = {
                        "location": geo,
                        "results": json.loads(shops_raw)
                    }
                else:
                    logger.info(f"Geocoding failed for location: {loc} -> {geo}")
            else:
                # If user asked for shops but no location provided, use current location
                logger.info("User requested shops but no location provided, using current location")
                geo_raw = get_current_location.invoke({})
                geo = json.loads(geo_raw)
                if not geo.get("error"):
                    shops_raw = find_fertilizer_shops.invoke({
                        "lat": geo["lat"],
                        "lon": geo["lon"],
                        "radius_m": 20000
                    })
                    shops_info = {
                        "location": geo,
                        "results": json.loads(shops_raw)
                    }
                else:
                    logger.info(f"Failed to get current location: {geo.get('error')}")
        return shops_info

class PriceAgentNode:
    """Price Agent - Provides market price information and selling advice"""
    
    def __init__(self):
        pass  # No LLM needed for this agent
    
    def process(self, crop: str) -> Dict[str, Any]:
        """Process price information request"""
        logger.info(f"PriceAgent processing crop: {crop}")
        
        try:
            price_result = price_tool.invoke({"crop_name": crop})
            price_info = json.loads(price_result)
            
            result = {
                "price_info": price_info,
                "agent": "price_agent"
            }
            
            logger.info(f"PriceAgent output: {result}")
            return result
            
        except Exception as e:
            logger.error(f"PriceAgent error: {e}")
            return {
                "price_info": {"error": str(e)},
                "agent": "price_agent"
            }

class CoordinatorNode:
    """Coordinator Node - Synthesizes outputs from all agents"""
    
    def __init__(self):
        self.llm = AgriMitraLLM()
    
    def process(self, user_input: str, agent_outputs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process and synthesize all agent outputs"""
        logger.info(f"Coordinator processing {len(agent_outputs)} agent outputs")
        
        # Prepare context for synthesis
        context = f"Original farmer query: {user_input}\n\n"
        context += "Agent outputs:\n"
        
        for output in agent_outputs:
            context += f"- {output.get('agent', 'unknown')}: {json.dumps(output, indent=2)}\n"
        
        try:
            # If reasoner marked out_of_scope, bypass LLM and return refusal
            try:
                # Attempt to detect out_of_scope from agent_outputs context
                # The reasoner output is added to workflow state, but not passed here directly
                # We infer out_of_scope if there are no agent outputs and the query seems non-agri
                non_agri_clues = [
                    'who is ', 'what is ', 'when was ', 'biography', 'president', 'movie', 'capital of', 'history', 'celebrity'
                ]
                u_lower = user_input.lower()
                if len(agent_outputs) == 0 and any(k in u_lower for k in non_agri_clues):
                    refusal = (
                        "This system only answers agriculture-focused questions (crops, plant diseases, farm practices, "
                        "and market prices). Your query appears to be outside this scope. Please ask something related to "
                        "plants or agriculture."
                    )
                    return {
                        "final_response": refusal,
                        "agent_outputs": agent_outputs,
                        "agent": "coordinator"
                    }
            except Exception:
                pass

            response = self.llm.chat(COORDINATOR_SYSTEM_PROMPT, context)
            
            result = {
                "final_response": response,
                "agent_outputs": agent_outputs,
                "agent": "coordinator"
            }
            
            logger.info("Coordinator synthesis completed")
            return result
            
        except Exception as e:
            logger.error(f"Coordinator error: {e}")
            return {
                "final_response": f"Error in coordination: {e}",
                "agent_outputs": agent_outputs,
                "agent": "coordinator"
            }

"""
AgriMitra Agentic Prototype - Agent implementations
"""
import json
import logging
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

@tool
def find_fertilizer_shops(lat: float, lon: float, radius_m: int = 5000) -> str:
    """Find nearby fertilizer/agro input shops around a location using Overpass API."""
    try:
        # Overpass query: search for shops that likely sell agro inputs/fertilizers
        # We search for nodes/ways with shop=agrarian OR name tags containing 'fertilizer'/'agro'
        overpass_url = "https://overpass-api.de/api/interpreter"
        query = f"""
        [out:json][timeout:25];
        (
          node["shop"="agrarian"](around:{radius_m},{lat},{lon});
          node["shop"="farm"](around:{radius_m},{lat},{lon});
          node["name"~"fertilizer|fertiliser|agro", i](around:{radius_m},{lat},{lon});
          way["shop"="agrarian"](around:{radius_m},{lat},{lon});
          way["shop"="farm"](around:{radius_m},{lat},{lon});
          way["name"~"fertilizer|fertiliser|agro", i](around:{radius_m},{lat},{lon});
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
                "lon": center.get("lon")
            })
        return json.dumps({"count": len(results), "shops": results[:20]}, indent=2)
    except Exception as e:
        logger.error(f"Overpass error: {e}")
        return json.dumps({"error": f"Overpass query failed: {e}"})

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
    
    def process(self, user_input: str, crop: Optional[str] = None) -> Dict[str, Any]:
        """Process disease diagnosis request"""
        logger.info(f"DiseaseAgent processing: {user_input}")
        
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
            
            remedy_info = None
            if diagnosis.get("needs_remedy", False):
                remedy_result = remedy_tool.invoke({"disease_name": diagnosis["disease"]})
                remedy_info = json.loads(remedy_result)

            # Detect if user asked for nearby shops and try to fetch using tools
            shops_info = None
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
                            "radius_m": 10000
                        })
                        shops_info = {
                            "location": geo,
                            "results": json.loads(shops_raw)
                        }
                    else:
                        logger.info(f"Geocoding failed for location: {loc} -> {geo}")
                else:
                    logger.info("No location detected for shop search request")
            
            result = {
                "disease_diagnosis": diagnosis,
                "remedy_info": remedy_info,
                "shops_info": shops_info,
                "agent": "disease_agent"
            }
            
            logger.info(f"DiseaseAgent output: {result}")
            return result
            
        except Exception as e:
            logger.error(f"DiseaseAgent error: {e}")
            return {
                "disease_diagnosis": {"error": str(e)},
                "remedy_info": None,
                "agent": "disease_agent"
            }

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

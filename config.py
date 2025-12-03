"""
Configuration file for AgriMitra Agentic Prototype
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# LLM Configuration - Groq API
LLM_MODEL = "openai/gpt-oss-20b"  # Using Groq's GPT-OSS-20B model
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_BASE = "https://api.groq.com/openai/v1"  # Groq API endpoint

# File paths
REMEDIES_FILE = "remedies.json"
LOGS_FILE = "agrimitra_logs.json"

# Mock price data (in USD per kg)
MOCK_PRICE_DATA = {
    "tomato": {
        "current_price": 2.50,
        "trend": "increasing",
        "last_week_price": 2.20,
        "advice": "sell",
        "reasoning": "Prices are rising, good time to sell"
    },
    "potato": {
        "current_price": 1.80,
        "trend": "stable",
        "last_week_price": 1.85,
        "advice": "wait",
        "reasoning": "Prices are stable, consider waiting for better opportunity"
    },
    "onion": {
        "current_price": 3.20,
        "trend": "decreasing",
        "last_week_price": 3.50,
        "advice": "wait",
        "reasoning": "Prices are declining, wait for recovery"
    },
    "rice": {
        "current_price": 1.20,
        "trend": "stable",
        "last_week_price": 1.18,
        "advice": "sell",
        "reasoning": "Stable prices with slight increase, good to sell"
    },
    "wheat": {
        "current_price": 0.95,
        "trend": "increasing",
        "last_week_price": 0.90,
        "advice": "sell",
        "reasoning": "Prices trending up, favorable selling conditions"
    },
    "corn": {
        "current_price": 1.45,
        "trend": "stable",
        "last_week_price": 1.43,
        "advice": "wait",
        "reasoning": "Stable prices, monitor for better opportunities"
    }
}

# Agent configuration
REASONER_SYSTEM_PROMPT = """You are an AI coordinator for AgriMitra, an agricultural assistance system. 
Given a user's text input, determine if the query is within AgriMitra's domain (plants/crops, plant diseases, farm practices, produce market/prices). If the query is outside this domain (e.g., history, celebrities, politics, generic facts), mark it as out_of_scope.

Identify and output:
1. User intent: disease diagnosis, market pricing, both, or out_of_scope
2. Which agent(s) should be triggered (empty if out_of_scope)
3. Crop mentioned (if any)

Output your analysis as a JSON object with this exact format:
{
  "intent": ["disease", "market"] or ["disease"] or ["market"] or ["out_of_scope"],
  "crop": "crop_name" or null,
  "agents_to_trigger": ["disease_agent", "price_agent"] or [] if out_of_scope
}

Examples:
- "My tomato plants have yellow spots on leaves" → {"intent": ["disease"], "crop": "tomato", "agents_to_trigger": ["disease_agent"]}
- "What's the current price of rice?" → {"intent": ["market"], "crop": "rice", "agents_to_trigger": ["price_agent"]}
- "My wheat is sick and I want to know the market price" → {"intent": ["disease", "market"], "crop": "wheat", "agents_to_trigger": ["disease_agent", "price_agent"]}
- "Who is Mahatma Gandhi?" → {"intent": ["out_of_scope"], "crop": null, "agents_to_trigger": []}
"""

DISEASE_AGENT_SYSTEM_PROMPT = """You are a plant disease diagnosis expert. 
Given symptoms and crop information, identify the most probable disease and determine if remedy information is needed.

Analyze the symptoms and provide:
1. Probable disease name
2. Confidence level (high/medium/low)
3. Whether remedy tool should be called (true/false)
4. Brief explanation of your diagnosis

Output as JSON:
{
  "disease": "disease_name",
  "confidence": "high/medium/low",
  "needs_remedy": true/false,
  "explanation": "brief explanation"
}"""

COORDINATOR_SYSTEM_PROMPT = """You are a coordinator that synthesizes information from multiple agricultural agents.
Given outputs from disease and price agents, create a comprehensive, actionable response for the farmer.

Combine all available information into a clear, helpful response that addresses the farmer's original query.
Be practical, encouraging, and provide specific actionable steps.

If shops_info is present from the disease agent, include a short section:
- Title: Nearby fertilizer shops
- List up to 5 shops: name, address (if available), approx. coordinates
- If no shops found, state that and suggest widening radius or clarifying location.
"""

# Logging configuration
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

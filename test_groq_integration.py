"""
Test script to verify Groq API integration
"""
import os
from agents import AgriMitraLLM

def test_groq_api():
    """Test Groq API integration"""
    print("🧪 Testing Groq API Integration")
    print("=" * 40)
    
    # Check if API key is available
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("⚠️ No GROQ_API_KEY found in environment variables")
        print("To test with real API, set your Groq API key:")
        print("export GROQ_API_KEY='your-groq-api-key-here'")
        print("\nTesting fallback logic instead...")
        
        # Test fallback logic
        llm = AgriMitraLLM()
        response = llm.chat(
            "You are an AI coordinator. Analyze user input and determine intent.",
            "My tomato plants have yellow spots"
        )
        print(f"✅ Fallback response: {response[:100]}...")
        return
    
    print(f"✅ GROQ_API_KEY found: {api_key[:10]}...")
    
    try:
        # Test real API
        llm = AgriMitraLLM()
        response = llm.chat(
            "You are an AI coordinator. Analyze user input and determine intent.",
            "My tomato plants have yellow spots"
        )
        print(f"✅ Groq API response: {response[:100]}...")
        print("🎉 Groq API integration successful!")
        
    except Exception as e:
        print(f"❌ Groq API test failed: {e}")
        print("This might be due to:")
        print("1. Invalid API key")
        print("2. Network connectivity issues")
        print("3. API rate limits")
        print("4. Model availability")

if __name__ == "__main__":
    test_groq_api()

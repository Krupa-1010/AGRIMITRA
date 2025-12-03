"""
AgriMitra System Test Script
Quick test to verify the system works correctly
"""
import json
import sys
from workflow import AgriMitraWorkflow
from agents import ReasonerNode, DiseaseAgentNode, PriceAgentNode
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_individual_components():
    """Test individual components"""
    print("🧪 Testing Individual Components")
    print("=" * 40)
    
    # Test Reasoner
    print("\n1. Testing Reasoner Node...")
    try:
        reasoner = ReasonerNode()
        result = reasoner.process("My tomato plants have yellow spots")
        print(f"✅ Reasoner works: {result.get('next_nodes', [])}")
    except Exception as e:
        print(f"❌ Reasoner failed: {e}")
    
    # Test Disease Agent
    print("\n2. Testing Disease Agent...")
    try:
        disease_agent = DiseaseAgentNode()
        result = disease_agent.process("Yellow spots on tomato leaves", "tomato")
        print(f"✅ Disease Agent works: {result.get('agent', 'unknown')}")
    except Exception as e:
        print(f"❌ Disease Agent failed: {e}")
    
    # Test Price Agent
    print("\n3. Testing Price Agent...")
    try:
        price_agent = PriceAgentNode()
        result = price_agent.process("tomato")
        print(f"✅ Price Agent works: {result.get('agent', 'unknown')}")
    except Exception as e:
        print(f"❌ Price Agent failed: {e}")

def test_full_workflow():
    """Test the complete workflow"""
    print("\n🔄 Testing Full Workflow")
    print("=" * 40)
    
    test_queries = [
        "My tomato plants have yellow spots on leaves",
        "What's the current price of rice?",
        "My wheat is sick and I want to know the market price"
    ]
    
    workflow = AgriMitraWorkflow()
    
    for i, query in enumerate(test_queries, 1):
        print(f"\nTest {i}: {query}")
        try:
            result = workflow.run(query)
            final_response = result.get('final_response', 'No response')
            print(f"✅ Workflow completed: {final_response[:100]}...")
            
            # Check execution log
            execution_log = result.get('execution_log', [])
            print(f"   Steps executed: {len(execution_log)}")
            
        except Exception as e:
            print(f"❌ Workflow failed: {e}")

def test_remedies_data():
    """Test remedies.json data"""
    print("\n📋 Testing Remedies Data")
    print("=" * 40)
    
    try:
        with open('remedies.json', 'r') as f:
            remedies = json.load(f)
        
        print(f"✅ Loaded {len(remedies)} remedies")
        
        # Test specific remedy
        if remedies:
            first_remedy = remedies[0]
            print(f"   Sample: {first_remedy.get('disease_name', 'Unknown')}")
            print(f"   Treatment: {first_remedy.get('remedy_name', 'Unknown')}")
        
    except Exception as e:
        print(f"❌ Remedies data failed: {e}")

def main():
    """Main test function"""
    print("🌱 AgriMitra System Test")
    print("=" * 50)
    
    # Run all tests
    test_remedies_data()
    test_individual_components()
    test_full_workflow()
    
    print("\n" + "=" * 50)
    print("🎉 System test completed!")
    print("\nIf all tests passed, your AgriMitra system is ready to use:")
    print("• python cli.py - Start the CLI interface")
    print("• python demo.py - Run the demo")
    print("• python setup.py - Verify setup")

if __name__ == "__main__":
    main()

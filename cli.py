"""
AgriMitra Agentic Prototype - CLI Interface
"""
import json
import os
import sys
from typing import Dict, Any
from workflow import AgriMitraWorkflow
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('agrimitra.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class AgriMitraCLI:
    """Command Line Interface for AgriMitra Agentic Prototype"""
    
    def __init__(self):
        self.workflow = AgriMitraWorkflow()
        self.session_log = []
    
    def display_welcome(self):
        """Display welcome message"""
        print("\n" + "="*60)
        print("🌱 AgriMitra Agentic Prototype 🌱")
        print("="*60)
        print("Welcome to your AI-powered agricultural assistant!")
        print("\nI can help you with:")
        print("• Plant disease diagnosis and treatment")
        print("• Market price information and selling advice")
        print("• Combined agricultural guidance")
        print("\nType 'help' for commands, 'quit' to exit")
        print("="*60)
    
    def display_help(self):
        """Display help information"""
        print("\n📋 Available Commands:")
        print("• help - Show this help message")
        print("• quit/exit - Exit the application")
        print("• debug - Show detailed execution logs")
        print("• log - Show session history")
        print("• clear - Clear session history")
        print("\n💬 Example Queries:")
        print("• 'My tomato plants have yellow spots on leaves'")
        print("• 'What's the current price of rice?'")
        print("• 'My wheat is sick and I want to know the market price'")
        print("• 'How to treat powdery mildew on my crops'")
    
    def display_debug_info(self, final_state: Dict[str, Any]):
        """Display detailed debug information"""
        print("\n🔍 DEBUG INFORMATION")
        print("="*50)
        
        # Show execution summary
        summary = self.workflow.get_execution_summary(final_state)
        print(summary)
        
        # Show raw state data
        print("\n📊 RAW STATE DATA:")
        print(json.dumps(final_state, indent=2, default=str))
    
    def display_session_log(self):
        """Display session interaction history"""
        if not self.session_log:
            print("\n📝 No interactions in this session yet.")
            return
        
        print(f"\n📝 Session History ({len(self.session_log)} interactions):")
        print("="*50)
        
        for i, log_entry in enumerate(self.session_log, 1):
            print(f"\n{i}. Query: {log_entry['user_input']}")
            print(f"   Response: {log_entry['final_response'][:100]}...")
            if log_entry.get('error'):
                print(f"   Error: {log_entry['error']}")
    
    def save_session_log(self):
        """Save session log to file"""
        if self.session_log:
            log_filename = f"agrimitra_session_{len(self.session_log)}_interactions.json"
            try:
                with open(log_filename, 'w') as f:
                    json.dump(self.session_log, f, indent=2, default=str)
                print(f"\n💾 Session log saved to: {log_filename}")
            except Exception as e:
                print(f"\n❌ Error saving session log: {e}")
    
    def process_query(self, user_input: str, debug_mode: bool = False) -> str:
        """Process a user query through the workflow"""
        print(f"\n🤖 Processing: {user_input}")
        print("-" * 50)
        
        try:
            # Run the workflow
            final_state = self.workflow.run(user_input)
            
            # Log the interaction
            log_entry = {
                "user_input": user_input,
                "final_response": final_state.get("final_response", "No response"),
                "timestamp": self.workflow._get_timestamp(),
                "execution_log": final_state.get("execution_log", [])
            }
            
            if "error" in final_state:
                log_entry["error"] = final_state["error"]
            
            self.session_log.append(log_entry)
            
            # Display response
            response = final_state.get("final_response", "No response generated")
            print(f"\n🌾 AgriMitra Response:")
            print(response)
            
            # Show debug info if requested
            if debug_mode:
                self.display_debug_info(final_state)
            
            return response
            
        except Exception as e:
            error_msg = f"Error processing query: {e}"
            logger.error(error_msg)
            print(f"\n❌ {error_msg}")
            
            # Log the error
            self.session_log.append({
                "user_input": user_input,
                "error": error_msg,
                "timestamp": self.workflow._get_timestamp()
            })
            
            return error_msg
    
    def run(self):
        """Main CLI loop"""
        self.display_welcome()
        
        while True:
            try:
                # Get user input
                user_input = input("\n🌱 Your query: ").strip()
                
                # Handle commands
                if user_input.lower() in ['quit', 'exit']:
                    print("\n👋 Thank you for using AgriMitra!")
                    self.save_session_log()
                    break
                
                elif user_input.lower() == 'help':
                    self.display_help()
                    continue
                
                elif user_input.lower() == 'debug':
                    if self.session_log:
                        last_interaction = self.session_log[-1]
                        if 'execution_log' in last_interaction:
                            print("\n🔍 Last interaction debug info:")
                            self.display_debug_info({
                                "user_input": last_interaction["user_input"],
                                "final_response": last_interaction["final_response"],
                                "execution_log": last_interaction["execution_log"]
                            })
                        else:
                            print("\n❌ No debug information available for last interaction")
                    else:
                        print("\n❌ No interactions to debug yet")
                    continue
                
                elif user_input.lower() == 'log':
                    self.display_session_log()
                    continue
                
                elif user_input.lower() == 'clear':
                    self.session_log = []
                    print("\n🗑️ Session history cleared")
                    continue
                
                elif not user_input:
                    print("\n⚠️ Please enter a query or command")
                    continue
                
                # Process the query
                self.process_query(user_input)
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye! Session interrupted by user.")
                self.save_session_log()
                break
            
            except EOFError:
                print("\n\n👋 Goodbye! Session ended.")
                self.save_session_log()
                break
            
            except Exception as e:
                logger.error(f"Unexpected error in CLI: {e}")
                print(f"\n❌ Unexpected error: {e}")
                continue

def main():
    """Main entry point"""
    # Check for required files
    required_files = ['remedies.json']
    for file in required_files:
        if not os.path.exists(file):
            print(f"❌ Required file missing: {file}")
            print("Please ensure all required files are present.")
            sys.exit(1)
    
    # Check for Groq API key (optional for demo)
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("⚠️ Warning: GROQ_API_KEY not found in environment variables")
        print("The system will use fallback logic for demonstration purposes")
        print("For full functionality, set your Groq API key:")
        print("export GROQ_API_KEY='your-groq-api-key-here'")
        print("\nContinuing with demo mode...\n")
    
    # Start the CLI
    cli = AgriMitraCLI()
    cli.run()

if __name__ == "__main__":
    main()

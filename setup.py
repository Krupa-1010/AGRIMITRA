"""
AgriMitra Setup Script
Helps users set up the environment and run the system
"""
import os
import sys

def check_requirements():
    """Check if required packages are installed"""
    try:
        import langgraph
        import langchain
        import openai
        print("✅ All required packages are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing required package: {e}")
        print("Please run: pip install -r requirements.txt")
        return False

def check_files():
    """Check if required files exist"""
    required_files = [
        'requirements.txt',
        'config.py',
        'agents.py', 
        'workflow.py',
        'cli.py',
        'remedies.json'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing required files: {', '.join(missing_files)}")
        return False
    else:
        print("✅ All required files are present")
        return True

def check_api_key():
    """Check if Groq API key is configured"""
    api_key = os.getenv("GROQ_API_KEY")
    if api_key:
        print("✅ Groq API key is configured")
        return True
    else:
        print("⚠️ Groq API key not found")
        print("The system will work in demo mode with fallback logic")
        print("For full functionality, set your Groq API key:")
        print("export GROQ_API_KEY='your-groq-api-key-here'")
        return False

def main():
    """Main setup function"""
    print("🌱 AgriMitra Agentic Prototype - Setup Check")
    print("=" * 50)
    
    # Check all requirements
    files_ok = check_files()
    packages_ok = check_requirements()
    api_ok = check_api_key()
    
    print("\n" + "=" * 50)
    
    if files_ok and packages_ok:
        if api_ok:
            print("🎉 Setup complete! You can run the full system.")
            print("\nTo start the CLI interface:")
            print("python cli.py")
            print("\nTo run the demo:")
            print("python demo.py")
        else:
            print("🎉 Setup complete! Running in demo mode.")
            print("\nTo start the CLI interface (demo mode):")
            print("python cli.py")
            print("\nTo run the demo:")
            print("python demo.py")
    else:
        print("❌ Setup incomplete. Please fix the issues above.")
        sys.exit(1)

if __name__ == "__main__":
    main()

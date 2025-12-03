"""
Environment Setup Script for AgriMitra with Groq API
"""
import os

def setup_groq_env():
    """Interactive setup for Groq API key"""
    print("🌱 AgriMitra Groq API Setup")
    print("=" * 40)
    
    # Check if API key already exists
    existing_key = os.getenv("GROQ_API_KEY")
    if existing_key:
        print(f"✅ GROQ_API_KEY is already set: {existing_key[:10]}...")
        choice = input("Do you want to update it? (y/n): ").lower()
        if choice != 'y':
            print("Keeping existing API key.")
            return
    
    print("\nTo use Groq LLMs, you need a Groq API key.")
    print("1. Visit https://console.groq.com/ to get your API key")
    print("2. Sign up/login and create an API key")
    print("3. Copy the key and paste it below")
    print("4. Model: openai/gpt-oss-20b (fast inference)")
    
    api_key = input("\nEnter your Groq API key: ").strip()
    
    if not api_key:
        print("❌ No API key provided. You can set it later using:")
        print("export GROQ_API_KEY='your-api-key-here'")
        return
    
    # Create .env file
    env_content = f"# AgriMitra Environment Configuration\nGROQ_API_KEY={api_key}\n"
    
    try:
        with open('.env', 'w') as f:
            f.write(env_content)
        print("✅ .env file created successfully!")
        print("✅ Groq API key configured!")
        print("✅ Model: openai/gpt-oss-20b")
        
        # Set environment variable for current session
        os.environ["GROQ_API_KEY"] = api_key
        print("✅ API key set for current session")
        
    except Exception as e:
        print(f"❌ Error creating .env file: {e}")
        print("You can manually set the environment variable:")
        print(f"export GROQ_API_KEY='{api_key}'")

def main():
    """Main setup function"""
    setup_groq_env()
    
    print("\n" + "=" * 40)
    print("🎉 Setup complete!")
    print("\nNext steps:")
    print("1. python setup.py - Verify your setup")
    print("2. python test_system.py - Test the system")
    print("3. python cli.py - Start the CLI interface")
    print("4. python demo.py - Run the demo")

if __name__ == "__main__":
    main()

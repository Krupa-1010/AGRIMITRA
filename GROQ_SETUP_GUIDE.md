# 🚀 Groq API Setup Guide for AgriMitra

## Overview
Your AgriMitra Agentic Prototype has been successfully configured to use **Groq's API** with the **openai/gpt-oss-20b** model for fast inference.

## 🔑 Getting Your Groq API Key

1. **Visit Groq Console**: https://console.groq.com/
2. **Sign up/Login**: Create an account or sign in
3. **Create API Key**: Navigate to API Keys section and create a new key
4. **Copy the Key**: Save your API key securely

## ⚡ Quick Setup

### Option 1: Environment Variable (Recommended)
```bash
export GROQ_API_KEY="your-groq-api-key-here"
```

### Option 2: Interactive Setup
```bash
python setup_env.py
```

### Option 3: Manual .env File
Create a `.env` file in the project directory:
```
GROQ_API_KEY=your-groq-api-key-here
```

## 🧪 Testing Your Setup

### Test API Integration
```bash
python test_groq_integration.py
```

### Test Complete System
```bash
python setup.py          # Verify setup
python test_system.py     # Test all components
python cli.py             # Start interactive CLI
```

## 🌟 What You Get with Groq

- **Ultra-fast Inference**: Groq's LPU technology provides extremely fast responses
- **Cost-effective**: Competitive pricing for high-performance inference
- **Reliable**: High uptime and consistent performance
- **OpenAI Compatible**: Seamless integration with existing LangChain code

## 📊 Model Information

- **Model**: `openai/gpt-oss-20b`
- **Type**: 20B parameter model optimized for speed
- **API Endpoint**: `https://api.groq.com/openai/v1`
- **Features**: Fast inference, good reasoning capabilities

## 🔧 Configuration Details

The system is configured in `config.py`:
```python
LLM_MODEL = "openai/gpt-oss-20b"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_BASE = "https://api.groq.com/openai/v1"
```

## 🎯 Example Usage

Once configured, your AgriMitra system will:
1. **Analyze user queries** with lightning-fast reasoning
2. **Route to appropriate agents** based on intent detection
3. **Provide comprehensive agricultural advice** combining disease diagnosis and market information
4. **Deliver responses in seconds** thanks to Groq's speed

## 🆘 Troubleshooting

### Common Issues:
1. **Invalid API Key**: Double-check your key from Groq Console
2. **Rate Limits**: Groq has generous limits, but monitor usage
3. **Network Issues**: Ensure stable internet connection
4. **Model Availability**: The GPT-OSS-20B model should be available

### Fallback Mode:
If no API key is provided, the system automatically falls back to demo mode with keyword-based logic, ensuring it always works for testing and demonstration purposes.

## 🎉 Ready to Use!

Your AgriMitra system is now configured for high-performance agricultural AI assistance using Groq's fast inference engine. The system will provide rapid, intelligent responses to farmer queries about disease diagnosis and market information.

**Next Steps:**
1. Set your Groq API key
2. Run `python cli.py` to start using the system
3. Ask agricultural questions and experience the speed of Groq-powered AI!

---

*Powered by Groq's ultra-fast inference technology* ⚡

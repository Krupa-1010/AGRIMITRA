# 🌱 AgriMitra Agentic Prototype

A LangGraph-based Python project demonstrating **agentic AI behavior** for agricultural assistance. This prototype showcases multi-agent coordination, tool invocation, and synthesized decision-making for farmers.

## 🎯 Features

### Core Capabilities
- **Plant Disease Diagnosis**: AI-powered disease identification with treatment recommendations
- **Market Price Analysis**: Real-time crop pricing with selling advice
- **Multi-Agent Coordination**: Intelligent routing and synthesis of information
- **Tool Integration**: Dynamic tool invocation based on context
- **Comprehensive Logging**: Full execution tracking for debugging and learning
- **Fast Inference**: Powered by Groq's high-performance inference engine

### Agentic AI Behavior
1. **Reasoning** → Analyzes user intent and determines which agents to activate
2. **Tool Invocation** → Agents decide when to call specialized tools
3. **Multi-Agent Coordination** → Multiple agents work together seamlessly
4. **Synthesized Output** → Coordinator merges all information into actionable advice

## 🏗️ Architecture

### Nodes and Agents

#### 1. **Reasoner Node (LLMNode)**
- **Purpose**: Analyzes user input and determines routing
- **Model**: Configurable LLM (default: GPT-3.5-turbo)
- **Output**: JSON with intent, crop, and agents to trigger
- **Fallback**: Keyword-based reasoning if JSON parsing fails

#### 2. **DiseaseAgent Node**
- **Purpose**: Diagnoses plant diseases and provides treatments
- **LLM Integration**: Uses LLM for disease identification
- **Tool**: RemedyTool reads from `remedies.json`
- **Decision Making**: Determines if remedy information is needed

#### 3. **PriceAgent Node**
- **Purpose**: Provides market price information and selling advice
- **Data Source**: Mock price data (easily replaceable with real APIs)
- **Tool**: PriceTool returns current prices, trends, and recommendations

#### 4. **Coordinator Node**
- **Purpose**: Synthesizes outputs from all active agents
- **LLM Integration**: Creates human-readable, actionable responses
- **Output**: Coherent, comprehensive agricultural guidance

### Workflow Flow
```
User Input → Reasoner → [DiseaseAgent | PriceAgent | Both] → Coordinator → Final Response
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Groq API key (optional for demo mode)

### Installation

1. **Clone or download the project files**
```bash
# Ensure you have all these files:
# - requirements.txt
# - config.py
# - agents.py
# - workflow.py
# - cli.py
# - remedies.json
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up environment (optional)**
```bash
# For full functionality, set your Groq API key
export GROQ_API_KEY="your-groq-api-key-here"

# Or use the interactive setup
python setup_env.py
```

4. **Run the application**
```bash
python cli.py
```

## 💻 Usage

### CLI Commands
- `help` - Show available commands and examples
- `debug` - Show detailed execution logs for last interaction
- `log` - Display session interaction history
- `clear` - Clear session history
- `quit/exit` - Exit the application

### Example Queries

#### Disease Diagnosis
```
My tomato plants have yellow spots on leaves
```
**Expected Flow**: Reasoner → DiseaseAgent → RemedyTool → Coordinator

#### Market Price
```
What's the current price of rice?
```
**Expected Flow**: Reasoner → PriceAgent → Coordinator

#### Combined Query
```
My wheat is sick and I want to know the market price
```
**Expected Flow**: Reasoner → DiseaseAgent + PriceAgent → Coordinator

## 🔧 Configuration

### Model Configuration
Edit `config.py` to change:
- LLM model (openai/gpt-oss-20b, llama-3.1-70b-versatile, etc.)
- API keys and settings
- System prompts for each agent

### Data Sources
- **Disease Remedies**: Edit `remedies.json` to add/modify treatments
- **Price Data**: Modify `MOCK_PRICE_DATA` in `config.py`
- **Logging**: Configure log levels and outputs

### Adding New Agents
The modular design makes it easy to add new agents:

1. **Create new agent class** in `agents.py`
2. **Add node to workflow** in `workflow.py`
3. **Update routing logic** in the workflow
4. **Add to coordinator** synthesis

Example new agents:
- **WeatherAgent**: Weather forecasts and farming advice
- **BuyerAgent**: Connect with local buyers and markets
- **SoilAgent**: Soil analysis and fertilizer recommendations

## 📊 Output Examples

### Disease Diagnosis Output
```json
{
  "disease_diagnosis": {
    "disease": "Late Blight",
    "confidence": "high",
    "needs_remedy": true,
    "explanation": "Yellow spots with dark edges on tomato leaves indicate late blight"
  },
  "remedy_info": {
    "disease_name": "Late Blight",
    "remedy_name": "Copper Fungicide Treatment",
    "steps": ["Remove infected parts", "Apply copper fungicide", ...],
    "duration_days": 21
  }
}
```

### Price Information Output
```json
{
  "price_info": {
    "current_price": 2.50,
    "trend": "increasing",
    "advice": "sell",
    "reasoning": "Prices are rising, good time to sell"
  }
}
```

### Final Coordinated Response
```
Based on your description of yellow spots on tomato leaves, this appears to be Late Blight, 
a serious fungal disease. Here's your action plan:

DISEASE TREATMENT:
1. Remove and destroy infected plant parts immediately
2. Apply copper-based fungicide every 7-10 days for 3 weeks
3. Improve air circulation around plants
4. Avoid overhead watering

MARKET ADVICE:
Current tomato price is $2.50/kg (trending upward). This is a good time to sell 
any healthy produce you have.

Start treatment immediately to save your crop and maximize your market returns!
```

## 🔍 Debugging and Monitoring

### Execution Logging
- **Console Logs**: Real-time execution tracking
- **File Logs**: Saved to `agrimitra.log`
- **Session Logs**: JSON files with interaction history
- **Debug Mode**: Detailed execution traces

### Debug Information
Use the `debug` command to see:
- Step-by-step execution flow
- Agent inputs and outputs
- Tool invocations
- Routing decisions
- Error details

## 🛠️ Development

### Project Structure
```
agrimitra/
├── requirements.txt      # Dependencies
├── config.py            # Configuration and prompts
├── agents.py            # Agent implementations and tools
├── workflow.py          # LangGraph workflow definition
├── cli.py              # Command-line interface
├── remedies.json       # Disease treatment database
├── README.md           # This file
└── demo.py            # Demo script (optional)
```

### Key Design Principles
1. **Modularity**: Easy to add new agents and tools
2. **Fallback Logic**: Graceful degradation when LLM fails
3. **Comprehensive Logging**: Full traceability for learning
4. **User-Friendly**: Clear CLI interface and helpful error messages
5. **Extensible**: Simple to integrate with real APIs and data sources

## 🎯 Future Enhancements

### Planned Features
- **Real API Integration**: Replace mock data with live APIs
- **Multi-Language Support**: Local language prompts and responses
- **Image Analysis**: Photo-based disease diagnosis
- **Weather Integration**: Real-time weather data and farming advice
- **Local Database**: Persistent storage for user interactions
- **Web Interface**: Browser-based UI for broader accessibility

### Integration Opportunities
- **Agricultural APIs**: Weather, market prices, disease databases
- **IoT Sensors**: Soil moisture, temperature monitoring
- **Satellite Data**: Crop monitoring and yield prediction
- **Local Experts**: Connect with agricultural extension services

## 🤝 Contributing

This prototype is designed to be easily extensible. Key areas for contribution:

1. **New Agents**: Weather, soil, pest management agents
2. **Tool Development**: Integration with real agricultural APIs
3. **UI/UX**: Web interface, mobile app, voice interface
4. **Data Sources**: More comprehensive disease and remedy databases
5. **Localization**: Multi-language support and regional customization

## 📄 License

This project is a prototype for educational and demonstration purposes. Feel free to use, modify, and extend for your agricultural AI applications.

## 🙏 Acknowledgments

- Built with [LangGraph](https://github.com/langchain-ai/langgraph) for agentic AI workflows
- Powered by [LangChain](https://github.com/langchain-ai/langchain) for LLM integration
- Designed for agricultural AI applications and farmer assistance

---

**🌱 AgriMitra Agentic Prototype** - Demonstrating the future of AI-powered agricultural assistance through intelligent multi-agent coordination and decision-making.

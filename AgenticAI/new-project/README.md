# TechTrend Innovations Customer Support Agent

## Overview
AI-powered customer support agent using LangGraph, OpenAI, and Streamlit. Features intelligent query classification, escalation handling, and persistent user memory.

## Project Structure
```
new-project/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── __init__.py
│   └── main.py
├── tests/
│   ├── __init__.py
│   └── test_main.py
├── docs/
├── data/
└── config/
```

## Installation
1. Create a virtual environment:
   ```bash
   python -m venv venv
   ```

2. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - Unix/macOS: `source venv/bin/activate`

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
Run the customer support agent:
```bash
# Using Streamlit directly
streamlit run src/customer_support_agent.py

# Or using the main entry point
python src/main.py
```

## Features
- 🤖 AI-powered customer support using OpenAI GPT-4
- 📊 Intelligent query classification and routing
- 🔄 LangGraph workflow for complex conversation handling  
- 💾 Persistent user profiles and conversation history
- 🚨 Automatic escalation detection for complex issues
- 🎨 Modern Streamlit UI with dark theme
- 📈 Session statistics and analytics
- 🔒 Secure API key management

## Configuration
1. Get an OpenAI API key from [OpenAI Platform](https://platform.openai.com/api-keys)
2. Launch the application and enter your API key in the sidebar
3. Optionally update user profile information for personalized support

## Testing
Run tests:
```bash
python -m pytest tests/
```
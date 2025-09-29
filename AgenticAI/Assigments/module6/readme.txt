# Customer Support Triage Agent

An AI-powered customer support system that automatically analyzes, categorizes, and generates responses for customer support tickets using advanced NLP and machine learning techniques.

## 🎯 Features

### Core Capabilities
- **Automated Ticket Triage**: Classify customer intents and sentiment analysis
- **Intelligent Response Generation**: Context-aware response suggestions
- **Multi-File Processing**: Support for CSV, PDF, and TXT files
- **Semantic Search**: Vector-based knowledge base search
- **Real-time Analytics**: Dashboard with insights and metrics
- **Batch Processing**: Handle multiple tickets efficiently

### AI Components
- **Sentiment Analysis**: Detect customer emotions, urgency, and frustration levels
- **Intent Classification**: Categorize tickets into refund, delivery, product issues, etc.
- **Response Generation**: Generate empathetic, solution-focused responses
- **Vector Search**: Semantic similarity search across knowledge base

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone or create the project directory
mkdir customer-support-agent
cd customer-support-agent

# Copy all the files from the artifacts into this directory
# (All .py files, requirements.txt, .env.example, etc.)

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

**Note**: The system is designed to work with or without optional dependencies. If packages like `openai`, `pinecone-client`, or `sentence-transformers` are not installed, the system will use fallback methods.

### 3. Configuration (Optional)

For full AI functionality, create a `.env` file:

```bash
cp .env.example .env
```

Edit `.env` and add your API keys:
```env
OPENAI_API_KEY=your_openai_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_ENVIRONMENT=us-west1-gcp-free
```

### 4. Run the Application

#### Option 1: Web Interface (Recommended)
```bash
streamlit run streamlit_app.py
```
Open your browser to `http://localhost:8501`

#### Option 2: Command Line Interface
```bash
python main.py --mode interactive
```

#### Option 3: Process Sample Data
```bash
python main.py --mode batch --file sample_data.csv
```

## 📁 Project Structure

```
customer-support-agent/
├── main.py                     # Main entry point
├── streamlit_app.py            # Web interface
├── config.py                   # Configuration settings
├── support_agent.py            # Main agent class
├── sentiment_analyzer.py       # Sentiment analysis tool
├── intent_classifier.py       # Intent classification tool
├── response_generator.py       # Response generation tool
├── csv_processor.py           # CSV file processor
├── pdf_processor.py           # PDF file processor (basic)
├── text_processor.py          # Text file processor (basic)
├── chunking_strategies.py     # Document chunking
├── vector_store.py            # Vector database integration
├── requirements.txt           # Dependencies
├── .env.example              # Environment template
├── README.md                 # This file
└── data/                     # Data directory (auto-created)
    ├── uploads/              # Uploaded files
    └── processed/            # Processed data
```

## 🔧 Modes of Operation

### 1. Full AI Mode (Recommended)
- Requires OpenAI API key for advanced response generation
- Requires Pinecone API key for vector search
- All features available

### 2. Basic AI Mode
- Works with OpenAI API key only
- Uses in-memory vector storage
- Most features available

### 3. Demo Mode
- Works without any API keys
- Uses rule-based analysis and template responses
- Great for testing and learning

## 💻 Usage Examples

### Web Interface
1. Launch: `streamlit run streamlit_app.py`
2. Navigate through different pages:
   - **Dashboard**: View analytics and metrics
   - **Process Tickets**: Analyze individual or batch tickets
   - **File Management**: Upload knowledge base files
   - **Knowledge Search**: Search through uploaded content
   - **Settings**: Configure system and view status

### Command Line
```bash
# Interactive mode
python main.py --mode interactive

# Validate setup
python main.py --validate

# Process a file
python main.py --mode batch --file path/to/tickets.csv

# Launch web interface
python main.py --mode web
```

### Python API
```python
from support_agent import SupportTriageAgent

# Initialize agent
agent = SupportTriageAgent()

# Process a single ticket
result = agent.process_support_ticket(
    "I need a refund for my broken laptop order #12345",
    ticket_id="TICKET-001"
)

# Access results
sentiment = result['sentiment_analysis']
intent = result['intent_analysis']
response = result['response_data']['response_text']

print(f"Intent: {intent['primary_intent']}")
print(f"Sentiment: {sentiment['sentiment_category']}")
print(f"Response: {response}")
```

## 📊 File Formats Supported

### CSV Files (Support Tickets)
Required columns:
- `text`, `message`, `description`, or `content`: Customer message
- `timestamp` (optional): When ticket was created
- `id` or `ticket_id` (optional): Unique identifier

### PDF Files (Basic Support)
- Simple text extraction
- Section-based processing
- Metadata preservation

### TXT Files (Chat Logs)
- Multi-line conversations
- Timestamped logs
- Speaker identification

## 🛠️ Customization

### Adding New Intents
Edit `intent_classifier.py`:
```python
self.intent_patterns['new_intent'] = {
    'keywords': ['keyword1', 'keyword2'],
    'patterns': [r'\b(pattern1|pattern2)\b'],
    'department': 'new_department'
}
```

### Custom Response Templates
Edit `response_generator.py`:
```python
self.response_templates['new_intent'] = {
    'opening': "Thank you for...",
    'empathy': "I understand...",
    'action': "I'll help you...",
    'closing': "Is there anything else..."
}
```

### Configuration Options
Edit `config.py` or use environment variables:
```python
# Model settings
DEFAULT_LLM_MODEL = "gpt-4"
TEMPERATURE = 0.3
MAX_TOKENS = 2000

# Processing settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K_RESULTS = 5
```

## 🔍 API Keys Setup

### OpenAI API
1. Visit [OpenAI Platform](https://platform.openai.com/)
2. Create an account and add billing information
3. Generate an API key
4. Add to `.env` file: `OPENAI_API_KEY=sk-...`

### Pinecone (Vector Database)
1. Visit [Pinecone](https://www.pinecone.io/)
2. Create a free account
3. Create a new index (dimension: 384)
4. Get API key and environment
5. Add to `.env` file:
   ```
   PINECONE_API_KEY=your-key
   PINECONE_ENVIRONMENT=us-west1-gcp-free
   ```

## 📚 Dependencies Explained

### Required (Core Functionality)
- `streamlit`: Web interface
- `pandas`: Data processing
- `numpy`: Numerical operations
- `plotly`: Visualizations

### Optional (Enhanced Features)
- `openai`: Advanced response generation
- `sentence-transformers`: Semantic embeddings
- `pinecone-client`: Vector database
- `textblob`: Enhanced sentiment analysis
- `pdfplumber`: PDF processing

### Fallback Behavior
- **No OpenAI**: Uses template-based responses
- **No Pinecone**: Uses in-memory vector storage
- **No sentence-transformers**: Uses random embeddings
- **No textblob**: Uses pattern-based sentiment analysis

## 🎯 Use Cases

### Customer Support Teams
- Automatically categorize incoming tickets
- Generate consistent response drafts
- Prioritize urgent issues
- Route tickets to appropriate departments

### Training and Learning
- Learn about LLM engineering patterns
- Understand AI agent architecture
- Practice with real-world NLP problems
- Experiment with different AI models

### Development and Integration
- Use as a foundation for custom support systems
- Integrate components into existing workflows
- Extend functionality for specific business needs
- Test different AI approaches

## 🔧 Troubleshooting

### Common Issues

**1. Import Errors**
```
ModuleNotFoundError: No module named 'openai'
```
- Solution: Install dependencies or run in demo mode
- Check: `pip install -r requirements.txt`

**2. API Key Errors**
```
Authentication error with OpenAI/Pinecone
```
- Solution: Check API keys in `.env` file
- Ensure keys are valid and have proper billing setup

**3. File Processing Errors**
```
Error processing CSV/PDF files
```
- Solution: Check file format and encoding
- Use UTF-8 encoding for text files
- Ensure CSV has required columns

**4. Vector Database Issues**
```
Connection timeout to Pinecone
```
- Solution: Check internet connection and API credentials
- System will fall back to in-memory storage

### Performance Tips

1. **Large Files**: Process in smaller batches
2. **Memory Issues**: Reduce chunk size in config
3. **Slow Responses**: Use lighter models (gpt-3.5-turbo)
4. **Rate Limits**: Add delays between API calls

## 🚦 System Status Indicators

### Web Interface
- ✅ Green: Component working properly
- ⚠️ Yellow: Component working with limitations
- ❌ Red: Component not functional
- 🎯 Demo: Using fallback/mock functionality

### Command Line
- Agent status shows component health
- Validation mode checks configuration
- Error messages provide specific guidance

## 📈 Analytics and Insights

### Dashboard Metrics
- Total tickets processed
- File processing statistics
- Intent and sentiment distributions
- Response generation metrics

### Batch Analysis
- Processing trends over time
- Common issue categories
- Customer satisfaction estimates
- Escalation requirements

## 🤝 Contributing

### Development Setup
1. Fork or clone the repository
2. Set up virtual environment
3. Install development dependencies
4. Make changes and test thoroughly
5. Update documentation as needed

### Code Style
- Follow PEP 8 guidelines
- Use type hints where applicable
- Add docstrings to functions
- Include error handling
- Write clear variable names

### Testing
- Test with sample data first
- Verify both API and demo modes work
- Check error handling for edge cases
- Validate UI responsiveness

## 📝 License and Usage

This project is designed for:
- Educational purposes
- Development and learning
- Commercial use (with proper API licensing)
- Research and experimentation

**Important**: Ensure compliance with:
- OpenAI API Terms of Service
- Pinecone Terms of Service
- Data privacy regulations (GDPR, CCPA, etc.)
- Your organization's security policies

## 🔮 Future Enhancements

### Planned Features
- Multi-language support
- Advanced emotion detection
- Integration with popular ticketing systems
- Real-time customer chat integration
- Custom model training interface

### Integration Possibilities
- Slack/Teams notifications
- Email automation systems
- CRM integration (Salesforce, HubSpot)
- Webhook support for external systems
- REST API endpoints

## 📞 Support and Resources

### Getting Help
1. Check this README for common issues
2. Review error messages and logs
3. Test with sample data first
4. Verify API key configuration

### Learning Resources
- [OpenAI Documentation](https://platform.openai.com/docs)
- [Pinecone Documentation](https://docs.pinecone.io/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)

### Best Practices
- Start with demo mode to learn the interface
- Use sample tickets to test functionality
- Configure API keys for full capabilities
- Monitor usage and costs
- Keep data secure and compliant

---

**Happy coding!** 🎯 This system provides a solid foundation for learning LLM engineering and building production-ready customer support automation.
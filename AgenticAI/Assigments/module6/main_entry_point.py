#!/usr/bin/env python3
"""
Main entry point for Customer Support Triage Agent
"""
import os
import sys
import argparse
import logging
from datetime import datetime

# Import all components
from support_agent import SupportTriageAgent
from config import settings

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('agent.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )

def validate_setup():
    """Validate that all required components are available"""
    validation = settings.validate_config()
    
    if not validation['openai_key']:
        print("WARNING: OpenAI API key not configured. Set OPENAI_API_KEY in .env file")
    
    if not validation['pinecone_key']:
        print("WARNING: Pinecone API key not configured. Set PINECONE_API_KEY in .env file")
    
    if not validation['directories_created']:
        print("ERROR: Could not create required directories")
        return False
    
    return True

def run_interactive_demo():
    """Run interactive demo mode"""
    print("\n🎯 Customer Support Triage Agent - Interactive Demo")
    print("=" * 50)
    
    # Initialize agent
    print("Initializing agent...")
    try:
        agent = SupportTriageAgent()
        print("✅ Agent initialized successfully!")
    except Exception as e:
        print(f"❌ Error initializing agent: {e}")
        return
    
    # Demo loop
    while True:
        print("\nChoose an option:")
        print("1. Process a support ticket")
        print("2. Upload a file")
        print("3. Search knowledge base")
        print("4. View agent status")
        print("5. Run sample tickets")
        print("6. Exit")
        
        choice = input("\nEnter choice (1-6): ").strip()
        
        if choice == '1':
            ticket_text = input("\nEnter customer message: ")
            if ticket_text.strip():
                print("Processing ticket...")
                result = agent.process_support_ticket(ticket_text)
                print_ticket_analysis(result)
        
        elif choice == '2':
            file_path = input("\nEnter file path: ")
            if os.path.exists(file_path):
                print("Processing file...")
                result = agent.upload_and_process_file(file_path)
                if result['status'] == 'success':
                    print(f"✅ File processed successfully!")
                    print(f"   Created {result['chunks_created']} chunks")
                else:
                    print(f"❌ Error: {result.get('error', 'Unknown error')}")
            else:
                print("❌ File not found")
        
        elif choice == '3':
            query = input("\nEnter search query: ")
            if query.strip():
                print("Searching...")
                results = agent.search_knowledge_base(query)
                print(f"Found {results['results_count']} results")
                for i, result in enumerate(results['results'][:3], 1):
                    print(f"\n{i}. {result['text'][:200]}...")
        
        elif choice == '4':
            status = agent.get_agent_status()
            print(f"\nAgent Status: {status['status']}")
            print(f"Tickets Processed: {status['session_stats']['tickets_processed']}")
            print(f"Files Uploaded: {status['session_stats']['files_uploaded']}")
            print(f"KB Files: {status['knowledge_base_files']}")
        
        elif choice == '5':
            run_sample_tickets(agent)
        
        elif choice == '6':
            print("Goodbye!")
            break
        
        else:
            print("Invalid choice. Please try again.")

def run_sample_tickets(agent):
    """Run sample tickets for demonstration"""
    sample_tickets = [
        "I ordered a laptop 2 weeks ago and it still hasn't arrived. This is unacceptable! I need it for work urgently. Order #12345",
        "Hi, I received my order but the item is damaged. The screen is cracked. Can you help me get a replacement? Thanks!",
        "I was charged twice for the same order. Please refund the duplicate charge of $299.99.",
        "Great service! Just wanted to say thank you. The delivery was fast and the product is exactly what I expected."
    ]
    
    print("\n🎯 Processing sample tickets...")
    results = agent.batch_process_tickets([{'text': ticket, 'id': f'SAMPLE-{i+1:03d}'} for i, ticket in enumerate(sample_tickets)])
    
    print(f"\n✅ Processed {len(results)} sample tickets")
    
    # Show summary
    intents = [r.get('triage_summary', {}).get('primary_intent', 'Unknown') for r in results]
    sentiments = [r.get('triage_summary', {}).get('customer_sentiment', 'Unknown') for r in results]
    
    print("\nIntent Distribution:")
    for intent in set(intents):
        count = intents.count(intent)
        print(f"  {intent}: {count}")
    
    print("\nSentiment Distribution:")
    for sentiment in set(sentiments):
        count = sentiments.count(sentiment)
        print(f"  {sentiment}: {count}")

def print_ticket_analysis(result):
    """Print formatted ticket analysis"""
    print(f"\n📊 Analysis for Ticket: {result['ticket_id']}")
    print("-" * 40)
    
    triage = result.get('triage_summary', {})
    
    print(f"Intent: {triage.get('primary_intent', 'Unknown')}")
    print(f"Sentiment: {triage.get('customer_sentiment', 'Unknown')}")
    print(f"Mood: {triage.get('customer_mood', 'Unknown')}")
    print(f"Urgency: {triage.get('urgency_level', 'Unknown')}")
    print(f"Priority: {triage.get('priority_score', 'Unknown')}")
    print(f"Department: {triage.get('recommended_department', 'Unknown')}")
    
    response_data = result.get('response_data', {})
    if 'response_text' in response_data:
        print(f"\nSuggested Response:")
        print(response_data['response_text'])

def run_web_interface():
    """Launch the Streamlit web interface"""
    print("🌐 Launching web interface...")
    os.system("streamlit run streamlit_app.py")

def run_batch_processing(file_path):
    """Run batch processing on a file"""
    print(f"📄 Processing file: {file_path}")
    
    if not os.path.exists(file_path):
        print("❌ File not found")
        return
    
    # Initialize agent
    try:
        agent = SupportTriageAgent()
        print("✅ Agent initialized")
    except Exception as e:
        print(f"❌ Error initializing agent: {e}")
        return
    
    # Process file
    try:
        result = agent.upload_and_process_file(file_path)
        if result['status'] == 'success':
            print(f"✅ File processed successfully!")
            print(f"   File Type: {result['file_type']}")
            print(f"   Chunks Created: {result['chunks_created']}")
            print(f"   Vectors Stored: {result['vector_storage'].get('successful_upserts', 0)}")
        else:
            print(f"❌ Error: {result.get('error', 'Unknown error')}")
    except Exception as e:
        print(f"❌ Processing error: {e}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Customer Support Triage Agent")
    parser.add_argument('--mode', choices=['interactive', 'web', 'batch'], default='interactive',
                       help='Run mode: interactive demo, web interface, or batch processing')
    parser.add_argument('--file', help='File to process (for batch mode)')
    parser.add_argument('--validate', action='store_true', help='Validate setup only')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    print("🎯 Customer Support Triage Agent")
    print(f"Started at: {datetime.now()}")
    
    # Validate setup
    if not validate_setup():
        print("❌ Setup validation failed")
        sys.exit(1)
    
    if args.validate:
        print("✅ Setup validation passed")
        return
    
    # Run based on mode
    try:
        if args.mode == 'interactive':
            run_interactive_demo()
        elif args.mode == 'web':
            run_web_interface()
        elif args.mode == 'batch':
            if not args.file:
                print("❌ --file argument required for batch mode")
                sys.exit(1)
            run_batch_processing(args.file)
    except KeyboardInterrupt:
        print("\n\nExiting...")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    main()
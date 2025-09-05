"""
Main application entry point
Launch the Customer Support Agent
"""

import os
import sys
from pathlib import Path

# Add the src directory to path for imports
sys.path.append(str(Path(__file__).parent))

from dotenv import load_dotenv
from customer_support_agent import main as run_support_agent


def main():
    """Main function - runs the customer support agent"""
    load_dotenv()
    
    print("Starting TechTrend Innovations Customer Support Agent...")
    print("Open your browser to view the Streamlit interface.")
    
    # Run the Streamlit app
    run_support_agent()


if __name__ == "__main__":
    main()
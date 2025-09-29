#!/usr/bin/env python3
"""
Launch script for the Customer Support Triage Agent
Provides multiple ways to start the application
"""

import sys
import subprocess
import argparse
from pathlib import Path

def launch_streamlit():
    """Launch the Streamlit web interface"""
    print("🌐 Launching Customer Support Triage Agent Web Interface...")
    print("The application will open in your browser at http://localhost:8501")
    print("Press Ctrl+C to stop the application")
    print("Note: If you see any import warnings, they can be safely ignored.")
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "streamlit_app.py",
            "--server.headless", "false",
            "--server.port", "8501",
            "--server.runOnSave", "true"
        ])
    except KeyboardInterrupt:
        print("\n👋 Application stopped.")
    except Exception as e:
        print(f"\n❌ Error launching Streamlit: {e}")
        print("Try running: streamlit run streamlit_app.py")

def launch_cli():
    """Launch the command-line interface"""
    print("💻 Launching Customer Support Triage Agent CLI...")
    
    try:
        subprocess.run([sys.executable, "main_entry_point.py", "--mode", "interactive"])
    except KeyboardInterrupt:
        print("\n👋 Application stopped.")

def run_tests():
    """Run the comprehensive test suite"""
    print("🧪 Running comprehensive tests...")
    
    try:
        result = subprocess.run([sys.executable, "comprehensive_test.py"])
        if result.returncode == 0:
            print("\n✅ All tests passed!")
        else:
            print("\n❌ Some tests failed. Check the output above.")
    except Exception as e:
        print(f"\n❌ Error running tests: {e}")

def main():
    parser = argparse.ArgumentParser(description="Customer Support Triage Agent Launcher")
    parser.add_argument(
        "mode", 
        nargs="?", 
        choices=["web", "cli", "test"], 
        default="web",
        help="Launch mode: web interface (default), command-line interface, or run tests"
    )
    
    args = parser.parse_args()
    
    print("🎯 Customer Support Triage Agent")
    print("=" * 50)
    
    if args.mode == "web":
        launch_streamlit()
    elif args.mode == "cli":
        launch_cli()
    elif args.mode == "test":
        run_tests()

if __name__ == "__main__":
    main()
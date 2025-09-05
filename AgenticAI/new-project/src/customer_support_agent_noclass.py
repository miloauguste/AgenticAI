# Import streamlit - a Python library for creating web applications
import streamlit as st
# Import os module for operating system interface (environment variables)
import os
# Import type hints for better code documentation and IDE support
from typing import List, Dict, Any, Optional
# Import datetime utilities for handling timestamps
from datetime import datetime, timezone
# Import json for working with JSON data format
import json
# Import uuid for generating unique identifiers
import uuid
# Import LangGraph components for building AI agent workflows
from langgraph.graph import StateGraph, END
# Import memory saver for persisting conversation state
from langgraph.checkpoint.memory import MemorySaver
# Import prebuilt tool node for LangGraph
from langgraph.prebuilt import ToolNode
# Import message types for chat conversations
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
# Import OpenAI chat model integration
from langchain_openai import ChatOpenAI
# Import tool decorator for creating LangChain tools
from langchain_core.tools import tool
# Import SQLite database for data storage
import sqlite3
# Import Path for file system operations
from pathlib import Path
# Import tempfile for creating temporary files
import tempfile
# Import base64 for encoding binary data as text
import base64
# Import PIL (Python Imaging Library) for image processing
from PIL import Image
# Import io for input/output operations with bytes
import io

# Configure Streamlit page settings (this sets up how the web page looks)
st.set_page_config(
    page_title="TechTrend Innovations - Customer Support",  # Title shown in browser tab
    page_icon="🤖",  # Icon shown in browser tab (robot emoji)
    layout="wide",  # Use full width of browser window
    initial_sidebar_state="expanded"  # Show sidebar when page loads
)

# Custom CSS styles to make the web interface look better
# st.markdown with unsafe_allow_html=True lets us add custom HTML/CSS
st.markdown("""
<style>
    /* Style for the main header at the top of the page */
    .main-header {
        background: linear-gradient(90deg, #1e3c72, #2a5298);  /* Blue gradient background */
        padding: 1rem;  /* Add space inside the header */
        border-radius: 10px;  /* Rounded corners */
        color: white;  /* White text */
        text-align: center;  /* Center the text */
        margin-bottom: 2rem;  /* Space below the header */
    }
    /* Style for chat message containers */
    .chat-container {
        background-color: #1a1a1a;  /* Dark gray background */
        padding: 1rem;  /* Space inside the container */
        border-radius: 10px;  /* Rounded corners */
        border-left: 4px solid #2a5298;  /* Blue left border */
        color: white;  /* White text */
    }
    /* Style for messages from the user */
    .user-message {
        background-color: #2d2d2d;  /* Darker gray background */
        padding: 0.8rem;  /* Space inside the message box */
        border-radius: 10px;  /* Rounded corners */
        margin: 0.5rem 0;  /* Space above and below */
        border-left: 4px solid #1976d2;  /* Blue left border */
        color: white;  /* White text */
    }
    /* Style for messages from the AI assistant */
    .ai-message {
        background-color: #1e1e1e;  /* Very dark gray background */
        padding: 0.8rem;  /* Space inside the message box */
        border-radius: 10px;  /* Rounded corners */
        margin: 0.5rem 0;  /* Space above and below */
        border-left: 4px solid #388e3c;  /* Green left border */
        color: white;  /* White text */
    }
    /* Style for escalation notices (when human support is needed) */
    .escalation-notice {
        background-color: #fff3e0;  /* Light orange background */
        padding: 1rem;  /* Space inside the notice box */
        border-radius: 10px;  /* Rounded corners */
        border-left: 4px solid #f57c00;  /* Orange left border */
        margin: 1rem 0;  /* Space above and below */
    }
</style>
""", unsafe_allow_html=True)  # unsafe_allow_html=True allows custom HTML/CSS

# Data Structure Functions - Instead of classes, we'll use dictionaries and functions
# These functions create and manage data structures without using classes

# Function to create a customer's profile information as a dictionary
def create_user_profile(user_id: str, name: Optional[str] = None, email: Optional[str] = None, 
                       subscription_type: Optional[str] = None, created_at: Optional[str] = None) -> Dict[str, Any]:
    """Create a user profile dictionary"""
    # If no creation time provided, use current time
    if created_at is None:
        created_at = datetime.now(timezone.utc).isoformat()
    
    # Return a dictionary with all user information
    return {
        "user_id": user_id,  # Required: unique identifier for each user
        "name": name,  # Optional: user's name (can be None/empty)
        "email": email,  # Optional: user's email address
        "subscription_type": subscription_type,  # Optional: Basic/Premium/Enterprise
        "created_at": created_at  # When the profile was created
    }

# Function to create a support ticket resolution as a dictionary
def create_query_resolution(query_id: str, query: str, resolution: str, 
                           resolution_type: str, timestamp: Optional[str] = None,
                           satisfaction_score: Optional[int] = None) -> Dict[str, Any]:
    """Create a query resolution dictionary"""
    # If no timestamp provided, use current time
    if timestamp is None:
        timestamp = datetime.now(timezone.utc).isoformat()
    
    # Return a dictionary with all resolution information
    return {
        "query_id": query_id,  # Required: unique identifier for this support ticket
        "query": query,  # Required: the customer's original question/problem
        "resolution": resolution,  # Required: how the problem was solved
        "timestamp": timestamp,  # Required: when this resolution was created
        "resolution_type": resolution_type,  # Required: "automated", "escalated", "resolved"
        "satisfaction_score": satisfaction_score  # Optional: customer rating (1-5)
    }

# Define the structure of data that flows through our AI agent
# This is just a type hint - we'll use regular dictionaries
# AgentState represents: messages, user_id, thread_id, user_profile, query_history, 
#                       current_query_type, escalation_required, session_metadata
AgentState = Dict[str, Any]

# Database connection path - shared across all database functions
DB_PATH = "customer_support.db"

# Database initialization function (replaces CustomerSupportMemory.__init__ and init_database)
def init_database(db_path: str = DB_PATH) -> None:
    """Initialize SQLite database for long-term memory storage"""
    # Connect to the SQLite database file (creates file if it doesn't exist)
    conn = sqlite3.connect(db_path)
    # Create a cursor object to execute SQL commands
    cursor = conn.cursor()
    
    # Create user profiles table to store customer information
    # CREATE TABLE IF NOT EXISTS means "only create if table doesn't already exist"
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_profiles (
            user_id TEXT PRIMARY KEY,  -- Unique ID for each user
            name TEXT,  -- Customer's name
            email TEXT,  -- Customer's email address
            subscription_type TEXT,  -- Type of subscription (Basic/Premium/Enterprise)
            created_at TEXT  -- When the profile was created
        )
    """)
    
    # Create query history table to store all support interactions
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS query_history (
            query_id TEXT PRIMARY KEY,  -- Unique ID for each support ticket
            user_id TEXT,  -- Which user this ticket belongs to
            query TEXT,  -- The customer's question/problem
            resolution TEXT,  -- How the problem was solved
            timestamp TEXT,  -- When this interaction happened
            resolution_type TEXT,  -- How it was resolved (automated/escalated/resolved)
            satisfaction_score INTEGER,  -- Customer rating (1-5)
            FOREIGN KEY (user_id) REFERENCES user_profiles (user_id)  -- Links to user_profiles table
        )
    """)
    
    # Save all changes to the database
    conn.commit()
    # Close the database connection to free up resources
    conn.close()

# Function to save a user's profile information to the database
def save_user_profile(profile: Dict[str, Any], db_path: str = DB_PATH) -> None:
    """Save user profile dictionary to database"""
    # Connect to the SQLite database
    conn = sqlite3.connect(db_path)
    # Create cursor to execute SQL commands
    cursor = conn.cursor()
    
    # INSERT OR REPLACE means "add new record or update if it already exists"
    # The ? marks are placeholders that prevent SQL injection attacks
    cursor.execute("""
        INSERT OR REPLACE INTO user_profiles 
        (user_id, name, email, subscription_type, created_at)
        VALUES (?, ?, ?, ?, ?)
    """, (profile["user_id"], profile["name"], profile["email"], 
          profile["subscription_type"], profile["created_at"]))  # Values to insert
    
    # Save changes to database
    conn.commit()
    # Close connection to free up resources
    conn.close()

# Function to look up a user's profile information from the database
def get_user_profile(user_id: str, db_path: str = DB_PATH) -> Optional[Dict[str, Any]]:
    """Retrieve user profile dictionary from database"""
    # Connect to the SQLite database
    conn = sqlite3.connect(db_path)
    # Create cursor to execute SQL commands
    cursor = conn.cursor()
    
    # SELECT statement gets data from the database
    # WHERE clause filters to only get the specific user we want
    cursor.execute("""
        SELECT user_id, name, email, subscription_type, created_at
        FROM user_profiles WHERE user_id = ?
    """, (user_id,))  # The ? is replaced with user_id safely
    
    # fetchone() gets the first matching record (should be only one)
    result = cursor.fetchone()
    # Close the database connection
    conn.close()
    
    # If we found a user profile, create a profile dictionary and return it
    if result:
        # Convert database row (tuple) to dictionary
        return {
            "user_id": result[0],
            "name": result[1], 
            "email": result[2],
            "subscription_type": result[3],
            "created_at": result[4]
        }
    # If no user found, return None
    return None

# Function to save a completed support ticket to the database
def save_query_resolution(resolution: Dict[str, Any], user_id: str, db_path: str = DB_PATH) -> None:
    """Save query resolution dictionary to database"""
    # Connect to the SQLite database
    conn = sqlite3.connect(db_path)
    # Create cursor to execute SQL commands
    cursor = conn.cursor()
    
    # INSERT adds a new record to the query_history table
    # All the ? marks are safely replaced with actual values
    cursor.execute("""
        INSERT INTO query_history 
        (query_id, user_id, query, resolution, timestamp, resolution_type, satisfaction_score)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (resolution["query_id"], user_id, resolution["query"], resolution["resolution"],
          resolution["timestamp"], resolution["resolution_type"], resolution["satisfaction_score"]))
    
    # Save changes to database
    conn.commit()
    # Close connection to free up resources
    conn.close()

# Function to get the customer's previous support tickets
def get_user_history(user_id: str, limit: int = 5, db_path: str = DB_PATH) -> List[Dict[str, Any]]:
    """Retrieve user's query history as list of dictionaries"""
    # Connect to the SQLite database
    conn = sqlite3.connect(db_path)
    # Create cursor to execute SQL commands
    cursor = conn.cursor()
    
    # SELECT gets records from query_history table
    # WHERE filters to only this user's tickets
    # ORDER BY timestamp DESC sorts newest first
    # LIMIT restricts how many records we get back (default is 5)
    cursor.execute("""
        SELECT query_id, query, resolution, timestamp, resolution_type, satisfaction_score
        FROM query_history 
        WHERE user_id = ? 
        ORDER BY timestamp DESC 
        LIMIT ?
    """, (user_id, limit))  # Replace ? marks with actual values
    
    # fetchall() gets all matching records as a list of tuples
    results = cursor.fetchall()
    # Close the database connection
    conn.close()
    
    # Convert each database row into a dictionary
    # This is a "list comprehension" - it creates a list by processing each row
    return [
        {
            "query_id": row[0], 
            "query": row[1], 
            "resolution": row[2],
            "timestamp": row[3], 
            "resolution_type": row[4], 
            "satisfaction_score": row[5]
        }
        for row in results  # "for row in results" means "do this for each row"
    ]

# Initialize the database when the module is imported
init_database()

# Tools for the agent - these are functions the AI can call to get information
# @tool decorator makes this function available to the LangChain AI agent
@tool
def get_user_history_tool(user_id: str) -> str:
    """Fetch user's previous interaction history"""
    # Get the last 3 support tickets for this user
    history = get_user_history(user_id, limit=3)
    
    # If the user has no previous tickets, return a message saying so
    if not history:
        return "No previous interaction history found for this user."
    
    # Build a text summary of the user's history
    history_text = "Previous interactions:\n"
    # Loop through each ticket in the history
    for item in history:
        history_text += f"- Query: {item['query']}\n"  # Add the customer's question
        history_text += f"  Resolution: {item['resolution']}\n"  # Add how it was resolved
        history_text += f"  Date: {item['timestamp'][:10]}\n\n"  # Add date (first 10 chars = YYYY-MM-DD)
    
    # Return the formatted history text
    return history_text

# Another tool function for checking if we have a quick answer ready
@tool
def check_common_issues_tool(query: str) -> str:
    """Check if query matches common issues and provide quick resolution"""
    # Dictionary (like a lookup table) of common problems and their solutions
    # Key = keyword to look for, Value = ready-made solution
    common_issues = {
        "password": "To reset your password: 1) Go to login page 2) Click 'Forgot Password' 3) Enter your email 4) Check your inbox for reset link",
        "billing": "For billing inquiries: 1) Check your account dashboard 2) Review recent transactions 3) Contact billing@techtrend.com for disputes",
        "login": "Login issues: 1) Clear browser cache 2) Try incognito mode 3) Check if Caps Lock is on 4) Reset password if needed",
        "feature": "For feature requests: 1) Check our roadmap at techtrend.com/roadmap 2) Submit requests via feedback form 3) Join our beta program for early access",
        "bug": "To report bugs: 1) Note exact steps to reproduce 2) Include screenshots if applicable 3) Submit via support portal 4) Include browser/OS details"
    }
    
    # Convert the query to lowercase for easier matching
    query_lower = query.lower()
    # Loop through each keyword and solution in our dictionary
    for keyword, solution in common_issues.items():
        # Check if this keyword appears anywhere in the customer's question
        if keyword in query_lower:
            # If we find a match, return the pre-written solution
            return f"Quick Resolution Available:\n{solution}"
    
    # If no common issues match, let the AI agent handle it
    return "No quick resolution found. Proceeding with detailed analysis."

# Agent nodes - these are the main processing steps of our AI support agent
# Each node takes the current state, processes it, and returns the updated state

# This function figures out what type of question the customer is asking
def classify_query_node(state: AgentState) -> AgentState:
    """Classify the type of customer query"""
    # If there are no messages yet, just return the state unchanged
    if not state["messages"]:
        return state
    
    # Get the most recent message (the last one in the list)
    last_message = state["messages"][-1]
    # Check if it's a message from a human (not the AI)
    if isinstance(last_message, HumanMessage):
        # Convert the message to lowercase for easier keyword matching
        query = last_message.content.lower()
        
        # Simple classification logic - check for specific keywords
        # any() function returns True if ANY of the words are found in the query
        if any(word in query for word in ["password", "login", "access"]):
            state["current_query_type"] = "authentication"  # Login/password problems
        elif any(word in query for word in ["billing", "payment", "invoice"]):
            state["current_query_type"] = "billing"  # Money/payment problems
        elif any(word in query for word in ["bug", "error", "crash", "not working"]):
            state["current_query_type"] = "technical_issue"  # Software problems
        elif any(word in query for word in ["feature", "request", "enhance"]):
            state["current_query_type"] = "feature_request"  # Asking for new features
        else:
            state["current_query_type"] = "general"  # Everything else
        
        # Check if this needs to go to a human support agent
        complex_indicators = ["refund", "legal", "urgent", "emergency", "complaint", "cancel subscription"]
        # If ANY of these words appear, set escalation_required to True
        state["escalation_required"] = any(indicator in query for indicator in complex_indicators)
    
    # Return the updated state with classification information
    return state

# This function gets a response from the OpenAI AI model
def llm_response_node(state: AgentState) -> AgentState:
    """Generate response using OpenAI"""
    # Check if the user has provided an OpenAI API key
    # (API key is needed to communicate with OpenAI's servers)
    if "openai_api_key" not in st.session_state or not st.session_state.openai_api_key:
        # If no API key, create an error message and add it to the conversation
        error_msg = "OpenAI API key not configured. Please add it in the sidebar."
        state["messages"].append(AIMessage(content=error_msg))
        return state  # Exit early since we can't proceed without API key
    
    # try/except block handles any errors that might occur
    try:
        # Create an OpenAI language model object
        # This is what actually communicates with ChatGPT
        llm = ChatOpenAI(
            model="gpt-4",  # Use GPT-4 model (most capable)
            api_key=st.session_state.openai_api_key,  # User's API key
            temperature=0.1  # Low temperature = more consistent, less creative responses
        )
        
        # Get the user's previous support tickets to provide better context
        history = get_user_history(state["user_id"], limit=3)  # Get last 3 tickets
        history_context = ""  # Start with empty string
        
        # If the user has previous tickets, format them nicely
        if history:
            history_context = "\n\nUser's Recent History:\n"
            # Loop through each previous ticket
            for item in history:
                history_context += f"- Previous query: {item['query']}\n"  # What they asked
                history_context += f"  Resolution: {item['resolution']}\n"  # How it was solved
        
        # Create instructions for the AI (this tells ChatGPT how to behave)
        system_prompt = f"""You are a helpful customer support agent for TechTrend Innovations, a software solutions company.

Current user: {state['user_id']}
Query type: {state.get('current_query_type', 'general')}

Guidelines:
1. Be friendly, professional, and helpful
2. Provide clear, actionable solutions
3. Reference user history when relevant
4. If the issue is complex or requires account access, suggest escalating to human support
5. For technical issues, provide step-by-step troubleshooting
6. Keep responses concise but thorough

{history_context}

Respond to the user's query with a helpful solution."""

        # Prepare the list of messages to send to OpenAI
        # Start with the system prompt (instructions for the AI)
        messages = [SystemMessage(content=system_prompt)]
        
        # Add the recent conversation messages (last 5 to save on API costs)
        # [-5:] means "get the last 5 items from the list"
        recent_messages = state["messages"][-5:]
        # Add each message to our list
        for msg in recent_messages:
            messages.append(msg)
        
        # Send all messages to OpenAI and get a response
        response = llm.invoke(messages)
        # Add the AI's response to our conversation
        state["messages"].append(AIMessage(content=response.content))
        
        # Save this support interaction to the database for future reference
        # Only save if we have at least a question and answer
        if len(state["messages"]) >= 2:
            last_user_msg = None
            # Look through messages in reverse order (newest first) to find user's question
            for msg in reversed(state["messages"]):
                # Check if this is a message from the human user
                if isinstance(msg, HumanMessage):
                    last_user_msg = msg  # Found the user's question
                    break  # Stop looking
            
            # If we found the user's question, save the complete interaction
            if last_user_msg:
                # Create a query resolution dictionary with all the details
                resolution = create_query_resolution(
                    query_id=str(uuid.uuid4()),  # Generate unique ID for this ticket
                    query=last_user_msg.content,  # The user's question
                    resolution=response.content,  # The AI's answer
                    resolution_type="automated"  # Mark this as solved by AI
                )
                # Save to database
                save_query_resolution(resolution, state["user_id"])
        
    # If any error occurs during AI processing, handle it gracefully
    except Exception as e:
        # Create a user-friendly error message (don't show technical details)
        error_msg = f"Sorry, I'm experiencing technical difficulties: {str(e)}"
        # Add the error message to the conversation
        state["messages"].append(AIMessage(content=error_msg))
    
    # Return the updated state
    return state

# This function checks if the customer's question needs human support
def escalation_check_node(state: AgentState) -> AgentState:
    """Check if escalation to human agent is needed"""
    # Check if the previous classification step marked this for escalation
    # get() method safely gets a value or returns False if key doesn't exist
    if state.get("escalation_required", False):
        # Create a message explaining that human support will help
        escalation_msg = """
        🚨 **Escalation Required**
        
        Your query has been flagged for human review due to its complexity or urgency. 
        A human support specialist will review your case and respond within 2 business hours.
        
        **Ticket ID:** """ + str(uuid.uuid4())[:8].upper() + """  
        **Priority:** High
        **Estimated Response Time:** 2 hours
        
        In the meantime, you can:
        - Check our FAQ at techtrend.com/faq
        - Browse our knowledge base
        - Contact emergency support at +1-800-TECHTREND for urgent issues
        """  # str(uuid.uuid4())[:8].upper() creates a short ticket ID like "A1B2C3D4"
        
        # Add the escalation message to the conversation
        state["messages"].append(AIMessage(content=escalation_msg))
        
        # Save escalation record to database for tracking
        # Create a query resolution dictionary to log that this was escalated
        resolution = create_query_resolution(
            query_id=str(uuid.uuid4()),  # Unique ID for this escalation
            # Try to get the user's question, or use default text if none found
            query=state["messages"][-2].content if len(state["messages"]) >= 2 else "Escalation request",
            resolution="Escalated to human support",  # Note that this was escalated
            resolution_type="escalated"  # Mark as escalated (not automated)
        )
        # Save the escalation record
        save_query_resolution(resolution, state["user_id"])
    
    # Return the updated state
    return state

# This function prevents the conversation from getting too long
# (Long conversations cost more money and can slow down the AI)
def trim_messages_node(state: AgentState) -> AgentState:
    """Trim messages to manage context window size"""
    # Only trim if we have more than 10 messages in the conversation
    # len() function counts how many items are in the messages list
    if len(state["messages"]) > 10:
        # Check if the first message is system instructions (we want to keep those)
        if state["messages"] and isinstance(state["messages"][0], SystemMessage):
            # Keep the first message + the last 9 messages
            # [0] gets first item, [-9:] gets last 9 items
            state["messages"] = [state["messages"][0]] + state["messages"][-9:]
        else:
            # If no system message, just keep the last 10 messages
            state["messages"] = state["messages"][-10:]
    
    # Return the state with trimmed messages
    return state

# Create global objects that will be used throughout the application
# MemorySaver helps LangGraph remember conversation context
memory_saver = MemorySaver()

# This function creates the AI agent workflow - the steps it follows for each customer question
def create_support_agent():
    """Create the LangGraph workflow for customer support"""
    # Create a StateGraph - this defines how our AI agent processes requests
    # It's like a flowchart that the AI follows step by step
    workflow = StateGraph(AgentState)
    
    # Add nodes (steps) to the workflow
    # Each node is a function that processes the customer's request
    workflow.add_node("classify_query", classify_query_node)  # Step 1: Figure out what type of question
    workflow.add_node("llm_response", llm_response_node)  # Step 2: Get AI response
    workflow.add_node("escalation_check", escalation_check_node)  # Step 3: Check if human help needed
    workflow.add_node("trim_messages", trim_messages_node)  # Step 4: Clean up conversation
    
    # Define the flow - what order the steps happen in
    workflow.set_entry_point("classify_query")  # Always start with classifying the question
    # add_edge means "after step A, go to step B"
    workflow.add_edge("classify_query", "llm_response")  # After classifying, get AI response
    workflow.add_edge("llm_response", "escalation_check")  # After AI response, check escalation
    workflow.add_edge("escalation_check", "trim_messages")  # After escalation check, trim messages
    workflow.add_edge("trim_messages", END)  # After trimming, we're done (END)
    
    # Compile the workflow into a runnable application
    # checkpointer=memory_saver means "remember conversations between user visits"
    app = workflow.compile(checkpointer=memory_saver)
    return app  # Return the completed workflow

# This function creates a visual diagram showing how our AI agent works
def generate_workflow_diagram():
    """Generate and return the workflow diagram as base64 image"""
    # try/except handles errors if diagram generation fails
    try:
        # Create a temporary workflow just for making the diagram
        # (we don't want to interfere with the real workflow)
        temp_workflow = StateGraph(AgentState)
        # Add dummy nodes using lambda functions (simple placeholder functions)
        # lambda x: x means "take input x and return x unchanged"
        temp_workflow.add_node("classify_query", lambda x: x)  # Step 1 placeholder
        temp_workflow.add_node("llm_response", lambda x: x)  # Step 2 placeholder
        temp_workflow.add_node("escalation_check", lambda x: x)  # Step 3 placeholder
        temp_workflow.add_node("trim_messages", lambda x: x)  # Step 4 placeholder
        
        # Set up the same flow as our real workflow
        temp_workflow.set_entry_point("classify_query")  # Start here
        temp_workflow.add_edge("classify_query", "llm_response")  # Go from step 1 to 2
        temp_workflow.add_edge("llm_response", "escalation_check")  # Go from step 2 to 3
        temp_workflow.add_edge("escalation_check", "trim_messages")  # Go from step 3 to 4
        temp_workflow.add_edge("trim_messages", END)  # End after step 4
        
        # Generate the actual diagram image
        compiled_graph = temp_workflow.compile()  # Convert workflow to executable form
        diagram_bytes = compiled_graph.get_graph().draw_mermaid_png()  # Create PNG image
        
        # Convert image to base64 text format (needed for web display)
        # base64 encoding converts binary data (image) to text
        img_b64 = base64.b64encode(diagram_bytes).decode()
        return img_b64  # Return the encoded image
    # If anything goes wrong, return None (no diagram)
    except Exception as e:
        return None  # Caller will handle showing fallback text diagram

# This function sets up all the data we need when the app first starts
# Streamlit's session_state keeps data alive as the user interacts with the app
def initialize_session_state():
    """Initialize Streamlit session state"""
    # Check if we already have a user_id, if not create one
    # st.session_state is like a dictionary that remembers values between page refreshes
    if "user_id" not in st.session_state:
        # uuid.uuid4() creates a unique random ID like "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
        st.session_state.user_id = str(uuid.uuid4())
    
    # Check if we have a conversation thread ID, if not create one
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())  # Each conversation gets unique ID
    
    # Check if we have a messages list, if not create empty list
    if "messages" not in st.session_state:
        st.session_state.messages = []  # Empty list to store all chat messages
    
    # Check if we have created the AI agent, if not create one
    if "agent" not in st.session_state:
        st.session_state.agent = create_support_agent()  # Create the workflow we defined earlier
    
    # Set up user profile - either load existing one or create new
    if "user_profile" not in st.session_state:
        # Try to find existing profile in database using user_id
        existing_profile = get_user_profile(st.session_state.user_id)
        if existing_profile:
            # If found, store the profile dictionary
            st.session_state.user_profile = existing_profile
        else:
            # If no existing profile, create a new one
            new_profile = create_user_profile(user_id=st.session_state.user_id)
            # Store in session
            st.session_state.user_profile = new_profile
            # Save the new profile to database for next time
            save_user_profile(new_profile)

# This is the main function that creates the web interface
def main():
    """Main Streamlit application"""
    # Set up all the data we need (user ID, conversation history, etc.)
    initialize_session_state()
    
    # Create the header section at the top of the page
    st.markdown("""
    <div class="main-header">
        <h1>🤖 TechTrend Innovations Customer Support</h1>
        <p>AI-Powered Support Agent | Available 24/7</p>
    </div>
    """, unsafe_allow_html=True)  # Display the HTML header
    
    # Create two tabs: one for chatting, one for viewing the workflow
    # tabs() creates clickable tabs at the top of the content area
    tab1, tab2 = st.tabs(["💬 Customer Support Chat", "🔄 Workflow Diagram"])
    
    # Everything in this "with" block appears in the second tab
    with tab2:
        st.header("🔄 Agent Workflow Diagram")  # Tab title
        st.write("Visual representation of the customer support agent workflow:")  # Description
        
        # Try to create and show a visual diagram of how the AI agent works
        diagram_b64 = generate_workflow_diagram()  # Get diagram as base64 text
        if diagram_b64:  # If diagram was successfully created
            # Display the diagram as an image using HTML
            st.markdown(f'<img src="data:image/png;base64,{diagram_b64}" style="max-width: 100%; height: auto; background-color: white; padding: 20px; border-radius: 10px;">', unsafe_allow_html=True)
        else:  # If diagram creation failed
           
            # Show a text-based diagram instead
            st.markdown("""
            ```
            Workflow Steps:
            
            1. [ENTRY] → classify_query
                ↓
            2. classify_query → llm_response
                ↓
            3. llm_response → escalation_check
                ↓
            4. escalation_check → trim_messages
                ↓
            5. trim_messages → [END]
            ```
            """)  # Three quotes allow multi-line text
    
    # Everything in this "with" block appears in the first tab (main chat)
    with tab1:
        # Create a sidebar (panel on the left side of the screen)
        with st.sidebar:
            st.header("⚙️ Configuration")  # Section title
            
            # Create an input box for the OpenAI API key
            # type="password" hides the text as user types (shows *** instead)
            api_key = st.text_input(
                "OpenAI API Key",  # Label for the input box
                type="password",  # Hide the text for security
                value=st.session_state.get("openai_api_key", ""),  # Show current value if any
                help="Enter your OpenAI API key to enable AI responses"  # Tooltip help text
            )
            # If user entered an API key, save it for use
            if api_key:
                st.session_state.openai_api_key = api_key  # Save in Streamlit session
                os.environ["OPENAI_API_KEY"] = api_key  # Also save in environment variables
            
            st.divider()
            
            # User Profile Management section
            st.header("👤 User Profile")  # Section title with user icon
            # Show first 8 characters of user ID (full ID would be too long)
            st.write(f"**User ID:** `{st.session_state.user_id[:8]}...`")
            
            # Create an expandable section for updating profile info
            # Expander creates a collapsible section that users can open/close
            with st.expander("Update Profile"):
                name = st.text_input("Name", value=st.session_state.user_profile.get("name", ""))
                email = st.text_input("Email", value=st.session_state.user_profile.get("email", ""))
                subscription = st.selectbox(
                    "Subscription Type",
                    ["Basic", "Premium", "Enterprise"],
                    index=0 if not st.session_state.user_profile.get("subscription_type") 
                          else ["Basic", "Premium", "Enterprise"].index(st.session_state.user_profile.get("subscription_type", "Basic"))
                )
                
                # If user clicks Update Profile button
                if st.button("Update Profile"):
                    # Create updated profile dictionary
                    profile = create_user_profile(
                        user_id=st.session_state.user_id,
                        name=name or None,  # Use entered name or None if empty
                        email=email or None,  # Use entered email or None if empty
                        subscription_type=subscription,
                        created_at=st.session_state.user_profile.get("created_at")  # Keep original creation time
                    )
                    # Save updated profile to database
                    save_user_profile(profile)
                    # Update session state with new profile
                    st.session_state.user_profile = profile
                    # Show success message to user
                    st.success("Profile updated!")
            
            # Session management section
            st.divider()  # Add a horizontal line to separate sections
            st.header("🔄 Session Management")  # Section title
            
            # Create a button to start a new conversation
            # type="secondary" makes it a less prominent button style
            if st.button("New Session", type="secondary"):
                st.session_state.thread_id = str(uuid.uuid4())  # Create new conversation ID
                st.session_state.messages = []  # Clear all messages
                st.rerun()  # Refresh the page to show changes
            
            # Create a button to clear conversation history
            if st.button("Clear All History", type="secondary"):
                st.session_state.messages = []  # Clear all messages from session
                # Note: This doesn't clear database history, just current session
                st.rerun()  # Refresh the page
            
            # Display session information for debugging/info
            st.write(f"**Thread ID:** `{st.session_state.thread_id[:8]}...`")
            st.write(f"**Messages in session:** {len(st.session_state.messages)}")
        
        # Create the main chat interface with two columns
        # [3, 1] means left column is 3 times wider than right column
        col1, col2 = st.columns([3, 1])
        
        # Everything in this block appears in the left (wider) column
        with col1:
            st.header("💬 Customer Support Chat")  # Main chat title
            
            # Create a container to hold all the chat messages
            chat_container = st.container()
            with chat_container:
                # Loop through each message in our conversation history
                for message in st.session_state.messages:
                    # Check if this message is from the human user
                    if isinstance(message, HumanMessage):
                        # Display user messages with blue styling
                        st.markdown(f"""
                        <div class="user-message">
                            <strong>You:</strong> {message.content}
                        </div>
                        """, unsafe_allow_html=True)  # Use our custom CSS class
                    # Check if this message is from the AI
                    elif isinstance(message, AIMessage):
                        # Display AI messages with green styling
                        st.markdown(f"""
                        <div class="ai-message">
                            <strong>Support Agent:</strong> {message.content}
                        </div>
                        """, unsafe_allow_html=True)  # Use our custom CSS class
            
            # Create an input box at the bottom for typing new messages
            user_input = st.chat_input("Type your support question here...")
            
            # This code runs when the user types a message and presses Enter
            if user_input:
                # Convert user's text into a HumanMessage object
                user_message = HumanMessage(content=user_input)
                # Add the user's message to our conversation history
                st.session_state.messages.append(user_message)
                
                # Prepare all the information the AI agent needs to process the request
                # This creates an AgentState dictionary with all current data
                current_state = {
                    "messages": st.session_state.messages,  # All conversation messages
                    "user_id": st.session_state.user_id,  # Who is asking the question
                    "thread_id": st.session_state.thread_id,  # Which conversation this is
                    "user_profile": st.session_state.user_profile,  # User's profile info
                    "query_history": [],  # Previous tickets (will be filled by agent)
                    "current_query_type": None,  # Type of question (will be determined by agent)
                    "escalation_required": False,  # Whether human help needed (will be determined)
                    "session_metadata": {  # Extra info about this session
                        "timestamp": datetime.now(timezone.utc).isoformat(),  # Current time
                        "session_length": len(st.session_state.messages)  # How many messages so far
                    }
                }
                
                # Send the user's question through our AI agent workflow
                # st.spinner shows a loading animation while processing
                with st.spinner("Processing your request..."):
                    try:
                        # Configuration tells the agent which conversation thread this is
                        config = {"configurable": {"thread_id": st.session_state.thread_id}}
                        # Run the entire workflow (classify -> AI response -> escalation check -> trim)
                        result = st.session_state.agent.invoke(current_state, config)
                        
                        # Update our stored messages with any new messages from the agent
                        st.session_state.messages = result["messages"]
                        
                    # If something goes wrong during processing
                    except Exception as e:
                        # Show error message to user
                        st.error(f"Error processing request: {str(e)}")
                        # Add a fallback message to the conversation
                        error_message = AIMessage(content="I apologize, but I'm experiencing technical difficulties. Please try again or contact human support.")
                        st.session_state.messages.append(error_message)
                
                # Refresh the page to show the new messages
                st.rerun()
        
        # Everything in this block appears in the right (narrower) column
        with col2:
            st.header("📊 Session Stats")  # Statistics section title
            
            # Show conversation statistics if there are any messages
            if st.session_state.messages:
                # Count how many messages are from the user
                # List comprehension: [item for item in list if condition]
                user_messages = [msg for msg in st.session_state.messages if isinstance(msg, HumanMessage)]
                # Count how many messages are from the AI
                ai_messages = [msg for msg in st.session_state.messages if isinstance(msg, AIMessage)]
                
                # st.metric creates a nice display with large numbers
                st.metric("User Messages", len(user_messages))  # Show count of user messages
                st.metric("AI Responses", len(ai_messages))  # Show count of AI responses
                
                # Current query type classification for display
                if len(st.session_state.messages) > 0:
                    # Create a temporary state for classification display
                    current_state = {
                        "messages": st.session_state.messages,
                        "user_id": st.session_state.user_id,
                        "thread_id": st.session_state.thread_id,
                        "user_profile": st.session_state.user_profile,
                        "query_history": [],
                        "current_query_type": None,
                        "escalation_required": False,
                        "session_metadata": {}
                    }
                    
                    # Quick classification for display purposes
                    classified_state = classify_query_node(current_state)
                    if classified_state.get("current_query_type"):
                        # Show the detected query type in a nice format
                        st.write(f"**Query Type:** {classified_state['current_query_type'].replace('_', ' ').title()}")
                    
                    # Show warning if escalation is required
                    if classified_state.get("escalation_required"):
                        st.warning("⚠️ Escalation Required")
            
            # User history preview section
            st.header("📝 Recent History")
            # Get the user's last 3 support tickets from database
            history = get_user_history(st.session_state.user_id, limit=3)
            
            # If user has previous tickets, show them
            if history:
                # Loop through each ticket with enumerate to get index numbers
                for i, item in enumerate(history):
                    # Create an expandable section for each ticket
                    # Show first 30 characters of the query as preview
                    with st.expander(f"Query {i+1}: {item['query'][:30]}..."):
                        st.write(f"**Query:** {item['query']}")  # Full question
                        st.write(f"**Resolution:** {item['resolution']}")  # How it was solved
                        st.write(f"**Date:** {item['timestamp'][:10]}")  # Date (YYYY-MM-DD)
                        st.write(f"**Type:** {item['resolution_type']}")  # automated/escalated/resolved
            else:
                # If no previous tickets, show info message
                st.info("No previous interactions found.")
    
    # Create a footer section with helpful information for users
    st.divider()  # Add horizontal line to separate footer from main content
    # Create three equal-width columns for footer information
    col1, col2, col3 = st.columns(3)
    
    

# This special Python pattern means "only run main() if this file is run directly"
# (not if this file is imported by another Python file)
if __name__ == "__main__":
    main()  # Start the Streamlit web application
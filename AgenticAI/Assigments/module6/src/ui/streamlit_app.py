"""
Streamlit UI for Customer Support Triage Agent - Single Directory Version
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import os
import logging

# Import components from current directory
from support_agent import SupportTriageAgent
from config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)

# Page configuration
st.set_page_config(
    page_title=settings.PAGE_TITLE,
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
def initialize_session_state():
    """Initialize session state variables"""
    if 'agent' not in st.session_state:
        try:
            st.session_state.agent = SupportTriageAgent()
            st.session_state.agent_initialized = True
        except Exception as e:
            st.session_state.agent = None
            st.session_state.agent_initialized = False
            st.session_state.initialization_error = str(e)
    
    if 'processed_tickets' not in st.session_state:
        st.session_state.processed_tickets = []
    
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = []

def main():
    """Main application function"""
    initialize_session_state()
    
    st.title("🎯 Customer Support Triage Agent")
    st.markdown("*AI-powered ticket analysis and response generation*")
    
    # Check agent initialization
    if not st.session_state.agent_initialized:
        st.error("❌ Failed to initialize agent")
        if hasattr(st.session_state, 'initialization_error'):
            st.error(f"Error: {st.session_state.initialization_error}")
        
        st.info("💡 This might be due to missing API keys. Check your .env file or use the system in demo mode.")
        
        if st.button("🔄 Retry Initialization"):
            # Clear the session state and retry
            for key in ['agent', 'agent_initialized', 'initialization_error']:
                if key in st.session_state:
                    del st.session_state[key]
            st.experimental_rerun()
        
        # Still show the interface for demo purposes
        st.warning("⚠️ Running in limited demo mode")
    
    # Sidebar navigation
    page = st.sidebar.selectbox(
        "Navigate to:",
        ["📊 Dashboard", "📝 Process Tickets", "📁 File Management", "🔍 Knowledge Search", "⚙️ Settings"]
    )
    
    # Display agent status in sidebar
    display_agent_status()
    
    # Route to appropriate page
    if page == "📊 Dashboard":
        dashboard_page()
    elif page == "📝 Process Tickets":
        process_tickets_page()
    elif page == "📁 File Management":
        file_management_page()
    elif page == "🔍 Knowledge Search":
        knowledge_search_page()
    elif page == "⚙️ Settings":
        settings_page()

def display_agent_status():
    """Display agent status in sidebar"""
    st.sidebar.markdown("---")
    st.sidebar.subheader("🤖 Agent Status")
    
    try:
        if st.session_state.agent_initialized and st.session_state.agent:
            status = st.session_state.agent.get_agent_status()
            
            # Status indicator
            if status['status'] == 'active':
                st.sidebar.success("✅ Active")
            else:
                st.sidebar.error("❌ Error")
            
            # Key metrics
            stats = status.get('session_stats', {})
            st.sidebar.metric("Tickets Processed", stats.get('tickets_processed', 0))
            st.sidebar.metric("Files Uploaded", stats.get('files_uploaded', 0))
            st.sidebar.metric("KB Files", status.get('knowledge_base_files', 0))
        else:
            st.sidebar.warning("⚠️ Agent not initialized")
            
    except Exception as e:
        st.sidebar.error(f"Status Error: {str(e)}")

def dashboard_page():
    """Main dashboard page"""
    st.header("📊 Dashboard")
    
    if not st.session_state.agent_initialized:
        st.warning("Agent not initialized - showing demo data")
        show_demo_dashboard()
        return
    
    # Get insights report
    with st.spinner("Loading dashboard data..."):
        try:
            insights = st.session_state.agent.generate_insights_report()
        except Exception as e:
            st.error(f"Error loading dashboard: {str(e)}")
            insights = {}
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    session_stats = insights.get('session_statistics', {})
    kb_stats = insights.get('knowledge_base_stats', {})
    
    with col1:
        st.metric("Total Tickets", session_stats.get('tickets_processed', 0))
    
    with col2:
        st.metric("Files in KB", kb_stats.get('total_files_processed', 0))
    
    with col3:
        st.metric("Responses Generated", session_stats.get('responses_generated', 0))
    
    with col4:
        vector_stats = kb_stats.get('vector_store_stats', {})
        st.metric("Vector Count", vector_stats.get('total_vector_count', 0))
    
    st.markdown("---")
    
    # Charts row
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📁 File Types Distribution")
        file_types = kb_stats.get('file_types_distribution', {})
        if file_types:
            fig = px.pie(
                values=list(file_types.values()),
                names=list(file_types.keys()),
                title="Uploaded File Types"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No files uploaded yet")
    
    with col2:
        st.subheader("📈 Processing Trends")
        if st.session_state.processed_tickets:
            # Create sample trend data
            dates = pd.date_range(start=datetime.now().date() - timedelta(days=7), end=datetime.now().date())
            counts = [len(st.session_state.processed_tickets) // len(dates)] * len(dates)
            
            trend_df = pd.DataFrame({'date': dates, 'count': counts})
            
            fig = px.line(trend_df, x='date', y='count', title="Daily Ticket Processing")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No tickets processed yet")
    
    # Recommendations
    st.subheader("💡 Recommendations")
    recommendations = insights.get('recommendations', [])
    if recommendations:
        for rec in recommendations:
            st.info(f"💡 {rec}")
    else:
        st.success("✅ System is operating optimally")

def show_demo_dashboard():
    """Show demo dashboard when agent is not initialized"""
    # Demo metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Tickets", 0, help="Demo Mode")
    with col2:
        st.metric("Files in KB", 0, help="Demo Mode")
    with col3:
        st.metric("Responses Generated", 0, help="Demo Mode")
    with col4:
        st.metric("Vector Count", 0, help="Demo Mode")
    
    st.info("🎯 Dashboard will show real data once the agent is properly initialized with API keys.")

def process_tickets_page():
    """Ticket processing page"""
    st.header("📝 Process Support Tickets")
    
    if not st.session_state.agent_initialized:
        st.warning("⚠️ Agent not fully initialized. You can still test with sample tickets.")
    
    # Ticket input methods
    input_method = st.radio(
        "Choose input method:",
        ["Single Ticket", "Sample Tickets", "Batch Upload"]
    )
    
    if input_method == "Single Ticket":
        process_single_ticket()
    elif input_method == "Sample Tickets":
        process_sample_tickets()
    elif input_method == "Batch Upload":
        process_batch_tickets()
    
    # Display processed tickets
    if st.session_state.processed_tickets:
        st.markdown("---")
        st.subheader("🎯 Recent Ticket Analysis")
        display_processed_tickets()

def process_single_ticket():
    """Process a single support ticket"""
    st.subheader("Single Ticket Processing")
    
    # Ticket input
    ticket_text = st.text_area(
        "Enter customer message:",
        height=150,
        placeholder="Paste the customer's message here..."
    )
    
    ticket_id = st.text_input("Ticket ID (optional):", placeholder="AUTO-GENERATED")
    
    if st.button("🔍 Analyze Ticket", type="primary"):
        if ticket_text.strip():
            with st.spinner("Analyzing ticket..."):
                try:
                    if st.session_state.agent_initialized and st.session_state.agent:
                        result = st.session_state.agent.process_support_ticket(
                            ticket_text, ticket_id or None
                        )
                    else:
                        # Demo mode - create mock result
                        result = create_mock_ticket_result(ticket_text, ticket_id)
                    
                
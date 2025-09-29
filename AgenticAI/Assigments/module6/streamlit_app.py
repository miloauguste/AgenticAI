#!/usr/bin/env python3
"""
Comprehensive Customer Service Dashboard
Advanced UI with file upload, analytics, chat interface, and response suggestions
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import io
import time
import re
from typing import Dict, List, Any

# Import our comprehensive system
from support_agent import SupportTriageAgent
from enhanced_support_agent import EnhancedSupportTriageAgent
from database_manager import DatabaseManager
from config import settings

# Page configuration
st.set_page_config(
    page_title=settings.PAGE_TITLE,
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cloud deployment optimizations
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_demo_data():
    """Cache demo data for better cloud performance"""
    return {
        'analytics': True,
        'cached_at': datetime.now().isoformat()
    }

def is_cloud_deployment():
    """Detect if running on Streamlit Cloud"""
    import os
    return 'STREAMLIT_SHARING' in os.environ or 'STREAMLIT_CLOUD' in os.environ

# Cloud-optimized error handling
def safe_import_with_fallback(module_name, fallback_name=None):
    """Safely import modules with cloud deployment fallbacks"""
    try:
        return __import__(module_name)
    except ImportError as e:
        if fallback_name:
            st.warning(f"⚠️ {module_name} not available, using {fallback_name} fallback")
        else:
            st.warning(f"⚠️ {module_name} not available, some features may be limited")
        return None

# Helper function for safe chart creation
def create_safe_chart(chart_func, *args, **kwargs):
    """Create charts with error handling"""
    try:
        return chart_func(*args, **kwargs)
    except Exception as e:
        st.error(f"Error creating chart: {str(e)}")
        return None

def generate_demo_analytics(query_type: str, title: str) -> Dict[str, Any]:
    """Generate demo analytics for fallback when methods don't exist"""
    import plotly.express as px
    
    if query_type == "weekly_ticket_volume":
        # Generate demo weekly ticket data
        dates = pd.date_range(start=datetime.now().date() - timedelta(days=7), end=datetime.now().date())
        counts = [15, 23, 18, 31, 27, 19, 22]
        df = pd.DataFrame({'Date': dates, 'Tickets': counts})
        
        fig = px.line(df, x='Date', y='Tickets', title="Weekly Ticket Volume")
        return {
            'status': 'success',
            'chart_data': fig,
            'summary': f"Total tickets this week: {sum(counts)}. Peak day: {dates[counts.index(max(counts))].strftime('%A')} with {max(counts)} tickets."
        }
    
    elif query_type == "satisfaction_analysis":
        # Generate demo satisfaction data
        ratings = ['1 Star', '2 Stars', '3 Stars', '4 Stars', '5 Stars']
        counts = [5, 8, 15, 35, 45]
        df = pd.DataFrame({'Rating': ratings, 'Count': counts})
        
        fig = px.bar(df, x='Rating', y='Count', title="Customer Satisfaction Distribution")
        avg_rating = sum(i * c for i, c in enumerate(counts, 1)) / sum(counts)
        return {
            'status': 'success',
            'chart_data': fig,
            'summary': f"Average satisfaction: {avg_rating:.1f}/5. {(counts[3] + counts[4])/sum(counts)*100:.1f}% of customers rated 4+ stars."
        }
    
    elif query_type == "top_issues_by_category":
        # Generate demo issue data
        categories = ['Billing', 'Technical', 'Delivery', 'Account', 'Refund']
        counts = [45, 38, 32, 28, 25]
        df = pd.DataFrame({'Category': categories, 'Issues': counts})
        
        fig = px.bar(df, x='Category', y='Issues', title="Top Issues by Category")
        return {
            'status': 'success',
            'chart_data': fig,
            'data': df,
            'summary': f"Most common issue category: {categories[0]} with {counts[0]} cases."
        }
    
    elif query_type == "response_time_analysis":
        # Generate demo response time data
        categories = ['Low', 'Medium', 'High', 'Critical']
        times = [8.5, 4.2, 1.8, 0.5]
        df = pd.DataFrame({'Urgency': categories, 'Avg Response Time (hours)': times})
        
        fig = px.bar(df, x='Urgency', y='Avg Response Time (hours)', title="Response Time by Urgency")
        return {
            'status': 'success',
            'chart_data': fig,
            'data': df,
            'summary': f"Average response time across all urgency levels: {sum(times)/len(times):.1f} hours."
        }
    
    elif query_type == "refund_analysis":
        # Generate demo refund data
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
        requested = [25, 30, 28, 35, 32, 29]
        approved = [20, 24, 22, 28, 26, 23]
        df = pd.DataFrame({'Month': months, 'Requested': requested, 'Approved': approved})
        
        fig = px.line(df, x='Month', y=['Requested', 'Approved'], title="Refund Requests vs Approvals")
        approval_rate = sum(approved) / sum(requested) * 100
        return {
            'status': 'success',
            'chart_data': fig,
            'data': df,
            'summary': f"Overall refund approval rate: {approval_rate:.1f}%. Total refunds approved: {sum(approved)}"
        }
    
    else:
        return {
            'status': 'success',
            'summary': f"Demo analytics for {title}. This would show real data when connected to your support system."
        }

# Initialize session state
def initialize_session_state():
    """Initialize session state variables"""
    if 'agent' not in st.session_state:
        try:
            st.session_state.agent = EnhancedSupportTriageAgent()
            st.session_state.agent_initialized = True
        except Exception as e:
            st.session_state.agent = None
            st.session_state.agent_initialized = False
            st.session_state.initialization_error = str(e)
    
    if 'processed_tickets' not in st.session_state:
        st.session_state.processed_tickets = []
    
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = []
    
    if 'kb_file_count' not in st.session_state:
        st.session_state.kb_file_count = 0
    
    if 'last_kb_update' not in st.session_state:
        st.session_state.last_kb_update = None
    
    if 'kb_file_types' not in st.session_state:
        st.session_state.kb_file_types = {'csv': 0, 'txt': 0, 'pdf': 0, 'docx': 0, 'md': 0}

def sync_kb_status():
    """Sync session state with agent status"""
    if st.session_state.agent_initialized and st.session_state.agent:
        try:
            status = st.session_state.agent.get_agent_status()
            agent_count = status.get('knowledge_base_files', 0)
            # Update session state if agent has more files (e.g., from previous sessions)
            if agent_count > st.session_state.kb_file_count:
                st.session_state.kb_file_count = agent_count
                st.session_state.last_kb_update = datetime.now()
                
                # Also sync file types if available
                kb_stats = status.get('knowledge_base_stats', {})
                agent_file_types = kb_stats.get('file_types_distribution', {})
                if agent_file_types:
                    for file_type, count in agent_file_types.items():
                        if file_type in st.session_state.kb_file_types:
                            st.session_state.kb_file_types[file_type] = max(
                                st.session_state.kb_file_types[file_type], count
                            )
        except:
            pass  # Ignore sync errors

def main():
    """Main application function"""
    initialize_session_state()
    sync_kb_status()  # Sync knowledge base status on page load
    
    # Cloud deployment indicator
    if is_cloud_deployment():
        st.success("☁️ **Running on Streamlit Cloud** - Optimized for cloud performance")
    
    st.title("🎯 Customer Support Triage Agent")
    st.markdown("*AI-powered ticket analysis and response generation with advanced escalation detection*")
    
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
            st.rerun()
        
        # Still show the interface for demo purposes
        st.warning("⚠️ Running in limited demo mode")
    
    # Sidebar navigation
    page = st.sidebar.selectbox(
        "Navigate to:",
        ["📊 Dashboard", "📝 Process Tickets", "🚨 Escalation Center", "📁 File Management", "🔍 Knowledge Search", "📈 Historical Data", "🗄️ Persistent Store", "👥 Supervisor Dashboard", "⚙️ Settings"]
    )
    
    # Display agent status in sidebar
    display_agent_status()
    
    # Route to appropriate page
    if page == "📊 Dashboard":
        dashboard_page()
    elif page == "📝 Process Tickets":
        process_tickets_page()
    elif page == "🚨 Escalation Center":
        escalation_center_page()
    elif page == "📁 File Management":
        file_management_page()
    elif page == "🔍 Knowledge Search":
        knowledge_search_page()
    elif page == "📈 Historical Data":
        historical_data_page()
    elif page == "🗄️ Persistent Store":
        persistent_store_page()
    elif page == "👥 Supervisor Dashboard":
        supervisor_dashboard_page()
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
            
            # Add persistent store status
            if hasattr(st.session_state.agent, 'persistent_enabled'):
                st.sidebar.markdown("**🗄️ Persistent Store:**")
                if st.session_state.agent.persistent_enabled:
                    st.sidebar.success("✅ Active")
                    try:
                        insights = st.session_state.agent.get_persistent_store_insights()
                        if insights and 'error' not in insights:
                            project_stats = insights.get('project_stats', {})
                            st.sidebar.metric("Stored Complaints", project_stats.get('total_complaints', 0))
                            st.sidebar.metric("Resolution Templates", project_stats.get('total_resolutions', 0))
                    except:
                        pass
                else:
                    st.sidebar.error("❌ Disabled")
            # Use session state counter for more accurate real-time count
            kb_files = max(st.session_state.kb_file_count, status.get('knowledge_base_files', 0))
            delta_text = f"+{st.session_state.kb_file_count}" if st.session_state.kb_file_count > 0 else None
            st.sidebar.metric("KB Files", kb_files, 
                            delta=delta_text,
                            help="Files stored in knowledge base")
        else:
            st.sidebar.warning("⚠️ Agent not initialized")
            
    except Exception as e:
        st.sidebar.error(f"Status Error: {str(e)}")

def dashboard_page():
    """Comprehensive summary dashboard with complaint types, top issues, and resolution times"""
    st.header("📊 Summary Dashboard")
    
    # Time range selector
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        time_range = st.selectbox(
            "Time Range:",
            ["Last 24 Hours", "Last 7 Days", "Last 30 Days", "Last 90 Days", "All Time"],
            index=2
        )
    with col2:
        auto_refresh = st.checkbox("Auto-refresh", value=False)
    with col3:
        if st.button("🔄 Refresh Data"):
            st.rerun()
    
    if not st.session_state.agent_initialized:
        st.warning("Agent not initialized - showing demo data")
        show_demo_dashboard()
        return
    
    # Get comprehensive insights report
    with st.spinner("Loading dashboard data..."):
        try:
            # Map time_range to time_period parameter
            time_period_mapping = {
                "Last 24 Hours": "day",
                "Last 7 Days": "week", 
                "Last 30 Days": "month",
                "Last 90 Days": "quarter",
                "All Time": "session"
            }
            time_period = time_period_mapping.get(time_range, "session")
            insights = st.session_state.agent.generate_insights_report(time_period=time_period)
        except Exception as e:
            st.error(f"Error loading dashboard: {str(e)}")
            insights = {}
    
    # Key Performance Metrics
    st.subheader("🎯 Key Performance Metrics")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    session_stats = insights.get('session_statistics', {})
    kb_stats = insights.get('knowledge_base_stats', {})
    
    with col1:
        total_tickets = session_stats.get('tickets_processed', 0)
        st.metric("Total Tickets", total_tickets, delta=session_stats.get('tickets_change', 0))
    
    with col2:
        avg_resolution_time = session_stats.get('avg_resolution_time_hours', 0)
        st.metric("Avg Resolution Time", f"{avg_resolution_time:.1f}h", delta=f"{session_stats.get('resolution_time_change', 0):.1f}h")
    
    with col3:
        satisfaction_score = session_stats.get('avg_satisfaction', 0)
        st.metric("Customer Satisfaction", f"{satisfaction_score:.1f}/5", delta=f"{session_stats.get('satisfaction_change', 0):.1f}")
    
    with col4:
        resolution_rate = session_stats.get('resolution_rate', 0)
        st.metric("Resolution Rate", f"{resolution_rate:.1f}%", delta=f"{session_stats.get('resolution_rate_change', 0):.1f}%")
    
    with col5:
        escalation_rate = session_stats.get('escalation_rate', 0)
        st.metric("Escalation Rate", f"{escalation_rate:.1f}%", delta=f"{session_stats.get('escalation_change', 0):.1f}%")
    
    st.markdown("---")
    
    # Complaint Types and Top Issues
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 Complaint Types Distribution")
        complaint_types = insights.get('complaint_types', {})
        if complaint_types:
            # Create pie chart for complaint types
            fig = px.pie(
                values=list(complaint_types.values()),
                names=list(complaint_types.keys()),
                title="Tickets by Category",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
            
            # Show detailed breakdown
            st.write("**Detailed Breakdown:**")
            for category, count in sorted(complaint_types.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / sum(complaint_types.values())) * 100
                st.write(f"• {category}: {count} tickets ({percentage:.1f}%)")
        else:
            st.info("No complaint data available")
    
    with col2:
        st.subheader("🔥 Top Issues This Period")
        top_issues = insights.get('top_issues', [])
        if top_issues:
            # Create bar chart for top issues
            issue_names = [issue['name'] for issue in top_issues[:10]]
            issue_counts = [issue['count'] for issue in top_issues[:10]]
            
            fig = px.bar(
                x=issue_counts,
                y=issue_names,
                orientation='h',
                title="Most Common Issues",
                color=issue_counts,
                color_continuous_scale='Reds'
            )
            fig.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
            
            # Show top 5 in detail
            st.write("**Top 5 Issues:**")
            for i, issue in enumerate(top_issues[:5], 1):
                trend = "📈" if issue.get('trending', False) else "📊"
                st.write(f"{i}. {trend} {issue['name']} - {issue['count']} occurrences")
        else:
            st.info("No issue data available")
    
    # Resolution Times Analysis
    st.markdown("---")
    st.subheader("⏱️ Resolution Times Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**Resolution Time by Urgency**")
        urgency_times = insights.get('resolution_times_by_urgency', {})
        if urgency_times:
            urgency_df = pd.DataFrame(list(urgency_times.items()), columns=['Urgency', 'Hours'])
            fig = px.bar(urgency_df, x='Urgency', y='Hours', title="Avg Resolution Time by Urgency",
                        color='Hours', color_continuous_scale='RdYlGn_r')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No urgency data available")
    
    with col2:
        st.write("**Resolution Time by Category**")
        category_times = insights.get('resolution_times_by_category', {})
        if category_times:
            category_df = pd.DataFrame(list(category_times.items()), columns=['Category', 'Hours'])
            fig = px.bar(category_df, x='Category', y='Hours', title="Avg Resolution Time by Category",
                        color='Hours', color_continuous_scale='Viridis')
            fig.update_layout(xaxis_tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No category data available")
    
    with col3:
        st.write("**Resolution Time Trend**")
        time_trend = insights.get('resolution_time_trend', [])
        if time_trend:
            trend_df = pd.DataFrame(time_trend)
            fig = px.line(trend_df, x='date', y='avg_hours', title="Resolution Time Trend",
                         markers=True)
            st.plotly_chart(fig, use_container_width=True)
        else:
            # Create mock trend data
            dates = pd.date_range(start=datetime.now().date() - timedelta(days=30), end=datetime.now().date(), freq='D')
            mock_times = [24 + (i % 7) * 2 for i in range(len(dates))]
            trend_df = pd.DataFrame({'date': dates, 'avg_hours': mock_times})
            fig = px.line(trend_df, x='date', y='avg_hours', title="Resolution Time Trend (Demo)")
            st.plotly_chart(fig, use_container_width=True)
    
    # Customer Satisfaction Analysis
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("😊 Customer Satisfaction Breakdown")
        satisfaction_data = insights.get('satisfaction_breakdown', {})
        if satisfaction_data:
            sat_df = pd.DataFrame(list(satisfaction_data.items()), columns=['Rating', 'Count'])
            fig = px.bar(sat_df, x='Rating', y='Count', title="Satisfaction Ratings Distribution",
                        color='Rating', color_continuous_scale='RdYlGn')
            st.plotly_chart(fig, use_container_width=True)
        else:
            # Demo satisfaction data
            demo_sat = {'1 Star': 5, '2 Stars': 8, '3 Stars': 15, '4 Stars': 25, '5 Stars': 30}
            sat_df = pd.DataFrame(list(demo_sat.items()), columns=['Rating', 'Count'])
            fig = px.bar(sat_df, x='Rating', y='Count', title="Satisfaction Ratings (Demo)")
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📈 Performance Trends")
        performance_metrics = insights.get('performance_trends', {})
        
        metrics_data = {
            'Metric': ['Resolution Rate', 'First Contact Resolution', 'Customer Satisfaction', 'Response Time'],
            'This Period': [85, 65, 4.2, 2.5],
            'Last Period': [82, 60, 4.0, 3.0],
            'Target': [90, 70, 4.5, 2.0]
        }
        
        metrics_df = pd.DataFrame(metrics_data)
        fig = px.bar(metrics_df, x='Metric', y=['This Period', 'Last Period', 'Target'],
                    title="Performance vs Targets", barmode='group')
        fig.update_layout(xaxis_tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
    
    # System Insights and Recommendations
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔍 System Insights")
        insights_list = insights.get('insights', [])
        if insights_list:
            for insight in insights_list:
                icon = "🔥" if insight.get('priority') == 'high' else "💡"
                st.write(f"{icon} {insight.get('text', '')}")
        else:
            st.write("🔥 Billing issues are trending upward this week")
            st.write("💡 Password reset requests spike on Mondays")
            st.write("📊 Customer satisfaction improved 8% this month")
            st.write("⚡ Response times are 15% faster than target")
    
    with col2:
        st.subheader("💡 Actionable Recommendations")
        recommendations = insights.get('recommendations', [])
        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                st.write(f"{i}. {rec}")
        else:
            st.write("1. Focus on reducing billing inquiry resolution time")
            st.write("2. Create self-service guide for password resets")
            st.write("3. Implement proactive outreach for delivery delays")
            st.write("4. Consider additional training for technical issues")
    
    # Recent Activity Summary
    st.markdown("---")
    st.subheader("📝 Recent Activity Summary")
    
    recent_activity = insights.get('recent_activity', [])
    if recent_activity:
        activity_df = pd.DataFrame(recent_activity)
        st.dataframe(activity_df, use_container_width=True)
    else:
        # Show processed tickets from session
        if st.session_state.processed_tickets:
            recent_df = pd.DataFrame([
                {
                    'Time': ticket.get('processing_timestamp', '')[:19],
                    'Ticket ID': ticket.get('ticket_id', 'N/A'),
                    'Category': ticket.get('intent_classification', {}).get('predicted_intent', 'Unknown'),
                    'Urgency': ticket.get('sentiment_analysis', {}).get('urgency_level', 'Unknown'),
                    'Status': 'Processed'
                } for ticket in st.session_state.processed_tickets[-10:]
            ])
            st.dataframe(recent_df, use_container_width=True)
        else:
            st.info("No recent activity to display")
    
    # 🔄 Workflow Analytics Section (Enhanced Agent Feature)
    if hasattr(st.session_state.agent, 'get_workflow_analytics'):
        st.subheader("🔄 Workflow Analytics")
        st.caption("Agentic Task Orchestration Performance")
        
        try:
            workflow_analytics = st.session_state.agent.get_workflow_analytics()
            
            total_workflows = workflow_analytics.get('total_workflows', 0) or 0
            if total_workflows > 0:
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "Total Workflows", 
                        total_workflows,
                        help="Total number of workflows executed"
                    )
                
                with col2:
                    st.metric(
                        "Success Rate", 
                        f"{workflow_analytics['success_rate']:.1f}%",
                        help="Percentage of successfully completed workflows"
                    )
                
                with col3:
                    st.metric(
                        "Successful Workflows", 
                        workflow_analytics['successful_workflows'],
                        help="Number of workflows that completed successfully"
                    )
                
                with col4:
                    st.metric(
                        "Failed Workflows", 
                        workflow_analytics['failed_workflows'],
                        help="Number of workflows that failed during execution"
                    )
                
                # Workflow types breakdown
                if workflow_analytics['workflow_types']:
                    st.write("**Workflow Types:**")
                    workflow_cols = st.columns(len(workflow_analytics['workflow_types']))
                    
                    for i, (workflow_type, count) in enumerate(workflow_analytics['workflow_types'].items()):
                        with workflow_cols[i]:
                            st.metric(
                                workflow_type.replace('_', ' ').title(),
                                count,
                                help=f"Number of {workflow_type} workflows executed"
                            )
                
                st.caption(f"Last workflow executed: {workflow_analytics.get('last_workflow', 'Never')}")
            else:
                st.info("No workflow executions yet. Process some tickets or files to see workflow analytics.")
                
        except Exception as e:
            st.error(f"Error loading workflow analytics: {str(e)}")

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
    
    # Display processed tickets with draft responses
    if st.session_state.processed_tickets:
        st.markdown("---")
        
        # Create tabs for different views
        tab1, tab2 = st.tabs(["📊 Analysis Summary", "💬 Draft Responses"])
        
        with tab1:
            st.subheader("🎯 Recent Ticket Analysis")
            display_processed_tickets()
            
        with tab2:
            st.subheader("🤖 Generated Draft Responses")
            display_ticket_draft_responses()

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
                        # Use supervisor analysis for enhanced capabilities including escalation detection
                        result = st.session_state.agent.process_ticket_with_supervisor_analysis(
                            ticket_text, ticket_id or None
                        )
                    else:
                        # Demo mode - create mock result
                        result = create_mock_ticket_result(ticket_text, ticket_id)
                except Exception as e:
                    st.error(f"Error analyzing ticket: {str(e)}")
                    return
                
                # Display results
                if result:
                    st.session_state.processed_tickets.append(result)
                    display_ticket_analysis(result)
        else:
            st.warning("Please enter a ticket message to analyze.")

def create_mock_ticket_result(ticket_text: str, ticket_id: str = None) -> dict:
    """Create mock result for demo purposes"""
    return {
        'ticket_id': ticket_id or f"DEMO_{datetime.now().strftime('%Y%m%d%H%M%S')}",
        'original_text': ticket_text,
        'sentiment_analysis': {
            'sentiment_score': 0.2,
            'sentiment_category': 'neutral',
            'urgency_level': 'medium',
            'customer_mood': 'neutral'
        },
        'intent_classification': {
            'predicted_intent': 'general_inquiry',
            'confidence': 0.75,
            'intent_category': 'General'
        },
        'response_generation': {
            'suggested_response': f"Thank you for contacting us. We understand your inquiry and will assist you promptly.",
            'response_tone': 'professional',
            'estimated_resolution_time': '24 hours'
        },
        'processing_timestamp': datetime.now().isoformat(),
        'processing_time_ms': 1500
    }

def display_ticket_analysis(result: dict):
    """Display ticket analysis results"""
    if not result:
        return
        
    st.success(f"✅ Ticket {result.get('ticket_id', 'N/A')} analyzed successfully!")
    
    # Check for escalation alerts and supervisor insights
    supervisor_analysis = result.get('supervisor_analysis', {})
    escalation_analysis = supervisor_analysis.get('escalation_analysis', {})
    
    # Enhanced escalation highlighting
    if escalation_analysis.get('escalation_needed', False):
        escalation_level = escalation_analysis.get('level', 'medium')
        priority_score = escalation_analysis.get('priority_score', 5)
        confidence = escalation_analysis.get('confidence_score', 0)
        
        # Color-coded escalation alerts with enhanced visibility
        if escalation_level in ['critical', 'immediate']:
            st.markdown("""
            <div style='background-color: #ff4b4b; color: white; padding: 20px; border-radius: 10px; 
                        border-left: 10px solid #ff0000; margin: 10px 0; text-align: center;
                        box-shadow: 0 4px 8px rgba(255,75,75,0.3); animation: pulse 2s infinite;'>
                <h2>🆘 CRITICAL ESCALATION REQUIRED 🆘</h2>
                <h3>Level: {} | Priority: {}/10 | Confidence: {:.0%}</h3>
                <p><strong>IMMEDIATE SUPERVISOR ATTENTION NEEDED</strong></p>
            </div>
            <style>
            @keyframes pulse {{
                0% {{ transform: scale(1); }}
                50% {{ transform: scale(1.02); }}
                100% {{ transform: scale(1); }}
            }}
            </style>
            """.format(escalation_level.upper(), priority_score, confidence), unsafe_allow_html=True)
            
        elif escalation_level == 'high':
            st.markdown("""
            <div style='background-color: #ff8c00; color: white; padding: 15px; border-radius: 8px; 
                        border-left: 8px solid #ff6600; margin: 10px 0; text-align: center;
                        box-shadow: 0 3px 6px rgba(255,140,0,0.3);'>
                <h3>🚨 HIGH PRIORITY ESCALATION 🚨</h3>
                <h4>Level: {} | Priority: {}/10 | Confidence: {:.0%}</h4>
                <p><strong>Supervisor Review Required</strong></p>
            </div>
            """.format(escalation_level.upper(), priority_score, confidence), unsafe_allow_html=True)
            
        elif escalation_level == 'medium':
            st.markdown("""
            <div style='background-color: #ffa500; color: white; padding: 12px; border-radius: 6px; 
                        border-left: 6px solid #ff9500; margin: 10px 0; text-align: center;
                        box-shadow: 0 2px 4px rgba(255,165,0,0.3);'>
                <h4>⚠️ ESCALATION RECOMMENDED ⚠️</h4>
                <p>Level: {} | Priority: {}/10 | Confidence: {:.0%}</p>
            </div>
            """.format(escalation_level.upper(), priority_score, confidence), unsafe_allow_html=True)
        else:
            st.info(f"ℹ️ **Low Priority Escalation** - Level: {escalation_level.title()} | Priority: {priority_score}/10")
        
        # Enhanced escalation details with better organization
        with st.expander("📋 **DETAILED ESCALATION ANALYSIS**", expanded=True):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("### 🎯 **Key Metrics**")
                st.metric("Escalation Level", escalation_level.upper(), help="Urgency of escalation needed")
                st.metric("Priority Score", f"{priority_score}/10", help="Overall priority ranking")
                st.metric("Confidence", f"{confidence:.1%}", help="AI confidence in escalation recommendation")
            
            with col2:
                st.markdown("### 🔍 **Escalation Reasons**")
                reasons = escalation_analysis.get('reasons', [])
                if reasons:
                    for i, reason in enumerate(reasons, 1):
                        reason_formatted = reason.replace('_', ' ').title()
                        if reason.lower() in ['legal_threat', 'safety_concern', 'regulatory_compliance']:
                            st.markdown(f"🔴 **{i}.** {reason_formatted}")
                        elif reason.lower() in ['vip_customer', 'high_financial_impact', 'severe_negative_sentiment']:
                            st.markdown(f"🟠 **{i}.** {reason_formatted}")
                        else:
                            st.markdown(f"🟡 **{i}.** {reason_formatted}")
                else:
                    st.write("No specific reasons identified")
            
            with col3:
                st.markdown("### 📋 **Action Required**")
                recommended_action = escalation_analysis.get('recommended_action', 'Review with supervisor')
                st.write(f"**Next Steps:** {recommended_action}")
                
                suggested_assignee = escalation_analysis.get('suggested_assignee', 'Team Lead')
                st.write(f"**Assign To:** {suggested_assignee}")
                
                timeframe = escalation_analysis.get('timeframe', 'Standard')
                st.write(f"**Timeframe:** {timeframe}")
            
            # Risk factors with enhanced visualization
            risk_factors = escalation_analysis.get('risk_factors', [])
            if risk_factors:
                st.markdown("### ⚠️ **Risk Factors Identified**")
                for factor in risk_factors:
                    st.markdown(f"• {factor}")
            
            # Key indicators that triggered escalation
            indicators = escalation_analysis.get('extracted_indicators', [])
            if indicators:
                st.markdown("### 🎯 **Key Text Indicators**")
                for indicator in indicators:
                    st.markdown(f'> "{indicator}"')

    # Create tabs for different analysis aspects  
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Analysis", "💬 Response", "📈 Details", "🔍 Similar Cases", "👥 Supervisor"])
    
    with tab1:
        # Sentiment analysis
        sentiment = result.get('sentiment_analysis', {})
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            score = sentiment.get('sentiment_score', 0)
            st.metric("Sentiment Score", f"{score:.2f}", f"{score - 0.5:.2f}")
        
        with col2:
            category = sentiment.get('sentiment_category', 'neutral')
            st.metric("Category", category.title())
        
        with col3:
            urgency = sentiment.get('urgency_level', 'low')
            st.metric("Urgency", urgency.title())
        
        with col4:
            mood = sentiment.get('customer_mood', 'neutral')
            st.metric("Customer Mood", mood.title())
    
    with tab2:
        # Response generation with editable drafts
        response_data = result.get('response_generation', {})
        suggested_response = response_data.get('suggested_response', 'No response generated')
        
        st.subheader("💬 Response Suggestion Panel")
        
        # Response editing interface
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Editable response text area
            edited_response = st.text_area(
                "Draft Response (editable):",
                value=suggested_response,
                height=150,
                help="Edit the suggested response to match your style and requirements"
            )
            
            # Response options
            response_col1, response_col2 = st.columns(2)
            with response_col1:
                response_tone = st.selectbox(
                    "Response Tone:",
                    ["Professional", "Friendly", "Empathetic", "Formal", "Apologetic"],
                    index=0 if response_data.get('response_tone', 'professional') == 'professional' else 0
                )
                
                include_escalation = st.checkbox("Include escalation option", value=False)
            
            with response_col2:
                add_follow_up = st.checkbox("Add follow-up reminder", value=True)
                include_survey = st.checkbox("Include satisfaction survey", value=False)
        
        with col2:
            st.write("**Response Info:**")
            tone = response_data.get('response_tone', 'neutral')
            st.info(f"**Original Tone:** {tone.title()}")
            
            est_time = response_data.get('estimated_resolution_time', 'N/A')
            st.info(f"**Est. Resolution:** {est_time}")
            
            # Response templates
            st.write("**Quick Templates:**")
            
            templates = {
                "Apologetic": "I sincerely apologize for the inconvenience you've experienced...",
                "Grateful": "Thank you for bringing this to our attention...",
                "Resolution": "I'm happy to help resolve this issue for you...",
                "Follow-up": "I'll follow up with you within 24 hours..."
            }
            
            for template_name, template_text in templates.items():
                if st.button(f"Add {template_name}", key=f"template_{template_name}_{result.get('ticket_id', 'unknown')}"):
                    current_response = st.session_state.get(f"response_draft_{result.get('ticket_id')}", edited_response)
                    new_response = f"{current_response}\n\n{template_text}"
                    st.session_state[f"response_draft_{result.get('ticket_id')}"] = new_response
                    st.rerun()
        
        # Response actions
        st.markdown("---")
        action_col1, action_col2, action_col3, action_col4 = st.columns(4)
        
        with action_col1:
            if st.button("💾 Save Draft", key=f"save_{result.get('ticket_id', 'unknown')}"):
                # Save draft to session state
                ticket_id = result.get('ticket_id', 'unknown')
                if 'saved_drafts' not in st.session_state:
                    st.session_state.saved_drafts = {}
                
                st.session_state.saved_drafts[ticket_id] = {
                    'response': edited_response,
                    'tone': response_tone,
                    'timestamp': datetime.now(),
                    'ticket_data': result
                }
                st.success("Draft saved!")
        
        with action_col2:
            if st.button("📋 Copy Response", key=f"copy_{result.get('ticket_id', 'unknown')}"):
                # This would copy to clipboard in a real implementation
                st.info("Response copied to clipboard!")
        
        with action_col3:
            if st.button("📧 Preview Email", key=f"preview_{result.get('ticket_id', 'unknown')}"):
                # Show email preview
                with st.expander("📧 Email Preview"):
                    st.write("**Subject:** Re: Your Support Request")
                    st.write("**To:** customer@example.com")
                    st.write("**From:** support@yourcompany.com")
                    st.markdown("---")
                    st.write(edited_response)
                    if add_follow_up:
                        st.write("\nWe'll follow up with you to ensure your issue is fully resolved.")
                    if include_survey:
                        st.write("\nPlease take a moment to rate your support experience: [Survey Link]")
        
        with action_col4:
            if st.button("🚀 Send Response", key=f"send_{result.get('ticket_id', 'unknown')}", type="primary"):
                # In a real implementation, this would send the response
                st.success("✅ Response sent to customer!")
                
                # Log the sent response
                if 'sent_responses' not in st.session_state:
                    st.session_state.sent_responses = []
                
                st.session_state.sent_responses.append({
                    'ticket_id': result.get('ticket_id'),
                    'response': edited_response,
                    'tone': response_tone,
                    'sent_at': datetime.now()
                })
        
        # Response alternatives
        if response_data.get('alternative_responses'):
            st.markdown("---")
            st.subheader("🔄 Alternative Response Options")
            
            alternatives = response_data.get('alternative_responses', [])
            for i, alt in enumerate(alternatives):
                with st.expander(f"Alternative {i+1}: {alt.get('tone', 'Standard')} Tone"):
                    st.write(alt.get('response', 'No alternative available'))
                    if st.button(f"Use This Response", key=f"alt_{i}_{result.get('ticket_id', 'unknown')}"):
                        st.session_state[f"response_draft_{result.get('ticket_id')}"] = alt.get('response', '')
                        st.rerun()
    
    with tab3:
        # Intent classification and processing details
        intent_data = result.get('intent_classification', {})
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🎯 Intent Classification")
            st.write(f"**Intent:** {intent_data.get('predicted_intent', 'unknown')}")
            st.write(f"**Category:** {intent_data.get('intent_category', 'General')}")
            st.write(f"**Confidence:** {intent_data.get('confidence', 0):.2%}")
        
        with col2:
            st.subheader("⏱️ Processing Details")
            processing_time = result.get('processing_time_ms', 0)
            st.write(f"**Processing Time:** {processing_time}ms")
            timestamp = result.get('processing_timestamp', '')
            if timestamp:
                st.write(f"**Timestamp:** {timestamp}")
    
    with tab4:
        # Similar cases and resolutions
        st.subheader("🔍 Similar Past Cases")
        
        similar_cases = result.get('similar_cases', [])
        if similar_cases:
            st.write(f"Found {len(similar_cases)} similar cases:")
            for i, case in enumerate(similar_cases[:3]):  # Show top 3
                with st.expander(f"Similar Case {i+1}: {case.get('ticket_id', 'N/A')}"):
                    st.write(f"**Message:** {case.get('customer_message', 'N/A')[:200]}...")
                    st.write(f"**Category:** {case.get('category', 'N/A')}")
                    st.write(f"**Resolution:** {case.get('resolution_status', 'N/A')}")
                    if case.get('customer_satisfaction'):
                        st.write(f"**Satisfaction:** {'⭐' * case.get('customer_satisfaction', 0)}")
        else:
            st.info("No similar cases found in database")
        
        st.subheader("📚 Relevant Resolutions")
        relevant_resolutions = result.get('relevant_resolutions', [])
        if relevant_resolutions:
            for i, resolution in enumerate(relevant_resolutions):
                with st.expander(f"Resolution {i+1}: {resolution.get('issue_category', 'General')}"):
                    st.write(f"**Issue:** {resolution.get('issue_description', 'N/A')}")
                    st.write(f"**Steps:** {resolution.get('resolution_steps', 'N/A')}")
                    st.write(f"**Effectiveness:** {resolution.get('effectiveness_score', 0):.1%}")
                    st.write(f"**Used:** {resolution.get('used_count', 0)} times")
        else:
            st.info("No relevant resolutions found in knowledge base")
    
    with tab5:
        # Supervisor insights and metrics
        st.subheader("👥 Supervisor Insights")
        
        if supervisor_analysis:
            # Display response suggestions
            suggestions = supervisor_analysis.get('response_suggestions', {})
            if suggestions.get('suggestions'):
                st.write("**Alternative Response Options:**")
                for i, suggestion in enumerate(suggestions['suggestions'][:3], 1):
                    with st.expander(f"Response Option {i}: {suggestion.get('tone', 'Professional')} Tone"):
                        st.write(suggestion.get('response', 'No response available'))
                        st.write(f"**Confidence:** {suggestion.get('confidence', 0):.1%}")
            
            # SLA and priority information
            if escalation_analysis:
                st.write("**Priority Assessment:**")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    priority_score = escalation_analysis.get('priority_score', 5)
                    st.metric("Priority Score", f"{priority_score}/10")
                
                with col2:
                    timeframe = escalation_analysis.get('timeframe', '24 hours')
                    st.metric("Response Timeframe", timeframe)
                
                with col3:
                    suggested_assignee = escalation_analysis.get('suggested_assignee', 'Agent')
                    st.metric("Suggested Assignee", suggested_assignee)
                
                # Risk factors
                risk_factors = escalation_analysis.get('risk_factors', [])
                if risk_factors:
                    st.write("**Risk Factors:**")
                    for factor in risk_factors:
                        st.write(f"⚠️ {factor}")
        else:
            st.info("Supervisor analysis not available - enable supervisor tools for advanced insights")

def process_sample_tickets():
    """Process predefined sample tickets"""
    st.subheader("Sample Tickets")
    
    sample_tickets = [
        {
            'text': "I'm really frustrated! My order was supposed to arrive yesterday and it's still not here. This is the third time this has happened. I need a refund immediately!",
            'id': 'SAMPLE_001'
        },
        {
            'text': "Hello, I'd like to know more about your return policy. Specifically, how long do I have to return an item if I'm not satisfied with it?",
            'id': 'SAMPLE_002'
        },
        {
            'text': "URGENT: My payment was charged twice for the same order! I need this fixed ASAP as it's caused an overdraft in my account.",
            'id': 'SAMPLE_003'
        }
    ]
    
    for i, ticket in enumerate(sample_tickets):
        with st.expander(f"Sample Ticket {i+1}: {ticket['id']}"):
            st.text_area(f"Message {i+1}:", ticket['text'], height=100, disabled=True)
            
            if st.button(f"Analyze Sample {i+1}", key=f"sample_{i}"):
                with st.spinner(f"Analyzing sample ticket {i+1}..."):
                    try:
                        if st.session_state.agent_initialized and st.session_state.agent:
                            result = st.session_state.agent.process_support_ticket_with_workflow(
                                ticket['text'], ticket['id']
                            )
                        else:
                            result = create_mock_ticket_result(ticket['text'], ticket['id'])
                        
                        if result:
                            st.session_state.processed_tickets.append(result)
                            st.rerun()
                    except Exception as e:
                        st.error(f"Error processing sample ticket: {str(e)}")

def process_batch_tickets():
    """Process batch uploaded tickets"""
    st.subheader("Batch Upload Processing")
    st.info("Upload a CSV file with columns: 'text' (required) and 'id' (optional)")
    
    uploaded_file = st.file_uploader("Choose CSV file", type=['csv'])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file, parse_dates=False, date_format=None)
            
            if 'text' not in df.columns:
                st.error("CSV file must contain a 'text' column with ticket messages.")
                return
            
            st.write(f"Found {len(df)} tickets to process:")
            st.dataframe(df.head())
            
            if st.button("Process All Tickets", type="primary"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, row in df.iterrows():
                    status_text.text(f"Processing ticket {i+1}/{len(df)}...")
                    
                    ticket_text = row['text']
                    ticket_id = row.get('id', f'BATCH_{i+1}')
                    
                    try:
                        if st.session_state.agent_initialized and st.session_state.agent:
                            result = st.session_state.agent.process_support_ticket_with_workflow(
                                ticket_text, ticket_id
                            )
                        else:
                            result = create_mock_ticket_result(ticket_text, ticket_id)
                        
                        if result:
                            st.session_state.processed_tickets.append(result)
                    
                    except Exception as e:
                        st.error(f"Error processing ticket {i+1}: {str(e)}")
                    
                    progress_bar.progress((i + 1) / len(df))
                
                status_text.text(f"Completed processing {len(df)} tickets!")
                st.success(f"✅ Processed {len(df)} tickets successfully!")
                
                # Display draft responses immediately after processing
                st.subheader("📝 Generated Draft Responses")
                display_ticket_draft_responses()
                
        except Exception as e:
            st.error(f"Error reading CSV file: {str(e)}")

def display_ticket_draft_responses():
    """Display draft responses for uploaded and processed tickets with edit and approval functionality"""
    if not st.session_state.processed_tickets:
        st.info("No processed tickets with draft responses available. Upload and process tickets first.")
        return
    
    # Initialize approval tracking if not exists
    if 'response_approvals' not in st.session_state:
        st.session_state.response_approvals = {}
    
    # Bulk actions
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("✅ Approve All", help="Approve all draft responses"):
            for ticket in st.session_state.processed_tickets[-10:]:
                ticket_id = ticket.get('ticket_id', '')
                st.session_state.response_approvals[ticket_id] = {
                    'status': 'approved',
                    'final_response': ticket.get('response_generation', {}).get('suggested_response', ''),
                    'approved_at': datetime.now().isoformat()
                }
            st.success("All responses approved!")
            st.rerun()
    
    with col2:
        if st.button("❌ Reject All", help="Reject all draft responses"):
            for ticket in st.session_state.processed_tickets[-10:]:
                ticket_id = ticket.get('ticket_id', '')
                st.session_state.response_approvals[ticket_id] = {
                    'status': 'rejected',
                    'rejected_at': datetime.now().isoformat()
                }
            st.warning("All responses rejected!")
            st.rerun()
    
    with col3:
        approved_count = sum(1 for v in st.session_state.response_approvals.values() if v.get('status') == 'approved')
        st.metric("Approved", approved_count)
    
    with col4:
        pending_count = len(st.session_state.processed_tickets[-10:]) - len([k for k in st.session_state.response_approvals.keys() 
                                                                           if any(ticket.get('ticket_id') == k for ticket in st.session_state.processed_tickets[-10:])])
        st.metric("Pending", pending_count)
    
    st.markdown("---")
    
    # Export approved responses section
    approved_count = sum(1 for v in st.session_state.response_approvals.values() if v.get('status') == 'approved')
    if approved_count > 0:
        st.subheader(f"📤 Export {approved_count} Approved Responses")
        export_approved_responses()
        st.markdown("---")
    
    st.write(f"**{len(st.session_state.processed_tickets)} tickets processed with draft responses:**")
    
    for i, ticket in enumerate(st.session_state.processed_tickets[-10:], 1):  # Show last 10 tickets
        ticket_id = ticket.get('ticket_id', f'#{i}')
        approval_data = st.session_state.response_approvals.get(ticket_id, {})
        status = approval_data.get('status', 'pending')
        
        # Status indicator
        status_emoji = {"approved": "✅", "rejected": "❌", "modified": "✏️", "pending": "⏳"}
        status_color = {"approved": "green", "rejected": "red", "modified": "orange", "pending": "gray"}
        
        with st.expander(f"{status_emoji.get(status, '⏳')} Ticket {ticket_id} - {ticket.get('sentiment_analysis', {}).get('sentiment_category', 'neutral').title()} Sentiment [{status.title()}]"):
            # Original ticket text
            original_text = ticket.get('original_text', 'No original text available')
            st.text_area("Original Ticket:", original_text, height=100, disabled=True, key=f"orig_{i}")
            
            # Draft response editing
            response_gen = ticket.get('response_generation', {})
            original_response = response_gen.get('suggested_response', 'No draft response generated')
            
            # Show current response (edited or original)
            current_response = approval_data.get('final_response', original_response)
            
            st.markdown("### 💬 **Draft Response:**")
            
            # Edit mode toggle
            edit_mode = st.checkbox(f"✏️ Edit Response", key=f"edit_mode_{i}", value=status == 'modified')
            
            if edit_mode:
                # Editable text area
                edited_response = st.text_area(
                    "Edit Response:", 
                    current_response, 
                    height=150, 
                    key=f"edit_draft_{i}",
                    help="Modify the response as needed, then click Update to save changes"
                )
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button(f"💾 Update Response", key=f"update_{i}"):
                        st.session_state.response_approvals[ticket_id] = {
                            'status': 'modified',
                            'final_response': edited_response,
                            'original_response': original_response,
                            'modified_at': datetime.now().isoformat()
                        }
                        st.success("Response updated!")
                        st.rerun()
                
                with col2:
                    if st.button(f"🔄 Reset to Original", key=f"reset_{i}"):
                        if ticket_id in st.session_state.response_approvals:
                            del st.session_state.response_approvals[ticket_id]
                        st.info("Reset to original response!")
                        st.rerun()
            else:
                # Read-only display
                st.text_area("Current Response:", current_response, height=150, disabled=True, key=f"readonly_draft_{i}")
            
            # Response approval actions
            st.markdown("### 🎯 **Response Actions:**")
            action_col1, action_col2, action_col3 = st.columns(3)
            
            with action_col1:
                if st.button(f"✅ Approve", key=f"approve_{i}", disabled=status == 'approved'):
                    st.session_state.response_approvals[ticket_id] = {
                        'status': 'approved',
                        'final_response': current_response,
                        'original_response': original_response,
                        'approved_at': datetime.now().isoformat()
                    }
                    st.success(f"Response approved for ticket {ticket_id}!")
                    st.rerun()
            
            with action_col2:
                if st.button(f"❌ Reject", key=f"reject_{i}", disabled=status == 'rejected'):
                    st.session_state.response_approvals[ticket_id] = {
                        'status': 'rejected',
                        'original_response': original_response,
                        'rejected_at': datetime.now().isoformat(),
                        'rejection_reason': 'Manual rejection'
                    }
                    st.warning(f"Response rejected for ticket {ticket_id}!")
                    st.rerun()
            
            with action_col3:
                if status != 'pending':
                    if st.button(f"↩️ Reset", key=f"reset_status_{i}"):
                        if ticket_id in st.session_state.response_approvals:
                            del st.session_state.response_approvals[ticket_id]
                        st.info(f"Status reset for ticket {ticket_id}!")
                        st.rerun()
            
            # Status information
            if status != 'pending':
                st.markdown("#### 📊 **Status Information:**")
                info_col1, info_col2 = st.columns(2)
                with info_col1:
                    st.write(f"**Status:** :{status_color[status]}[{status.title()}]")
                    if status == 'approved':
                        st.write(f"**Approved:** {approval_data.get('approved_at', 'Unknown')[:19]}")
                    elif status == 'rejected':
                        st.write(f"**Rejected:** {approval_data.get('rejected_at', 'Unknown')[:19]}")
                    elif status == 'modified':
                        st.write(f"**Modified:** {approval_data.get('modified_at', 'Unknown')[:19]}")
                
                with info_col2:
                    if status == 'modified':
                        st.write("**Changes:** Response was edited")
                    elif status == 'approved':
                        st.write("**Action:** Ready to send")
                    elif status == 'rejected':
                        st.write("**Action:** Needs new response")
            
            # Response details
            col1, col2, col3 = st.columns(3)
            with col1:
                tone = response_gen.get('response_tone', 'professional')
                st.metric("Response Tone", tone.title())
            with col2:
                resolution_time = response_gen.get('estimated_resolution_time', 'Unknown')
                st.metric("Est. Resolution", resolution_time)
            with col3:
                confidence = response_gen.get('confidence', 0)
                st.metric("Confidence", f"{confidence:.1%}" if isinstance(confidence, (int, float)) else str(confidence))
            
            # Analysis results
            if ticket.get('sentiment_analysis'):
                sentiment = ticket['sentiment_analysis']
                st.markdown("**Analysis:**")
                analysis_col1, analysis_col2 = st.columns(2)
                with analysis_col1:
                    st.write(f"• **Sentiment:** {sentiment.get('sentiment_category', 'neutral').title()}")
                    st.write(f"• **Urgency:** {sentiment.get('urgency_level', 'low').title()}")
                with analysis_col2:
                    intent = ticket.get('intent_classification', {})
                    st.write(f"• **Intent:** {intent.get('predicted_intent', 'unknown')}")
                    st.write(f"• **Priority:** {ticket.get('priority', 'Normal')}")
            
            # Alternative responses if available
            alternatives = response_gen.get('alternative_responses', [])
            if alternatives:
                st.markdown("**Alternative Response Options:**")
                for j, alt in enumerate(alternatives[:2], 1):  # Show up to 2 alternatives
                    alt_col1, alt_col2 = st.columns([3, 1])
                    with alt_col1:
                        with st.expander(f"Alternative {j}: {alt.get('tone', 'Unknown')} Tone"):
                            st.write(alt.get('response', 'No alternative response'))
                    with alt_col2:
                        if st.button(f"Use Alt {j}", key=f"use_alt_{i}_{j}"):
                            st.session_state.response_approvals[ticket_id] = {
                                'status': 'modified',
                                'final_response': alt.get('response', ''),
                                'original_response': original_response,
                                'modified_at': datetime.now().isoformat(),
                                'modification_source': f'alternative_{j}'
                            }
                            st.success(f"Switched to alternative response {j}!")
                            st.rerun()

def export_approved_responses():
    """Export approved responses"""
    if 'response_approvals' not in st.session_state:
        return
    
    approved_responses = []
    for ticket in st.session_state.processed_tickets:
        ticket_id = ticket.get('ticket_id', '')
        approval_data = st.session_state.response_approvals.get(ticket_id, {})
        
        if approval_data.get('status') == 'approved':
            approved_responses.append({
                'ticket_id': ticket_id,
                'original_ticket': ticket.get('original_text', ''),
                'final_response': approval_data.get('final_response', ''),
                'response_tone': ticket.get('response_generation', {}).get('response_tone', 'professional'),
                'approved_at': approval_data.get('approved_at', ''),
                'sentiment': ticket.get('sentiment_analysis', {}).get('sentiment_category', 'neutral'),
                'urgency': ticket.get('sentiment_analysis', {}).get('urgency_level', 'low'),
                'intent': ticket.get('intent_classification', {}).get('predicted_intent', 'unknown'),
                'was_modified': approval_data.get('original_response') != approval_data.get('final_response')
            })
    
    if approved_responses:
        df = pd.DataFrame(approved_responses)
        
        # Download CSV
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download Approved Responses (CSV)",
            data=csv,
            file_name=f"approved_responses_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
        
        # Download JSON
        import json
        json_data = json.dumps(approved_responses, indent=2)
        st.download_button(
            label="📥 Download Approved Responses (JSON)",
            data=json_data,
            file_name=f"approved_responses_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
        
        # Store in persistent store if agent available
        if st.session_state.agent_initialized and st.session_state.agent:
            if st.button("💾 Save to Persistent Store"):
                try:
                    import uuid
                    saved_count = 0
                    for response in approved_responses:
                        # Create Resolution object for the persistent store
                        
                        resolution_data = {
                            'id': str(uuid.uuid4()),
                            'complaint_id': str(response['ticket_id']),  # Ensure string
                            'title': f"Approved Response for {response['ticket_id']}",
                            'description': str(response['final_response']),
                            'solution_steps': [str(response['final_response'])],  # List of strings
                            'resolution_type': 'manual',  # Use valid enum value
                            'category': str(response['intent']),
                            'effectiveness_score': float(0.9),  # Ensure float
                            'resolution_time_hours': float(1.0),  # Ensure float
                            'created_by': 'streamlit_user',
                            'created_at': datetime.now()
                        }
                        
                        resolution_id = st.session_state.agent.persistent_store.store.add_resolution(resolution_data)
                        if resolution_id:
                            saved_count += 1
                    
                    st.success(f"✅ Saved {saved_count} approved responses to persistent store!")
                except Exception as e:
                    st.error(f"Error saving to persistent store: {str(e)}")
        
        return df
    else:
        st.info("No approved responses to export.")
        return None

def display_processed_tickets():
    """Display all processed tickets"""
    if not st.session_state.processed_tickets:
        return
    
    # Create summary dataframe
    tickets_data = []
    for ticket in st.session_state.processed_tickets:
        sentiment = ticket.get('sentiment_analysis', {})
        intent = ticket.get('intent_classification', {})
        
        tickets_data.append({
            'ID': ticket.get('ticket_id', 'N/A'),
            'Sentiment': sentiment.get('sentiment_category', 'neutral').title(),
            'Urgency': sentiment.get('urgency_level', 'low').title(),
            'Intent': intent.get('predicted_intent', 'unknown'),
            'Confidence': f"{intent.get('confidence', 0):.1%}",
            'Timestamp': ticket.get('processing_timestamp', '')[:19] if ticket.get('processing_timestamp') else ''
        })
    
    df = pd.DataFrame(tickets_data)
    st.dataframe(df, use_container_width=True)
    
    # Clear processed tickets button
    if st.button("Clear All Results"):
        st.session_state.processed_tickets = []
        st.rerun()

def file_management_page():
    """Streamlined file management page"""
    st.header("📁 File Upload & Management")
    
    # Create simplified tabs
    tab1, tab2 = st.tabs(["📂 Upload Files", "📊 Analytics Data"])
    
    with tab1:
        st.subheader("📂 Universal File Upload")
        st.info("Upload any support-related files: tickets (CSV), policies (PDF/DOCX), chat logs (TXT/JSON), or other documents. The system will automatically detect and process them appropriately.")
        
        uploaded_files = st.file_uploader(
            "Choose files to upload",
            type=['csv', 'json', 'txt', 'xlsx', 'pdf', 'docx', 'md'],
            accept_multiple_files=True,
            key="all_files",
            help="Supported formats: CSV (tickets), PDF/DOCX (policies), TXT/JSON (chat logs), MD (documentation)"
        )
        
        if uploaded_files:
            # Automatically categorize files by type
            file_categories = {'tickets': [], 'policies': [], 'chat_logs': [], 'other': []}
            
            for file in uploaded_files:
                if file.name.endswith(('.csv', '.xlsx')):
                    file_categories['tickets'].append(file)
                elif file.name.endswith(('.pdf', '.docx', '.md')):
                    file_categories['policies'].append(file)
                elif file.name.endswith(('.txt', '.json')):
                    file_categories['chat_logs'].append(file)
                else:
                    file_categories['other'].append(file)
            
            # Display categorized files
            total_files = len(uploaded_files)
            st.write(f"**📁 {total_files} files uploaded and categorized:**")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("🎫 Tickets", len(file_categories['tickets']))
            with col2:
                st.metric("📄 Policies", len(file_categories['policies']))
            with col3:
                st.metric("💬 Chat Logs", len(file_categories['chat_logs']))
            with col4:
                st.metric("📎 Other", len(file_categories['other']))
            
            # File preview section
            with st.expander(f"📋 File Details ({total_files} files)"):
                for uploaded_file in uploaded_files:
                    with st.expander(f"📄 {uploaded_file.name}"):
                        st.write(f"**Size:** {uploaded_file.size:,} bytes")
                        st.write(f"**Type:** {uploaded_file.type}")
                        
                        # Preview file content
                        if uploaded_file.name.endswith('.csv'):
                            try:
                                df = pd.read_csv(uploaded_file, parse_dates=False, date_format=None)
                                st.write(f"**Rows:** {len(df)}")
                                st.write(f"**Columns:** {list(df.columns)}")
                                st.dataframe(df.head(3))
                            except Exception as e:
                                st.error(f"Error reading CSV: {str(e)}")
                        
                        elif uploaded_file.name.endswith('.json'):
                            try:
                                content = json.loads(uploaded_file.read().decode())
                                st.write(f"**Records:** {len(content) if isinstance(content, list) else 'Single object'}")
                                st.json(content if not isinstance(content, list) else content[0])
                            except Exception as e:
                                st.error(f"Error reading JSON: {str(e)}")
                        
                        elif uploaded_file.name.endswith(('.pdf', '.docx')):
                            st.info("Document preview - content will be processed when uploaded")
                        
                        else:
                            st.info("File will be processed based on content type")
            
            # Smart processing options
            st.markdown("### 🔧 Processing Options")
            col1, col2 = st.columns(2)
            with col1:
                generate_responses = st.checkbox("🤖 Generate Draft Responses", value=True, 
                                               help="Generate draft responses for support tickets found in CSV files")
                extract_insights = st.checkbox("📊 Extract Insights", value=True,
                                             help="Analyze sentiment, urgency, and categorize content")
            with col2:
                add_to_knowledge = st.checkbox("📚 Add to Knowledge Base", value=True,
                                             help="Store processed content in searchable knowledge base")
                generate_summaries = st.checkbox("📝 Generate Summaries", value=False,
                                                help="Create summaries for policy documents and chat logs")
            
            if st.button("🚀 Process All Files", type="primary"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                ticket_responses_generated = 0
                knowledge_files_processed = 0
                
                # Process files by category
                all_files = uploaded_files
                for i, uploaded_file in enumerate(all_files):
                    file_ext = uploaded_file.name.split('.')[-1].lower()
                    status_text.text(f"Processing {uploaded_file.name} ({file_ext.upper()})...")
                    
                    # Determine file type and processing approach
                    is_ticket_csv = False
                    if file_ext in ['csv', 'xlsx'] and generate_responses:
                        uploaded_file.seek(0)
                        try:
                            # Read CSV with proper date parsing to avoid warnings
                            df = pd.read_csv(uploaded_file, parse_dates=False, date_format=None)
                            # Check if it looks like a ticket file (has 'text' or 'description' or 'message' column)
                            ticket_columns = ['text', 'description', 'message', 'content', 'complaint', 'issue']
                            if any(col.lower() in [c.lower() for c in df.columns] for col in ticket_columns):
                                is_ticket_csv = True
                        except:
                            pass
                    
                    if is_ticket_csv:
                        # Process individual tickets and generate draft responses
                        status_text.text(f"Generating draft responses for tickets in {uploaded_file.name}...")
                        uploaded_file.seek(0)
                        df = pd.read_csv(uploaded_file, parse_dates=False, date_format=None)
                        
                        # Find the text column
                        text_col = None
                        for col in df.columns:
                            if col.lower() in ['text', 'description', 'message', 'content', 'complaint', 'issue']:
                                text_col = col
                                break
                        
                        if text_col:
                            ticket_progress = st.progress(0)
                            for j, row in df.iterrows():
                                ticket_text = str(row[text_col])
                                if ticket_text and ticket_text.strip() and ticket_text != 'nan':
                                    ticket_id = row.get('id', row.get('ticket_id', f'{uploaded_file.name}_ticket_{j+1}'))
                                    
                                    try:
                                        if st.session_state.agent_initialized and st.session_state.agent:
                                            result = st.session_state.agent.process_support_ticket_with_workflow(
                                                ticket_text, str(ticket_id)
                                            )
                                            if result:
                                                st.session_state.processed_tickets.append(result)
                                                ticket_responses_generated += 1
                                    except Exception as e:
                                        st.warning(f"Could not process ticket {j+1}: {str(e)}")
                                
                                ticket_progress.progress((j + 1) / len(df))
                            
                            st.success(f"✅ Generated draft responses for {ticket_responses_generated} tickets from {uploaded_file.name}")
                        else:
                            st.warning(f"Could not find ticket text column in {uploaded_file.name}")
                    
                    # Also process the file normally for knowledge base
                    if st.session_state.agent_initialized and st.session_state.agent:
                        try:
                            # Reset file pointer to beginning
                            uploaded_file.seek(0)
                            
                            # Try UTF-8 first, fallback to other encodings if needed
                            try:
                                file_content = uploaded_file.read().decode('utf-8')
                            except UnicodeDecodeError:
                                uploaded_file.seek(0)
                                file_content = uploaded_file.read().decode('latin-1')
                            
                            result = st.session_state.agent.process_file_with_workflow(
                                file_content, uploaded_file.name, uploaded_file.type
                            )
                            
                            if result.get('status') == 'completed':
                                st.success(f"✅ {uploaded_file.name} processed successfully with workflow!")
                                
                                # Show workflow processing results
                                workflow_data = result.get('workflow_data', {})
                                artifacts = workflow_data.get('data_artifacts', {})
                                
                                st.write(f"  • Created {artifacts.get('chunks_created', 0)} chunks")
                                st.write(f"  • Generated {artifacts.get('embeddings_count', 0)} embeddings")
                                st.write(f"  • Workflow ID: {result.get('workflow_id', 'N/A')}")
                                
                                # Show execution summary
                                exec_summary = result.get('execution_summary', {})
                                st.write(f"  • Completed {exec_summary.get('completed_tasks', 0)} workflow tasks")
                                
                                # Show knowledge base storage status
                                if result.get('knowledge_base_stored'):
                                    st.write(f"  • ✅ Added to knowledge base ({result.get('records_processed', 0)} records)")
                                    # Update session state counter and file type
                                    st.session_state.kb_file_count += 1
                                    st.session_state.last_kb_update = datetime.now()
                                    
                                    # Track file type - support logs are typically CSV/TXT
                                    file_ext = uploaded_file.name.split('.')[-1].lower()
                                    if file_ext in st.session_state.kb_file_types:
                                        st.session_state.kb_file_types[file_ext] += 1
                                else:
                                    st.write(f"  • ❌ Knowledge base storage failed")
                                    if result.get('kb_error'):
                                        st.write(f"    Error: {result.get('kb_error')}")
                                
                            else:
                                st.error(f"❌ Error processing {uploaded_file.name}: {result.get('error', 'Unknown error')}")
                        except Exception as e:
                            st.error(f"Error processing {uploaded_file.name}: {str(e)}")
                    else:
                        st.info(f"Demo mode: {uploaded_file.name} would be processed")
                    
                    progress_bar.progress((i + 1) / len(all_files))
                
                status_text.text("Processing complete!")
                progress_bar.progress(1.0)
                
                # Show comprehensive summary
                summary_col1, summary_col2 = st.columns(2)
                with summary_col1:
                    st.metric("📁 Files Processed", len(all_files))
                    st.metric("🎫 Draft Responses", ticket_responses_generated)
                with summary_col2:
                    st.metric("📚 Knowledge Base", knowledge_files_processed)
                    st.metric("✅ Success Rate", f"{((len(all_files) - 0) / len(all_files) * 100):.0f}%")
                
                # Show results based on what was processed
                if ticket_responses_generated > 0:
                    st.success(f"✅ Successfully processed {len(all_files)} file(s) and generated {ticket_responses_generated} draft responses!")
                    # Display the generated draft responses
                    st.subheader("🤖 Generated Draft Responses")
                    display_ticket_draft_responses()
                elif knowledge_files_processed > 0:
                    st.success(f"✅ Successfully processed {len(all_files)} file(s) and added {knowledge_files_processed} to knowledge base!")
                else:
                    st.success(f"✅ Successfully processed {len(all_files)} file(s)!")
                
                # Force UI refresh to update knowledge base stats
                time.sleep(1)  # Brief pause for user to see success message
                st.rerun()
    
    with tab2:
        st.subheader("📊 Analytics & Data Import")
        st.info("Upload additional data for analytics and reporting")
        
        analytics_files = st.file_uploader(
            "Choose analytics files",
            type=['csv', 'xlsx', 'json'],
            accept_multiple_files=True,
            key="analytics_data"
        )
        
        if analytics_files:
            for uploaded_file in analytics_files:
                st.write(f"📊 {uploaded_file.name} - {uploaded_file.size:,} bytes")
            
            if st.button("📈 Process Analytics Data"):
                st.info("Analytics data processing feature - ready for implementation")
    
    # Current knowledge base status
    st.markdown("---")
    st.subheader("📚 Current Knowledge Base Status")
    
    if st.session_state.agent_initialized and st.session_state.agent:
        try:
            status = st.session_state.agent.get_agent_status()
            kb_stats = status.get('knowledge_base_stats', {})
            
            # Use session state counter as primary source, agent status as backup
            total_files = max(st.session_state.kb_file_count, kb_stats.get('total_files_processed', 0))
            
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                delta_text = None
                if st.session_state.last_kb_update:
                    time_since = datetime.now() - st.session_state.last_kb_update
                    if time_since.total_seconds() < 60:  # Show delta if updated within last minute
                        delta_text = f"+{st.session_state.kb_file_count}"
                
                st.metric("Total Files", total_files, 
                         delta=delta_text,
                         help=f"Total files processed and stored in knowledge base. Last updated: {st.session_state.last_kb_update.strftime('%H:%M:%S') if st.session_state.last_kb_update else 'Never'}")
            with col2:
                # Use session state file types with agent status as backup
                agent_file_types = kb_stats.get('file_types_distribution', {})
                session_types = st.session_state.kb_file_types
                
                # Support logs: primarily CSV files from support tab
                support_logs = max(
                    session_types.get('csv', 0),
                    agent_file_types.get('csv', 0)
                )
                st.metric("Support Logs", support_logs, 
                         help="CSV files containing support ticket data")
            with col3:
                policy_docs = max(
                    session_types.get('pdf', 0) + session_types.get('docx', 0) + session_types.get('md', 0),
                    agent_file_types.get('pdf', 0) + agent_file_types.get('docx', 0) + agent_file_types.get('md', 0)
                )
                st.metric("Policy Docs", policy_docs,
                         help="PDF, DOCX, and MD policy documents")
            with col4:
                # Internal notes: primarily TXT files from internal notes tab
                internal_notes = max(
                    session_types.get('txt', 0) + session_types.get('json', 0),
                    agent_file_types.get('txt', 0) + agent_file_types.get('json', 0)
                )
                st.metric("Internal Notes", internal_notes,
                         help="TXT, JSON, and internal communication files")
            with col5:
                # Try to get vector count from workflow analytics if available
                if hasattr(st.session_state.agent, 'get_workflow_analytics'):
                    workflow_analytics = st.session_state.agent.get_workflow_analytics()
                    vector_count = workflow_analytics.get('total_workflows', 0) * 10  # Estimate
                else:
                    vector_count = total_files * 5  # Rough estimate
                st.metric("Vector Records", vector_count,
                         help="Estimated number of vector embeddings created")
                
        except Exception as e:
            st.error(f"Error getting knowledge base status: {str(e)}")
            
        # Add refresh button for manual updates
        col1, col2 = st.columns([3, 1])
        with col1:
            if st.button("🔄 Refresh Knowledge Base Status", help="Manually refresh the knowledge base statistics"):
                st.rerun()
        with col2:
            if st.button("🔧 Debug Info"):
                st.json({
                    "session_kb_count": st.session_state.kb_file_count,
                    "agent_kb_count": kb_stats.get('total_files_processed', 0),
                    "session_file_types": st.session_state.kb_file_types,
                    "agent_file_types": kb_stats.get('file_types_distribution', {}),
                    "last_update": str(st.session_state.last_kb_update),
                    "agent_status": status.get('knowledge_base_files', 0),
                    "calculated_support_logs": session_types.get('csv', 0),
                    "calculated_policy_docs": session_types.get('pdf', 0) + session_types.get('docx', 0) + session_types.get('md', 0),
                    "calculated_internal_notes": session_types.get('txt', 0) + session_types.get('json', 0)
                })
    else:
        st.info("Agent not initialized - knowledge base status unavailable")

def knowledge_search_page():
    """Comprehensive knowledge base search with chat interface"""
    st.header("🔍 Knowledge Base Search & Chat")
    
    # Create tabs for different search modes
    tab1, tab2, tab3 = st.tabs(["💬 Chat Interface", "🔍 Advanced Search", "📊 Analytics Queries"])
    
    with tab1:
        st.subheader("Chat with Your Support Data")
        st.info("Ask questions in natural language about your support logs and customer data")
        
        # Initialize chat history
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        # Predefined example queries
        st.write("**Example queries you can try:**")
        example_col1, example_col2 = st.columns(2)
        
        with example_col1:
            if st.button("📊 Summarize refund-related complaints from last week"):
                example_query = "Summarize refund-related complaints from last week"
                st.session_state.chat_input = example_query
            
            if st.button("📈 What are top 3 customer pain points this month?"):
                example_query = "What are top 3 customer pain points this month?"
                st.session_state.chat_input = example_query
        
        with example_col2:
                st.rerun()
    
    with tab3:
        st.subheader("Internal Notes & Chat Logs Upload")
        st.info("Upload internal team notes, chat logs, and communication records")
        
        # File upload for internal communications
        internal_files = st.file_uploader(
            "Choose internal communication files",
            type=['txt', 'csv', 'json', 'md', 'docx'],
            accept_multiple_files=True,
            key="internal_notes",
            help="Upload internal notes, chat logs, team communications, or meeting notes"
        )
        
        if internal_files:
            st.write(f"Selected {len(internal_files)} internal communication files:")
            
            for uploaded_file in internal_files:
                with st.expander(f"💬 {uploaded_file.name}"):
                    st.write(f"**Size:** {uploaded_file.size:,} bytes")
                    st.write(f"**Type:** {uploaded_file.type}")
                    
                    # Preview content based on file type
                    try:
                        uploaded_file.seek(0)
                        if uploaded_file.name.endswith('.json'):
                            content = json.loads(uploaded_file.read().decode())
                            st.write(f"**Records:** {len(content) if isinstance(content, list) else 'Single object'}")
                            st.json(content if not isinstance(content, list) else content[0])
                        elif uploaded_file.name.endswith('.csv'):
                            df = pd.read_csv(uploaded_file, parse_dates=False, date_format=None)
                            st.write(f"**Rows:** {len(df)}, **Columns:** {list(df.columns)}")
                            st.dataframe(df.head(3))
                        else:
                            # Text-based files
                            content = uploaded_file.read().decode('utf-8')[:500]
                            st.text_area("Preview:", content, height=100, disabled=True)
                    except Exception as e:
                        st.warning(f"Could not preview file: {str(e)}")
        
            # Processing options for internal communications
            col1, col2 = st.columns(2)
            with col1:
                extract_topics = st.checkbox("Extract key topics", value=True)
                identify_action_items = st.checkbox("Identify action items", value=True)
            with col2:
                track_decisions = st.checkbox("Track decisions made", value=True)
                categorize_communications = st.checkbox("Categorize by type", value=True)
            
            if st.button("💬 Process Internal Communications", type="primary"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, uploaded_file in enumerate(internal_files):
                    status_text.text(f"Processing {uploaded_file.name}...")
                    
                    if st.session_state.agent_initialized and st.session_state.agent:
                        try:
                            # Reset file pointer and read content
                            uploaded_file.seek(0)
                            try:
                                file_content = uploaded_file.read().decode('utf-8')
                            except UnicodeDecodeError:
                                uploaded_file.seek(0)
                                file_content = uploaded_file.read().decode('latin-1')
                            
                            # Process with workflow orchestration
                            result = st.session_state.agent.process_file_with_workflow(
                                file_content, uploaded_file.name, uploaded_file.type or "text/plain"
                            )
                            
                            if result.get('status') == 'completed':
                                st.success(f"✅ {uploaded_file.name} processed successfully!")
                                
                                # Show processing results
                                workflow_data = result.get('workflow_data', {})
                                artifacts = workflow_data.get('data_artifacts', {})
                                
                                st.write(f"  • Created {artifacts.get('chunks_created', 0)} text chunks")
                                st.write(f"  • Generated {artifacts.get('embeddings_count', 0)} embeddings")
                                st.write(f"  • Workflow ID: {result.get('workflow_id', 'N/A')}")
                                
                                # Update knowledge base tracking
                                if result.get('knowledge_base_stored'):
                                    st.write(f"  • ✅ Added to knowledge base")
                                    st.session_state.kb_file_count += 1
                                    st.session_state.last_kb_update = datetime.now()
                                    
                                    # Track as internal notes type
                                    file_ext = uploaded_file.name.split('.')[-1].lower()
                                    if file_ext in st.session_state.kb_file_types:
                                        st.session_state.kb_file_types[file_ext] += 1
                                else:
                                    st.write(f"  • ❌ Knowledge base storage failed")
                                
                            else:
                                st.error(f"❌ Error processing {uploaded_file.name}: {result.get('error', 'Unknown error')}")
                                
                        except Exception as e:
                            st.error(f"Error processing {uploaded_file.name}: {str(e)}")
                    else:
                        st.info(f"Demo mode: {uploaded_file.name} would be processed")
                    
                    progress_bar.progress((i + 1) / len(internal_files))
                
                status_text.text("Processing complete!")
                progress_bar.progress(1.0)
                
                # Force UI refresh
                st.success(f"✅ Successfully processed {len(internal_files)} internal communication file(s)!")
                time.sleep(1)
                st.rerun()
        
        # Text input for quick notes
        st.markdown("---")
        st.subheader("📝 Quick Internal Note Entry")
        st.caption("Add internal notes directly without uploading files")
        
        note_title = st.text_input("Note Title:", placeholder="e.g., Team Meeting Notes - 2024-01-15")
        note_content = st.text_area("Note Content:", height=150, 
                                   placeholder="Enter internal notes, decisions, action items, or team communications...")
        
        if st.button("💾 Save Internal Note") and note_content:
            if st.session_state.agent_initialized and st.session_state.agent:
                try:
                    # Create note with metadata
                    note_with_metadata = f"Title: {note_title or 'Untitled Note'}\n"
                    note_with_metadata += f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                    note_with_metadata += f"Type: Internal Note\n\n"
                    note_with_metadata += note_content
                    
                    # Process the note using workflow
                    result = st.session_state.agent.process_file_with_workflow(
                        note_with_metadata, f"internal_note_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt", "text/plain"
                    )
                    
                    if result.get('status') == 'completed':
                        st.success("✅ Internal note saved successfully!")
                        st.session_state.kb_file_count += 1
                        st.session_state.last_kb_update = datetime.now()
                        st.session_state.kb_file_types['txt'] += 1
                        
                        # Clear the form
                        st.session_state.note_title = ""
                        st.session_state.note_content = ""
                        st.rerun()
                    else:
                        st.error(f"❌ Failed to save note: {result.get('error', 'Unknown error')}")
                        
                except Exception as e:
                    st.error(f"Error saving note: {str(e)}")
            else:
                st.info("Demo mode: Note would be saved to knowledge base")
    
    with tab4:
        st.subheader("Analytics & Reports")
        st.info("Upload additional data for analytics and reporting")
        
        analytics_files = st.file_uploader(
            "Choose analytics files",
            type=['csv', 'xlsx', 'json'],
            accept_multiple_files=True,
            key="analytics_data"
        )
        
        if analytics_files:
            for uploaded_file in analytics_files:
                st.write(f"📊 {uploaded_file.name} - {uploaded_file.size:,} bytes")
            
            if st.button("📈 Process Analytics Data"):
                st.info("Analytics data processing feature - ready for implementation")
    
    # Current knowledge base status
    st.markdown("---")
    st.subheader("📚 Current Knowledge Base Status")
    
    if st.session_state.agent_initialized and st.session_state.agent:
        try:
            status = st.session_state.agent.get_agent_status()
            kb_stats = status.get('knowledge_base_stats', {})
            
            # Use session state counter as primary source, agent status as backup
            total_files = max(st.session_state.kb_file_count, kb_stats.get('total_files_processed', 0))
            
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                delta_text = None
                if st.session_state.last_kb_update:
                    time_since = datetime.now() - st.session_state.last_kb_update
                    if time_since.total_seconds() < 60:  # Show delta if updated within last minute
                        delta_text = f"+{st.session_state.kb_file_count}"
                
                st.metric("Total Files", total_files, 
                         delta=delta_text,
                         help=f"Total files processed and stored in knowledge base. Last updated: {st.session_state.last_kb_update.strftime('%H:%M:%S') if st.session_state.last_kb_update else 'Never'}")
            with col2:
                # Use session state file types with agent status as backup
                agent_file_types = kb_stats.get('file_types_distribution', {})
                session_types = st.session_state.kb_file_types
                
                # Support logs: primarily CSV files from support tab
                support_logs = max(
                    session_types.get('csv', 0),
                    agent_file_types.get('csv', 0)
                )
                st.metric("Support Logs", support_logs, 
                         help="CSV files containing support ticket data")
            with col3:
                policy_docs = max(
                    session_types.get('pdf', 0) + session_types.get('docx', 0) + session_types.get('md', 0),
                    agent_file_types.get('pdf', 0) + agent_file_types.get('docx', 0) + agent_file_types.get('md', 0)
                )
                st.metric("Policy Docs", policy_docs,
                         help="PDF, DOCX, and MD policy documents")
            with col4:
                # Internal notes: primarily TXT files from internal notes tab
                internal_notes = max(
                    session_types.get('txt', 0) + session_types.get('json', 0),
                    agent_file_types.get('txt', 0) + agent_file_types.get('json', 0)
                )
                st.metric("Internal Notes", internal_notes,
                         help="TXT, JSON, and internal communication files")
            with col5:
                # Try to get vector count from workflow analytics if available
                if hasattr(st.session_state.agent, 'get_workflow_analytics'):
                    workflow_analytics = st.session_state.agent.get_workflow_analytics()
                    vector_count = workflow_analytics.get('total_workflows', 0) * 10  # Estimate
                else:
                    vector_count = total_files * 5  # Rough estimate
                st.metric("Vector Records", vector_count,
                         help="Estimated number of vector embeddings created")
                
        except Exception as e:
            st.error(f"Error getting knowledge base status: {str(e)}")
            
        # Add refresh button for manual updates
        col1, col2 = st.columns([3, 1])
        with col1:
            if st.button("🔄 Refresh Knowledge Base Status", help="Manually refresh the knowledge base statistics"):
                st.rerun()
        with col2:
            if st.button("🔧 Debug Info"):
                st.json({
                    "session_kb_count": st.session_state.kb_file_count,
                    "agent_kb_count": kb_stats.get('total_files_processed', 0),
                    "session_file_types": st.session_state.kb_file_types,
                    "agent_file_types": kb_stats.get('file_types_distribution', {}),
                    "last_update": str(st.session_state.last_kb_update),
                    "agent_status": status.get('knowledge_base_files', 0),
                    "calculated_support_logs": session_types.get('csv', 0),
                    "calculated_policy_docs": session_types.get('pdf', 0) + session_types.get('docx', 0) + session_types.get('md', 0),
                    "calculated_internal_notes": session_types.get('txt', 0) + session_types.get('json', 0)
                })
    else:
        st.info("Agent not initialized - knowledge base status unavailable")

def knowledge_search_page():
    """Comprehensive knowledge base search with chat interface"""
    st.header("🔍 Knowledge Base Search & Chat")
    
    # Create tabs for different search modes
    tab1, tab2, tab3 = st.tabs(["💬 Chat Interface", "🔍 Advanced Search", "📊 Analytics Queries"])
    
    with tab1:
        st.subheader("Chat with Your Support Data")
        st.info("Ask questions in natural language about your support logs and customer data")
        
        # Initialize chat history
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        # Predefined example queries
        st.write("**Example queries you can try:**")
        example_col1, example_col2 = st.columns(2)
        
        with example_col1:
            if st.button("📊 Summarize refund-related complaints from last week"):
                example_query = "Summarize refund-related complaints from last week"
                st.session_state.chat_input = example_query
            
            if st.button("📈 What are top 3 customer pain points this month?"):
                example_query = "What are top 3 customer pain points this month?"
                st.session_state.chat_input = example_query
        
        with example_col2:
            if st.button("🚨 Show me all high urgency tickets today"):
                example_query = "Show me all high urgency tickets today"
                st.session_state.chat_input = example_query
            
            if st.button("📋 How many billing disputes were resolved successfully?"):
                example_query = "How many billing disputes were resolved successfully?"
                st.session_state.chat_input = example_query
        
        # Chat input
        chat_query = st.text_input(
            "Ask a question about your support data:",
            placeholder="e.g., 'What are the most common issues this week?' or 'Show me frustrated customers'",
            key="chat_input"
        )
        
        col1, col2 = st.columns([3, 1])
        with col1:
            if st.button("💬 Ask", type="primary") and chat_query:
                #st.write("🐛 DEBUG: Ask button clicked!")
                #st.write(f"🐛 DEBUG: Query = '{chat_query}'")
                
                # Add user message to history
                st.session_state.chat_history.append({
                    "role": "user",
                    "content": chat_query,
                    "timestamp": datetime.now()
                })
                
                #st.write("🐛 DEBUG: Added to chat history")
                
                # Process the query
                #st.write("🐛 DEBUG: About to start processing query")
                
                try:
                    with st.spinner("Analyzing your question..."):
                        #st.write("🐛 DEBUG: Inside spinner")
                        
                        # DEBUG: Show initialization status
                        #st.write(f"🐛 DEBUG: agent_initialized = {st.session_state.agent_initialized}")
                        #st.write(f"🐛 DEBUG: agent exists = {st.session_state.agent is not None}")
                        
                        if st.session_state.agent_initialized and st.session_state.agent:
                            #st.write("🐛 DEBUG: Agent check passed - entering agent processing")
                            try:
                                # Use search functionality to process natural language query
                                has_nlq_method = hasattr(st.session_state.agent, 'process_natural_language_query')
                                #st.write(f"🐛 DEBUG: Has process_natural_language_query method: {has_nlq_method}")
                                
                                if has_nlq_method:
                                    #st.write("🐛 DEBUG: Using process_natural_language_query method")
                                    response = st.session_state.agent.process_natural_language_query(chat_query)
                                    #st.write(f"🐛 DEBUG: NLQ response: {response[:100]}...")
                                else:
                                    # Fallback: use search knowledge base for natural language queries
                                    #st.write(f"🐛 DEBUG: About to call search_knowledge_base with query: '{chat_query}'")
                                    
                                    search_results = st.session_state.agent.search_knowledge_base(chat_query)
                                    #st.write(f"🐛 DEBUG: search_results = {search_results}")
                                    
                                    # Generate a response based on search results
                                    total_results = search_results.get('total_results', 0)
                                    #st.write(f"🐛 DEBUG: total_results = {total_results}")
                                    similar_tickets = search_results.get('similar_tickets', [])
                                    #st.write(f"🐛 DEBUG: similar_tickets count = {len(similar_tickets)}")
                                    
                                    if total_results > 0:
                                        response = f"I found {total_results} relevant results for your query '{chat_query}'. Based on the data, here's what I can tell you:\n\n"
                                        
                                        # Add insights from search results
                                        if similar_tickets:
                                            response += f"• Found {len(similar_tickets)} similar tickets\n"
                                            top_categories = {}
                                            for ticket in similar_tickets[:5]:
                                                category = ticket.get('category', 'Unknown')
                                                top_categories[category] = top_categories.get(category, 0) + 1
                                            
                                            if top_categories:
                                                response += f"• Top categories: {', '.join(top_categories.keys())}\n"
                                        
                                        resolutions = search_results.get('relevant_resolutions', [])
                                        if resolutions:
                                            response += f"• Found {len(resolutions)} relevant resolution procedures"
                                    else:
                                        response = f"I couldn't find specific data matching your query '{chat_query}'. This might be because:\n• No tickets have been processed yet\n• The query needs to be more specific\n• The data hasn't been uploaded to the system"
                                
                                # Add response to history
                                st.session_state.chat_history.append({
                                    "role": "assistant", 
                                    "content": response,
                                    "timestamp": datetime.now()
                                })
                                
                            except Exception as e:
                                error_response = f"I encountered an error processing your query: {str(e)}"
                                st.session_state.chat_history.append({
                                    "role": "assistant",
                                    "content": error_response,
                                    "timestamp": datetime.now()
                                })
                        else:
                            #st.write("🐛 DEBUG: Agent check FAILED - entering demo mode")
                            #st.write(f"🐛 DEBUG: agent_initialized = {st.session_state.agent_initialized}")
                            #st.write(f"🐛 DEBUG: agent = {st.session_state.agent}")
                            
                            # Demo response
                            demo_response = f"Demo mode: I would analyze your data to answer '{chat_query}'. This feature requires agent initialization with your actual support data."
                            st.session_state.chat_history.append({
                                "role": "assistant",
                                "content": demo_response,
                                "timestamp": datetime.now()
                            })
                
                except Exception as outer_e:
                 #   st.error(f"🐛 DEBUG: Outer exception occurred: {outer_e}")
                 #   st.write("🐛 DEBUG: Full traceback:")
                    import traceback
                    st.code(traceback.format_exc())
                
                # DON'T auto-refresh so we can see debug messages
                #st.write("🐛 DEBUG: Processing complete - debug messages above")
        
        with col2:
            if st.button("🔄 Refresh Chat"):
                st.rerun()
            if st.button("🗑️ Clear Chat"):
                st.session_state.chat_history = []
                st.rerun()
        
        # Display chat history
        if st.session_state.chat_history:
            st.markdown("---")
            st.subheader("💬 Conversation")
            
            for i, message in enumerate(reversed(st.session_state.chat_history[-10:])):  # Show last 10 messages
                if message["role"] == "user":
                    st.write(f"**👤 You ({message['timestamp'].strftime('%H:%M')}):**")
                    st.write(message["content"])
                else:
                    st.write(f"**🤖 Assistant ({message['timestamp'].strftime('%H:%M')}):**")
                    st.write(message["content"])
                
                if i < len(st.session_state.chat_history[-10:]) - 1:
                    st.write("")
    
    with tab2:
        st.subheader("Advanced Knowledge Base Search")
        
        if not st.session_state.agent_initialized:
            st.warning("⚠️ Agent not fully initialized. Search functionality limited.")
        
        # Search interface
        query = st.text_input("Search query:", placeholder="Enter your search terms...")
        
        # Search help
        with st.expander("💡 Search Tips"):
            st.write("""
            **Try searching for:**
            - Keywords: "order delay", "password reset", "billing issue", "app crash"
            - Customer emotions: "frustrated", "angry", "satisfied", "happy" 
            - Product issues: "login problem", "payment error", "refund request"
            - Categories: Use category filter to narrow down results
            
            **Available resolutions in knowledge base:**
            - Order delay handling procedures
            - Password reset instructions
            - Billing dispute resolution
            - Technical troubleshooting steps
            - Customer feedback responses
            """)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            search_type = st.selectbox("Search type:", ["semantic", "hybrid", "metadata"])
        with col2:
            category_filter = st.selectbox("Category:", ["All Categories", "General", "Account", "Billing", "Technical", "Product"])
        with col3:
            top_k = st.number_input("Max results:", min_value=1, max_value=20, value=5)
        
        if st.button("🔍 Search") and query:
            with st.spinner("Searching knowledge base..."):
                if st.session_state.agent_initialized and st.session_state.agent:
                    # Real search using database
                    category_to_search = None if category_filter == "All Categories" else category_filter
                    search_results = st.session_state.agent.search_knowledge_base(query, category_to_search)
                    
                    similar_tickets = search_results.get('similar_tickets', [])
                    resolutions = search_results.get('relevant_resolutions', [])
                    total_results = search_results.get('total_results', 0)
                    
                    if total_results > 0:
                        st.success(f"Found {total_results} results")
                        
                        if similar_tickets:
                            st.subheader("📋 Similar Tickets")
                            for i, ticket in enumerate(similar_tickets):
                                with st.expander(f"Ticket {i+1}: {ticket.get('ticket_id', 'N/A')}"):
                                    # Get the message content from either customer_message or content field
                                    message = ticket.get('customer_message', '') or ticket.get('content', '')
                                    if message:
                                        st.write(f"**Message:** {message[:200]}...")
                                    else:
                                        st.write("**Message:** No content available")
                                    st.write(f"**Category:** {ticket.get('category', 'N/A')}")
                                    st.write(f"**Urgency:** {ticket.get('urgency_level', 'N/A')}")
                                    st.write(f"**Status:** {ticket.get('resolution_status', 'N/A')}")
                                    if ticket.get('customer_satisfaction'):
                                        st.write(f"**Satisfaction:** {'⭐' * ticket.get('customer_satisfaction', 0)}")
                        
                        if resolutions:
                            st.subheader("💡 Relevant Resolutions")
                            for i, resolution in enumerate(resolutions):
                                with st.expander(f"Resolution {i+1}: {resolution.get('issue_category', 'General')}"):
                                    st.write(f"**Issue:** {resolution.get('issue_description', '')}")
                                    st.write(f"**Steps:** {resolution.get('resolution_steps', '')}")
                                    st.write(f"**Effectiveness:** {resolution.get('effectiveness_score', 0):.1%}")
                                    st.write(f"**Times Used:** {resolution.get('used_count', 0)}")
                    else:
                        st.info("No results found for your query.")
                        
                else:
                    # Demo mode
                    st.info("Demo mode: Database search requires agent initialization")
                    mock_results = [
                        {
                            'content': f"Mock result for query: {query}",
                            'source': 'demo_database',
                            'relevance_score': 0.85
                        }
                    ]
                    
                    for i, result in enumerate(mock_results):
                        with st.expander(f"Demo Result {i+1}"):
                            st.write(f"**Content:** {result['content']}")
                            st.write(f"**Source:** {result['source']}")
                            st.write(f"**Score:** {result['relevance_score']:.2f}")
    
    with tab3:
        st.subheader("📊 Analytics & Reporting Queries")
        st.info("Predefined analytics queries for common business insights")
        
        # Predefined analytics queries
        analytics_queries = [
            {
                "title": "📈 Weekly Ticket Volume Trend",
                "description": "Shows ticket volume by day for the last 7 days",
                "query": "weekly_ticket_volume"
            },
            {
                "title": "😡 Customer Satisfaction Analysis",
                "description": "Breakdown of customer satisfaction scores and trends",
                "query": "satisfaction_analysis"
            },
            {
                "title": "🔥 Top Issues by Category",
                "description": "Most common issues grouped by category",
                "query": "top_issues_by_category"
            },
            {
                "title": "⚡ Response Time Performance", 
                "description": "Average response times by category and urgency",
                "query": "response_time_analysis"
            },
            {
                "title": "💰 Refund Request Analysis",
                "description": "Refund request patterns and approval rates",
                "query": "refund_analysis"
            }
        ]
        
        for query_item in analytics_queries:
            with st.expander(f"{query_item['title']}"):
                st.write(query_item['description'])
                
                if st.button(f"Run Analysis", key=f"analytics_{query_item['query']}"):
                    if st.session_state.agent_initialized and st.session_state.agent:
                        with st.spinner("Generating analytics..."):
                            try:
                                # Check if the method exists, otherwise create a demo response
                                if hasattr(st.session_state.agent, 'run_analytics_query'):
                                    result = st.session_state.agent.run_analytics_query(query_item['query'])
                                else:
                                    # Fallback: generate demo analytics based on query type
                                    result = generate_demo_analytics(query_item['query'], query_item['title'])
                                
                                if result.get('status') == 'success':
                                    st.success("✅ Analysis complete!")
                                    
                                    # Display results based on query type
                                    if 'chart_data' in result:
                                        st.plotly_chart(result['chart_data'], use_container_width=True)
                                    
                                    if 'summary' in result:
                                        st.write("**Summary:**")
                                        st.write(result['summary'])
                                    
                                    if 'data' in result:
                                        st.dataframe(result['data'])
                                
                                else:
                                    st.error(f"Analysis failed: {result.get('error', 'Unknown error')}")
                            
                            except Exception as e:
                                st.error(f"Error running analysis: {str(e)}")
                    else:
                        st.info(f"Demo mode: {query_item['title']} analysis would be generated here")

def settings_page():
    """Application settings page"""
    st.header("⚙️ Settings")
    
    # Agent settings
    st.subheader("🤖 Agent Configuration")
    
    with st.expander("Environment Variables"):
        st.code("""
# Required environment variables:
OPENAI_API_KEY=your_openai_api_key_here
HUGGINGFACE_API_KEY=your_huggingface_key_here (optional)
        """)
    
    # Processing settings
    st.subheader("⚙️ Processing Settings")
    
    col1, col2 = st.columns(2)
    with col1:
        st.number_input("Max file size (MB):", min_value=1, max_value=100, value=10)
        st.selectbox("Default response tone:", ["professional", "friendly", "formal", "empathetic"])
    
    with col2:
        st.number_input("Response timeout (seconds):", min_value=5, max_value=60, value=30)
        st.checkbox("Enable detailed logging", value=True)
    
    # System information
    st.subheader("📊 System Information")
    
    if st.session_state.agent_initialized and st.session_state.agent:
        try:
            status = st.session_state.agent.get_agent_status()
            st.json(status)
        except Exception as e:
            st.error(f"Error getting agent status: {str(e)}")
    else:
        st.info("Agent not initialized - status unavailable")
    
    # Actions
    st.subheader("🔧 Actions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Restart Agent"):
            # Clear agent from session state to force reinitialization
            if 'agent' in st.session_state:
                del st.session_state.agent
            if 'agent_initialized' in st.session_state:
                del st.session_state.agent_initialized
            st.success("Agent restart initiated. Please refresh the page.")
    
    with col2:
        if st.button("🧹 Clear Cache"):
            st.cache_data.clear()
            st.success("Cache cleared successfully.")
    
    with col3:
        if st.button("📋 Export Logs"):
            st.info("Log export functionality - demo mode")

def historical_data_page():
    """Historical data and analytics page"""
    st.header("📈 Historical Data & Analytics")
    
    if not st.session_state.agent_initialized:
        st.warning("⚠️ Agent not initialized - showing demo data")
    
    # Time period selector
    col1, col2 = st.columns(2)
    with col1:
        days_back = st.selectbox("Time Period:", [7, 30, 90, 365], index=1)
    with col2:
        auto_refresh = st.checkbox("Auto-refresh data", value=False)
    
    if st.button("🔄 Refresh Data") or auto_refresh:
        if st.session_state.agent_initialized and st.session_state.agent:
            # Get real historical data
            with st.spinner("Loading historical data..."):
                historical_data = st.session_state.agent.get_historical_data(days_back)
                
                stats = historical_data.get('statistics', {})
                recent_tickets = historical_data.get('recent_tickets', [])
                trends = historical_data.get('trends', {})
                recommendations = historical_data.get('recommendations', [])
                
                # Display statistics
                st.subheader("📊 Statistics Overview")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Tickets", stats.get('total_tickets', 0))
                with col2:
                    st.metric("Resolution Rate", f"{stats.get('resolution_rate', 0):.1f}%")
                with col3:
                    st.metric("Avg Satisfaction", f"{stats.get('average_satisfaction', 0):.1f}/5")
                with col4:
                    st.metric("Period", f"Last {days_back} days")
                
                # Category distribution
                category_dist = stats.get('category_distribution', {})
                if category_dist:
                    st.subheader("📋 Ticket Categories")
                    
                    # Create pie chart
                    fig = px.pie(
                        values=list(category_dist.values()),
                        names=list(category_dist.keys()),
                        title="Tickets by Category"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Urgency distribution
                urgency_dist = stats.get('urgency_distribution', {})
                if urgency_dist:
                    st.subheader("🚨 Urgency Levels")
                    
                    # Create bar chart
                    fig = px.bar(
                        x=list(urgency_dist.keys()),
                        y=list(urgency_dist.values()),
                        title="Tickets by Urgency Level",
                        color=list(urgency_dist.keys()),
                        color_discrete_map={
                            'low': 'green',
                            'medium': 'orange', 
                            'high': 'red',
                            'critical': 'darkred'
                        }
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Trends analysis
                if trends:
                    st.subheader("📈 Trend Analysis")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if 'most_common_category' in trends:
                            most_common = trends['most_common_category']
                            st.info(f"**Most Common Category:** {most_common['category']} ({most_common['percentage']}%)")
                        
                        if 'high_urgency_percentage' in trends:
                            st.warning(f"**High Urgency:** {trends['high_urgency_percentage']}% of tickets")
                    
                    with col2:
                        if 'resolution_trend' in trends:
                            trend_color = {
                                'excellent': '🟢',
                                'good': '🟡', 
                                'needs_improvement': '🔴'
                            }
                            st.info(f"**Resolution Trend:** {trend_color.get(trends['resolution_trend'], '⚪')} {trends['resolution_trend'].title()}")
                
                # Persistent Store Analytics (if available)
                if hasattr(st.session_state.agent, 'persistent_enabled') and st.session_state.agent.persistent_enabled:
                    st.markdown("---")
                    st.subheader("🗄️ Persistent Store Analytics")
                    
                    try:
                        persistent_insights = st.session_state.agent.get_persistent_store_insights()
                        if persistent_insights and 'project_stats' in persistent_insights:
                            stats = persistent_insights['project_stats']
                            
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Stored Complaints", stats.get('total_complaints', 0))
                            with col2:
                                st.metric("Resolution Templates", stats.get('total_resolutions', 0))
                            with col3:
                                st.metric("Knowledge Articles", stats.get('total_knowledge_articles', 0))
                            with col4:
                                st.metric("This Week", stats.get('complaints_this_week', 0))
                        
                        # Analytics data
                        if persistent_insights:
                            analytics = persistent_insights.get('analytics', {})
                        else:
                            analytics = {}
                        if analytics:
                            complaint_stats = analytics.get('complaints', {})
                            if complaint_stats:
                                st.subheader("📊 Complaint Resolution Analytics")
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    total_complaints = complaint_stats.get('total_complaints', 0) or 0
                                    resolved = complaint_stats.get('resolved_complaints', 0) or 0
                                    if total_complaints and total_complaints > 0:
                                        resolution_rate = (resolved / total_complaints) * 100
                                        st.metric("Resolution Rate", f"{resolution_rate:.1f}%")
                                        avg_time = complaint_stats.get('avg_resolution_time', 0) or 0
                                        st.metric("Avg Resolution Time", f"{avg_time:.1f}h")
                                
                                with col2:
                                    avg_satisfaction = complaint_stats.get('avg_satisfaction', 0) or 0
                                    if avg_satisfaction and avg_satisfaction > 0:
                                        st.metric("Customer Satisfaction", f"{avg_satisfaction:.1f}/5")
                        
                        # Category trends
                        categories = analytics.get('categories', [])
                        if categories:
                            st.subheader("📈 Category Trends")
                            category_data = pd.DataFrame(categories)
                            if not category_data.empty:
                                fig = px.bar(
                                    category_data, 
                                    x='category', 
                                    y='count',
                                    title="Complaints by Category",
                                    color='count',
                                    color_continuous_scale='Blues'
                                )
                                st.plotly_chart(fig, use_container_width=True)
                        
                        # Top resolutions
                        top_resolutions = analytics.get('top_resolutions', [])
                        if top_resolutions:
                            st.subheader("🏆 Most Effective Resolutions")
                            for i, res in enumerate(top_resolutions[:5], 1):
                                with st.expander(f"{i}. {res.get('title', 'Untitled')} (Used {res.get('used_count', 0)} times)"):
                                    st.write(f"**Effectiveness Score:** {res.get('effectiveness_score', 0):.1%}")
                                    st.write(f"**Usage Count:** {res.get('used_count', 0)}")
                    
                    except Exception as e:
                        st.error(f"Error loading persistent store data: {str(e)}")
                
                # Recommendations
                if recommendations:
                    st.subheader("💡 Recommendations")
                    for rec in recommendations:
                        st.info(f"• {rec}")
                
                # Recent tickets
                if recent_tickets:
                    st.subheader("🕐 Recent Tickets")
                    
                    # Convert to DataFrame for display
                    tickets_df = pd.DataFrame(recent_tickets)
                    if not tickets_df.empty:
                        # Format the DataFrame
                        display_columns = ['ticket_id', 'category', 'urgency_level', 'resolution_status', 'created_at']
                        available_columns = [col for col in display_columns if col in tickets_df.columns]
                        
                        if available_columns:
                            st.dataframe(
                                tickets_df[available_columns].head(10),
                                use_container_width=True
                            )
                
                # Export options
                st.subheader("📤 Export Data")
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("Export as JSON"):
                        try:
                            export_path = st.session_state.agent.export_historical_data('json')
                            st.success(f"Data exported to: {export_path}")
                        except Exception as e:
                            st.error(f"Export failed: {str(e)}")
                
                with col2:
                    if st.button("Export as CSV"):
                        try:
                            export_path = st.session_state.agent.export_historical_data('csv')
                            st.success(f"Data exported to: {export_path}")
                        except Exception as e:
                            st.error(f"Export failed: {str(e)}")
        
        else:
            # Demo mode with mock data
            st.subheader("📊 Demo Statistics")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Tickets", 150)
            with col2:
                st.metric("Resolution Rate", "85.2%")
            with col3:
                st.metric("Avg Satisfaction", "4.1/5")
            with col4:
                st.metric("Period", f"Last {days_back} days")
            
            # Demo charts
            st.subheader("📋 Demo Categories")
            demo_categories = {'Technical': 45, 'Billing': 30, 'Account': 25, 'General': 50}
            fig = px.pie(values=list(demo_categories.values()), names=list(demo_categories.keys()))
            st.plotly_chart(fig, use_container_width=True)
            
            st.info("💡 Initialize the agent to see real historical data from the database")
    
    # Knowledge base management
    st.markdown("---")
    st.subheader("📚 Knowledge Base Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Add Resolution to Knowledge Base**")
        with st.form("add_resolution"):
            issue_category = st.selectbox("Category:", ["Technical", "Billing", "Account", "General", "Product"])
            issue_desc = st.text_area("Issue Description:")
            resolution_steps = st.text_area("Resolution Steps:")
            effectiveness = st.slider("Effectiveness Score:", 0.0, 1.0, 0.8)
            
            if st.form_submit_button("Add Resolution"):
                if st.session_state.agent_initialized and st.session_state.agent:
                    success = st.session_state.agent.add_resolution_to_knowledge_base(
                        issue_category, issue_desc, resolution_steps, effectiveness
                    )
                    if success:
                        st.success("Resolution added to knowledge base!")
                    else:
                        st.error("Failed to add resolution")
                else:
                    st.info("Agent not initialized - resolution not saved")
    
    with col2:
        st.write("**Update Ticket Resolution**")
        with st.form("update_ticket"):
            ticket_id = st.text_input("Ticket ID:")
            actual_response = st.text_area("Actual Response Sent:")
            satisfaction = st.selectbox("Customer Satisfaction:", [None, 1, 2, 3, 4, 5])
            
            if st.form_submit_button("Update Ticket"):
                if st.session_state.agent_initialized and st.session_state.agent and ticket_id:
                    success = st.session_state.agent.update_ticket_resolution(
                        ticket_id, actual_response, satisfaction
                    )
                    if success:
                        st.success("Ticket updated successfully!")
                    else:
                        st.error("Failed to update ticket - check ticket ID")
                else:
                    st.warning("Please provide ticket ID and ensure agent is initialized")

def persistent_store_page():
    """Persistent store management and analytics page"""
    st.header("🗄️ Persistent Store Management")
    
    if not st.session_state.agent_initialized:
        st.warning("⚠️ Agent not initialized - persistent store features unavailable")
        return
    
    # Check if persistent store is enabled
    agent = st.session_state.agent
    if not hasattr(agent, 'persistent_enabled') or not agent.persistent_enabled:
        st.error("❌ Persistent store not enabled in the agent")
        st.info("The persistent store provides cross-session storage of complaints and resolutions for learning and analytics.")
        return
    
    st.success("✅ Persistent store is active and ready")
    
    # Main tabs for different persistent store features
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Analytics", "🔍 Search History", "💡 Resolution Templates", "📤 Export/Import", "📁 File Management", "⚙️ Management"
    ])
    
    with tab1:
        st.subheader("📊 Persistent Store Analytics")
        
        if st.button("🔄 Refresh Analytics"):
            with st.spinner("Loading persistent store analytics..."):
                try:
                    insights = agent.get_persistent_store_insights()
                    
                    if 'error' in insights:
                        st.error(f"Error loading analytics: {insights['error']}")
                        return
                    
                    # Project statistics
                    project_stats = insights.get('project_stats', {})
                    if project_stats:
                        st.subheader("📈 Project Statistics")
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Total Complaints", project_stats.get('total_complaints', 0))
                        with col2:
                            st.metric("Resolution Templates", project_stats.get('total_resolutions', 0))
                        with col3:
                            st.metric("Knowledge Articles", project_stats.get('total_knowledge_articles', 0))
                        with col4:
                            st.metric("This Week", project_stats.get('complaints_this_week', 0))
                    
                    # Analytics overview
                    analytics = insights.get('analytics', {})
                    if analytics:
                        st.subheader("📊 Analytics Overview")
                        
                        # Complaints analytics
                        complaints = analytics.get('complaints', {})
                        if complaints:
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                total = complaints.get('total_complaints', 0) or 0
                                resolved = complaints.get('resolved_complaints', 0) or 0
                                if total and total > 0:
                                    resolution_rate = (resolved / total) * 100
                                    st.metric("Resolution Rate", f"{resolution_rate:.1f}%")
                            
                            with col2:
                                avg_time = complaints.get('avg_resolution_time', 0) or 0
                                st.metric("Avg Resolution Time", f"{avg_time:.1f}h")
                            
                            with col3:
                                satisfaction = complaints.get('avg_satisfaction', 0) or 0
                                if satisfaction and satisfaction > 0:
                                    st.metric("Avg Satisfaction", f"{satisfaction:.1f}/5")
                        
                        # Category trends
                        categories = analytics.get('categories', [])
                        if categories:
                            st.subheader("📈 Category Distribution")
                            category_df = pd.DataFrame(categories)
                            
                            if not category_df.empty:
                                fig = px.pie(
                                    category_df,
                                    values='count',
                                    names='category',
                                    title="Complaints by Category"
                                )
                                st.plotly_chart(fig, use_container_width=True)
                        
                        # Top resolutions
                        top_resolutions = analytics.get('top_resolutions', [])
                        if top_resolutions:
                            st.subheader("🏆 Most Effective Resolutions")
                            for i, res in enumerate(top_resolutions[:5], 1):
                                with st.expander(f"{i}. {res.get('title', 'Untitled')} (Effectiveness: {res.get('effectiveness_score', 0):.1%})"):
                                    st.write(f"**Used:** {res.get('used_count', 0)} times")
                                    st.write(f"**Effectiveness:** {res.get('effectiveness_score', 0):.1%}")
                    
                    # Trends and recommendations
                    trends = insights.get('trends', {})
                    if trends:
                        st.subheader("📈 Recent Trends")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric("Recent Volume", trends.get('recent_volume', 0))
                            st.metric("Volume Change", trends.get('volume_change', 0))
                        
                        with col2:
                            direction = trends.get('trend_direction', 'stable')
                            direction_emoji = {'increasing': '📈', 'decreasing': '📉', 'stable': '➡️'}
                            st.info(f"Trend Direction: {direction_emoji.get(direction, '➡️')} {direction.title()}")
                    
                    # Recommendations
                    recommendations = insights.get('recommendations', [])
                    if recommendations:
                        st.subheader("💡 AI Recommendations")
                        for rec in recommendations:
                            st.info(f"• {rec}")
                
                except Exception as e:
                    st.error(f"Error loading persistent store analytics: {str(e)}")
    
    with tab2:
        st.subheader("🔍 Historical Case Search")
        
        # Search interface
        col1, col2 = st.columns([3, 1])
        with col1:
            search_query = st.text_input("Search historical cases:", placeholder="e.g., billing issue, refund request")
        with col2:
            search_category = st.selectbox("Category:", ["All", "billing", "refund", "delivery", "technical", "general"])
        
        limit = st.slider("Max results:", 1, 20, 5)
        
        if st.button("🔍 Search Historical Cases") and search_query:
            with st.spinner("Searching historical cases..."):
                try:
                    category_filter = None if search_category == "All" else search_category
                    similar_cases = agent.get_similar_historical_cases(search_query, category_filter, limit)
                    
                    if similar_cases:
                        st.success(f"Found {len(similar_cases)} similar historical cases")
                        
                        for i, case in enumerate(similar_cases, 1):
                            with st.expander(f"Case {i}: {case.get('title', 'Untitled')[:50]}... (Similarity: {case.get('similarity_score', 0):.1%})"):
                                st.write(f"**Category:** {case.get('category', 'Unknown')}")
                                st.write(f"**Status:** {case.get('status', 'Unknown')}")
                                st.write(f"**Created:** {case.get('created_at', 'Unknown')}")
                                st.write(f"**Description:** {case.get('description', 'No description')[:200]}...")
                                
                                sentiment_score = case.get('sentiment_score')
                                if sentiment_score is not None:
                                    sentiment = "Positive" if sentiment_score > 0 else "Negative"
                                    st.write(f"**Sentiment:** {sentiment} ({sentiment_score:.2f})")
                                
                                # Show resolution if available
                                if case.get('resolution_time_hours'):
                                    st.write(f"**Resolution Time:** {case['resolution_time_hours']:.1f} hours")
                    else:
                        st.info("No similar historical cases found. Try different keywords or broader search terms.")
                
                except Exception as e:
                    st.error(f"Error searching historical cases: {str(e)}")
    
    with tab3:
        st.subheader("💡 Resolution Template Management")
        
        # Create new template
        with st.expander("➕ Create New Resolution Template"):
            with st.form("create_template"):
                template_category = st.selectbox("Category:", ["billing", "refund", "delivery", "technical", "general", "account"])
                template_title = st.text_input("Template Title:", placeholder="e.g., Standard Refund Process")
                template_description = st.text_area("Description:", placeholder="Brief description of when to use this template")
                
                st.write("**Solution Steps:**")
                step1 = st.text_input("Step 1:", placeholder="First action to take")
                step2 = st.text_input("Step 2:", placeholder="Second action to take")
                step3 = st.text_input("Step 3:", placeholder="Third action to take (optional)")
                step4 = st.text_input("Step 4:", placeholder="Fourth action to take (optional)")
                
                if st.form_submit_button("Create Template"):
                    if template_title and template_description and step1:
                        solution_steps = [step for step in [step1, step2, step3, step4] if step.strip()]
                        
                        try:
                            template_id = agent.create_resolution_template(
                                template_category, template_title, template_description, solution_steps
                            )
                            if template_id:
                                st.success(f"✅ Template created successfully! ID: {template_id}")
                            else:
                                st.error("Failed to create template")
                        except Exception as e:
                            st.error(f"Error creating template: {str(e)}")
                    else:
                        st.warning("Please fill in at least the title, description, and first step")
        
        # Show existing templates
        st.write("**Existing Templates:**")
        if st.button("🔄 Load Templates"):
            try:
                # Get recommended resolutions as templates
                for category in ["billing", "refund", "delivery", "technical", "general"]:
                    resolutions = agent.get_recommended_resolutions(category, limit=3)
                    if resolutions:
                        st.write(f"**{category.title()} Templates:**")
                        for res in resolutions:
                            with st.expander(f"{res.get('title', 'Untitled')} (Effectiveness: {res.get('effectiveness_score', 0):.1%})"):
                                st.write(f"**Description:** {res.get('description', 'No description')}")
                                st.write(f"**Category:** {res.get('category', 'Unknown')}")
                                st.write(f"**Used:** {res.get('used_count', 0)} times")
                                st.write(f"**Success Rate:** {res.get('success_rate', 0):.1%}")
                                
                                # Solution steps
                                steps = res.get('solution_steps', [])
                                if isinstance(steps, list) and steps:
                                    st.write("**Solution Steps:**")
                                    for i, step in enumerate(steps, 1):
                                        st.write(f"{i}. {step}")
            except Exception as e:
                st.error(f"Error loading templates: {str(e)}")
    
    with tab4:
        st.subheader("📤 Export & Import Data")
        
        # Export section
        st.write("**Export Learning Data:**")
        col1, col2 = st.columns(2)
        
        with col1:
            include_resolutions = st.checkbox("Include resolution data", value=True)
            min_effectiveness = st.slider("Minimum effectiveness score:", 0.0, 1.0, 0.7, 0.1)
        
        with col2:
            if st.button("📤 Export Learning Data"):
                with st.spinner("Exporting learning data..."):
                    try:
                        learning_data = agent.export_learning_data(include_resolutions, min_effectiveness)
                        
                        if 'error' not in learning_data:
                            # Show export summary
                            st.success("✅ Export completed!")
                            
                            metadata = learning_data.get('metadata', {})
                            st.write(f"**Project:** {metadata.get('project', 'Unknown')}")
                            st.write(f"**Exported at:** {metadata.get('exported_at', 'Unknown')}")
                            st.write(f"**Quality threshold:** {metadata.get('quality_threshold', 0)}")
                            
                            complaints_count = len(learning_data.get('complaints', []))
                            resolutions_count = len(learning_data.get('resolutions', []))
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Complaints Exported", complaints_count)
                            with col2:
                                st.metric("Resolutions Exported", resolutions_count)
                            
                            # Download button
                            import json
                            json_data = json.dumps(learning_data, indent=2, default=str)
                            st.download_button(
                                label="💾 Download Export File",
                                data=json_data,
                                file_name=f"learning_data_export_{metadata.get('exported_at', 'unknown')[:10]}.json",
                                mime="application/json"
                            )
                        else:
                            st.error(f"Export failed: {learning_data['error']}")
                    
                    except Exception as e:
                        st.error(f"Error exporting data: {str(e)}")
        
        st.markdown("---")
        
        # Import section
        st.write("**Import Historical Data:**")
        uploaded_file = st.file_uploader("Choose a JSON file to import:", type=['json'])
        
        if uploaded_file and st.button("📥 Import Data"):
            with st.spinner("Importing data..."):
                try:
                    import json
                    file_content = uploaded_file.read()
                    import_data = json.loads(file_content)
                    
                    # Import using the persistent store
                    success = agent.persistent_store.import_historical_data(import_data, format="dict")
                    
                    if success:
                        st.success("✅ Data imported successfully!")
                        # Show import summary
                        complaints_imported = len(import_data.get('complaints', []))
                        resolutions_imported = len(import_data.get('resolutions', []))
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Complaints Imported", complaints_imported)
                        with col2:
                            st.metric("Resolutions Imported", resolutions_imported)
                    else:
                        st.error("Failed to import data")
                
                except Exception as e:
                    st.error(f"Error importing data: {str(e)}")
    
    with tab5:
        st.subheader("📁 File Management")
        
        # Storage system selection
        storage_system = st.radio(
            "Select Storage System:",
            ["📊 SQLite Database (Structured)", "🔍 Vector Store (Search)", "🔄 Both Systems"],
            horizontal=True
        )
        
        # Debug information
        if st.checkbox("🔧 Show Debug Info"):
            import sqlite3
            import os
            
            if st.session_state.agent and hasattr(st.session_state.agent, 'persistent_store'):
                try:
                    db_path = st.session_state.agent.persistent_store.store.db_path
                    st.info(f"Database path: {db_path}")
                    st.info(f"Database exists: {os.path.exists(db_path)}")
                    
                    if os.path.exists(db_path):
                        conn = sqlite3.connect(db_path)
                        cursor = conn.cursor()
                        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                        tables = [row[0] for row in cursor.fetchall()]
                        st.info(f"Available tables: {', '.join(tables)}")
                        conn.close()
                except Exception as e:
                    st.error(f"Debug error: {e}")
            else:
                st.warning("Agent or persistent store not available")
        
        # File search and management interface
        col1, col2 = st.columns([2, 1])
        
        with col1:
            search_term = st.text_input("🔍 Search files by name:", placeholder="Enter filename or pattern...")
        
        with col2:
            refresh_files = st.button("🔄 Refresh File List")

        # Handle different storage systems
        if storage_system == "🔍 Vector Store (Search)":
            # Vector Store Management
            st.subheader("🔍 Vector Store Files")
            
            try:
                # Import vector store
                from simple_vector_response import VectorStore
                vs = VectorStore()
                
                if hasattr(vs, 'in_memory_store') and vs.in_memory_store:
                    # Group documents by file
                    files_data = {}
                    for doc_id, data in vs.in_memory_store.items():
                        if isinstance(data, dict) and 'metadata' in data:
                            metadata = data['metadata']
                            file_name = metadata.get('file_name', 'Unknown')
                            
                            if search_term and search_term.lower() not in file_name.lower():
                                continue
                                
                            if file_name not in files_data:
                                files_data[file_name] = {
                                    'chunks': [],
                                    'file_type': metadata.get('file_type', 'unknown'),
                                    'created_at': metadata.get('created_at', 'unknown'),
                                    'file_id': metadata.get('file_id', 'unknown')
                                }
                            
                            files_data[file_name]['chunks'].append({
                                'doc_id': doc_id,
                                'content': data.get('content', data.get('text', '')),
                                'chunk_index': metadata.get('chunk_index', 0)
                            })
                    
                    if files_data:
                        st.write(f"**Found {len(files_data)} files** in vector store")
                        
                        for file_name, file_info in files_data.items():
                            with st.expander(f"🔍 {file_name} ({file_info['file_type'].upper()}) - {len(file_info['chunks'])} chunks"):
                                col1, col2, col3 = st.columns([2, 1, 1])
                                
                                with col1:
                                    st.write(f"**Filename:** {file_name}")
                                    st.write(f"**Type:** {file_info['file_type']}")
                                    st.write(f"**Created:** {file_info['created_at']}")
                                    st.write(f"**File ID:** {file_info['file_id']}")
                                
                                with col2:
                                    st.write(f"**Chunks:** {len(file_info['chunks'])}")
                                
                                with col3:
                                    if st.button(f"🔍 View Content", key=f"vs_view_{file_name}"):
                                        st.write("**Chunks Content:**")
                                        for chunk in sorted(file_info['chunks'], key=lambda x: x['chunk_index']):
                                            st.write(f"**Chunk {chunk['chunk_index']}:**")
                                            st.text_area("", chunk['content'][:300] + "..." if len(chunk['content']) > 300 else chunk['content'], 
                                                       disabled=True, key=f"vs_content_{chunk['doc_id']}")
                                    
                                    if st.button(f"🗑️ Delete from Vector Store", key=f"vs_delete_{file_name}", type="secondary"):
                                        # Delete all chunks for this file
                                        try:
                                            deleted_count = 0
                                            for chunk in file_info['chunks']:
                                                if chunk['doc_id'] in vs.in_memory_store:
                                                    del vs.in_memory_store[chunk['doc_id']]
                                                    deleted_count += 1
                                            
                                            # Save the updated store
                                            vs._save_persistent_data()
                                            
                                            st.success(f"✅ Deleted {file_name} ({deleted_count} chunks) from vector store")
                                            st.rerun()
                                        except Exception as e:
                                            st.error(f"Error deleting from vector store: {str(e)}")
                    else:
                        if search_term:
                            st.info(f"No files found matching '{search_term}' in vector store")
                        else:
                            st.info("No files found in vector store")
                
                else:
                    st.info("Vector store is empty or not accessible")
                    
            except Exception as e:
                st.error(f"Error accessing vector store: {str(e)}")
        
        elif storage_system == "📊 SQLite Database (Structured)":
            # SQLite Database Management 
            st.subheader("📊 SQLite Database Files")
            
            # Get file list from database
            try:
                import sqlite3
                import os
                db_path = "./data/support_agent.db"
                
                if os.path.exists(db_path):
                    conn = sqlite3.connect(db_path)
                    cursor = conn.cursor()
                    
                    # Build query based on search term
                    if search_term:
                        query = """
                        SELECT DISTINCT file_name, file_type, COUNT(*) as sections, 
                               MIN(created_at) as uploaded, SUM(file_size) as total_size, 
                               GROUP_CONCAT(DISTINCT category) as categories
                        FROM file_content 
                        WHERE file_name LIKE ? 
                        GROUP BY file_name, file_type
                        ORDER BY uploaded DESC
                        """
                        cursor.execute(query, (f"%{search_term}%",))
                    else:
                        query = """
                        SELECT DISTINCT file_name, file_type, COUNT(*) as sections, 
                               MIN(created_at) as uploaded, SUM(file_size) as total_size,
                               GROUP_CONCAT(DISTINCT category) as categories
                        FROM file_content 
                        GROUP BY file_name, file_type
                        ORDER BY uploaded DESC
                        LIMIT 50
                        """
                        cursor.execute(query)
                    
                    files = cursor.fetchall()
                    conn.close()
                    
                    if files:
                        st.write(f"**Found {len(files)} files** in SQLite database")
                        
                        for i, (filename, file_type, sections, uploaded, total_size, categories) in enumerate(files):
                            with st.expander(f"📄 {filename} ({file_type.upper()}) - {sections} sections"):
                                col1, col2, col3 = st.columns([2, 1, 1])
                                
                                with col1:
                                    st.write(f"**Filename:** {filename}")
                                    st.write(f"**Type:** {file_type}")
                                    st.write(f"**Uploaded:** {uploaded}")
                                    st.write(f"**Categories:** {categories or 'None'}")
                                
                                with col2:
                                    st.write(f"**Sections:** {sections}")
                                    st.write(f"**Size:** {total_size or 0} bytes")
                                
                                with col3:
                                    if st.button(f"🔍 View Content", key=f"db_view_{i}"):
                                        # Show file content
                                        conn = sqlite3.connect(db_path)
                                        cursor = conn.cursor()
                                        cursor.execute(
                                            "SELECT section_title, content_text FROM file_content WHERE file_name = ? ORDER BY section_number",
                                            (filename,)
                                        )
                                        content = cursor.fetchall()
                                        conn.close()
                                        
                                        st.write("**File Content:**")
                                        for section_title, content_text in content:
                                            st.write(f"**{section_title}:**")
                                            st.text_area("", content_text[:500] + "..." if len(content_text) > 500 else content_text, 
                                                       disabled=True, key=f"db_content_{i}_{section_title}")
                                    
                                    if st.button(f"🗑️ Delete", key=f"db_delete_{i}", type="secondary"):
                                        try:
                                            conn = sqlite3.connect(db_path)
                                            cursor = conn.cursor()
                                            cursor.execute("DELETE FROM file_content WHERE file_name = ?", (filename,))
                                            deleted_count = cursor.rowcount
                                            conn.commit()
                                            conn.close()
                                            
                                            st.success(f"✅ Deleted {filename} ({deleted_count} sections)")
                                            st.rerun()
                                        except Exception as e:
                                            st.error(f"Error deleting file: {str(e)}")
                    else:
                        if search_term:
                            st.info(f"No files found matching '{search_term}' in database")
                        else:
                            st.info("No files found in database")
                else:
                    st.error("Database not found")
                    
            except Exception as e:
                st.error(f"Error accessing database: {str(e)}")
        
        else:  # Both Systems
            st.subheader("🔄 Both Storage Systems")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**🔍 Vector Store (Search Engine)**")
                try:
                    from simple_vector_response import VectorStore
                    vs = VectorStore()
                    
                    if hasattr(vs, 'in_memory_store'):
                        vector_files = set()
                        for doc_id, data in vs.in_memory_store.items():
                            if isinstance(data, dict) and 'metadata' in data:
                                file_name = data['metadata'].get('file_name', 'Unknown')
                                if not search_term or search_term.lower() in file_name.lower():
                                    vector_files.add(file_name)
                        
                        st.metric("Files in Vector Store", len(vector_files))
                        if vector_files:
                            for file_name in sorted(list(vector_files))[:10]:
                                st.write(f"• {file_name}")
                                
                except Exception as e:
                    st.error(f"Vector store error: {str(e)}")
            
            with col2:
                st.write("**📊 SQLite Database (Structured)**")
                try:
                    import sqlite3
                    import os
                    
                    db_path = "./data/support_agent.db"
                    if os.path.exists(db_path):
                        conn = sqlite3.connect(db_path)
                        cursor = conn.cursor()
                        
                        if search_term:
                            cursor.execute("SELECT COUNT(DISTINCT file_name) FROM file_content WHERE file_name LIKE ?", (f"%{search_term}%",))
                        else:
                            cursor.execute("SELECT COUNT(DISTINCT file_name) FROM file_content")
                        
                        db_file_count = cursor.fetchone()[0]
                        
                        cursor.execute("SELECT DISTINCT file_name FROM file_content ORDER BY created_at DESC LIMIT 10")
                        db_files = [row[0] for row in cursor.fetchall()]
                        
                        conn.close()
                        
                        st.metric("Files in Database", db_file_count)
                        if db_files:
                            for file_name in db_files:
                                if not search_term or search_term.lower() in file_name.lower():
                                    st.write(f"• {file_name}")
                                    
                except Exception as e:
                    st.error(f"Database error: {str(e)}")
        
        # Bulk operations for all storage systems
        st.markdown("---")
        st.subheader("🔧 Bulk Operations")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📊 Get Statistics"):
                try:
                    # Vector store stats
                    from simple_vector_response import VectorStore
                    vs = VectorStore()
                    vector_count = len(vs.in_memory_store) if hasattr(vs, 'in_memory_store') else 0
                    
                    # Database stats
                    import sqlite3, os
                    db_path = "./data/support_agent.db"
                    if os.path.exists(db_path):
                        conn = sqlite3.connect(db_path)
                        cursor = conn.cursor()
                        cursor.execute("SELECT COUNT(DISTINCT file_name) FROM file_content")
                        db_files = cursor.fetchone()[0]
                        cursor.execute("SELECT COUNT(*) FROM file_content")
                        db_sections = cursor.fetchone()[0]
                        conn.close()
                        
                        st.metric("Vector Store Documents", vector_count)
                        st.metric("Database Files", db_files)
                        st.metric("Database Sections", db_sections)
                    else:
                        st.error("Database not found")
                        
                except Exception as e:
                    st.error(f"Error getting statistics: {str(e)}")
        
        with col2:
            if st.button("🧹 Clean All Temporary Files"):
                try:
                    deleted_total = 0
                    
                    # Clean vector store
                    from simple_vector_response import VectorStore
                    vs = VectorStore()
                    if hasattr(vs, 'in_memory_store'):
                        to_delete = []
                        for doc_id, data in vs.in_memory_store.items():
                            if isinstance(data, dict) and 'metadata' in data:
                                file_name = data['metadata'].get('file_name', '')
                                if 'tmp' in file_name.lower() or 'temp' in file_name.lower():
                                    to_delete.append(doc_id)
                        
                        for doc_id in to_delete:
                            del vs.in_memory_store[doc_id]
                            deleted_total += 1
                        
                        vs._save_persistent_data()
                    
                    # Clean database
                    import sqlite3, os
                    db_path = "./data/support_agent.db"
                    if os.path.exists(db_path):
                        conn = sqlite3.connect(db_path)
                        cursor = conn.cursor()
                        cursor.execute("DELETE FROM file_content WHERE file_name LIKE '%tmp%' OR file_name LIKE '%temp%'")
                        deleted_total += cursor.rowcount
                        conn.commit()
                        conn.close()
                    
                    if deleted_total > 0:
                        st.success(f"✅ Cleaned up {deleted_total} temporary file entries")
                        st.rerun()
                    else:
                        st.info("No temporary files found to clean")
                        
                except Exception as e:
                    st.error(f"Error cleaning temporary files: {str(e)}")
        
        with col3:
            if st.button("⚠️ Clear All Files", type="secondary"):
                st.warning("This will delete ALL files from BOTH storage systems!")
                if st.button("🚨 CONFIRM: Delete Everything", type="primary"):
                    try:
                        deleted_total = 0
                        
                        # Clear vector store
                        from simple_vector_response import VectorStore
                        vs = VectorStore()
                        if hasattr(vs, 'in_memory_store'):
                            deleted_total += len(vs.in_memory_store)
                            vs.in_memory_store.clear()
                            vs._save_persistent_data()
                        
                        # Clear database
                        import sqlite3, os
                        db_path = "./data/support_agent.db"
                        if os.path.exists(db_path):
                            conn = sqlite3.connect(db_path)
                            cursor = conn.cursor()
                            cursor.execute("DELETE FROM file_content")
                            deleted_total += cursor.rowcount
                            conn.commit()
                            conn.close()
                        
                        st.success(f"✅ Deleted all files ({deleted_total} total entries)")
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Error clearing all files: {str(e)}")

    
    with tab6:
        st.subheader("⚙️ Store Management")
        
        # Store information
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Store Information:**")
            try:
                insights_data = agent.get_persistent_store_insights()
                if insights_data and 'error' not in insights_data:
                    project_stats = insights_data.get('project_stats', {})
                    st.info(f"Project: {agent.persistent_store.project_name}")
                    st.info(f"Database: {agent.persistent_store.store.db_path}")
                    st.info(f"Total Records: {project_stats.get('total_complaints', 0) + project_stats.get('total_resolutions', 0)}")
                else:
                    error_msg = insights_data.get('error', 'Unknown error') if insights_data else 'No data available'
                    st.error(f"Error loading store insights: {error_msg}")
            except Exception as e:
                st.error(f"Error getting store info: {str(e)}")
        
        with col2:
            st.write("**Store Actions:**")
            
            if st.button("🧹 Cleanup Old Logs"):
                try:
                    days_to_keep = st.number_input("Days to keep:", min_value=1, max_value=365, value=90)
                    deleted_count = agent.persistent_store.store.cleanup_old_logs(days_to_keep)
                    st.success(f"Cleaned up {deleted_count} old log entries")
                except Exception as e:
                    st.error(f"Error cleaning up logs: {str(e)}")
            
            if st.button("📊 Get Store Statistics"):
                with st.spinner("Loading statistics..."):
                    try:
                        insights = agent.get_persistent_store_insights()
                        st.json(insights)
                    except Exception as e:
                        st.error(f"Error getting statistics: {str(e)}")
        
        # Advanced settings
        with st.expander("🔧 Advanced Settings"):
            st.warning("⚠️ Advanced settings - use with caution")
            
            new_project_name = st.text_input("Change project name:", value=agent.persistent_store.project_name)
            if st.button("Update Project Name") and new_project_name != agent.persistent_store.project_name:
                try:
                    # This would require reinitializing the persistent store
                    st.info("Project name change would require restarting the application")
                except Exception as e:
                    st.error(f"Error updating project name: {str(e)}")

def supervisor_dashboard_page():
    """Supervisor dashboard with issue spikes, SLA violations, and team insights"""
    st.header("👥 Supervisor Dashboard")
    
    if not st.session_state.agent_initialized or not st.session_state.agent:
        st.warning("⚠️ Agent not initialized. Please go to Settings to configure the agent first.")
        return
    
    if not hasattr(st.session_state.agent, 'supervisor_tools_enabled') or not st.session_state.agent.supervisor_tools_enabled:
        st.error("🚫 Supervisor tools not available. Please ensure the EnhancedSupportTriageAgent is properly configured.")
        return
    
    # Time period selection
    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader("📊 Real-time Supervisor Insights")
    with col2:
        timeframe = st.selectbox("Timeframe:", ["Last 24h", "Last 7 days", "Last 30 days"], index=0)
    
    # Convert timeframe to hours
    timeframe_hours = {"Last 24h": 24, "Last 7 days": 168, "Last 30 days": 720}[timeframe]
    
    try:
        # Get comprehensive dashboard data
        dashboard_data = st.session_state.agent.get_supervisor_dashboard(timeframe_hours)
        
        if 'error' in dashboard_data:
            st.error(f"Error loading dashboard: {dashboard_data['error']}")
            return
        
        # Display key metrics
        st.subheader("🎯 Key Performance Indicators")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        metrics = dashboard_data.get('performance_metrics', {})
        
        with col1:
            avg_response_time = metrics.get('avg_response_time_hours', 0)
            st.metric("Avg Response Time", f"{avg_response_time:.1f}h")
        
        with col2:
            resolution_rate = metrics.get('resolution_rate', 0)
            st.metric("Resolution Rate", f"{resolution_rate:.1%}")
        
        with col3:
            customer_satisfaction = metrics.get('customer_satisfaction', 0)
            st.metric("Customer Satisfaction", f"{customer_satisfaction:.1f}/5")
        
        with col4:
            escalation_rate = metrics.get('escalation_rate', 0)
            st.metric("Escalation Rate", f"{escalation_rate:.1%}")
        
        with col5:
            active_tickets = metrics.get('active_tickets', 0)
            st.metric("Active Tickets", active_tickets)
        
        # SLA Violations and Alerts
        st.subheader("🚨 SLA Status & Violations")
        sla_data = dashboard_data.get('sla_violations', [])
        
        if sla_data:
            for violation in sla_data[:5]:  # Show top 5 violations
                severity = violation.get('status', 'compliant')
                if severity == 'violated':
                    st.error(f"🚨 **SLA VIOLATION**: {violation.get('metric_name', 'Unknown')} - Target: {violation.get('target_value', 0):.1f}, Current: {violation.get('current_value', 0):.1f}")
                elif severity == 'at_risk':
                    st.warning(f"⚠️ **AT RISK**: {violation.get('metric_name', 'Unknown')} - {violation.get('time_to_violation', 'Unknown')} until violation")
        else:
            st.success("✅ All SLA metrics are compliant")
        
        # Issue Spikes Detection
        st.subheader("📈 Issue Spikes & Trend Analysis")
        issue_spikes = dashboard_data.get('issue_spikes', [])
        
        if issue_spikes:
            for spike in issue_spikes:
                severity = spike.get('severity', 'low')
                volume_increase = spike.get('volume_increase', 0)
                category = spike.get('category', 'Unknown')
                
                if severity in ['high', 'critical']:
                    st.error(f"🔥 **CRITICAL SPIKE**: {category} - {volume_increase:.0f}% increase (Current: {spike.get('current_volume', 0)}, Baseline: {spike.get('baseline_volume', 0)})")
                elif severity == 'medium':
                    st.warning(f"📊 **Spike Detected**: {category} - {volume_increase:.0f}% increase")
                
                # Show probable causes and recommendations
                causes = spike.get('probable_causes', [])
                actions = spike.get('recommended_actions', [])
                
                if causes or actions:
                    with st.expander(f"Analysis for {category} spike"):
                        if causes:
                            st.write("**Probable Causes:**")
                            for cause in causes:
                                st.write(f"• {cause}")
                        if actions:
                            st.write("**Recommended Actions:**")
                            for action in actions:
                                st.write(f"• {action}")
        else:
            st.info("📊 No significant issue spikes detected in the selected timeframe")
        
        # Supervisor Alerts
        st.subheader("🔔 Active Supervisor Alerts")
        alerts = st.session_state.agent.get_supervisor_alerts('medium')
        
        if 'alerts' in alerts and alerts['alerts']:
            for alert in alerts['alerts'][:10]:  # Show top 10 alerts
                severity = alert.get('severity', 'low')
                title = alert.get('title', 'Unknown Alert')
                
                if severity == 'critical':
                    st.error(f"🚨 **CRITICAL**: {title}")
                elif severity == 'high':
                    st.error(f"❗ **HIGH**: {title}")
                elif severity == 'medium':
                    st.warning(f"⚠️ **MEDIUM**: {title}")
                
                with st.expander(f"Alert Details: {title}"):
                    st.write(f"**Description:** {alert.get('description', 'No description')}")
                    st.write(f"**Category:** {alert.get('category', 'General')}")
                    st.write(f"**Created:** {alert.get('created_at', 'Unknown')}")
                    
                    actions = alert.get('recommended_actions', [])
                    if actions:
                        st.write("**Recommended Actions:**")
                        for action in actions:
                            st.write(f"• {action}")
        else:
            st.success("✅ No active supervisor alerts")
        
        # Team Performance Summary
        st.subheader("👥 Team Performance Summary")
        team_stats = dashboard_data.get('team_performance', {})
        
        if team_stats:
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Top Performing Categories:**")
                top_categories = team_stats.get('top_categories', [])
                for cat in top_categories[:5]:
                    st.write(f"• {cat.get('name', 'Unknown')}: {cat.get('resolution_rate', 0):.1%} resolution")
            
            with col2:
                st.write("**Areas Needing Attention:**")
                attention_areas = team_stats.get('attention_areas', [])
                for area in attention_areas[:5]:
                    st.write(f"• {area.get('name', 'Unknown')}: {area.get('issue_description', 'No details')}")
        
        # Auto-refresh option
        st.markdown("---")
        if st.checkbox("🔄 Auto-refresh dashboard (30 seconds)", key="supervisor_auto_refresh"):
            import time
            time.sleep(30)
            st.rerun()
    
    except Exception as e:
        st.error(f"Error loading supervisor dashboard: {str(e)}")
        st.write("This might be due to missing supervisor tools or insufficient data.")

def escalation_center_page():
    """Enhanced escalation monitoring and management center"""
    st.title("🚨 Escalation Center")
    st.markdown("**Real-time escalation monitoring, alerts, and management dashboard**")
    
    if not st.session_state.get('agent_initialized', False):
        st.warning("⚠️ Agent not initialized. Showing demo escalation data.")
        display_demo_escalations()
        return
    
    # Escalation Overview Dashboard
    st.markdown("---")
    st.subheader("📊 Escalation Overview")
    
    # Mock escalation metrics for demo
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("🔴 Critical", "3", delta="1", help="Immediate attention required")
    with col2:
        st.metric("🟠 High", "8", delta="2", help="Supervisor review needed")
    with col3:
        st.metric("🟡 Medium", "15", delta="-1", help="Team lead oversight")
    with col4:
        st.metric("📈 Total Today", "26", delta="2", help="All escalations today")
    with col5:
        st.metric("⏱️ Avg Response", "3.2h", delta="-0.5h", help="Average response time")
    
    # Active Escalations Table
    st.markdown("---")
    st.subheader("🎯 Active Escalations Requiring Attention")
    
    # Enhanced escalation filter
    filter_col1, filter_col2, filter_col3 = st.columns(3)
    
    with filter_col1:
        level_filter = st.multiselect(
            "Filter by Level",
            ["Critical", "High", "Medium", "Low"],
            default=["Critical", "High"]
        )
    
    with filter_col2:
        reason_filter = st.multiselect(
            "Filter by Reason",
            ["Legal Threat", "VIP Customer", "Severe Sentiment", "Technical Issue", "Financial Impact"],
            default=[]
        )
    
    with filter_col3:
        time_filter = st.selectbox("Time Frame", ["All", "Last Hour", "Last 4 Hours", "Today"])
    
    # Demo escalation data with enhanced highlighting
    escalations = get_demo_escalations()
    
    # Display escalations with color coding and priority indicators
    for i, escalation in enumerate(escalations):
        level = escalation.get('level', 'medium')
        priority = escalation.get('priority_score', 5)
        
        # Color-coded container based on escalation level
        if level == 'critical':
            container_color = "#ff4b4b"
            border_color = "#ff0000"
            icon = "🆘"
        elif level == 'high':
            container_color = "#ff8c00"
            border_color = "#ff6600"
            icon = "🚨"
        elif level == 'medium':
            container_color = "#ffa500"
            border_color = "#ff9500"
            icon = "⚠️"
        else:
            container_color = "#ffd700"
            border_color = "#ffcc00"
            icon = "ℹ️"
        
        # Create escalation card
        st.markdown(f"""
        <div style='background-color: {container_color}; color: white; padding: 15px; 
                    border-radius: 8px; border-left: 8px solid {border_color}; 
                    margin: 10px 0; box-shadow: 0 3px 6px rgba(0,0,0,0.1);'>
            <div style='display: flex; justify-content: space-between; align-items: center;'>
                <h4>{icon} Ticket #{escalation.get('ticket_id', 'Unknown')} - {level.upper()}</h4>
                <span style='background: rgba(255,255,255,0.2); padding: 5px 10px; border-radius: 15px;'>
                    Priority: {priority}/10
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Escalation details in expander
        with st.expander(f"📋 Details for Ticket #{escalation.get('ticket_id', 'Unknown')}", expanded=(level in ['critical', 'high'])):
            detail_col1, detail_col2, detail_col3 = st.columns(3)
            
            with detail_col1:
                st.markdown("**🎯 Escalation Info**")
                st.write(f"**Level:** {level.title()}")
                st.write(f"**Priority:** {priority}/10")
                st.write(f"**Confidence:** {escalation.get('confidence', 0.85):.0%}")
                st.write(f"**Created:** {escalation.get('created_at', 'Just now')}")
            
            with detail_col2:
                st.markdown("**🔍 Reasons & Risks**")
                reasons = escalation.get('reasons', [])
                for reason in reasons:
                    if reason in ['Legal Threat', 'Safety Concern']:
                        st.markdown(f"🔴 {reason}")
                    elif reason in ['VIP Customer', 'High Financial Impact']:
                        st.markdown(f"🟠 {reason}")
                    else:
                        st.markdown(f"🟡 {reason}")
                
                risk_factors = escalation.get('risk_factors', [])
                if risk_factors:
                    st.write("**Risk Factors:**")
                    for risk in risk_factors[:3]:
                        st.write(f"• {risk}")
            
            with detail_col3:
                st.markdown("**📋 Actions Required**")
                st.write(f"**Assign To:** {escalation.get('suggested_assignee', 'Team Lead')}")
                st.write(f"**Timeframe:** {escalation.get('timeframe', 'Standard')}")
                st.write(f"**Action:** {escalation.get('recommended_action', 'Review with supervisor')[:50]}...")
                
                # Action buttons
                button_col1, button_col2 = st.columns(2)
                with button_col1:
                    if st.button(f"✅ Escalate Now", key=f"escalate_{i}"):
                        st.success(f"Ticket #{escalation.get('ticket_id')} escalated to {escalation.get('suggested_assignee')}")
                with button_col2:
                    if st.button(f"📋 View Ticket", key=f"view_{i}"):
                        st.info(f"Opening ticket #{escalation.get('ticket_id')} details...")
    
    # Escalation Analytics
    st.markdown("---")
    st.subheader("📈 Escalation Analytics & Trends")
    
    trend_col1, trend_col2 = st.columns(2)
    
    with trend_col1:
        st.markdown("**📊 Escalation Trends (Last 7 Days)**")
        # Create sample trend data
        import pandas as pd
        from datetime import datetime, timedelta
        
        dates = [(datetime.now() - timedelta(days=i)).strftime('%m/%d') for i in range(6, -1, -1)]
        critical_counts = [1, 2, 0, 3, 1, 2, 3]
        high_counts = [5, 7, 4, 9, 6, 8, 8]
        medium_counts = [12, 15, 11, 18, 14, 16, 15]
        
        trend_df = pd.DataFrame({
            'Date': dates,
            'Critical': critical_counts,
            'High': high_counts,
            'Medium': medium_counts
        })
        
        st.line_chart(trend_df.set_index('Date'))
    
    with trend_col2:
        st.markdown("**🎯 Top Escalation Reasons**")
        reasons_data = {
            'Severe Negative Sentiment': 35,
            'Repeated Contact': 25,
            'VIP Customer': 22,
            'Complex Technical Issue': 18,
            'High Financial Impact': 15,
            'Legal Threat': 8
        }
        
        for reason, count in reasons_data.items():
            percentage = (count / sum(reasons_data.values())) * 100
            st.markdown(f"**{reason}:** {count} cases ({percentage:.1f}%)")
            st.progress(percentage / 100)
    
    # Quick Actions Panel
    st.markdown("---")
    st.subheader("⚡ Quick Actions")
    
    action_col1, action_col2, action_col3, action_col4 = st.columns(4)
    
    with action_col1:
        if st.button("🔍 Scan for New Escalations"):
            with st.spinner("Scanning tickets for escalation indicators..."):
                import time
                time.sleep(2)
                st.success("Scan complete! 2 new escalations detected.")
    
    with action_col2:
        if st.button("📊 Generate Escalation Report"):
            st.success("Escalation report generated and sent to supervisors.")
    
    with action_col3:
        if st.button("🔔 Send Escalation Alerts"):
            st.success("Escalation alerts sent to relevant team members.")
    
    with action_col4:
        if st.button("⚙️ Configure Escalation Rules"):
            st.info("Escalation rules configuration panel would open here.")

def get_demo_escalations():
    """Get demo escalation data for the escalation center"""
    return [
        {
            'ticket_id': 'TK-2024-0892',
            'level': 'critical',
            'priority_score': 9,
            'confidence': 0.95,
            'reasons': ['Legal Threat', 'VIP Customer'],
            'risk_factors': ['Customer threatened legal action', 'High-value enterprise customer', 'Public reputation risk'],
            'suggested_assignee': 'Legal Team + Senior Manager',
            'timeframe': 'Immediate (within 1 hour)',
            'recommended_action': 'Immediately escalate to legal team and senior management. Document all communications.',
            'created_at': '15 minutes ago'
        },
        {
            'ticket_id': 'TK-2024-0891',
            'level': 'high',
            'priority_score': 8,
            'confidence': 0.87,
            'reasons': ['Severe Negative Sentiment', 'High Financial Impact'],
            'risk_factors': ['Customer extremely frustrated', '$5000 order value', 'Multiple failed resolution attempts'],
            'suggested_assignee': 'Senior Customer Success Manager',
            'timeframe': 'High priority (within 4 hours)',
            'recommended_action': 'Assign to senior customer success manager. Expedite resolution with executive oversight.',
            'created_at': '32 minutes ago'
        },
        {
            'ticket_id': 'TK-2024-0890',
            'level': 'medium',
            'priority_score': 6,
            'confidence': 0.72,
            'reasons': ['Repeated Contact', 'Complex Technical Issue'],
            'risk_factors': ['Customer contacted 4 times this week', 'Integration API failure', 'Multiple systems involved'],
            'suggested_assignee': 'Technical Support Team Lead',
            'timeframe': 'Medium priority (within 8 hours)',
            'recommended_action': 'Route to team lead with timeline monitoring and technical specialist support.',
            'created_at': '1 hour ago'
        }
    ]

def display_demo_escalations():
    """Display demo escalation data when agent is not initialized"""
    st.info("📝 Displaying demo escalation data. Connect your agent for real-time escalation monitoring.")
    
    # Demo metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🔴 Critical", "1", help="Demo: Critical escalations")
    with col2:
        st.metric("🟠 High", "3", help="Demo: High priority escalations")
    with col3:
        st.metric("🟡 Medium", "5", help="Demo: Medium priority escalations")
    with col4:
        st.metric("📈 Total", "9", help="Demo: Total escalations")
    
    st.markdown("**Demo Escalation Cases:**")
    escalations = get_demo_escalations()
    
    for escalation in escalations:
        level = escalation.get('level', 'medium')
        if level == 'critical':
            st.error(f"🆘 **CRITICAL** - Ticket #{escalation.get('ticket_id')} - {', '.join(escalation.get('reasons', []))}")
        elif level == 'high':
            st.warning(f"🚨 **HIGH** - Ticket #{escalation.get('ticket_id')} - {', '.join(escalation.get('reasons', []))}")
        else:
            st.info(f"⚠️ **MEDIUM** - Ticket #{escalation.get('ticket_id')} - {', '.join(escalation.get('reasons', []))}")

# Main execution
if __name__ == "__main__":
    main()
else:
    # This ensures the main function runs when executed via streamlit run
    main()
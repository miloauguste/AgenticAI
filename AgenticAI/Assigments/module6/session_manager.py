"""
Session State Management for Customer Support Agent
Tracks session history, queries, file uploads, and responses with persistence
"""
import json
import pickle
import os
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import logging

class SessionManager:
    """
    Manages session state with JSON and Pickle-based persistence
    Tracks queries, file uploads, responses, and user interactions
    """
    
    def __init__(self, session_id: str = None, storage_format: str = 'json'):
        """
        Initialize session manager
        
        Args:
            session_id: Unique session identifier (auto-generated if None)
            storage_format: 'json' or 'pickle' for state persistence
        """
        self.logger = logging.getLogger(__name__)
        
        # Generate session ID if not provided
        if session_id is None:
            session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.session_id = session_id
        self.storage_format = storage_format
        
        # Create sessions directory
        self.sessions_dir = Path("data/sessions")
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        
        # Session file path
        extension = 'json' if storage_format == 'json' else 'pkl'
        self.session_file = self.sessions_dir / f"{session_id}.{extension}"
        
        # Initialize session state
        self.state = {
            'session_id': session_id,
            'created_at': datetime.now().isoformat(),
            'last_activity': datetime.now().isoformat(),
            'queries': [],
            'file_uploads': [],
            'ticket_analyses': [],
            'responses_generated': [],
            'search_history': [],
            'escalations': [],
            'session_stats': {
                'total_queries': 0,
                'total_uploads': 0,
                'total_tickets': 0,
                'total_searches': 0,
                'session_duration_seconds': 0
            },
            'user_preferences': {
                'default_category': None,
                'preferred_response_tone': 'professional',
                'auto_escalation_threshold': 0.8,
                'max_search_results': 10
            },
            'context': {
                'current_ticket_id': None,
                'active_categories': [],
                'recent_issues': [],
                'pending_escalations': []
            }
        }
        
        # Load existing session if available
        self.load_session()
    
    def save_session(self) -> bool:
        """
        Save current session state to disk
        
        Returns:
            Success boolean
        """
        try:
            # Update last activity
            self.state['last_activity'] = datetime.now().isoformat()
            
            # Calculate session duration
            created_at = datetime.fromisoformat(self.state['created_at'])
            duration = (datetime.now() - created_at).total_seconds()
            self.state['session_stats']['session_duration_seconds'] = int(duration)
            
            if self.storage_format == 'json':
                with open(self.session_file, 'w', encoding='utf-8') as f:
                    json.dump(self.state, f, indent=2, ensure_ascii=False)
            else:
                with open(self.session_file, 'wb') as f:
                    pickle.dump(self.state, f)
            
            self.logger.debug(f"Session {self.session_id} saved to {self.session_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving session: {str(e)}")
            return False
    
    def load_session(self) -> bool:
        """
        Load session state from disk if exists
        
        Returns:
            Success boolean
        """
        try:
            if not self.session_file.exists():
                self.logger.info(f"No existing session file found, starting fresh session {self.session_id}")
                return False
            
            if self.storage_format == 'json':
                with open(self.session_file, 'r', encoding='utf-8') as f:
                    saved_state = json.load(f)
            else:
                with open(self.session_file, 'rb') as f:
                    saved_state = pickle.load(f)
            
            # Merge saved state with current state
            self.state.update(saved_state)
            self.logger.info(f"Loaded session {self.session_id} from {self.session_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading session: {str(e)}")
            return False
    
    def track_query(self, query: str, query_type: str = 'search', metadata: Dict[str, Any] = None) -> None:
        """
        Track a user query
        
        Args:
            query: The query string
            query_type: Type of query ('search', 'ticket', 'file_upload', etc.)
            metadata: Additional metadata about the query
        """
        query_entry = {
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'query_type': query_type,
            'metadata': metadata or {},
            'query_id': len(self.state['queries']) + 1
        }
        
        self.state['queries'].append(query_entry)
        self.state['session_stats']['total_queries'] += 1
        self.save_session()
    
    def track_file_upload(self, file_name: str, file_type: str, file_size: int, 
                         processing_result: Dict[str, Any] = None) -> None:
        """
        Track file upload
        
        Args:
            file_name: Name of uploaded file
            file_type: Type/extension of file
            file_size: Size of file in bytes
            processing_result: Result of file processing
        """
        upload_entry = {
            'timestamp': datetime.now().isoformat(),
            'file_name': file_name,
            'file_type': file_type,
            'file_size': file_size,
            'processing_result': processing_result or {},
            'upload_id': len(self.state['file_uploads']) + 1
        }
        
        self.state['file_uploads'].append(upload_entry)
        self.state['session_stats']['total_uploads'] += 1
        self.save_session()
    
    def track_ticket_analysis(self, ticket_id: str, ticket_text: str, 
                             analysis_result: Dict[str, Any]) -> None:
        """
        Track ticket analysis
        
        Args:
            ticket_id: Unique ticket identifier
            ticket_text: Original ticket text
            analysis_result: Complete analysis result
        """
        ticket_entry = {
            'timestamp': datetime.now().isoformat(),
            'ticket_id': ticket_id,
            'ticket_text': ticket_text[:200] + "..." if len(ticket_text) > 200 else ticket_text,
            'analysis_result': analysis_result,
            'escalation_needed': analysis_result.get('escalation_indicators', {}).get('requires_escalation', False)
        }
        
        self.state['ticket_analyses'].append(ticket_entry)
        self.state['session_stats']['total_tickets'] += 1
        self.state['context']['current_ticket_id'] = ticket_id
        
        # Track recent issues
        category = analysis_result.get('intent_classification', {}).get('intent_category', 'Unknown')
        if category not in self.state['context']['recent_issues']:
            self.state['context']['recent_issues'].append(category)
            
        # Keep only last 10 recent issues
        if len(self.state['context']['recent_issues']) > 10:
            self.state['context']['recent_issues'] = self.state['context']['recent_issues'][-10:]
        
        self.save_session()
    
    def track_response_generation(self, ticket_id: str, suggested_response: str, 
                                 response_metadata: Dict[str, Any] = None) -> None:
        """
        Track response generation
        
        Args:
            ticket_id: Associated ticket ID
            suggested_response: Generated response
            response_metadata: Additional response metadata
        """
        response_entry = {
            'timestamp': datetime.now().isoformat(),
            'ticket_id': ticket_id,
            'suggested_response': suggested_response,
            'metadata': response_metadata or {},
            'response_id': len(self.state['responses_generated']) + 1
        }
        
        self.state['responses_generated'].append(response_entry)
        self.save_session()
    
    def track_search(self, query: str, search_type: str, results_count: int,
                    category: str = None) -> None:
        """
        Track knowledge base search
        
        Args:
            query: Search query
            search_type: Type of search performed
            results_count: Number of results returned
            category: Search category if used
        """
        search_entry = {
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'search_type': search_type,
            'results_count': results_count,
            'category': category,
            'search_id': len(self.state['search_history']) + 1
        }
        
        self.state['search_history'].append(search_entry)
        self.state['session_stats']['total_searches'] += 1
        self.save_session()
    
    def track_escalation(self, ticket_id: str, escalation_reason: str, 
                        escalation_metadata: Dict[str, Any] = None) -> None:
        """
        Track escalation event
        
        Args:
            ticket_id: Ticket requiring escalation
            escalation_reason: Reason for escalation
            escalation_metadata: Additional escalation data
        """
        escalation_entry = {
            'timestamp': datetime.now().isoformat(),
            'ticket_id': ticket_id,
            'escalation_reason': escalation_reason,
            'metadata': escalation_metadata or {},
            'status': 'pending',
            'escalation_id': len(self.state['escalations']) + 1
        }
        
        self.state['escalations'].append(escalation_entry)
        self.state['context']['pending_escalations'].append(ticket_id)
        self.save_session()
    
    def get_session_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive session summary
        
        Returns:
            Session summary with statistics and insights
        """
        # Recent activity (last 10 items)
        recent_queries = self.state['queries'][-10:] if self.state['queries'] else []
        recent_tickets = self.state['ticket_analyses'][-10:] if self.state['ticket_analyses'] else []
        recent_searches = self.state['search_history'][-10:] if self.state['search_history'] else []
        
        # Category analysis
        categories = {}
        for ticket in self.state['ticket_analyses']:
            category = ticket.get('analysis_result', {}).get('intent_classification', {}).get('intent_category', 'Unknown')
            categories[category] = categories.get(category, 0) + 1
        
        # Escalation summary
        escalations_pending = len(self.state['context']['pending_escalations'])
        escalations_total = len(self.state['escalations'])
        
        return {
            'session_id': self.session_id,
            'session_stats': self.state['session_stats'],
            'recent_activity': {
                'queries': recent_queries,
                'tickets': recent_tickets,
                'searches': recent_searches
            },
            'insights': {
                'most_common_categories': dict(sorted(categories.items(), key=lambda x: x[1], reverse=True)[:5]),
                'escalation_rate': escalations_total / max(self.state['session_stats']['total_tickets'], 1),
                'pending_escalations': escalations_pending,
                'active_categories': self.state['context']['active_categories'],
                'recent_issues': self.state['context']['recent_issues']
            },
            'duration_minutes': self.state['session_stats']['session_duration_seconds'] // 60,
            'created_at': self.state['created_at'],
            'last_activity': self.state['last_activity']
        }
    
    def get_context_for_ticket(self, current_category: str = None) -> Dict[str, Any]:
        """
        Get relevant context for current ticket processing
        
        Args:
            current_category: Current ticket category
            
        Returns:
            Context information for ticket processing
        """
        # Recent tickets in same category
        related_tickets = []
        if current_category:
            for ticket in self.state['ticket_analyses'][-20:]:  # Last 20 tickets
                ticket_category = ticket.get('analysis_result', {}).get('intent_classification', {}).get('intent_category')
                if ticket_category == current_category:
                    related_tickets.append(ticket)
        
        # Recent escalations
        recent_escalations = self.state['escalations'][-5:] if self.state['escalations'] else []
        
        return {
            'session_id': self.session_id,
            'current_ticket_id': self.state['context']['current_ticket_id'],
            'related_tickets_in_session': related_tickets[-5:],  # Last 5 related
            'recent_escalations': recent_escalations,
            'session_issue_trend': self.state['context']['recent_issues'],
            'pending_escalations_count': len(self.state['context']['pending_escalations']),
            'user_preferences': self.state['user_preferences']
        }
    
    def update_preferences(self, preferences: Dict[str, Any]) -> None:
        """
        Update user preferences
        
        Args:
            preferences: Dictionary of preference updates
        """
        self.state['user_preferences'].update(preferences)
        self.save_session()
    
    def clear_session(self) -> bool:
        """
        Clear current session data (but keep preferences)
        
        Returns:
            Success boolean
        """
        try:
            # Keep preferences and session metadata
            preferences = self.state['user_preferences'].copy()
            session_id = self.state['session_id']
            created_at = self.state['created_at']
            
            # Reset session state
            self.state = {
                'session_id': session_id,
                'created_at': created_at,
                'last_activity': datetime.now().isoformat(),
                'queries': [],
                'file_uploads': [],
                'ticket_analyses': [],
                'responses_generated': [],
                'search_history': [],
                'escalations': [],
                'session_stats': {
                    'total_queries': 0,
                    'total_uploads': 0,
                    'total_tickets': 0,
                    'total_searches': 0,
                    'session_duration_seconds': 0
                },
                'user_preferences': preferences,
                'context': {
                    'current_ticket_id': None,
                    'active_categories': [],
                    'recent_issues': [],
                    'pending_escalations': []
                }
            }
            
            self.save_session()
            return True
            
        except Exception as e:
            self.logger.error(f"Error clearing session: {str(e)}")
            return False
    
    @classmethod
    def get_recent_sessions(cls, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get list of recent sessions
        
        Args:
            limit: Maximum sessions to return
            
        Returns:
            List of session summaries
        """
        sessions_dir = Path("data/sessions")
        if not sessions_dir.exists():
            return []
        
        sessions = []
        session_files = list(sessions_dir.glob("*.json")) + list(sessions_dir.glob("*.pkl"))
        
        # Sort by modification time
        session_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        for session_file in session_files[:limit]:
            try:
                session_id = session_file.stem
                if session_file.suffix == '.json':
                    with open(session_file, 'r', encoding='utf-8') as f:
                        session_data = json.load(f)
                else:
                    with open(session_file, 'rb') as f:
                        session_data = pickle.load(f)
                
                sessions.append({
                    'session_id': session_id,
                    'created_at': session_data.get('created_at'),
                    'last_activity': session_data.get('last_activity'),
                    'total_tickets': session_data.get('session_stats', {}).get('total_tickets', 0),
                    'total_queries': session_data.get('session_stats', {}).get('total_queries', 0),
                    'file_path': str(session_file)
                })
                
            except Exception as e:
                logging.error(f"Error reading session file {session_file}: {str(e)}")
                continue
        
        return sessions
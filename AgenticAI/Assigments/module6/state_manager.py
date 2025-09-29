#!/usr/bin/env python3
"""
State Management System for Supervisor Tools
Tracks session history, queries, file uploads, responses with JSON-based persistence
"""

import json
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
import hashlib

class EventType(Enum):
    """Types of events to track"""
    QUERY = "query"
    FILE_UPLOAD = "file_upload"
    RESPONSE_GENERATED = "response_generated"
    ESCALATION_TRIGGERED = "escalation_triggered"
    SLA_VIOLATION = "sla_violation"
    ISSUE_SPIKE_DETECTED = "issue_spike_detected"
    SUPERVISOR_ALERT = "supervisor_alert"
    TICKET_PROCESSED = "ticket_processed"
    USER_ACTION = "user_action"
    SYSTEM_EVENT = "system_event"

class SessionStatus(Enum):
    """Session status states"""
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    EXPIRED = "expired"

@dataclass
class SessionEvent:
    """Individual session event"""
    event_id: str
    event_type: EventType
    timestamp: datetime
    user_id: str
    session_id: str
    data: Dict[str, Any]
    metadata: Dict[str, Any]
    duration_ms: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'event_id': self.event_id,
            'event_type': self.event_type.value,
            'timestamp': self.timestamp.isoformat(),
            'user_id': self.user_id,
            'session_id': self.session_id,
            'data': self.data,
            'metadata': self.metadata,
            'duration_ms': self.duration_ms
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SessionEvent':
        """Create from dictionary"""
        return cls(
            event_id=data['event_id'],
            event_type=EventType(data['event_type']),
            timestamp=datetime.fromisoformat(data['timestamp']),
            user_id=data['user_id'],
            session_id=data['session_id'],
            data=data['data'],
            metadata=data['metadata'],
            duration_ms=data.get('duration_ms')
        )

@dataclass
class SessionState:
    """Complete session state"""
    session_id: str
    user_id: str
    created_at: datetime
    last_activity: datetime
    status: SessionStatus
    events: List[SessionEvent]
    context: Dict[str, Any]
    statistics: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'session_id': self.session_id,
            'user_id': self.user_id,
            'created_at': self.created_at.isoformat(),
            'last_activity': self.last_activity.isoformat(),
            'status': self.status.value,
            'events': [event.to_dict() for event in self.events],
            'context': self.context,
            'statistics': self.statistics
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SessionState':
        """Create from dictionary"""
        return cls(
            session_id=data['session_id'],
            user_id=data['user_id'],
            created_at=datetime.fromisoformat(data['created_at']),
            last_activity=datetime.fromisoformat(data['last_activity']),
            status=SessionStatus(data['status']),
            events=[SessionEvent.from_dict(event) for event in data['events']],
            context=data['context'],
            statistics=data['statistics']
        )

class StateManager:
    """
    Comprehensive state management system for supervisor tools
    """
    
    def __init__(self, storage_dir: str = "session_data", auto_save: bool = True):
        """
        Initialize state manager
        
        Args:
            storage_dir: Directory for storing session files
            auto_save: Whether to automatically save state changes
        """
        self.logger = logging.getLogger(__name__)
        self.storage_dir = storage_dir
        self.auto_save = auto_save
        
        # Create storage directory
        os.makedirs(storage_dir, exist_ok=True)
        
        # Current session state
        self.current_session: Optional[SessionState] = None
        self.session_cache: Dict[str, SessionState] = {}
        
        # Configuration
        self.max_events_per_session = 10000
        self.session_timeout_hours = 24
        self.auto_cleanup_days = 30
        
        self.logger.info(f"State manager initialized with storage at {storage_dir}")
    
    def create_session(self, user_id: str, context: Dict[str, Any] = None) -> str:
        """
        Create a new session
        
        Args:
            user_id: User identifier
            context: Initial session context
            
        Returns:
            Session ID
        """
        session_id = self._generate_session_id()
        now = datetime.now()
        
        session_state = SessionState(
            session_id=session_id,
            user_id=user_id,
            created_at=now,
            last_activity=now,
            status=SessionStatus.ACTIVE,
            events=[],
            context=context or {},
            statistics={
                'total_events': 0,
                'queries_count': 0,
                'files_uploaded': 0,
                'responses_generated': 0,
                'escalations_triggered': 0,
                'session_duration_minutes': 0
            }
        )
        
        self.current_session = session_state
        self.session_cache[session_id] = session_state
        
        # Log session creation
        self.log_event(
            EventType.SYSTEM_EVENT,
            {
                'action': 'session_created',
                'user_id': user_id,
                'context': context or {}
            }
        )
        
        if self.auto_save:
            self._save_session(session_state)
        
        self.logger.info(f"Created new session {session_id} for user {user_id}")
        return session_id
    
    def load_session(self, session_id: str) -> Optional[SessionState]:
        """
        Load session from storage
        
        Args:
            session_id: Session identifier
            
        Returns:
            Session state or None if not found
        """
        # Check cache first
        if session_id in self.session_cache:
            return self.session_cache[session_id]
        
        # Load from file
        session_file = os.path.join(self.storage_dir, f"session_{session_id}.json")
        
        if not os.path.exists(session_file):
            self.logger.warning(f"Session file not found: {session_id}")
            return None
        
        try:
            with open(session_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            session_state = SessionState.from_dict(data)
            self.session_cache[session_id] = session_state
            
            self.logger.info(f"Loaded session {session_id} with {len(session_state.events)} events")
            return session_state
            
        except Exception as e:
            self.logger.error(f"Error loading session {session_id}: {str(e)}")
            return None
    
    def set_current_session(self, session_id: str) -> bool:
        """
        Set the current active session
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if session was set successfully
        """
        session_state = self.load_session(session_id)
        if session_state:
            self.current_session = session_state
            
            # Update last activity
            session_state.last_activity = datetime.now()
            session_state.status = SessionStatus.ACTIVE
            
            if self.auto_save:
                self._save_session(session_state)
            
            self.logger.info(f"Set current session to {session_id}")
            return True
        
        return False
    
    def log_event(self, event_type: EventType, data: Dict[str, Any], 
                  metadata: Dict[str, Any] = None, duration_ms: int = None) -> str:
        """
        Log an event to the current session
        
        Args:
            event_type: Type of event
            data: Event data
            metadata: Additional metadata
            duration_ms: Event duration in milliseconds
            
        Returns:
            Event ID
        """
        if not self.current_session:
            self.logger.warning("No current session - creating default session")
            self.create_session("system")
        
        event_id = str(uuid.uuid4())
        now = datetime.now()
        
        event = SessionEvent(
            event_id=event_id,
            event_type=event_type,
            timestamp=now,
            user_id=self.current_session.user_id,
            session_id=self.current_session.session_id,
            data=data,
            metadata=metadata or {},
            duration_ms=duration_ms
        )
        
        # Add event to session
        self.current_session.events.append(event)
        self.current_session.last_activity = now
        
        # Update statistics
        self._update_session_statistics(event_type)
        
        # Enforce event limit
        if len(self.current_session.events) > self.max_events_per_session:
            self.current_session.events = self.current_session.events[-self.max_events_per_session:]
            self.logger.warning(f"Session event limit reached, removed oldest events")
        
        if self.auto_save:
            self._save_session(self.current_session)
        
        self.logger.debug(f"Logged event {event_type.value} with ID {event_id}")
        return event_id
    
    def log_query(self, query: str, query_type: str = "general", 
                  response: str = None, response_time_ms: int = None) -> str:
        """
        Log a user query and response
        
        Args:
            query: User query text
            query_type: Type of query
            response: System response
            response_time_ms: Response time in milliseconds
            
        Returns:
            Event ID
        """
        data = {
            'query': query,
            'query_type': query_type,
            'query_length': len(query),
            'response': response,
            'response_length': len(response) if response else 0
        }
        
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'has_response': response is not None
        }
        
        return self.log_event(EventType.QUERY, data, metadata, response_time_ms)
    
    def log_file_upload(self, filename: str, file_size: int, file_type: str,
                       processing_result: Dict[str, Any] = None) -> str:
        """
        Log a file upload event
        
        Args:
            filename: Name of uploaded file
            file_size: File size in bytes
            file_type: Type of file
            processing_result: Results of file processing
            
        Returns:
            Event ID
        """
        data = {
            'filename': filename,
            'file_size': file_size,
            'file_type': file_type,
            'file_hash': self._generate_file_hash(filename, file_size),
            'processing_result': processing_result or {}
        }
        
        metadata = {
            'upload_timestamp': datetime.now().isoformat(),
            'processed': processing_result is not None
        }
        
        return self.log_event(EventType.FILE_UPLOAD, data, metadata)
    
    def log_response_generated(self, response_type: str, confidence: float,
                             suggestions_count: int = 0, escalation_triggered: bool = False) -> str:
        """
        Log a response generation event
        
        Args:
            response_type: Type of response generated
            confidence: Confidence score
            suggestions_count: Number of suggestions generated
            escalation_triggered: Whether escalation was triggered
            
        Returns:
            Event ID
        """
        data = {
            'response_type': response_type,
            'confidence_score': confidence,
            'suggestions_count': suggestions_count,
            'escalation_triggered': escalation_triggered
        }
        
        metadata = {
            'generation_timestamp': datetime.now().isoformat(),
            'quality_tier': self._classify_confidence(confidence)
        }
        
        return self.log_event(EventType.RESPONSE_GENERATED, data, metadata)
    
    def log_escalation(self, escalation_level: str, reasons: List[str],
                      priority_score: int, recommended_action: str) -> str:
        """
        Log an escalation event
        
        Args:
            escalation_level: Level of escalation
            reasons: Reasons for escalation
            priority_score: Priority score
            recommended_action: Recommended action
            
        Returns:
            Event ID
        """
        data = {
            'escalation_level': escalation_level,
            'reasons': reasons,
            'priority_score': priority_score,
            'recommended_action': recommended_action,
            'reason_count': len(reasons)
        }
        
        metadata = {
            'escalation_timestamp': datetime.now().isoformat(),
            'severity': self._classify_escalation_severity(escalation_level, priority_score)
        }
        
        return self.log_event(EventType.ESCALATION_TRIGGERED, data, metadata)
    
    def get_session_history(self, session_id: str = None, 
                           event_types: List[EventType] = None,
                           limit: int = 100) -> List[SessionEvent]:
        """
        Get session history with optional filtering
        
        Args:
            session_id: Session ID (current session if None)
            event_types: Filter by event types
            limit: Maximum number of events to return
            
        Returns:
            List of session events
        """
        if session_id:
            session = self.load_session(session_id)
        else:
            session = self.current_session
        
        if not session:
            return []
        
        events = session.events
        
        # Filter by event types
        if event_types:
            events = [e for e in events if e.event_type in event_types]
        
        # Sort by timestamp (newest first) and limit
        events.sort(key=lambda x: x.timestamp, reverse=True)
        return events[:limit]
    
    def get_session_statistics(self, session_id: str = None) -> Dict[str, Any]:
        """
        Get session statistics
        
        Args:
            session_id: Session ID (current session if None)
            
        Returns:
            Session statistics
        """
        if session_id:
            session = self.load_session(session_id)
        else:
            session = self.current_session
        
        if not session:
            return {}
        
        # Update duration
        duration_minutes = (session.last_activity - session.created_at).total_seconds() / 60
        session.statistics['session_duration_minutes'] = round(duration_minutes, 2)
        
        # Add event type breakdown
        event_breakdown = {}
        for event in session.events:
            event_type = event.event_type.value
            event_breakdown[event_type] = event_breakdown.get(event_type, 0) + 1
        
        session.statistics['event_breakdown'] = event_breakdown
        
        return session.statistics
    
    def export_session(self, session_id: str = None, 
                      include_events: bool = True) -> Dict[str, Any]:
        """
        Export session data for analysis
        
        Args:
            session_id: Session ID (current session if None)
            include_events: Whether to include all events
            
        Returns:
            Complete session data
        """
        if session_id:
            session = self.load_session(session_id)
        else:
            session = self.current_session
        
        if not session:
            return {}
        
        export_data = session.to_dict()
        
        if not include_events:
            export_data['events'] = []
            export_data['event_count'] = len(session.events)
        
        return export_data
    
    def cleanup_old_sessions(self, days: int = None) -> int:
        """
        Clean up old session files
        
        Args:
            days: Sessions older than this many days (default: auto_cleanup_days)
            
        Returns:
            Number of sessions cleaned up
        """
        cleanup_days = days or self.auto_cleanup_days
        cutoff_date = datetime.now() - timedelta(days=cleanup_days)
        
        cleaned_count = 0
        
        for filename in os.listdir(self.storage_dir):
            if filename.startswith("session_") and filename.endswith(".json"):
                filepath = os.path.join(self.storage_dir, filename)
                
                try:
                    # Check file modification time
                    mod_time = datetime.fromtimestamp(os.path.getmtime(filepath))
                    
                    if mod_time < cutoff_date:
                        os.remove(filepath)
                        cleaned_count += 1
                        
                        # Remove from cache
                        session_id = filename.replace("session_", "").replace(".json", "")
                        self.session_cache.pop(session_id, None)
                        
                except Exception as e:
                    self.logger.warning(f"Error cleaning up {filename}: {str(e)}")
        
        self.logger.info(f"Cleaned up {cleaned_count} old session files")
        return cleaned_count
    
    def save_current_session(self) -> bool:
        """
        Manually save the current session
        
        Returns:
            True if saved successfully
        """
        if self.current_session:
            return self._save_session(self.current_session)
        return False
    
    def close_session(self, session_id: str = None) -> bool:
        """
        Close a session
        
        Args:
            session_id: Session ID (current session if None)
            
        Returns:
            True if closed successfully
        """
        if session_id:
            session = self.load_session(session_id)
        else:
            session = self.current_session
        
        if not session:
            return False
        
        session.status = SessionStatus.COMPLETED
        session.last_activity = datetime.now()
        
        # Log session closure
        self.log_event(
            EventType.SYSTEM_EVENT,
            {
                'action': 'session_closed',
                'session_duration_minutes': session.statistics.get('session_duration_minutes', 0),
                'total_events': len(session.events)
            }
        )
        
        if self.auto_save:
            self._save_session(session)
        
        # Clear current session if it's the one being closed
        if self.current_session and self.current_session.session_id == session.session_id:
            self.current_session = None
        
        self.logger.info(f"Closed session {session.session_id}")
        return True
    
    def _save_session(self, session: SessionState) -> bool:
        """Save session to file"""
        try:
            session_file = os.path.join(self.storage_dir, f"session_{session.session_id}.json")
            
            with open(session_file, 'w', encoding='utf-8') as f:
                json.dump(session.to_dict(), f, indent=2, ensure_ascii=False)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving session {session.session_id}: {str(e)}")
            return False
    
    def _generate_session_id(self) -> str:
        """Generate unique session ID"""
        return f"sess_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
    
    def _generate_file_hash(self, filename: str, file_size: int) -> str:
        """Generate file hash for tracking"""
        content = f"{filename}_{file_size}_{datetime.now().isoformat()}"
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    def _update_session_statistics(self, event_type: EventType):
        """Update session statistics based on event type"""
        stats = self.current_session.statistics
        stats['total_events'] += 1
        
        if event_type == EventType.QUERY:
            stats['queries_count'] += 1
        elif event_type == EventType.FILE_UPLOAD:
            stats['files_uploaded'] += 1
        elif event_type == EventType.RESPONSE_GENERATED:
            stats['responses_generated'] += 1
        elif event_type == EventType.ESCALATION_TRIGGERED:
            stats['escalations_triggered'] += 1
    
    def _classify_confidence(self, confidence: float) -> str:
        """Classify confidence score"""
        if confidence >= 0.9:
            return "high"
        elif confidence >= 0.7:
            return "medium"
        else:
            return "low"
    
    def _classify_escalation_severity(self, level: str, priority: int) -> str:
        """Classify escalation severity"""
        if level in ["critical", "immediate"] or priority >= 9:
            return "critical"
        elif level == "high" or priority >= 7:
            return "high"
        elif level == "medium" or priority >= 5:
            return "medium"
        else:
            return "low"
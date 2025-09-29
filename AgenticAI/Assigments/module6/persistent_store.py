#!/usr/bin/env python3
"""
Persistent Store for Complaints and Resolutions
Reusable across sessions and projects with advanced features
"""

import sqlite3
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from pathlib import Path
import hashlib
from dataclasses import dataclass, asdict
from enum import Enum

class ComplaintStatus(Enum):
    """Status enum for complaints"""
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
    CLOSED = "closed"
    ESCALATED = "escalated"

class ResolutionType(Enum):
    """Type enum for resolutions"""
    AUTOMATIC = "automatic"
    MANUAL = "manual"
    ESCALATED = "escalated"
    KNOWLEDGE_BASE = "knowledge_base"

class Priority(Enum):
    """Priority levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"
    CRITICAL = "critical"

@dataclass
class Complaint:
    """Data class for complaint records"""
    id: str
    customer_id: str
    title: str
    description: str
    category: str
    subcategory: str
    priority: str
    status: str
    source: str
    channel: str
    created_at: datetime
    updated_at: datetime
    resolved_at: Optional[datetime] = None
    assigned_to: Optional[str] = None
    tags: List[str] = None
    metadata: Dict[str, Any] = None
    sentiment_score: Optional[float] = None
    urgency_score: Optional[float] = None
    customer_satisfaction: Optional[int] = None
    resolution_time_hours: Optional[float] = None

@dataclass
class Resolution:
    """Data class for resolution records"""
    id: str
    complaint_id: str
    title: str
    description: str
    solution_steps: List[str]
    resolution_type: str
    category: str
    effectiveness_score: float
    resolution_time_hours: float
    created_by: str
    created_at: datetime
    used_count: int = 0
    last_used_at: Optional[datetime] = None
    success_rate: float = 0.0
    prerequisites: List[str] = None
    related_articles: List[str] = None
    tags: List[str] = None
    metadata: Dict[str, Any] = None

class PersistentStore:
    """
    Comprehensive persistent store for complaints and resolutions
    Designed for reuse across sessions and projects
    """
    
    def __init__(self, db_path: Optional[str] = None, project_name: str = "default"):
        """
        Initialize persistent store
        
        Args:
            db_path: Path to database file (optional)
            project_name: Name of the project for multi-project support
        """
        self.logger = logging.getLogger(__name__)
        self.project_name = project_name
        
        # Create project-specific database path
        if db_path is None:
            data_dir = Path("persistent_data")
            data_dir.mkdir(exist_ok=True)
            db_path = str(data_dir / f"{project_name}_complaints_resolutions.db")
        
        self.db_path = db_path
        self.init_database()
        self.logger.info(f"Initialized persistent store for project '{project_name}' at {db_path}")
    
    def init_database(self):
        """Initialize database with comprehensive schema"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                # Projects table for multi-project support
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS projects (
                        id TEXT PRIMARY KEY,
                        name TEXT UNIQUE NOT NULL,
                        description TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        is_active BOOLEAN DEFAULT 1,
                        settings TEXT,
                        metadata TEXT
                    )
                """)
                
                # Complaints table (enhanced)
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS complaints (
                        id TEXT PRIMARY KEY,
                        project_id TEXT,
                        customer_id TEXT NOT NULL,
                        title TEXT NOT NULL,
                        description TEXT NOT NULL,
                        category TEXT NOT NULL,
                        subcategory TEXT,
                        priority TEXT NOT NULL,
                        status TEXT NOT NULL,
                        source TEXT,
                        channel TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        resolved_at TIMESTAMP,
                        assigned_to TEXT,
                        tags TEXT,
                        metadata TEXT,
                        sentiment_score REAL,
                        urgency_score REAL,
                        customer_satisfaction INTEGER,
                        resolution_time_hours REAL,
                        content_hash TEXT UNIQUE,
                        FOREIGN KEY (project_id) REFERENCES projects (id)
                    )
                """)
                
                # Resolutions table (enhanced)
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS resolutions (
                        id TEXT PRIMARY KEY,
                        project_id TEXT,
                        complaint_id TEXT,
                        title TEXT NOT NULL,
                        description TEXT NOT NULL,
                        solution_steps TEXT NOT NULL,
                        resolution_type TEXT NOT NULL,
                        category TEXT NOT NULL,
                        effectiveness_score REAL DEFAULT 0.0,
                        resolution_time_hours REAL,
                        created_by TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        used_count INTEGER DEFAULT 0,
                        last_used_at TIMESTAMP,
                        success_rate REAL DEFAULT 0.0,
                        prerequisites TEXT,
                        related_articles TEXT,
                        tags TEXT,
                        metadata TEXT,
                        is_template BOOLEAN DEFAULT 0,
                        template_usage_count INTEGER DEFAULT 0,
                        FOREIGN KEY (project_id) REFERENCES projects (id),
                        FOREIGN KEY (complaint_id) REFERENCES complaints (id)
                    )
                """)
                
                # Knowledge base articles
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS knowledge_base (
                        id TEXT PRIMARY KEY,
                        project_id TEXT,
                        title TEXT NOT NULL,
                        content TEXT NOT NULL,
                        category TEXT NOT NULL,
                        subcategory TEXT,
                        type TEXT DEFAULT 'article',
                        status TEXT DEFAULT 'active',
                        created_by TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        view_count INTEGER DEFAULT 0,
                        helpful_votes INTEGER DEFAULT 0,
                        unhelpful_votes INTEGER DEFAULT 0,
                        tags TEXT,
                        related_complaints TEXT,
                        related_resolutions TEXT,
                        metadata TEXT,
                        FOREIGN KEY (project_id) REFERENCES projects (id)
                    )
                """)
                
                # Analytics and metrics table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS analytics (
                        id TEXT PRIMARY KEY,
                        project_id TEXT,
                        metric_name TEXT NOT NULL,
                        metric_value REAL NOT NULL,
                        metric_type TEXT NOT NULL,
                        category TEXT,
                        period_start TIMESTAMP,
                        period_end TIMESTAMP,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        metadata TEXT,
                        FOREIGN KEY (project_id) REFERENCES projects (id)
                    )
                """)
                
                # Session logs for tracking usage
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS session_logs (
                        id TEXT PRIMARY KEY,
                        project_id TEXT,
                        session_id TEXT,
                        action TEXT NOT NULL,
                        entity_type TEXT,
                        entity_id TEXT,
                        user_id TEXT,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        details TEXT,
                        FOREIGN KEY (project_id) REFERENCES projects (id)
                    )
                """)
                
                # Create comprehensive indexes
                self._create_indexes(cursor)
                
                # Insert current project if not exists
                self._ensure_project_exists(cursor)
                
                conn.commit()
                self.logger.info("Database schema initialized successfully")
                
        except Exception as e:
            self.logger.error(f"Error initializing database: {str(e)}")
            raise
    
    def _create_indexes(self, cursor):
        """Create performance indexes"""
        indexes = [
            # Complaints indexes
            "CREATE INDEX IF NOT EXISTS idx_complaints_project ON complaints (project_id)",
            "CREATE INDEX IF NOT EXISTS idx_complaints_status ON complaints (status)",
            "CREATE INDEX IF NOT EXISTS idx_complaints_category ON complaints (category)",
            "CREATE INDEX IF NOT EXISTS idx_complaints_priority ON complaints (priority)",
            "CREATE INDEX IF NOT EXISTS idx_complaints_created_at ON complaints (created_at)",
            "CREATE INDEX IF NOT EXISTS idx_complaints_customer ON complaints (customer_id)",
            "CREATE INDEX IF NOT EXISTS idx_complaints_hash ON complaints (content_hash)",
            
            # Resolutions indexes
            "CREATE INDEX IF NOT EXISTS idx_resolutions_project ON resolutions (project_id)",
            "CREATE INDEX IF NOT EXISTS idx_resolutions_complaint ON resolutions (complaint_id)",
            "CREATE INDEX IF NOT EXISTS idx_resolutions_category ON resolutions (category)",
            "CREATE INDEX IF NOT EXISTS idx_resolutions_type ON resolutions (resolution_type)",
            "CREATE INDEX IF NOT EXISTS idx_resolutions_effectiveness ON resolutions (effectiveness_score)",
            "CREATE INDEX IF NOT EXISTS idx_resolutions_usage ON resolutions (used_count)",
            
            # Knowledge base indexes
            "CREATE INDEX IF NOT EXISTS idx_knowledge_project ON knowledge_base (project_id)",
            "CREATE INDEX IF NOT EXISTS idx_knowledge_category ON knowledge_base (category)",
            "CREATE INDEX IF NOT EXISTS idx_knowledge_status ON knowledge_base (status)",
            "CREATE INDEX IF NOT EXISTS idx_knowledge_views ON knowledge_base (view_count)",
            
            # Analytics indexes
            "CREATE INDEX IF NOT EXISTS idx_analytics_project ON analytics (project_id)",
            "CREATE INDEX IF NOT EXISTS idx_analytics_metric ON analytics (metric_name)",
            "CREATE INDEX IF NOT EXISTS idx_analytics_period ON analytics (period_start, period_end)",
            
            # Session logs indexes
            "CREATE INDEX IF NOT EXISTS idx_sessions_project ON session_logs (project_id)",
            "CREATE INDEX IF NOT EXISTS idx_sessions_action ON session_logs (action)",
            "CREATE INDEX IF NOT EXISTS idx_sessions_timestamp ON session_logs (timestamp)"
        ]
        
        for index_sql in indexes:
            cursor.execute(index_sql)
    
    def _ensure_project_exists(self, cursor):
        """Ensure current project exists in database"""
        project_id = self._generate_project_id()
        
        cursor.execute("SELECT id FROM projects WHERE name = ?", (self.project_name,))
        if not cursor.fetchone():
            cursor.execute("""
                INSERT INTO projects (id, name, description, settings, metadata)
                VALUES (?, ?, ?, ?, ?)
            """, (
                project_id,
                self.project_name,
                f"Project for {self.project_name}",
                json.dumps({"auto_created": True}),
                json.dumps({"created_by": "persistent_store"})
            ))
    
    def _generate_project_id(self) -> str:
        """Generate unique project ID"""
        return f"proj_{hashlib.md5(self.project_name.encode()).hexdigest()[:8]}"
    
    def _get_project_id(self) -> str:
        """Get project ID for current project"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM projects WHERE name = ?", (self.project_name,))
            result = cursor.fetchone()
            return result[0] if result else self._generate_project_id()
    
    def add_complaint(self, complaint: Union[Complaint, Dict[str, Any]]) -> str:
        """
        Add a new complaint to the persistent store
        
        Args:
            complaint: Complaint object or dictionary
            
        Returns:
            Complaint ID
        """
        try:
            # Convert dict to Complaint if needed
            if isinstance(complaint, dict):
                complaint_data = complaint.copy()
                complaint_data.setdefault('id', str(uuid.uuid4()))
                complaint_data.setdefault('created_at', datetime.now())
                complaint_data.setdefault('updated_at', datetime.now())
                complaint_data.setdefault('status', ComplaintStatus.OPEN.value)
                complaint_data.setdefault('priority', Priority.MEDIUM.value)
                
                # Convert datetime strings if needed
                for field in ['created_at', 'updated_at', 'resolved_at']:
                    if field in complaint_data and isinstance(complaint_data[field], str):
                        complaint_data[field] = datetime.fromisoformat(complaint_data[field])
                
                complaint = Complaint(**complaint_data)
            
            # Generate content hash for deduplication
            content_hash = self._generate_content_hash(complaint.description + complaint.title)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                project_id = self._get_project_id()
                
                # Check for duplicates
                cursor.execute("SELECT id FROM complaints WHERE content_hash = ?", (content_hash,))
                if cursor.fetchone():
                    self.logger.warning(f"Duplicate complaint detected: {complaint.title[:50]}...")
                    return None
                
                cursor.execute("""
                    INSERT INTO complaints (
                        id, project_id, customer_id, title, description, category, subcategory,
                        priority, status, source, channel, created_at, updated_at, resolved_at,
                        assigned_to, tags, metadata, sentiment_score, urgency_score,
                        customer_satisfaction, resolution_time_hours, content_hash
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    complaint.id, project_id, complaint.customer_id, complaint.title,
                    complaint.description, complaint.category, complaint.subcategory,
                    complaint.priority, complaint.status, complaint.source, complaint.channel,
                    complaint.created_at, complaint.updated_at, complaint.resolved_at,
                    complaint.assigned_to, json.dumps(complaint.tags or []),
                    json.dumps(complaint.metadata or {}), complaint.sentiment_score,
                    complaint.urgency_score, complaint.customer_satisfaction,
                    complaint.resolution_time_hours, content_hash
                ))
                
                conn.commit()
                self._log_action("add_complaint", "complaint", complaint.id)
                self.logger.info(f"Added complaint: {complaint.id}")
                return complaint.id
                
        except Exception as e:
            self.logger.error(f"Error adding complaint: {str(e)}")
            raise
    
    def add_resolution(self, resolution: Union[Resolution, Dict[str, Any]]) -> str:
        """
        Add a new resolution to the persistent store
        
        Args:
            resolution: Resolution object or dictionary
            
        Returns:
            Resolution ID
        """
        try:
            # Convert dict to Resolution if needed
            if isinstance(resolution, dict):
                resolution_data = resolution.copy()
                resolution_data.setdefault('id', str(uuid.uuid4()))
                resolution_data.setdefault('created_at', datetime.now())
                resolution_data.setdefault('resolution_type', ResolutionType.MANUAL.value)
                resolution_data.setdefault('effectiveness_score', 0.0)
                resolution_data.setdefault('used_count', 0)
                resolution_data.setdefault('success_rate', 0.0)
                
                # Ensure solution_steps is a list
                if isinstance(resolution_data.get('solution_steps'), str):
                    resolution_data['solution_steps'] = [resolution_data['solution_steps']]
                
                resolution = Resolution(**resolution_data)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                project_id = self._get_project_id()
                
                cursor.execute("""
                    INSERT INTO resolutions (
                        id, project_id, complaint_id, title, description, solution_steps,
                        resolution_type, category, effectiveness_score, resolution_time_hours,
                        created_by, created_at, used_count, last_used_at, success_rate,
                        prerequisites, related_articles, tags, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    resolution.id, project_id, resolution.complaint_id, resolution.title,
                    resolution.description, json.dumps(resolution.solution_steps),
                    resolution.resolution_type, resolution.category, resolution.effectiveness_score,
                    resolution.resolution_time_hours, resolution.created_by, resolution.created_at,
                    resolution.used_count, resolution.last_used_at, resolution.success_rate,
                    json.dumps(resolution.prerequisites or []),
                    json.dumps(resolution.related_articles or []),
                    json.dumps(resolution.tags or []), json.dumps(resolution.metadata or {})
                ))
                
                conn.commit()
                self._log_action("add_resolution", "resolution", resolution.id)
                self.logger.info(f"Added resolution: {resolution.id}")
                return resolution.id
                
        except Exception as e:
            self.logger.error(f"Error adding resolution: {str(e)}")
            raise
    
    def search_complaints(self, 
                         query: str = None,
                         category: str = None,
                         status: str = None,
                         priority: str = None,
                         date_from: datetime = None,
                         date_to: datetime = None,
                         customer_id: str = None,
                         limit: int = 50) -> List[Dict[str, Any]]:
        """
        Search complaints with multiple filters
        
        Args:
            query: Text search in title/description
            category: Filter by category
            status: Filter by status
            priority: Filter by priority
            date_from: Filter from date
            date_to: Filter to date
            customer_id: Filter by customer
            limit: Maximum results
            
        Returns:
            List of complaint records
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                project_id = self._get_project_id()
                
                # Build dynamic query
                where_clauses = ["project_id = ?"]
                params = [project_id]
                
                if query:
                    where_clauses.append("(title LIKE ? OR description LIKE ?)")
                    params.extend([f"%{query}%", f"%{query}%"])
                
                if category:
                    where_clauses.append("category = ?")
                    params.append(category)
                
                if status:
                    where_clauses.append("status = ?")
                    params.append(status)
                
                if priority:
                    where_clauses.append("priority = ?")
                    params.append(priority)
                
                if date_from:
                    where_clauses.append("created_at >= ?")
                    params.append(date_from)
                
                if date_to:
                    where_clauses.append("created_at <= ?")
                    params.append(date_to)
                
                if customer_id:
                    where_clauses.append("customer_id = ?")
                    params.append(customer_id)
                
                sql = f"""
                    SELECT * FROM complaints 
                    WHERE {' AND '.join(where_clauses)}
                    ORDER BY created_at DESC
                    LIMIT ?
                """
                params.append(limit)
                
                cursor.execute(sql, params)
                results = [dict(row) for row in cursor.fetchall()]
                
                # Parse JSON fields
                for result in results:
                    result['tags'] = json.loads(result['tags']) if result['tags'] else []
                    result['metadata'] = json.loads(result['metadata']) if result['metadata'] else {}
                
                self._log_action("search_complaints", "search", f"query:{query or 'all'}")
                return results
                
        except Exception as e:
            self.logger.error(f"Error searching complaints: {str(e)}")
            return []
    
    def search_resolutions(self,
                          query: str = None,
                          category: str = None,
                          resolution_type: str = None,
                          min_effectiveness: float = None,
                          sort_by: str = "effectiveness_score",
                          limit: int = 50) -> List[Dict[str, Any]]:
        """
        Search resolutions with filters
        
        Args:
            query: Text search in title/description
            category: Filter by category
            resolution_type: Filter by type
            min_effectiveness: Minimum effectiveness score
            sort_by: Sort field
            limit: Maximum results
            
        Returns:
            List of resolution records
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                project_id = self._get_project_id()
                
                where_clauses = ["project_id = ?"]
                params = [project_id]
                
                if query:
                    where_clauses.append("(title LIKE ? OR description LIKE ?)")
                    params.extend([f"%{query}%", f"%{query}%"])
                
                if category:
                    where_clauses.append("category = ?")
                    params.append(category)
                
                if resolution_type:
                    where_clauses.append("resolution_type = ?")
                    params.append(resolution_type)
                
                if min_effectiveness is not None:
                    where_clauses.append("effectiveness_score >= ?")
                    params.append(min_effectiveness)
                
                # Validate sort field
                valid_sort_fields = ["effectiveness_score", "used_count", "success_rate", "created_at"]
                if sort_by not in valid_sort_fields:
                    sort_by = "effectiveness_score"
                
                sql = f"""
                    SELECT * FROM resolutions 
                    WHERE {' AND '.join(where_clauses)}
                    ORDER BY {sort_by} DESC
                    LIMIT ?
                """
                params.append(limit)
                
                cursor.execute(sql, params)
                results = [dict(row) for row in cursor.fetchall()]
                
                # Parse JSON fields
                for result in results:
                    result['solution_steps'] = json.loads(result['solution_steps'])
                    result['prerequisites'] = json.loads(result['prerequisites']) if result['prerequisites'] else []
                    result['related_articles'] = json.loads(result['related_articles']) if result['related_articles'] else []
                    result['tags'] = json.loads(result['tags']) if result['tags'] else []
                    result['metadata'] = json.loads(result['metadata']) if result['metadata'] else {}
                
                self._log_action("search_resolutions", "search", f"query:{query or 'all'}")
                return results
                
        except Exception as e:
            self.logger.error(f"Error searching resolutions: {str(e)}")
            return []
    
    def get_complaint_by_id(self, complaint_id: str) -> Optional[Dict[str, Any]]:
        """Get complaint by ID"""
        results = self.search_complaints()
        for complaint in results:
            if complaint['id'] == complaint_id:
                return complaint
        return None
    
    def get_resolution_by_id(self, resolution_id: str) -> Optional[Dict[str, Any]]:
        """Get resolution by ID"""
        results = self.search_resolutions()
        for resolution in results:
            if resolution['id'] == resolution_id:
                return resolution
        return None
    
    def update_complaint_status(self, complaint_id: str, status: str, resolution_time: float = None) -> bool:
        """Update complaint status"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                update_fields = ["status = ?", "updated_at = ?"]
                params = [status, datetime.now()]
                
                if status in ["resolved", "closed"] and resolution_time:
                    update_fields.extend(["resolved_at = ?", "resolution_time_hours = ?"])
                    params.extend([datetime.now(), resolution_time])
                
                sql = f"UPDATE complaints SET {', '.join(update_fields)} WHERE id = ?"
                params.append(complaint_id)
                
                cursor.execute(sql, params)
                conn.commit()
                
                self._log_action("update_complaint_status", "complaint", complaint_id)
                return cursor.rowcount > 0
                
        except Exception as e:
            self.logger.error(f"Error updating complaint status: {str(e)}")
            return False
    
    def mark_resolution_used(self, resolution_id: str, effectiveness_feedback: float = None) -> bool:
        """Mark resolution as used and update metrics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Update usage count and last used time
                cursor.execute("""
                    UPDATE resolutions 
                    SET used_count = used_count + 1, 
                        last_used_at = ?,
                        effectiveness_score = CASE 
                            WHEN ? IS NOT NULL THEN (effectiveness_score + ?) / 2
                            ELSE effectiveness_score
                        END
                    WHERE id = ?
                """, (datetime.now(), effectiveness_feedback, effectiveness_feedback or 0, resolution_id))
                
                conn.commit()
                self._log_action("mark_resolution_used", "resolution", resolution_id)
                return cursor.rowcount > 0
                
        except Exception as e:
            self.logger.error(f"Error marking resolution used: {str(e)}")
            return False
    
    def get_analytics(self, days: int = 30) -> Dict[str, Any]:
        """Get analytics for the specified period"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                project_id = self._get_project_id()
                date_from = datetime.now() - timedelta(days=days)
                
                analytics = {}
                
                # Complaint statistics
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total_complaints,
                        COUNT(CASE WHEN status = 'resolved' THEN 1 END) as resolved_complaints,
                        COUNT(CASE WHEN status = 'open' THEN 1 END) as open_complaints,
                        AVG(resolution_time_hours) as avg_resolution_time,
                        AVG(customer_satisfaction) as avg_satisfaction
                    FROM complaints 
                    WHERE project_id = ? AND created_at >= ?
                """, (project_id, date_from))
                
                complaint_stats = dict(cursor.fetchone())
                analytics['complaints'] = complaint_stats
                
                # Category distribution
                cursor.execute("""
                    SELECT category, COUNT(*) as count
                    FROM complaints 
                    WHERE project_id = ? AND created_at >= ?
                    GROUP BY category
                    ORDER BY count DESC
                """, (project_id, date_from))
                
                analytics['categories'] = [dict(row) for row in cursor.fetchall()]
                
                # Resolution effectiveness
                cursor.execute("""
                    SELECT 
                        AVG(effectiveness_score) as avg_effectiveness,
                        COUNT(*) as total_resolutions,
                        SUM(used_count) as total_usage
                    FROM resolutions 
                    WHERE project_id = ?
                """, (project_id,))
                
                resolution_stats = dict(cursor.fetchone())
                analytics['resolutions'] = resolution_stats
                
                # Most used resolutions
                cursor.execute("""
                    SELECT title, used_count, effectiveness_score
                    FROM resolutions 
                    WHERE project_id = ?
                    ORDER BY used_count DESC
                    LIMIT 10
                """, (project_id,))
                
                analytics['top_resolutions'] = [dict(row) for row in cursor.fetchall()]
                
                return analytics
                
        except Exception as e:
            self.logger.error(f"Error getting analytics: {str(e)}")
            return {}
    
    def export_data(self, format: str = "json", include_projects: List[str] = None) -> str:
        """Export data for backup or migration"""
        try:
            export_data = {
                'metadata': {
                    'exported_at': datetime.now().isoformat(),
                    'version': '1.0',
                    'projects': include_projects or [self.project_name]
                },
                'projects': [],
                'complaints': [],
                'resolutions': [],
                'knowledge_base': []
            }
            
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                # Export projects
                if include_projects:
                    placeholders = ','.join(['?'] * len(include_projects))
                    cursor.execute(f"SELECT * FROM projects WHERE name IN ({placeholders})", include_projects)
                else:
                    cursor.execute("SELECT * FROM projects WHERE name = ?", (self.project_name,))
                
                export_data['projects'] = [dict(row) for row in cursor.fetchall()]
                
                # Export complaints and resolutions for each project
                for project in export_data['projects']:
                    project_id = project['id']
                    
                    cursor.execute("SELECT * FROM complaints WHERE project_id = ?", (project_id,))
                    export_data['complaints'].extend([dict(row) for row in cursor.fetchall()])
                    
                    cursor.execute("SELECT * FROM resolutions WHERE project_id = ?", (project_id,))
                    export_data['resolutions'].extend([dict(row) for row in cursor.fetchall()])
                    
                    cursor.execute("SELECT * FROM knowledge_base WHERE project_id = ?", (project_id,))
                    export_data['knowledge_base'].extend([dict(row) for row in cursor.fetchall()])
            
            # Save to file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"persistent_store_export_{timestamp}.{format}"
            
            export_dir = Path("exports")
            export_dir.mkdir(exist_ok=True)
            export_path = export_dir / filename
            
            if format == "json":
                import json
                with open(export_path, 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, indent=2, default=str)
            else:
                raise ValueError(f"Unsupported export format: {format}")
            
            self.logger.info(f"Data exported to {export_path}")
            return str(export_path)
            
        except Exception as e:
            self.logger.error(f"Error exporting data: {str(e)}")
            raise
    
    def import_data(self, file_path: str) -> bool:
        """Import data from exported file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                import_data = json.load(f)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Import projects
                for project in import_data.get('projects', []):
                    cursor.execute("""
                        INSERT OR REPLACE INTO projects 
                        (id, name, description, created_at, updated_at, is_active, settings, metadata)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (project['id'], project['name'], project['description'],
                          project['created_at'], project['updated_at'], project['is_active'],
                          project['settings'], project['metadata']))
                
                # Import complaints
                for complaint in import_data.get('complaints', []):
                    cursor.execute("""
                        INSERT OR REPLACE INTO complaints 
                        (id, project_id, customer_id, title, description, category, subcategory,
                         priority, status, source, channel, created_at, updated_at, resolved_at,
                         assigned_to, tags, metadata, sentiment_score, urgency_score,
                         customer_satisfaction, resolution_time_hours, content_hash)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, tuple(complaint[field] for field in [
                        'id', 'project_id', 'customer_id', 'title', 'description', 'category',
                        'subcategory', 'priority', 'status', 'source', 'channel', 'created_at',
                        'updated_at', 'resolved_at', 'assigned_to', 'tags', 'metadata',
                        'sentiment_score', 'urgency_score', 'customer_satisfaction',
                        'resolution_time_hours', 'content_hash'
                    ]))
                
                # Import resolutions
                for resolution in import_data.get('resolutions', []):
                    cursor.execute("""
                        INSERT OR REPLACE INTO resolutions 
                        (id, project_id, complaint_id, title, description, solution_steps,
                         resolution_type, category, effectiveness_score, resolution_time_hours,
                         created_by, created_at, updated_at, used_count, last_used_at, success_rate,
                         prerequisites, related_articles, tags, metadata, is_template, template_usage_count)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, tuple(resolution.get(field) for field in [
                        'id', 'project_id', 'complaint_id', 'title', 'description', 'solution_steps',
                        'resolution_type', 'category', 'effectiveness_score', 'resolution_time_hours',
                        'created_by', 'created_at', 'updated_at', 'used_count', 'last_used_at',
                        'success_rate', 'prerequisites', 'related_articles', 'tags', 'metadata',
                        'is_template', 'template_usage_count'
                    ]))
                
                conn.commit()
                self.logger.info(f"Successfully imported data from {file_path}")
                return True
                
        except Exception as e:
            self.logger.error(f"Error importing data: {str(e)}")
            return False
    
    def _generate_content_hash(self, content: str) -> str:
        """Generate hash for content deduplication"""
        return hashlib.sha256(content.encode('utf-8')).hexdigest()
    
    def _log_action(self, action: str, entity_type: str, entity_id: str, details: str = None):
        """Log action for audit trail"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO session_logs (id, project_id, session_id, action, entity_type, entity_id, details)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    str(uuid.uuid4()),
                    self._get_project_id(),
                    "default_session",  # Could be enhanced with real session tracking
                    action,
                    entity_type,
                    entity_id,
                    details
                ))
                conn.commit()
        except Exception as e:
            self.logger.warning(f"Failed to log action: {str(e)}")
    
    def get_project_stats(self) -> Dict[str, Any]:
        """Get comprehensive project statistics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                project_id = self._get_project_id()
                
                stats = {}
                
                # Total counts
                cursor.execute("SELECT COUNT(*) as count FROM complaints WHERE project_id = ?", (project_id,))
                stats['total_complaints'] = cursor.fetchone()['count']
                
                cursor.execute("SELECT COUNT(*) as count FROM resolutions WHERE project_id = ?", (project_id,))
                stats['total_resolutions'] = cursor.fetchone()['count']
                
                cursor.execute("SELECT COUNT(*) as count FROM knowledge_base WHERE project_id = ?", (project_id,))
                stats['total_knowledge_articles'] = cursor.fetchone()['count']
                
                # Recent activity
                week_ago = datetime.now() - timedelta(days=7)
                cursor.execute("""
                    SELECT COUNT(*) as count FROM complaints 
                    WHERE project_id = ? AND created_at >= ?
                """, (project_id, week_ago))
                stats['complaints_this_week'] = cursor.fetchone()['count']
                
                return stats
                
        except Exception as e:
            self.logger.error(f"Error getting project stats: {str(e)}")
            return {}
    
    def cleanup_old_logs(self, days_to_keep: int = 90):
        """Clean up old session logs"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cutoff_date = datetime.now() - timedelta(days=days_to_keep)
                
                cursor.execute("DELETE FROM session_logs WHERE timestamp < ?", (cutoff_date,))
                deleted_count = cursor.rowcount
                conn.commit()
                
                self.logger.info(f"Cleaned up {deleted_count} old log entries")
                return deleted_count
                
        except Exception as e:
            self.logger.error(f"Error cleaning up logs: {str(e)}")
            return 0
    
    def get_all_complaints(self) -> List[Dict[str, Any]]:
        """
        Get all complaints from the database
        
        Returns:
            List of all complaints with their details
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                # First check what columns exist
                cursor.execute("PRAGMA table_info(complaints)")
                columns = [row[1] for row in cursor.fetchall()]
                
                # Build query with available columns
                base_columns = ['id', 'title', 'description', 'category', 'priority', 'status', 'created_at', 'updated_at']
                optional_columns = ['sentiment_score', 'urgency_level', 'resolution_time_hours', 'satisfaction_rating']
                
                select_columns = base_columns + [col for col in optional_columns if col in columns]
                column_str = ', '.join(select_columns)
                
                cursor.execute(f"""
                    SELECT {column_str}
                    FROM complaints 
                    ORDER BY created_at DESC
                """)
                
                complaints = []
                for row in cursor.fetchall():
                    complaint = {}
                    # Safely add columns that exist
                    for col in select_columns:
                        try:
                            complaint[col] = row[col]
                        except (IndexError, KeyError):
                            complaint[col] = None
                    
                    # Get resolution details
                    resolution_cursor = conn.cursor()
                    try:
                        resolution_cursor.execute("""
                            SELECT solution_steps, effectiveness_score 
                            FROM resolutions 
                            WHERE complaint_id = ?
                        """, (row['id'],))
                        resolution_row = resolution_cursor.fetchone()
                        
                        if resolution_row:
                            complaint['resolution'] = resolution_row['solution_steps'] or 'No resolution steps available'
                            complaint['resolution_quality'] = resolution_row['effectiveness_score'] or 0
                        else:
                            complaint['resolution'] = 'No resolution available'
                            complaint['resolution_quality'] = 0
                    except Exception as res_error:
                        self.logger.warning(f"Could not get resolution for complaint {row['id']}: {str(res_error)}")
                        complaint['resolution'] = 'No resolution available'
                        complaint['resolution_quality'] = 0
                    
                    complaints.append(complaint)
                
                self.logger.info(f"Retrieved {len(complaints)} complaints from database")
                return complaints
                
        except Exception as e:
            self.logger.error(f"Error getting all complaints: {str(e)}")
            return []
    
    def get_complaints_by_category(self, category: str) -> List[Dict[str, Any]]:
        """
        Get complaints filtered by category
        
        Args:
            category: Category to filter by
            
        Returns:
            List of complaints in the specified category
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                # First check what columns exist
                cursor.execute("PRAGMA table_info(complaints)")
                columns = [row[1] for row in cursor.fetchall()]
                
                # Build query with available columns
                base_columns = ['id', 'title', 'description', 'category', 'priority', 'status', 'created_at', 'updated_at']
                optional_columns = ['sentiment_score', 'urgency_level', 'resolution_time_hours', 'satisfaction_rating']
                
                select_columns = base_columns + [col for col in optional_columns if col in columns]
                column_str = ', '.join(select_columns)
                
                cursor.execute(f"""
                    SELECT {column_str}
                    FROM complaints 
                    WHERE category = ? 
                    ORDER BY created_at DESC
                """, (category,))
                
                complaints = []
                for row in cursor.fetchall():
                    complaint = {}
                    # Safely add columns that exist
                    for col in select_columns:
                        try:
                            complaint[col] = row[col]
                        except (IndexError, KeyError):
                            complaint[col] = None
                    
                    # Get resolution details
                    resolution_cursor = conn.cursor()
                    try:
                        resolution_cursor.execute("""
                            SELECT solution_steps, effectiveness_score 
                            FROM resolutions 
                            WHERE complaint_id = ?
                        """, (row['id'],))
                        resolution_row = resolution_cursor.fetchone()
                        
                        if resolution_row:
                            complaint['resolution'] = resolution_row['solution_steps'] or 'No resolution steps available'
                            complaint['resolution_quality'] = resolution_row['effectiveness_score'] or 0
                        else:
                            complaint['resolution'] = 'No resolution available'
                            complaint['resolution_quality'] = 0
                    except Exception as res_error:
                        self.logger.warning(f"Could not get resolution for complaint {row['id']}: {str(res_error)}")
                        complaint['resolution'] = 'No resolution available'
                        complaint['resolution_quality'] = 0
                    
                    complaints.append(complaint)
                
                self.logger.info(f"Retrieved {len(complaints)} complaints for category: {category}")
                return complaints
                
        except Exception as e:
            self.logger.error(f"Error getting complaints by category: {str(e)}")
            return []
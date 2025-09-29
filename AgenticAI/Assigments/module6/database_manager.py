"""
Database manager for persistent storage of support tickets and resolutions
"""
import sqlite3
import json
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import logging
from pathlib import Path

class DatabaseManager:
    """Manages persistent storage of support tickets, resolutions, and knowledge base"""
    
    def __init__(self, db_path: str = None):
        """
        Initialize database manager
        
        Args:
            db_path: Path to SQLite database file
        """
        self.logger = logging.getLogger(__name__)
        
        # Default database path in data directory
        if db_path is None:
            data_dir = Path("data")
            data_dir.mkdir(exist_ok=True)
            db_path = str(data_dir / "support_agent.db")
        
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database with required tables"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Support tickets table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS support_tickets (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        ticket_id TEXT UNIQUE NOT NULL,
                        customer_message TEXT NOT NULL,
                        sentiment_analysis TEXT,
                        intent_classification TEXT,
                        suggested_response TEXT,
                        actual_response TEXT,
                        resolution_status TEXT DEFAULT 'pending',
                        urgency_level TEXT,
                        priority_score REAL,
                        tags TEXT,
                        category TEXT,
                        subcategory TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        resolved_at TIMESTAMP,
                        processing_time_ms INTEGER,
                        agent_confidence REAL,
                        customer_satisfaction INTEGER,
                        follow_up_required BOOLEAN DEFAULT 0,
                        escalated BOOLEAN DEFAULT 0,
                        similar_cases TEXT
                    )
                """)
                
                # Resolutions knowledge base table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS resolutions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        ticket_id TEXT,
                        issue_category TEXT NOT NULL,
                        issue_description TEXT NOT NULL,
                        resolution_steps TEXT NOT NULL,
                        effectiveness_score REAL,
                        resolution_time_hours REAL,
                        created_by TEXT DEFAULT 'system',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        used_count INTEGER DEFAULT 0,
                        last_used_at TIMESTAMP,
                        success_rate REAL DEFAULT 0.0,
                        customer_feedback TEXT,
                        resolution_type TEXT,
                        prerequisites TEXT,
                        related_cases TEXT,
                        FOREIGN KEY (ticket_id) REFERENCES support_tickets (ticket_id)
                    )
                """)
                
                # Customer history table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS customer_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        customer_id TEXT,
                        customer_email TEXT,
                        ticket_id TEXT,
                        interaction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        issue_type TEXT,
                        resolution_outcome TEXT,
                        satisfaction_rating INTEGER,
                        notes TEXT,
                        FOREIGN KEY (ticket_id) REFERENCES support_tickets (ticket_id)
                    )
                """)
                
                # Knowledge base articles table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS knowledge_articles (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        title TEXT NOT NULL,
                        content TEXT NOT NULL,
                        category TEXT,
                        tags TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        views INTEGER DEFAULT 0,
                        helpful_votes INTEGER DEFAULT 0,
                        unhelpful_votes INTEGER DEFAULT 0,
                        author TEXT,
                        status TEXT DEFAULT 'active'
                    )
                """)
                
                # File content storage table for search
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS file_content (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        file_id TEXT NOT NULL,
                        file_name TEXT NOT NULL,
                        file_type TEXT NOT NULL,
                        file_path TEXT,
                        content_text TEXT NOT NULL,
                        section_title TEXT,
                        section_number INTEGER,
                        category TEXT,
                        tags TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        file_size INTEGER,
                        content_type TEXT DEFAULT 'document'
                    )
                """)
                
                # Create indexes for better performance
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_tickets_created_at ON support_tickets (created_at)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_tickets_category ON support_tickets (category)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_tickets_urgency ON support_tickets (urgency_level)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_resolutions_category ON resolutions (issue_category)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_customer_history_customer ON customer_history (customer_id)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_file_content_file_id ON file_content (file_id)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_file_content_category ON file_content (category)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_file_content_type ON file_content (file_type)")
                
                conn.commit()
                self.logger.info(f"Database initialized successfully at {self.db_path}")
                
        except Exception as e:
            self.logger.error(f"Error initializing database: {str(e)}")
            raise
    
    def store_ticket(self, ticket_data: Dict[str, Any]) -> str:
        """
        Store a support ticket in the database
        
        Args:
            ticket_data: Ticket information dictionary
            
        Returns:
            ticket_id: Generated or provided ticket ID
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Generate ticket ID if not provided
                ticket_id = ticket_data.get('ticket_id') or f"TKT_{datetime.now().strftime('%Y%m%d%H%M%S')}"
                
                # Extract data with defaults
                customer_message = ticket_data.get('original_text', ticket_data.get('customer_message', ''))
                sentiment = json.dumps(ticket_data.get('sentiment_analysis', {}))
                intent = json.dumps(ticket_data.get('intent_classification', {}))
                suggested_response = ticket_data.get('response_generation', {}).get('suggested_response', '')
                urgency_level = ticket_data.get('sentiment_analysis', {}).get('urgency_level', 'medium')
                category = ticket_data.get('intent_classification', {}).get('intent_category', 'General')
                processing_time = ticket_data.get('processing_time_ms', 0)
                
                # Insert ticket
                cursor.execute("""
                    INSERT OR REPLACE INTO support_tickets 
                    (ticket_id, customer_message, sentiment_analysis, intent_classification, 
                     suggested_response, urgency_level, category, processing_time_ms, 
                     agent_confidence, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    ticket_id, customer_message, sentiment, intent, suggested_response,
                    urgency_level, category, processing_time, 
                    ticket_data.get('confidence', 0.5),
                    datetime.now().isoformat(),
                    datetime.now().isoformat()
                ))
                
                conn.commit()
                self.logger.info(f"Stored ticket {ticket_id} in database")
                return ticket_id
                
        except Exception as e:
            self.logger.error(f"Error storing ticket: {str(e)}")
            raise
    
    def store_resolution(self, resolution_data: Dict[str, Any]) -> int:
        """
        Store a resolution in the knowledge base
        
        Args:
            resolution_data: Resolution information
            
        Returns:
            resolution_id: Database ID of stored resolution
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO resolutions 
                    (ticket_id, issue_category, issue_description, resolution_steps,
                     effectiveness_score, resolution_time_hours, resolution_type)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    resolution_data.get('ticket_id'),
                    resolution_data.get('issue_category', 'General'),
                    resolution_data.get('issue_description', ''),
                    resolution_data.get('resolution_steps', ''),
                    resolution_data.get('effectiveness_score', 0.5),
                    resolution_data.get('resolution_time_hours', 0),
                    resolution_data.get('resolution_type', 'standard')
                ))
                
                resolution_id = cursor.lastrowid
                conn.commit()
                self.logger.info(f"Stored resolution {resolution_id} in database")
                return resolution_id
                
        except Exception as e:
            self.logger.error(f"Error storing resolution: {str(e)}")
            raise
    
    def search_similar_tickets(self, query: str, category: str = None, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search for similar tickets based on content and category
        
        Args:
            query: Search query
            category: Optional category filter
            limit: Maximum results to return
            
        Returns:
            List of similar tickets
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                sql = """
                    SELECT ticket_id, customer_message, sentiment_analysis, intent_classification,
                           suggested_response, actual_response, resolution_status, category,
                           urgency_level, created_at, customer_satisfaction
                    FROM support_tickets
                    WHERE customer_message LIKE ?
                """
                params = [f"%{query}%"]
                
                if category:
                    sql += " AND category = ?"
                    params.append(category)
                
                sql += " ORDER BY created_at DESC LIMIT ?"
                params.append(limit)
                
                cursor.execute(sql, params)
                rows = cursor.fetchall()
                
                tickets = []
                for row in rows:
                    tickets.append({
                        'ticket_id': row[0],
                        'customer_message': row[1],
                        'sentiment_analysis': json.loads(row[2]) if row[2] else {},
                        'intent_classification': json.loads(row[3]) if row[3] else {},
                        'suggested_response': row[4],
                        'actual_response': row[5],
                        'resolution_status': row[6],
                        'category': row[7],
                        'urgency_level': row[8],
                        'created_at': row[9],
                        'customer_satisfaction': row[10]
                    })
                
                return tickets
                
        except Exception as e:
            self.logger.error(f"Error searching tickets: {str(e)}")
            return []
    
    def get_resolutions_by_category(self, category: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get resolutions for a specific issue category
        
        Args:
            category: Issue category
            limit: Maximum results
            
        Returns:
            List of resolutions
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT id, issue_category, issue_description, resolution_steps,
                           effectiveness_score, resolution_time_hours, used_count,
                           success_rate, created_at
                    FROM resolutions
                    WHERE issue_category = ?
                    ORDER BY effectiveness_score DESC, used_count DESC
                    LIMIT ?
                """, (category, limit))
                
                rows = cursor.fetchall()
                
                resolutions = []
                for row in rows:
                    resolutions.append({
                        'id': row[0],
                        'issue_category': row[1],
                        'issue_description': row[2],
                        'resolution_steps': row[3],
                        'effectiveness_score': row[4],
                        'resolution_time_hours': row[5],
                        'used_count': row[6],
                        'success_rate': row[7],
                        'created_at': row[8]
                    })
                
                return resolutions
                
        except Exception as e:
            self.logger.error(f"Error fetching resolutions: {str(e)}")
            return []
    
    def update_ticket_resolution(self, ticket_id: str, actual_response: str, status: str = 'resolved', 
                                customer_satisfaction: int = None) -> bool:
        """
        Update ticket with actual response and resolution status
        
        Args:
            ticket_id: Ticket ID
            actual_response: The actual response sent
            status: Resolution status
            customer_satisfaction: Customer satisfaction rating (1-5)
            
        Returns:
            Success boolean
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                update_fields = [
                    "actual_response = ?",
                    "resolution_status = ?",
                    "updated_at = ?",
                    "resolved_at = ?"
                ]
                params = [actual_response, status, datetime.now().isoformat(), datetime.now().isoformat()]
                
                if customer_satisfaction is not None:
                    update_fields.append("customer_satisfaction = ?")
                    params.append(customer_satisfaction)
                
                params.append(ticket_id)  # for WHERE clause
                
                sql = f"UPDATE support_tickets SET {', '.join(update_fields)} WHERE ticket_id = ?"
                cursor.execute(sql, params)
                
                conn.commit()
                return cursor.rowcount > 0
                
        except Exception as e:
            self.logger.error(f"Error updating ticket resolution: {str(e)}")
            return False
    
    def get_ticket_statistics(self, days: int = 30) -> Dict[str, Any]:
        """
        Get ticket statistics for the specified period
        
        Args:
            days: Number of days to look back
            
        Returns:
            Statistics dictionary
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Date filter
                cutoff_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
                cutoff_date = cutoff_date.replace(day=cutoff_date.day - days)
                
                # Total tickets
                cursor.execute("SELECT COUNT(*) FROM support_tickets WHERE created_at >= ?", 
                             (cutoff_date.isoformat(),))
                total_tickets = cursor.fetchone()[0]
                
                # Tickets by category
                cursor.execute("""
                    SELECT category, COUNT(*) 
                    FROM support_tickets 
                    WHERE created_at >= ?
                    GROUP BY category
                """, (cutoff_date.isoformat(),))
                category_counts = dict(cursor.fetchall())
                
                # Tickets by urgency
                cursor.execute("""
                    SELECT urgency_level, COUNT(*) 
                    FROM support_tickets 
                    WHERE created_at >= ?
                    GROUP BY urgency_level
                """, (cutoff_date.isoformat(),))
                urgency_counts = dict(cursor.fetchall())
                
                # Average satisfaction
                cursor.execute("""
                    SELECT AVG(customer_satisfaction) 
                    FROM support_tickets 
                    WHERE customer_satisfaction IS NOT NULL AND created_at >= ?
                """, (cutoff_date.isoformat(),))
                avg_satisfaction = cursor.fetchone()[0] or 0
                
                # Resolution rate
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total,
                        SUM(CASE WHEN resolution_status = 'resolved' THEN 1 ELSE 0 END) as resolved
                    FROM support_tickets 
                    WHERE created_at >= ?
                """, (cutoff_date.isoformat(),))
                total, resolved = cursor.fetchone()
                resolution_rate = (resolved / total * 100) if total > 0 else 0
                
                return {
                    'total_tickets': total_tickets,
                    'category_distribution': category_counts,
                    'urgency_distribution': urgency_counts,
                    'average_satisfaction': round(avg_satisfaction, 2),
                    'resolution_rate': round(resolution_rate, 1),
                    'period_days': days
                }
                
        except Exception as e:
            self.logger.error(f"Error getting statistics: {str(e)}")
            return {}
    
    def get_recent_tickets(self, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Get most recent tickets
        
        Args:
            limit: Maximum tickets to return
            
        Returns:
            List of recent tickets
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT ticket_id, customer_message, category, urgency_level,
                           resolution_status, created_at, customer_satisfaction
                    FROM support_tickets
                    ORDER BY created_at DESC
                    LIMIT ?
                """, (limit,))
                
                rows = cursor.fetchall()
                
                tickets = []
                for row in rows:
                    tickets.append({
                        'ticket_id': row[0],
                        'customer_message': row[1][:100] + "..." if len(row[1]) > 100 else row[1],
                        'category': row[2],
                        'urgency_level': row[3],
                        'resolution_status': row[4],
                        'created_at': row[5],
                        'customer_satisfaction': row[6]
                    })
                
                return tickets
                
        except Exception as e:
            self.logger.error(f"Error fetching recent tickets: {str(e)}")
            return []
    
    def export_data(self, format: str = 'json', filename: str = None) -> str:
        """
        Export all data to specified format
        
        Args:
            format: Export format ('json' or 'csv')
            filename: Optional filename
            
        Returns:
            Path to exported file
        """
        try:
            if filename is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"support_data_export_{timestamp}.{format}"
            
            data_dir = Path("data")
            data_dir.mkdir(exist_ok=True)
            export_path = data_dir / filename
            
            with sqlite3.connect(self.db_path) as conn:
                if format.lower() == 'json':
                    # Export to JSON
                    export_data = {}
                    
                    # Export tickets
                    cursor = conn.cursor()
                    cursor.execute("SELECT * FROM support_tickets")
                    columns = [desc[0] for desc in cursor.description]
                    tickets = [dict(zip(columns, row)) for row in cursor.fetchall()]
                    export_data['tickets'] = tickets
                    
                    # Export resolutions
                    cursor.execute("SELECT * FROM resolutions")
                    columns = [desc[0] for desc in cursor.description]
                    resolutions = [dict(zip(columns, row)) for row in cursor.fetchall()]
                    export_data['resolutions'] = resolutions
                    
                    with open(export_path, 'w', encoding='utf-8') as f:
                        json.dump(export_data, f, indent=2, ensure_ascii=False)
                
                elif format.lower() == 'csv':
                    import csv
                    # Export tickets to CSV
                    cursor = conn.cursor()
                    cursor.execute("SELECT * FROM support_tickets")
                    
                    with open(export_path, 'w', newline='', encoding='utf-8') as f:
                        writer = csv.writer(f)
                        # Write headers
                        writer.writerow([desc[0] for desc in cursor.description])
                        # Write data
                        writer.writerows(cursor.fetchall())
            
            self.logger.info(f"Data exported to {export_path}")
            return str(export_path)
            
        except Exception as e:
            self.logger.error(f"Error exporting data: {str(e)}")
            raise
    
    def store_file_content(self, file_id: str, file_name: str, file_type: str, 
                          content_sections: List[Dict[str, Any]], file_path: str = None) -> bool:
        """
        Store file content in searchable format with source references
        
        Args:
            file_id: Unique identifier for the file
            file_name: Original filename
            file_type: Type of file (pdf, csv, txt, etc.)
            content_sections: List of content sections with text and metadata
            file_path: Optional path to original file
            
        Returns:
            Success boolean
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                for i, section in enumerate(content_sections):
                    cursor.execute("""
                        INSERT INTO file_content 
                        (file_id, file_name, file_type, file_path, content_text,
                         section_title, section_number, category, tags, file_size)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        file_id,
                        file_name,
                        file_type,
                        file_path,
                        section.get('content', ''),
                        section.get('title', f'Section {i+1}'),
                        i + 1,
                        section.get('category', 'General'),
                        section.get('tags', ''),
                        len(section.get('content', ''))
                    ))
                
                conn.commit()
                self.logger.info(f"Stored {len(content_sections)} content sections for file {file_id}")
                return True
                
        except Exception as e:
            self.logger.error(f"Error storing file content: {str(e)}")
            return False
    
    def hybrid_search(self, query: str, category: str = None, limit: int = 10) -> Dict[str, Any]:
        """
        Hybrid search combining keyword matching and semantic similarity
        
        Args:
            query: Search query
            category: Optional category filter
            limit: Maximum results to return
            
        Returns:
            Combined search results with source references
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Keyword search in tickets
                ticket_sql = """
                    SELECT ticket_id, customer_message, category, urgency_level, 
                           resolution_status, created_at, customer_satisfaction,
                           'ticket' as source_type, ticket_id as source_ref
                    FROM support_tickets
                    WHERE customer_message LIKE ?
                """
                params = [f"%{query}%"]
                
                if category:
                    ticket_sql += " AND category = ?"
                    params.append(category)
                
                ticket_sql += " ORDER BY created_at DESC LIMIT ?"
                params.append(limit // 2)
                
                cursor.execute(ticket_sql, params)
                ticket_results = cursor.fetchall()
                
                # Keyword search in file content
                file_sql = """
                    SELECT file_id, file_name, file_type, content_text, section_title,
                           section_number, category, created_at,
                           'file' as source_type, file_name as source_ref
                    FROM file_content
                    WHERE content_text LIKE ?
                """
                file_params = [f"%{query}%"]
                
                if category:
                    file_sql += " AND category = ?"
                    file_params.append(category)
                
                file_sql += " ORDER BY created_at DESC LIMIT ?"
                file_params.append(limit // 2)
                
                cursor.execute(file_sql, file_params)
                file_results = cursor.fetchall()
                
                # Keyword search in resolutions
                resolution_sql = """
                    SELECT id, issue_category, issue_description, resolution_steps,
                           effectiveness_score, used_count, success_rate, created_at,
                           'resolution' as source_type, 
                           CAST(id as TEXT) as source_ref
                    FROM resolutions
                    WHERE issue_description LIKE ? OR resolution_steps LIKE ?
                """
                resolution_params = [f"%{query}%", f"%{query}%"]
                
                if category:
                    resolution_sql += " AND issue_category = ?"
                    resolution_params.append(category)
                
                resolution_sql += " ORDER BY effectiveness_score DESC, used_count DESC LIMIT ?"
                resolution_params.append(limit // 3)
                
                cursor.execute(resolution_sql, resolution_params)
                resolution_results = cursor.fetchall()
                
                # Format results with relevance scoring
                results = []
                
                # Process ticket results
                for row in ticket_results:
                    relevance_score = self._calculate_relevance_score(query, row[1])
                    results.append({
                        'type': 'ticket',
                        'id': row[0],
                        'content': row[1][:200] + "..." if len(row[1]) > 200 else row[1],
                        'full_content': row[1],
                        'category': row[2],
                        'metadata': {
                            'urgency': row[3],
                            'status': row[4],
                            'date': row[5],
                            'satisfaction': row[6]
                        },
                        'source_type': row[7],
                        'source_reference': row[8],
                        'relevance_score': relevance_score
                    })
                
                # Process file results  
                for row in file_results:
                    relevance_score = self._calculate_relevance_score(query, row[3])
                    results.append({
                        'type': 'file_content',
                        'id': row[0],
                        'content': row[3][:200] + "..." if len(row[3]) > 200 else row[3],
                        'full_content': row[3],
                        'category': row[6],
                        'metadata': {
                            'file_name': row[1],
                            'file_type': row[2],
                            'section': row[4],
                            'section_number': row[5],
                            'date': row[7]
                        },
                        'source_type': row[8],
                        'source_reference': f"{row[9]} (Section {row[5]})",
                        'relevance_score': relevance_score
                    })
                
                # Process resolution results
                for row in resolution_results:
                    relevance_score = self._calculate_relevance_score(query, row[2] + " " + row[3])
                    results.append({
                        'type': 'resolution',
                        'id': row[0],
                        'content': row[2][:100] + "..." if len(row[2]) > 100 else row[2],
                        'full_content': row[3],
                        'category': row[1],
                        'metadata': {
                            'effectiveness': row[4],
                            'usage_count': row[5],
                            'success_rate': row[6],
                            'date': row[7]
                        },
                        'source_type': row[8],
                        'source_reference': f"Resolution #{row[0]}",
                        'relevance_score': relevance_score
                    })
                
                # Sort by relevance score
                results.sort(key=lambda x: x['relevance_score'], reverse=True)
                
                return {
                    'query': query,
                    'category': category,
                    'total_results': len(results),
                    'results': results[:limit],
                    'search_type': 'hybrid',
                    'timestamp': datetime.now().isoformat()
                }
                
        except Exception as e:
            self.logger.error(f"Error in hybrid search: {str(e)}")
            return {
                'query': query,
                'category': category,
                'total_results': 0,
                'results': [],
                'error': str(e),
                'search_type': 'hybrid',
                'timestamp': datetime.now().isoformat()
            }
    
    def _calculate_relevance_score(self, query: str, content: str) -> float:
        """
        Calculate relevance score based on keyword matching and positioning
        
        Args:
            query: Search query
            content: Content to score
            
        Returns:
            Relevance score between 0 and 1
        """
        if not query or not content:
            return 0.0
        
        query_lower = query.lower()
        content_lower = content.lower()
        
        # Exact phrase match gets highest score
        if query_lower in content_lower:
            position = content_lower.find(query_lower)
            # Earlier position gets higher score
            position_score = max(0, 1 - (position / len(content_lower)))
            return min(1.0, 0.8 + position_score * 0.2)
        
        # Individual word matches
        query_words = query_lower.split()
        content_words = content_lower.split()
        
        matches = sum(1 for word in query_words if word in content_words)
        word_score = matches / len(query_words) if query_words else 0
        
        return word_score * 0.6
    
    def get_file_content_by_category(self, category: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get file content by category for reference
        
        Args:
            category: Content category
            limit: Maximum results
            
        Returns:
            List of file content entries
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT file_id, file_name, file_type, content_text, section_title,
                           section_number, category, created_at
                    FROM file_content
                    WHERE category = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                """, (category, limit))
                
                rows = cursor.fetchall()
                
                content = []
                for row in rows:
                    content.append({
                        'file_id': row[0],
                        'file_name': row[1],
                        'file_type': row[2],
                        'content': row[3],
                        'section_title': row[4],
                        'section_number': row[5],
                        'category': row[6],
                        'created_at': row[7],
                        'source_reference': f"{row[1]} (Section {row[5]})"
                    })
                
                return content
                
        except Exception as e:
            self.logger.error(f"Error getting file content by category: {str(e)}")
            return []
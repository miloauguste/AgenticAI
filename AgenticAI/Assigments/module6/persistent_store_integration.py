#!/usr/bin/env python3
"""
Integration layer between Persistent Store and Support Agent
Provides seamless integration for complaints and resolutions across sessions
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import uuid

from persistent_store import PersistentStore, Complaint, Resolution, ComplaintStatus, Priority
from config import settings

class PersistentStoreIntegration:
    """
    Integration layer that extends the support agent with persistent storage capabilities
    """
    
    def __init__(self, project_name: str = None):
        """
        Initialize integration layer
        
        Args:
            project_name: Name of the project (defaults to settings or 'default')
        """
        self.logger = logging.getLogger(__name__)
        
        # Determine project name from settings or use default
        self.project_name = project_name or getattr(settings, 'PROJECT_NAME', 'support_system')
        
        # Initialize persistent store
        self.store = PersistentStore(project_name=self.project_name)
        
        self.logger.info(f"Initialized persistent store integration for project: {self.project_name}")
    
    def store_processed_ticket(self, ticket_result: Dict[str, Any]) -> str:
        """
        Store a processed support ticket as a complaint with resolution
        
        Args:
            ticket_result: Result from support agent processing
            
        Returns:
            Complaint ID
        """
        try:
            # Extract data from ticket result
            ticket_id = ticket_result.get('ticket_id', str(uuid.uuid4()))
            original_text = ticket_result.get('original_text', '')
            
            # Extract sentiment analysis
            sentiment = ticket_result.get('sentiment_analysis', {})
            sentiment_score = sentiment.get('confidence', 0.0) if sentiment.get('sentiment') == 'positive' else -sentiment.get('confidence', 0.0)
            
            # Extract intent classification
            intent = ticket_result.get('intent_classification', {})
            category = intent.get('intent_category', 'general')
            subcategory = intent.get('predicted_intent', 'unknown')
            
            # Determine priority based on urgency and sentiment
            urgency_score = self._calculate_urgency_score(ticket_result)
            priority = self._determine_priority(urgency_score, sentiment_score)
            
            # Extract customer information
            customer_id = self._extract_customer_id(ticket_result)
            
            # Create complaint record
            complaint_data = {
                'id': ticket_id,
                'customer_id': customer_id,
                'title': self._generate_title_from_text(original_text),
                'description': original_text,
                'category': category,
                'subcategory': subcategory,
                'priority': priority,
                'status': ComplaintStatus.OPEN.value,
                'source': 'support_agent',
                'channel': 'automated_processing',
                'sentiment_score': sentiment_score,
                'urgency_score': urgency_score,
                'tags': self._extract_tags(ticket_result),
                'metadata': {
                    'processing_timestamp': ticket_result.get('processing_timestamp'),
                    'processing_time_ms': ticket_result.get('processing_time_ms'),
                    'agent_confidence': intent.get('confidence', 0.0),
                    'workflow_id': ticket_result.get('workflow_id'),
                    'tools_used': ticket_result.get('comprehensive_analysis', {}).get('tools_used', [])
                }
            }
            
            complaint_id = self.store.add_complaint(complaint_data)
            
            # If there's a suggested response, create a resolution record
            response_data = ticket_result.get('response_generation', {})
            if response_data and response_data.get('suggested_response'):
                self._create_resolution_from_response(complaint_id, response_data, category)
            
            self.logger.info(f"Stored ticket {ticket_id} as complaint {complaint_id}")
            return complaint_id
            
        except Exception as e:
            self.logger.error(f"Error storing processed ticket: {str(e)}")
            raise
    
    def find_similar_complaints(self, description: str, category: str = None, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Find similar complaints for pattern recognition and resolution reuse
        
        Args:
            description: Description to match against
            category: Optional category filter
            limit: Maximum results
            
        Returns:
            List of similar complaints with similarity scores
        """
        try:
            # Search for complaints with text similarity
            similar_complaints = self.store.search_complaints(
                query=self._extract_keywords_for_search(description),
                category=category,
                limit=limit * 2  # Get more to filter by similarity
            )
            
            # Calculate similarity scores and rank
            scored_complaints = []
            for complaint in similar_complaints:
                similarity = self._calculate_text_similarity(description, complaint['description'])
                if similarity > 0.3:  # Minimum similarity threshold
                    complaint['similarity_score'] = similarity
                    scored_complaints.append(complaint)
            
            # Sort by similarity and return top results
            scored_complaints.sort(key=lambda x: x['similarity_score'], reverse=True)
            return scored_complaints[:limit]
            
        except Exception as e:
            self.logger.error(f"Error finding similar complaints: {str(e)}")
            return []
    
    def get_best_resolutions(self, category: str, description: str = None, limit: int = 3) -> List[Dict[str, Any]]:
        """
        Get best resolutions for a given category and description
        
        Args:
            category: Issue category
            description: Optional description for better matching
            limit: Maximum results
            
        Returns:
            List of best matching resolutions
        """
        try:
            # Search resolutions by category and effectiveness
            resolutions = self.store.search_resolutions(
                query=self._extract_keywords_for_search(description) if description else None,
                category=category,
                min_effectiveness=0.5,
                sort_by="effectiveness_score",
                limit=limit
            )
            
            # Enhance with usage statistics
            for resolution in resolutions:
                resolution['recommendation_score'] = self._calculate_recommendation_score(resolution)
            
            # Sort by recommendation score
            resolutions.sort(key=lambda x: x['recommendation_score'], reverse=True)
            
            return resolutions[:limit]
            
        except Exception as e:
            self.logger.error(f"Error getting best resolutions: {str(e)}")
            return []
    
    def create_resolution_template(self, category: str, title: str, description: str, 
                                  solution_steps: List[str], created_by: str = "system") -> str:
        """
        Create a reusable resolution template
        
        Args:
            category: Resolution category
            title: Resolution title
            description: Resolution description
            solution_steps: List of solution steps
            created_by: Creator identifier
            
        Returns:
            Resolution ID
        """
        try:
            resolution_data = {
                'id': str(uuid.uuid4()),
                'complaint_id': None,  # Template not tied to specific complaint
                'title': title,
                'description': description,
                'solution_steps': solution_steps,
                'resolution_type': 'template',
                'category': category,
                'effectiveness_score': 0.8,  # Start with good score for templates
                'resolution_time_hours': 0.0,
                'created_by': created_by,
                'metadata': {
                    'is_template': True,
                    'created_via': 'persistent_store_integration'
                }
            }
            
            resolution_id = self.store.add_resolution(resolution_data)
            self.logger.info(f"Created resolution template: {resolution_id}")
            return resolution_id
            
        except Exception as e:
            self.logger.error(f"Error creating resolution template: {str(e)}")
            raise
    
    def track_resolution_usage(self, resolution_id: str, effectiveness_feedback: float = None, 
                              resolution_time: float = None) -> bool:
        """
        Track when a resolution is used and update its effectiveness
        
        Args:
            resolution_id: ID of the resolution used
            effectiveness_feedback: Feedback score (0.0 to 1.0)
            resolution_time: Time taken to resolve in hours
            
        Returns:
            Success status
        """
        try:
            success = self.store.mark_resolution_used(resolution_id, effectiveness_feedback)
            
            if success and resolution_time:
                # Update average resolution time
                resolution = self.store.get_resolution_by_id(resolution_id)
                if resolution:
                    # Calculate weighted average of resolution time
                    current_time = resolution.get('resolution_time_hours', 0.0)
                    usage_count = resolution.get('used_count', 1)
                    
                    if current_time > 0:
                        new_avg_time = ((current_time * (usage_count - 1)) + resolution_time) / usage_count
                    else:
                        new_avg_time = resolution_time
                    
                    # Update in database (would need additional method in store)
                    self.logger.info(f"Updated resolution {resolution_id} usage metrics")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error tracking resolution usage: {str(e)}")
            return False
    
    def get_complaint_history(self, customer_id: str = None, days: int = 30) -> List[Dict[str, Any]]:
        """
        Get complaint history for analysis and pattern recognition
        
        Args:
            customer_id: Optional customer filter
            days: Number of days to look back
            
        Returns:
            List of complaints with trends
        """
        try:
            date_from = datetime.now() - timedelta(days=days)
            
            complaints = self.store.search_complaints(
                customer_id=customer_id,
                date_from=date_from,
                limit=100
            )
            
            # Add trend analysis
            history_data = {
                'complaints': complaints,
                'summary': self._analyze_complaint_trends(complaints),
                'patterns': self._identify_patterns(complaints)
            }
            
            return history_data
            
        except Exception as e:
            self.logger.error(f"Error getting complaint history: {str(e)}")
            return []
    
    def export_learning_data(self, include_resolutions: bool = True, 
                           min_effectiveness: float = 0.7) -> Dict[str, Any]:
        """
        Export high-quality data for training or transfer learning
        
        Args:
            include_resolutions: Whether to include resolution data
            min_effectiveness: Minimum effectiveness score for resolutions
            
        Returns:
            Structured learning data
        """
        try:
            learning_data = {
                'metadata': {
                    'project': self.project_name,
                    'exported_at': datetime.now().isoformat(),
                    'quality_threshold': min_effectiveness
                },
                'complaints': [],
                'resolutions': [],
                'patterns': {},
                'statistics': {}
            }
            
            # Get all complaints for pattern analysis
            all_complaints = self.store.search_complaints(limit=1000)
            
            # Filter high-quality complaints (resolved with satisfaction)
            quality_complaints = [
                c for c in all_complaints 
                if c['status'] == 'resolved' and 
                (c.get('customer_satisfaction') or 0) >= 4
            ]
            
            learning_data['complaints'] = quality_complaints
            
            if include_resolutions:
                # Get high-effectiveness resolutions
                quality_resolutions = self.store.search_resolutions(
                    min_effectiveness=min_effectiveness,
                    limit=500
                )
                learning_data['resolutions'] = quality_resolutions
            
            # Generate patterns and statistics
            learning_data['patterns'] = self._extract_learning_patterns(quality_complaints)
            learning_data['statistics'] = self.store.get_analytics(days=90)
            
            return learning_data
            
        except Exception as e:
            self.logger.error(f"Error exporting learning data: {str(e)}")
            return {}
    
    def get_project_insights(self) -> Dict[str, Any]:
        """
        Get comprehensive insights about the current project
        
        Returns:
            Project insights and recommendations
        """
        try:
            insights = {
                'project_stats': self.store.get_project_stats(),
                'analytics': self.store.get_analytics(days=30),
                'trends': self._analyze_recent_trends(),
                'recommendations': self._generate_recommendations()
            }
            
            return insights
            
        except Exception as e:
            self.logger.error(f"Error getting project insights: {str(e)}")
            return {}
    
    def import_historical_data(self, data_source: str, format: str = "json") -> bool:
        """
        Import historical complaint and resolution data from external sources
        
        Args:
            data_source: Path to data file or data dictionary
            format: Data format ('json', 'csv', 'dict')
            
        Returns:
            Success status
        """
        try:
            if format == "json" and isinstance(data_source, str):
                return self.store.import_data(data_source)
            elif format == "dict" and isinstance(data_source, dict):
                # Handle direct dictionary import
                success_count = 0
                
                for complaint_data in data_source.get('complaints', []):
                    try:
                        self.store.add_complaint(complaint_data)
                        success_count += 1
                    except Exception as e:
                        self.logger.warning(f"Failed to import complaint: {str(e)}")
                
                for resolution_data in data_source.get('resolutions', []):
                    try:
                        self.store.add_resolution(resolution_data)
                        success_count += 1
                    except Exception as e:
                        self.logger.warning(f"Failed to import resolution: {str(e)}")
                
                self.logger.info(f"Successfully imported {success_count} records")
                return success_count > 0
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error importing historical data: {str(e)}")
            return False
    
    # Helper methods
    def _calculate_urgency_score(self, ticket_result: Dict[str, Any]) -> float:
        """Calculate urgency score from ticket analysis"""
        sentiment = ticket_result.get('sentiment_analysis', {})
        escalation = ticket_result.get('escalation_indicators', {})
        
        base_score = 0.5
        
        # Negative sentiment increases urgency
        if sentiment.get('sentiment') == 'negative':
            base_score += sentiment.get('confidence', 0.3)
        
        # Escalation indicators increase urgency
        if escalation.get('requires_escalation'):
            base_score += 0.3
        
        # Keyword-based urgency
        urgent_keywords = ['urgent', 'asap', 'immediately', 'emergency', 'critical']
        text = ticket_result.get('original_text', '').lower()
        for keyword in urgent_keywords:
            if keyword in text:
                base_score += 0.2
                break
        
        return min(1.0, base_score)
    
    def _determine_priority(self, urgency_score: float, sentiment_score: float) -> str:
        """Determine priority level based on scores"""
        if urgency_score > 0.8 or sentiment_score < -0.7:
            return Priority.CRITICAL.value
        elif urgency_score > 0.6 or sentiment_score < -0.5:
            return Priority.HIGH.value
        elif urgency_score > 0.4:
            return Priority.MEDIUM.value
        else:
            return Priority.LOW.value
    
    def _extract_customer_id(self, ticket_result: Dict[str, Any]) -> str:
        """Extract or generate customer ID"""
        # Try to extract from metadata or generate anonymous ID
        metadata = ticket_result.get('metadata', {})
        customer_id = metadata.get('customer_id')
        
        if not customer_id:
            # Generate anonymous ID based on ticket content hash
            text = ticket_result.get('original_text', '')
            import hashlib
            customer_id = f"anon_{hashlib.md5(text.encode()).hexdigest()[:8]}"
        
        return customer_id
    
    def _generate_title_from_text(self, text: str, max_length: int = 100) -> str:
        """Generate title from text content"""
        # Simple title generation - could be enhanced with NLP
        sentences = text.split('.')
        if sentences:
            title = sentences[0].strip()
            if len(title) > max_length:
                title = title[:max_length-3] + "..."
            return title
        return text[:max_length]
    
    def _extract_tags(self, ticket_result: Dict[str, Any]) -> List[str]:
        """Extract tags from ticket analysis"""
        tags = []
        
        # Add category as tag
        intent = ticket_result.get('intent_classification', {})
        if intent.get('intent_category'):
            tags.append(intent['intent_category'])
        
        # Add sentiment as tag
        sentiment = ticket_result.get('sentiment_analysis', {})
        if sentiment.get('sentiment'):
            tags.append(f"sentiment_{sentiment['sentiment']}")
        
        # Add escalation tag if needed
        escalation = ticket_result.get('escalation_indicators', {})
        if escalation.get('requires_escalation'):
            tags.append('escalation_required')
        
        return tags
    
    def _create_resolution_from_response(self, complaint_id: str, response_data: Dict[str, Any], category: str):
        """Create resolution record from response generation"""
        try:
            suggested_response = response_data.get('suggested_response', '')
            
            if suggested_response:
                resolution_data = {
                    'complaint_id': complaint_id,
                    'title': f"Automated Response - {category}",
                    'description': f"Generated response for {category} inquiry",
                    'solution_steps': [suggested_response],
                    'resolution_type': 'automatic',
                    'category': category,
                    'effectiveness_score': response_data.get('confidence', 0.7),
                    'resolution_time_hours': 0.1,  # Immediate automated response
                    'created_by': 'support_agent_ai',
                    'metadata': {
                        'response_tone': response_data.get('response_tone'),
                        'estimated_resolution_time': response_data.get('estimated_resolution_time'),
                        'alternative_responses_count': len(response_data.get('alternative_responses', []))
                    }
                }
                
                self.store.add_resolution(resolution_data)
                
        except Exception as e:
            self.logger.warning(f"Failed to create resolution from response: {str(e)}")
    
    def _extract_keywords_for_search(self, text: str) -> str:
        """Extract key search terms from text"""
        if not text:
            return ""
        
        # Simple keyword extraction - could be enhanced with NLP
        import re
        words = re.findall(r'\b\w{4,}\b', text.lower())
        
        # Filter common words
        stop_words = {'that', 'this', 'with', 'have', 'will', 'from', 'they', 'been', 'said'}
        keywords = [w for w in words if w not in stop_words]
        
        return ' '.join(keywords[:5])  # Top 5 keywords
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple text similarity score"""
        # Simple word overlap similarity - could be enhanced with embeddings
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _calculate_recommendation_score(self, resolution: Dict[str, Any]) -> float:
        """Calculate recommendation score for resolution"""
        effectiveness = resolution.get('effectiveness_score', 0.0)
        usage_count = resolution.get('used_count', 0)
        success_rate = resolution.get('success_rate', 0.0)
        
        # Weighted score considering effectiveness, usage, and success rate
        score = (effectiveness * 0.4) + (min(usage_count / 10.0, 1.0) * 0.3) + (success_rate * 0.3)
        
        return score
    
    def _analyze_complaint_trends(self, complaints: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze trends in complaints"""
        if not complaints:
            return {}
        
        from collections import Counter
        
        # Category trends
        categories = [c['category'] for c in complaints]
        category_trends = Counter(categories)
        
        # Priority trends
        priorities = [c['priority'] for c in complaints]
        priority_trends = Counter(priorities)
        
        # Status trends
        statuses = [c['status'] for c in complaints]
        status_trends = Counter(statuses)
        
        return {
            'category_distribution': dict(category_trends),
            'priority_distribution': dict(priority_trends),
            'status_distribution': dict(status_trends),
            'total_complaints': len(complaints),
            'resolution_rate': status_trends.get('resolved', 0) / len(complaints)
        }
    
    def _identify_patterns(self, complaints: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Identify patterns in complaint data"""
        patterns = {}
        
        # Time patterns (could be enhanced)
        if complaints:
            # Find peak complaint hours/days
            creation_times = [c['created_at'] for c in complaints if c.get('created_at')]
            if creation_times:
                patterns['peak_times'] = "Analysis would require datetime parsing"
        
        # Recurring issues
        descriptions = [c['description'][:100] for c in complaints]
        # Could implement more sophisticated pattern detection here
        
        return patterns
    
    def _analyze_recent_trends(self) -> Dict[str, Any]:
        """Analyze recent trends in the data"""
        try:
            # Get recent data for trend analysis
            recent_complaints = self.store.search_complaints(
                date_from=datetime.now() - timedelta(days=7),
                limit=100
            )
            
            previous_complaints = self.store.search_complaints(
                date_from=datetime.now() - timedelta(days=14),
                date_to=datetime.now() - timedelta(days=7),
                limit=100
            )
            
            trends = {
                'recent_volume': len(recent_complaints),
                'previous_volume': len(previous_complaints),
                'volume_change': len(recent_complaints) - len(previous_complaints),
                'trend_direction': 'increasing' if len(recent_complaints) > len(previous_complaints) else 'decreasing'
            }
            
            return trends
            
        except Exception as e:
            self.logger.error(f"Error analyzing trends: {str(e)}")
            return {}
    
    def _generate_recommendations(self) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        try:
            stats = self.store.get_project_stats()
            analytics = self.store.get_analytics()
            
            # Resolution rate recommendations
            complaint_stats = analytics.get('complaints', {})
            resolved_complaints = complaint_stats.get('resolved_complaints', 0) or 0
            total_complaints = complaint_stats.get('total_complaints', 1) or 1
            resolution_rate = resolved_complaints / max(total_complaints, 1)
            
            if resolution_rate < 0.8:
                recommendations.append("Consider reviewing resolution processes - current resolution rate is below 80%")
            
            # Category-based recommendations
            categories = analytics.get('categories', [])
            if categories and len(categories) > 0:
                top_category = categories[0]
                category_name = top_category.get('category', 'unknown') if top_category else 'unknown'
                category_count = top_category.get('count', 0) if top_category else 0
                if category_name and category_name != 'unknown':
                    recommendations.append(f"Focus on '{category_name}' category - highest volume with {category_count} cases")
            
            # Resolution effectiveness recommendations
            resolution_stats = analytics.get('resolutions', {})
            avg_effectiveness = resolution_stats.get('avg_effectiveness', 0)
            
            # Ensure avg_effectiveness is not None before comparison
            if avg_effectiveness is not None and avg_effectiveness < 0.7:
                recommendations.append("Review and improve resolution templates - current effectiveness is below 70%")
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error generating recommendations: {str(e)}")
            return ["Unable to generate recommendations at this time"]
    
    def _extract_learning_patterns(self, complaints: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract patterns for machine learning"""
        patterns = {}
        
        if complaints:
            # Category patterns
            categories = {}
            for complaint in complaints:
                cat = complaint['category']
                if cat not in categories:
                    categories[cat] = {
                        'count': 0,
                        'avg_resolution_time': 0,
                        'avg_satisfaction': 0,
                        'common_keywords': []
                    }
                categories[cat]['count'] += 1
                
                # Could extract more sophisticated patterns here
            
            patterns['category_patterns'] = categories
        
        return patterns
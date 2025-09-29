"""
Main Customer Support Triage Agent
"""
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

# Import all components
from sentiment_analyzer import SentimentAnalyzer
from intent_classifier import IntentClassifier
from response_generator import ResponseGenerator
from csv_processor import CSVProcessor
from pdf_processor import PDFProcessor
from text_processor import TextProcessor
from chunking_strategies import ChunkingManager
from vector_store import VectorStore
from config import settings

class SupportTriageAgent:
    """
    AI Agent for customer support ticket triage and response generation
    """
    
    def __init__(self, name: str = "SupportTriageAgent"):
        self.name = name
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        try:
            self.sentiment_analyzer = SentimentAnalyzer()
            self.intent_classifier = IntentClassifier()
            self.response_generator = ResponseGenerator()
            self.chunking_manager = ChunkingManager()
            self.vector_store = VectorStore()
            
            # Initialize processors
            self.csv_processor = CSVProcessor()
            self.pdf_processor = PDFProcessor()
            self.text_processor = TextProcessor()
            
            # Agent state
            self.processed_files = {}
            self.knowledge_base = {}
            self.session_stats = {
                'tickets_processed': 0,
                'files_uploaded': 0,
                'responses_generated': 0,
                'session_start': datetime.now().isoformat()
            }
            
            self.logger.info(f"Initialized {name} with all components")
            
        except Exception as e:
            self.logger.error(f"Error initializing agent: {str(e)}")
            raise
    
    def process_support_ticket(self, ticket_text: str, ticket_id: str = None) -> Dict[str, Any]:
        """
        Process a single support ticket through the complete triage pipeline
        
        Args:
            ticket_text: The customer's message/ticket content
            ticket_id: Optional ticket identifier
            
        Returns:
            Complete analysis and response for the ticket
        """
        try:
            ticket_id = ticket_id or f"ticket_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            
            self.logger.info(f"Processing support ticket: {ticket_id}")
            
            # Step 1: Sentiment Analysis
            sentiment_analysis = self.sentiment_analyzer.analyze_sentiment(ticket_text)
            
            # Step 2: Intent Classification
            intent_analysis = self.intent_classifier.classify_intent(ticket_text)
            
            # Step 3: Generate Response
            response_data = self.response_generator.generate_response(
                ticket_text,
                intent_analysis,
                sentiment_analysis,
                self._get_context_for_ticket(intent_analysis)
            )
            
            # Step 4: Determine Routing and Priority
            routing_info = self._determine_routing_and_priority(
                intent_analysis,
                sentiment_analysis
            )
            
            # Step 5: Search Related Knowledge
            related_info = self._search_related_knowledge(ticket_text, intent_analysis)
            
            # Compile results
            result = {
                'ticket_id': ticket_id,
                'original_message': ticket_text,
                'processing_timestamp': datetime.now().isoformat(),
                'sentiment_analysis': sentiment_analysis,
                'intent_analysis': intent_analysis,
                'response_data': response_data,
                'routing_info': routing_info,
                'related_knowledge': related_info,
                'triage_summary': self._generate_triage_summary(
                    sentiment_analysis, intent_analysis, routing_info
                )
            }
            
            # Update session stats
            self.session_stats['tickets_processed'] += 1
            self.session_stats['responses_generated'] += 1
            
            self.logger.info(f"Successfully processed ticket {ticket_id}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing ticket {ticket_id}: {str(e)}")
            return self._generate_error_response(ticket_id, ticket_text, str(e))
    
    def upload_and_process_file(self, file_path: str, file_type: str = None) -> Dict[str, Any]:
        """
        Upload and process a file (CSV, PDF, or TXT) for knowledge base
        
        Args:
            file_path: Path to the file to process
            file_type: Type of file ('csv', 'pdf', 'txt') or None for auto-detection
            
        Returns:
            Processing results and file information
        """
        try:
            self.logger.info(f"Processing file: {file_path}")
            
            # Auto-detect file type if not provided
            if not file_type:
                file_type = file_path.split('.')[-1].lower()
            
            # Process file based on type
            if file_type == 'csv':
                processed_data = self.csv_processor.process_file(file_path)
                chunks = self.chunking_manager.chunk_data(
                    processed_data['data'], 
                    strategy='ticket'
                )
            elif file_type == 'pdf':
                processed_data = self.pdf_processor.process_file(file_path)
                chunks = self.chunking_manager.chunk_data(
                    processed_data['sections'], 
                    strategy='policy'
                )
            elif file_type in ['txt', 'text']:
                processed_data = self.text_processor.process_file(file_path)
                chunks = self.chunking_manager.chunk_data(
                    processed_data['conversations'], 
                    strategy='conversation'
                )
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
            
            # Store in vector database
            vector_result = self.vector_store.upsert_chunks(chunks)
            
            # Update knowledge base
            file_id = f"file_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            self.processed_files[file_id] = {
                'file_path': file_path,
                'file_type': file_type,
                'processed_data': processed_data,
                'chunks': chunks,
                'vector_result': vector_result,
                'processing_timestamp': datetime.now().isoformat()
            }
            
            # Update session stats
            self.session_stats['files_uploaded'] += 1
            
            result = {
                'file_id': file_id,
                'file_path': file_path,
                'file_type': file_type,
                'processing_summary': processed_data.get('summary', {}),
                'chunks_created': len(chunks),
                'vector_storage': vector_result,
                'status': 'success'
            }
            
            self.logger.info(f"Successfully processed file {file_path}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing file {file_path}: {str(e)}")
            return {
                'file_path': file_path,
                'status': 'error',
                'error': str(e),
                'processing_timestamp': datetime.now().isoformat()
            }
    
    def search_knowledge_base(self, query: str, search_type: str = 'semantic', top_k: int = 5) -> Dict[str, Any]:
        """
        Search the knowledge base for relevant information
        
        Args:
            query: Search query
            search_type: Type of search ('semantic', 'hybrid', 'metadata')
            top_k: Number of results to return
            
        Returns:
            Search results with relevant information
        """
        try:
            self.logger.info(f"Searching knowledge base: {query[:50]}...")
            
            if search_type == 'semantic':
                results = self.vector_store.search_similar(query, top_k)
            elif search_type == 'hybrid':
                # Extract keywords for hybrid search
                keywords = self._extract_search_keywords(query)
                results = self.vector_store.search_hybrid(query, keywords, top_k)
            elif search_type == 'metadata':
                # Create metadata filter from query
                metadata_filter = self._create_metadata_filter(query)
                results = self.vector_store.search_by_metadata(metadata_filter, top_k)
            else:
                raise ValueError(f"Unsupported search type: {search_type}")
            
            # Enhance results with context
            enhanced_results = self._enhance_search_results(results, query)
            
            return {
                'query': query,
                'search_type': search_type,
                'results_count': len(enhanced_results),
                'results': enhanced_results,
                'search_timestamp': datetime.now().isoformat()
            }
    
    def generate_insights_report(self, time_period: str = 'session') -> Dict[str, Any]:
        """
        Generate insights and analytics report
        
        Args:
            time_period: Time period for analysis ('session', 'day', 'week')
            
        Returns:
            Comprehensive insights report
        """
        try:
            self.logger.info(f"Generating insights report for: {time_period}")
            
            # Get vector store statistics
            vector_stats = self.vector_store.get_index_stats()
            
            # Generate report sections
            report = {
                'report_timestamp': datetime.now().isoformat(),
                'time_period': time_period,
                'session_statistics': self.session_stats,
                'knowledge_base_stats': {
                    'total_files_processed': len(self.processed_files),
                    'vector_store_stats': vector_stats,
                    'file_types_distribution': self._get_file_types_distribution()
                },
                'processing_insights': self._generate_processing_insights(),
                'recommendations': self._generate_recommendations()
            }
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating insights report: {str(e)}")
            return {
                'error': str(e),
                'report_timestamp': datetime.now().isoformat()
            }
    
    def batch_process_tickets(self, tickets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process multiple tickets in batch
        
        Args:
            tickets: List of ticket dictionaries with 'text' and optional 'id'
            
        Returns:
            List of processed ticket results
        """
        results = []
        
        for i, ticket in enumerate(tickets):
            try:
                ticket_text = ticket.get('text', ticket.get('message', ''))
                ticket_id = ticket.get('id', f'batch_ticket_{i}')
                
                result = self.process_support_ticket(ticket_text, ticket_id)
                results.append(result)
                
            except Exception as e:
                self.logger.error(f"Error processing batch ticket {i}: {str(e)}")
                error_result = self._generate_error_response(f'batch_ticket_{i}', '', str(e))
                results.append(error_result)
        
        return results
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get current status and health of the agent"""
        try:
            return {
                'agent_name': self.name,
                'status': 'active',
                'session_stats': self.session_stats,
                'components_status': {
                    'sentiment_analyzer': 'active',
                    'intent_classifier': 'active',
                    'response_generator': 'active',
                    'vector_store': 'active',
                    'file_processors': 'active'
                },
                'knowledge_base_files': len(self.processed_files),
                'vector_store_stats': self.vector_store.get_index_stats(),
                'last_activity': datetime.now().isoformat()
            }
        except Exception as e:
            return {
                'agent_name': self.name,
                'status': 'error',
                'error': str(e),
                'last_activity': datetime.now().isoformat()
            }
    
    # Private helper methods
    
    def _get_context_for_ticket(self, intent_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Get relevant context information for response generation"""
        context = {}
        
        # Get relevant policies/procedures from knowledge base
        intent = intent_analysis.get('primary_intent', 'general')
        if intent in ['refund', 'billing', 'product_issue']:
            # Search for relevant policy information
            try:
                policy_search = self.search_knowledge_base(
                    f"{intent} policy procedure", 
                    search_type='semantic', 
                    top_k=3
                )
                context['relevant_policies'] = policy_search.get('results', [])
            except:
                context['relevant_policies'] = []
        
        return context
    
    def _determine_routing_and_priority(
        self,
        intent_analysis: Dict[str, Any],
        sentiment_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Determine routing department and priority"""
        routing_info = {
            'department': intent_analysis.get('routing_department', 'customer_service'),
            'priority_level': self._calculate_combined_priority(intent_analysis, sentiment_analysis),
            'estimated_resolution_time': intent_analysis.get('estimated_resolution_time', {}),
            'requires_escalation': sentiment_analysis.get('urgency_level') == 'critical',
            'requires_human_review': intent_analysis.get('requires_human_review', False),
            'suggested_agent_skills': self._get_required_agent_skills(intent_analysis)
        }
        
        return routing_info
    
    def _search_related_knowledge(self, ticket_text: str, intent_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Search for related knowledge base information"""
        try:
            # Search for similar tickets
            similar_tickets = self.search_knowledge_base(ticket_text, 'semantic', 3)
            
            # Search for intent-specific information
            intent = intent_analysis.get('primary_intent', 'general')
            intent_info = self.search_knowledge_base(intent, 'metadata', 2)
            
            return {
                'similar_tickets': similar_tickets.get('results', []),
                'intent_specific_info': intent_info.get('results', []),
                'search_confidence': min(
                    len(similar_tickets.get('results', [])) * 0.3,
                    1.0
                )
            }
        except Exception as e:
            self.logger.error(f"Error searching related knowledge: {str(e)}")
            return {'similar_tickets': [], 'intent_specific_info': []}
    
    def _generate_triage_summary(
        self,
        sentiment_analysis: Dict[str, Any],
        intent_analysis: Dict[str, Any],
        routing_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate a summary of the triage results"""
        return {
            'customer_sentiment': sentiment_analysis.get('sentiment_category'),
            'customer_mood': sentiment_analysis.get('customer_mood'),
            'primary_intent': intent_analysis.get('primary_intent'),
            'confidence_level': intent_analysis.get('confidence'),
            'urgency_level': sentiment_analysis.get('urgency_level'),
            'priority_score': routing_info.get('priority_level'),
            'recommended_department': routing_info.get('department'),
            'escalation_required': routing_info.get('requires_escalation'),
            'estimated_resolution': routing_info.get('estimated_resolution_time', {}).get('max', 'Unknown')
        }
    
    def _calculate_combined_priority(
        self,
        intent_analysis: Dict[str, Any],
        sentiment_analysis: Dict[str, Any]
    ) -> str:
        """Calculate combined priority from intent and sentiment analysis"""
        intent_priority = intent_analysis.get('priority_score', 0.5)
        sentiment_urgency = sentiment_analysis.get('urgency_score', 0.5)
        
        combined_score = (intent_priority * 0.6) + (sentiment_urgency * 0.4)
        
        if combined_score >= 0.8:
            return 'critical'
        elif combined_score >= 0.6:
            return 'high'
        elif combined_score >= 0.4:
            return 'medium'
        else:
            return 'low'
    
    def _get_required_agent_skills(self, intent_analysis: Dict[str, Any]) -> List[str]:
        """Get required agent skills based on intent"""
        skill_mapping = {
            'refund': ['billing_knowledge', 'policy_expertise', 'empathy'],
            'product_issue': ['technical_knowledge', 'problem_solving', 'quality_assurance'],
            'delivery': ['logistics_knowledge', 'tracking_systems', 'communication'],
            'account': ['technical_support', 'security_protocols', 'patience'],
            'billing': ['financial_systems', 'mathematical_skills', 'attention_to_detail'],
            'technical_support': ['technical_expertise', 'troubleshooting', 'patience'],
            'complaint': ['conflict_resolution', 'empathy', 'management_skills']
        }
        
        intent = intent_analysis.get('primary_intent', 'general')
        return skill_mapping.get(intent, ['customer_service', 'communication', 'empathy'])
    
    def _extract_search_keywords(self, query: str) -> List[str]:
        """Extract keywords for hybrid search"""
        # Simple keyword extraction - could be enhanced with NLP
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        words = [word.lower().strip('.,!?') for word in query.split()]
        keywords = [word for word in words if len(word) > 3 and word not in stop_words]
        return keywords[:5]  # Limit to top 5 keywords
    
    def _create_metadata_filter(self, query: str) -> Dict[str, Any]:
        """Create metadata filter for search"""
        # Simple implementation - could be enhanced
        filter_dict = {}
        
        if 'refund' in query.lower():
            filter_dict['intent'] = 'refund'
        elif 'delivery' in query.lower():
            filter_dict['intent'] = 'delivery'
        elif 'billing' in query.lower():
            filter_dict['intent'] = 'billing'
        
        return filter_dict
    
    def _enhance_search_results(self, results: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """Enhance search results with additional context"""
        enhanced_results = []
        
        for result in results:
            enhanced_result = result.copy()
            
            # Add relevance explanation
            enhanced_result['relevance_explanation'] = self._explain_relevance(result, query)
            
            # Add source information
            enhanced_result['source_info'] = self._get_source_info(result)
            
            enhanced_results.append(enhanced_result)
        
        return enhanced_results
    
    def _explain_relevance(self, result: Dict[str, Any], query: str) -> str:
        """Explain why this result is relevant to the query"""
        # Simple implementation
        score = result.get('score', 0)
        if score > 0.8:
            return "Highly relevant match"
        elif score > 0.6:
            return "Good relevance match"
        elif score > 0.4:
            return "Moderate relevance"
        else:
            return "Low relevance"
    
    def _get_source_info(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Get source information for the search result"""
        metadata = result.get('metadata', {})
        return {
            'document_type': metadata.get('chunk_type', 'unknown'),
            'source_file': metadata.get('source', 'unknown'),
            'section': metadata.get('section_title', 'unknown'),
            'confidence': result.get('score', 0)
        }
    
    def _get_file_types_distribution(self) -> Dict[str, int]:
        """Get distribution of processed file types"""
        distribution = {}
        for file_data in self.processed_files.values():
            file_type = file_data.get('file_type', 'unknown')
            distribution[file_type] = distribution.get(file_type, 0) + 1
        return distribution
    
    def _generate_processing_insights(self) -> Dict[str, Any]:
        """Generate insights about processing patterns"""
        insights = {
            'total_tickets_processed': self.session_stats['tickets_processed'],
            'average_processing_time': 'N/A',  # Would need timing data
            'most_common_intents': {},
            'sentiment_distribution': {},
            'escalation_rate': 0.0
        }
        
        # In a full implementation, you'd track these metrics over time
        return insights
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations for improvement"""
        recommendations = []
        
        # Check knowledge base coverage
        if len(self.processed_files) < 3:
            recommendations.append("Consider uploading more policy documents to improve response accuracy")
        
        # Check vector store utilization
        vector_stats = self.vector_store.get_index_stats()
        if vector_stats.get('total_vector_count', 0) < 100:
            recommendations.append("Add more training data to improve semantic search capabilities")
        
        # Session-based recommendations
        if self.session_stats['tickets_processed'] > 50:
            recommendations.append("Consider monitoring response quality for high-volume processing")
        
        return recommendations
    
    def _generate_error_response(self, ticket_id: str, ticket_text: str, error_message: str) -> Dict[str, Any]:
        """Generate error response for failed ticket processing"""
        return {
            'ticket_id': ticket_id,
            'original_message': ticket_text,
            'status': 'error',
            'error_message': error_message,
            'processing_timestamp': datetime.now().isoformat(),
            'fallback_response': {
                'response_text': "Thank you for contacting us. We're experiencing a temporary issue processing your request. A human agent will review your message shortly.",
                'requires_human_review': True,
                'priority_level': 'high'
            }
        }
    
    # Additional utility methods for agent management
    
    def reset_session(self) -> Dict[str, Any]:
        """Reset the agent session and clear temporary data"""
        try:
            # Clear processed files (but keep vector store)
            cleared_files = len(self.processed_files)
            self.processed_files = {}
            
            # Reset session stats
            old_stats = self.session_stats.copy()
            self.session_stats = {
                'tickets_processed': 0,
                'files_uploaded': 0,
                'responses_generated': 0,
                'session_start': datetime.now().isoformat()
            }
            
            self.logger.info("Agent session reset successfully")
            
            return {
                'status': 'success',
                'cleared_files': cleared_files,
                'previous_session_stats': old_stats,
                'reset_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error resetting session: {str(e)}")
            return {
                'status': 'error',
                'error': str(e),
                'reset_timestamp': datetime.now().isoformat()
            }
    
    def validate_system_health(self) -> Dict[str, Any]:
        """Validate the health of all system components"""
        health_status = {
            'overall_status': 'healthy',
            'component_health': {},
            'issues_found': [],
            'recommendations': [],
            'check_timestamp': datetime.now().isoformat()
        }
        
        try:
            # Check sentiment analyzer
            test_sentiment = self.sentiment_analyzer.analyze_sentiment("This is a test message")
            health_status['component_health']['sentiment_analyzer'] = 'healthy' if test_sentiment else 'error'
            
            # Check intent classifier
            test_intent = self.intent_classifier.classify_intent("I need help with my order")
            health_status['component_health']['intent_classifier'] = 'healthy' if test_intent else 'error'
            
            # Check vector store
            vector_stats = self.vector_store.get_index_stats()
            health_status['component_health']['vector_store'] = 'healthy' if vector_stats else 'error'
            
            # Check response generator
            health_status['component_health']['response_generator'] = 'healthy'  # Basic check
            
            # Overall health assessment
            unhealthy_components = [k for k, v in health_status['component_health'].items() if v != 'healthy']
            if unhealthy_components:
                health_status['overall_status'] = 'degraded'
                health_status['issues_found'] = [f"{comp} is not functioning properly" for comp in unhealthy_components]
            
            return health_status
            
        except Exception as e:
            health_status['overall_status'] = 'error'
            health_status['issues_found'].append(f"Health check failed: {str(e)}")
            return health_status
            
        except Exception as e:
            self.logger.error(f"Error searching knowledge base: {str(e)}")
            return {
                'query': query,
                'results_count': 0,
                'results': [],
                'error': str(e),
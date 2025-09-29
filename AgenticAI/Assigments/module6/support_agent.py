"""
Main Customer Support Triage Agent
"""
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

# Import all components
from sentiment_analyzer import SentimentAnalyzer
from intent_classifier import IntentClassifier, create_sample_classifier
from response_generator import ResponseGenerator, create_sample_response_generator
from csv_processor import CSVProcessor
from pdf_processor import PDFProcessor
from text_processor import TextProcessor
from database_manager import DatabaseManager
from session_manager import SessionManager
from agent_tools import AgentTools
from policy_matcher import PolicyMatcher
from external_lookup import ExternalLookup
from refund_calculator import RefundCalculator, RefundReason, ProductType, Purchase
from config import settings
from simple_vector_response import VectorStore

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
            self.intent_classifier = create_sample_classifier()
            self.response_generator = create_sample_response_generator()
            self.database = DatabaseManager()
            
            # Initialize processors
            self.csv_processor = CSVProcessor()
            self.pdf_processor = PDFProcessor()
            self.text_processor = TextProcessor()
            
            # Initialize session management and agent tools
            self.session_manager = SessionManager()
            self.agent_tools = AgentTools(self.database, self.session_manager)
            
            # Initialize new comprehensive tools
            self.policy_matcher = PolicyMatcher()
            self.external_lookup = ExternalLookup()
            self.refund_calculator = RefundCalculator()
            
            # Initialize vector store for semantic search
            self.vector_store = VectorStore()
            
            # Agent state
            self.processed_files = {}
            self.knowledge_base = {}
            self.session_stats = {
                'tickets_processed': 0,
                'files_uploaded': 0,
                'responses_generated': 0,
                'session_start': datetime.now().isoformat()
            }
            
            self.logger.info(f"Initialized {name} with all components including session management and agent tools")
            
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
            intent_result = self.intent_classifier.classify(ticket_text)
            # Map intents to proper categories
            intent_to_category = {
                'account_issue': 'Account',
                'billing_question': 'Billing', 
                'technical_issue': 'Technical',
                'general_inquiry': 'General',
                'complaint': 'General',
                'refund_request': 'Billing',
                'password_reset': 'Account',
                'delivery_issue': 'Shipping',
                'product_inquiry': 'Product',
                'order_status': 'Order',
                'cancellation_request': 'Billing',
                'compliment': 'General',
                'unknown': 'General'
            }
            
            predicted_intent = intent_result[0] if intent_result[0] else 'general_inquiry'
            intent_category = intent_to_category.get(predicted_intent.lower(), 'General')
            
            intent_analysis = {
                'predicted_intent': predicted_intent,
                'confidence': intent_result[1],
                'intent_category': intent_category,
                'details': intent_result[2] if len(intent_result) > 2 else {}
            }
            
            # Step 3: Policy Reference Matching
            try:
                policy_matches = self.policy_matcher.find_relevant_policies(
                    ticket_text, 
                    intent_analysis.get('predicted_intent'),
                    max_results=3
                )
                policy_summaries = [self.policy_matcher.get_policy_summary(match) for match in policy_matches]
            except Exception as e:
                self.logger.error(f"Error matching policies: {str(e)}")
                policy_matches = []
                policy_summaries = []
            
            # Step 4: External Information Lookup (for complex queries)
            external_info = {}
            if intent_analysis.get('confidence', 0) < 0.7 or predicted_intent in ['general_inquiry', 'product_inquiry']:
                try:
                    external_info = self.external_lookup.get_customer_service_context(
                        ticket_text, 
                        intent_analysis.get('predicted_intent')
                    )
                except Exception as e:
                    self.logger.error(f"Error in external lookup: {str(e)}")
                    external_info = {}
            
            # Step 5: Refund Eligibility Check (for refund requests)
            refund_calculation = None
            if predicted_intent == 'refund_request':
                try:
                    # Extract order information from ticket (this would be more sophisticated in production)
                    refund_calculation = self._check_refund_eligibility(ticket_text, sentiment_analysis)
                except Exception as e:
                    self.logger.error(f"Error calculating refund eligibility: {str(e)}")
                    refund_calculation = None
            
            # Step 6: Generate Response
            response_data = self.response_generator.generate_response(
                ticket_text,
                intent_analysis.get('predicted_intent', 'general_inquiry'),
                intent_analysis.get('confidence', 0.5),
                {
                    'sentiment': sentiment_analysis,
                    'context': self._get_context_for_ticket(intent_analysis),
                    'policies': policy_summaries,
                    'external_info': external_info,
                    'refund_info': refund_calculation
                }
            )
            
            # Step 7: Determine Routing and Priority
            routing_info = self._determine_routing_and_priority(
                intent_analysis,
                sentiment_analysis
            )
            
            # Step 8: Search Related Knowledge
            try:
                related_info = self._search_related_knowledge(ticket_text, intent_analysis)
            except Exception as e:
                self.logger.error(f"Error searching related knowledge: {str(e)}")
                related_info = {'related_articles': [], 'knowledge_used': False}
            
            # Step 9: Search for similar past cases
            similar_cases = self.database.search_similar_tickets(
                ticket_text, 
                intent_analysis.get('intent_category', 'General'),
                limit=5
            )
            
            # Step 10: Get relevant resolutions from knowledge base
            relevant_resolutions = self.database.get_resolutions_by_category(
                intent_analysis.get('intent_category', 'General'),
                limit=3
            )
            
            # Compile results
            result = {
                'ticket_id': ticket_id,
                'original_text': ticket_text,
                'processing_timestamp': datetime.now().isoformat(),
                'processing_time_ms': 0,  # Will be updated when processing ends
                'sentiment_analysis': sentiment_analysis,
                'intent_classification': intent_analysis,
                'policy_matches': policy_summaries,
                'external_information': external_info,
                'refund_calculation': refund_calculation,
                'response_generation': response_data,
                'routing_info': routing_info,
                'related_knowledge': related_info,
                'similar_cases': similar_cases,
                'relevant_resolutions': relevant_resolutions,
                'triage_summary': self._generate_triage_summary(
                    sentiment_analysis, intent_analysis, routing_info
                ),
                'comprehensive_analysis': {
                    'tools_used': [
                        'sentiment_analyzer',
                        'intent_classifier', 
                        'policy_matcher',
                        'response_generator'
                    ],
                    'policy_matches_found': len(policy_summaries),
                    'external_info_available': bool(external_info),
                    'refund_eligible': refund_calculation.get('eligible', False) if refund_calculation else None
                }
            }
            
            # Step 11: Get auto-suggested responses from agent tools
            try:
                auto_suggestions = self.agent_tools.auto_suggest_response(
                    ticket_text, sentiment_analysis, intent_analysis,
                    context=self.session_manager.get_context_for_ticket(intent_analysis.get('intent_category'))
                )
                result['auto_suggestions'] = auto_suggestions
            except Exception as e:
                self.logger.error(f"Error getting auto-suggestions: {str(e)}")
                result['auto_suggestions'] = {}
            
            # Step 12: Check for escalation needs
            try:
                escalation_analysis = self.agent_tools.analyze_escalation_need(
                    ticket_text, sentiment_analysis, intent_analysis
                )
                result['escalation_indicators'] = escalation_analysis
                
                # Track escalation if needed
                if escalation_analysis.get('requires_escalation'):
                    self.session_manager.track_escalation(
                        ticket_id,
                        escalation_analysis.get('escalation_level', 'high_priority'),
                        escalation_analysis
                    )
                    
            except Exception as e:
                self.logger.error(f"Error analyzing escalation: {str(e)}")
                result['escalation_indicators'] = {}
            
            # Step 13: Store ticket in database for future reference
            try:
                self.database.store_ticket(result)
                self.logger.info(f"Stored ticket {ticket_id} in persistent database")
            except Exception as db_error:
                self.logger.error(f"Failed to store ticket in database: {str(db_error)}")
            
            # Step 14: Track session activity
            try:
                self.session_manager.track_ticket_analysis(ticket_id, ticket_text, result)
                self.session_manager.track_response_generation(
                    ticket_id,
                    response_data.get('suggested_response', ''),
                    auto_suggestions.get('response_metadata', {})
                )
            except Exception as e:
                self.logger.error(f"Error tracking session activity: {str(e)}")
            
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
        Upload and process a file (CSV, PDF, or TXT) for knowledge base with database storage
        
        Args:
            file_path: Path to the file to process
            file_type: Type of file ('csv', 'pdf', 'txt') or None for auto-detection
            
        Returns:
            Processing results and file information
        """
        try:
            import os
            self.logger.info(f"Processing file: {file_path}")
            
            # Auto-detect file type if not provided
            if not file_type:
                file_type = file_path.split('.')[-1].lower()
            
            # Get file name and size
            file_name = os.path.basename(file_path)
            file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
            
            # Process file based on type
            if file_type == 'csv':
                processed_data = self.csv_processor.process_file(file_path)
            elif file_type == 'pdf':
                processed_data = self.pdf_processor.process_file(file_path)
            elif file_type in ['txt', 'text']:
                processed_data = self.text_processor.process_file(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
            
            # Store processed data
            processed_data['processing_status'] = 'completed'
            
            # Create file ID and store in memory
            file_id = f"file_{datetime.now().strftime('%Y%m%d%H%M%S')}_{file_name}"
            self.processed_files[file_id] = {
                'file_path': file_path,
                'file_type': file_type,
                'processed_data': processed_data,
                'processing_timestamp': datetime.now().isoformat()
            }
            
            # Extract content sections for database storage
            content_sections = self._extract_content_sections(processed_data, file_type)
            
            # Store file content in database for searchability
            try:
                success = self.database.store_file_content(
                    file_id=file_id,
                    file_name=file_name,
                    file_type=file_type,
                    content_sections=content_sections,
                    file_path=file_path
                )
                
                if success:
                    self.logger.info(f"Stored {len(content_sections)} content sections for {file_name} in database")
                else:
                    self.logger.warning(f"Failed to store file content in database for {file_name}")
                    
            except Exception as db_error:
                self.logger.error(f"Database storage error for {file_name}: {str(db_error)}")
            
            # Store content in vector store for semantic search
            try:
                vector_chunks = self._prepare_chunks_for_vector_store(content_sections, file_id, file_name, file_type)
                if vector_chunks:
                    vector_result = self.vector_store.upsert_chunks(vector_chunks)
                    if vector_result.get('successful_upserts', 0) > 0:
                        self.logger.info(f"Stored {vector_result['successful_upserts']} chunks in vector store for {file_name}")
                    else:
                        self.logger.warning(f"Failed to store chunks in vector store for {file_name}")
            except Exception as vector_error:
                self.logger.error(f"Vector store error for {file_name}: {str(vector_error)}")
            
            # Track file upload in session
            try:
                self.session_manager.track_file_upload(
                    file_name, file_type, file_size, 
                    {'sections': len(content_sections), 'database_stored': success if 'success' in locals() else False}
                )
            except Exception as e:
                self.logger.error(f"Error tracking file upload: {str(e)}")
            
            # Update session stats
            self.session_stats['files_uploaded'] += 1
            
            result = {
                'file_id': file_id,
                'file_path': file_path,
                'file_type': file_type,
                'file_name': file_name,
                'file_size': file_size,
                'content_sections': len(content_sections),
                'processing_summary': processed_data.get('summary', {}),
                'data_processed': True,
                'database_stored': success if 'success' in locals() else False,
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
    
    def search_knowledge_base(self, query: str, category: str = None, search_type: str = 'hybrid', top_k: int = 5) -> Dict[str, Any]:
        """
        Search the knowledge base for relevant information using hybrid database search
        
        Args:
            query: Search query
            search_type: Type of search ('semantic', 'hybrid', 'metadata')
            top_k: Number of results to return
            
        Returns:
            Search results with relevant information
        """
        try:
            self.logger.info(f"Searching knowledge base: {query[:50]}...")
            
            # Get results from multiple sources
            all_results = []
            
            # 1. Vector store semantic search
            vector_results = []
            if search_type in ['semantic', 'hybrid']:
                try:
                    vector_search = self.vector_store.search_similar(query, top_k=top_k)
                    for result in vector_search:
                        vector_results.append({
                            'content': result.get('text', ''),
                            'metadata': result.get('metadata', {}),
                            'relevance_score': result.get('score', 0.5),
                            'search_source': 'vector_store'
                        })
                except Exception as e:
                    self.logger.warning(f"Vector search failed: {str(e)}")
            
            # 2. Database search
            database_results = []
            try:
                db_search = self.database.hybrid_search(query, category=None, limit=top_k)
                database_results = db_search.get('results', [])
                for result in database_results:
                    result['search_source'] = 'database'
            except Exception as e:
                self.logger.warning(f"Database search failed: {str(e)}")
            
            # 3. Memory/processed files search
            memory_results = self._search_processed_files(query, search_type, top_k)
            for result in memory_results:
                result['search_source'] = 'memory'
            
            # Combine all results
            all_results = vector_results + database_results + memory_results
            enhanced_results = self._enhance_search_results(all_results, query)
            
            # Remove duplicates and sort by relevance
            unique_results = []
            seen_content = set()
            for result in enhanced_results:
                content_key = result.get('content', '')[:100]  # Use first 100 chars as key
                if content_key not in seen_content:
                    seen_content.add(content_key)
                    unique_results.append(result)
            
            # Sort by relevance score
            unique_results.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
            final_results = unique_results[:top_k]
            
            # Get database search results for category-specific queries
            ticket_results = []
            if category:
                ticket_search = self.search_ticket_database(query, category)
                ticket_results = ticket_search.get('similar_tickets', [])
            
            # Combine results in expected format
            all_similar_tickets = final_results + ticket_results
            
            return {
                'query': query,
                'category': category,
                'search_type': search_type,
                'similar_tickets': all_similar_tickets,
                'relevant_resolutions': [],  # Could be added later
                'total_results': len(all_similar_tickets),
                'vector_results': len(vector_results),
                'database_results': len(database_results),
                'memory_results': len(memory_results),
                'search_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error searching knowledge base: {str(e)}")
            return {
                'query': query,
                'category': category,
                'similar_tickets': [],
                'relevant_resolutions': [],
                'total_results': 0,
                'error': str(e),
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
            
            # Get processing statistics  
            processing_stats = {
                'total_files': len(self.processed_files),
                'successful_processing': len([f for f in self.processed_files.values() if f.get('processed_data', {}).get('processing_status') == 'completed'])
            }
            
            # Generate report sections
            report = {
                'report_timestamp': datetime.now().isoformat(),
                'time_period': time_period,
                'session_statistics': self.session_stats,
                'knowledge_base_stats': {
                    'total_files_processed': len(self.processed_files),
                    'processing_stats': processing_stats,
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
                    'file_processors': 'active'
                },
                'knowledge_base_files': len(self.processed_files),
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
            similar_tickets = self.search_knowledge_base(ticket_text, 'hybrid', 3)
            
            # Search for intent-specific information
            intent = intent_analysis.get('primary_intent', 'general')
            intent_info = self.search_knowledge_base(intent, 'hybrid', 2)
            
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
        
        # Check data processing coverage
        processed_data_count = len([f for f in self.processed_files.values() if f.get('processed_data')])
        if processed_data_count < 100:
            recommendations.append("Add more training data to improve response capabilities")
        
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
            test_intent = self.intent_classifier.classify("I need help with my order")
            health_status['component_health']['intent_classifier'] = 'healthy' if test_intent else 'error'
            
            # Check file processors
            health_status['component_health']['file_processors'] = 'healthy'  # Basic check
            
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
    
    def _search_processed_files(self, query: str, search_type: str = 'semantic', top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search through processed files (simplified version without vector storage)
        
        Args:
            query: Search query
            search_type: Type of search ('semantic', 'hybrid', 'metadata')  
            top_k: Maximum number of results
            
        Returns:
            List of search results
        """
        results = []
        query_lower = query.lower()
        
        for file_id, file_info in self.processed_files.items():
            processed_data = file_info.get('processed_data', {})
            
            # Simple keyword matching in processed data
            relevance_score = 0
            content_text = ""
            
            # Extract text content based on file type
            if file_info['file_type'] == 'csv':
                # Search in CSV data
                data = processed_data.get('data', [])
                for row in data[:10]:  # Limit to first 10 rows for performance
                    row_text = ' '.join([str(v) for v in row.values() if v])
                    content_text += row_text + " "
                    if query_lower in row_text.lower():
                        relevance_score += 0.5
                        
            elif file_info['file_type'] == 'pdf':
                # Search in PDF sections
                sections = processed_data.get('sections', [])
                for section in sections[:5]:  # Limit for performance
                    section_text = section.get('content', '')
                    content_text += section_text + " "
                    if query_lower in section_text.lower():
                        relevance_score += 0.7
                        
            elif file_info['file_type'] in ['txt', 'text']:
                # Search in text conversations
                conversations = processed_data.get('conversations', [])
                for conv in conversations[:5]:  # Limit for performance
                    conv_text = conv.get('message', '')
                    content_text += conv_text + " "
                    if query_lower in conv_text.lower():
                        relevance_score += 0.6
            
            # If relevant content found, add to results
            if relevance_score > 0:
                results.append({
                    'file_id': file_id,
                    'file_path': file_info['file_path'],
                    'file_type': file_info['file_type'],
                    'relevance_score': relevance_score,
                    'content_preview': content_text[:200] + "..." if len(content_text) > 200 else content_text,
                    'metadata': {
                        'processing_timestamp': file_info.get('processing_timestamp'),
                        'data_type': processed_data.get('data_type', 'unknown')
                    }
                })
        
        # Sort by relevance score and return top_k results
        results.sort(key=lambda x: x['relevance_score'], reverse=True)
        return results[:top_k]
    
    def update_ticket_resolution(self, ticket_id: str, actual_response: str, 
                                customer_satisfaction: int = None) -> bool:
        """
        Update a ticket with the actual response sent and customer feedback
        
        Args:
            ticket_id: The ticket ID
            actual_response: The actual response that was sent
            customer_satisfaction: Customer satisfaction rating (1-5)
            
        Returns:
            Success boolean
        """
        try:
            success = self.database.update_ticket_resolution(
                ticket_id, actual_response, 'resolved', customer_satisfaction
            )
            
            if success:
                self.logger.info(f"Updated resolution for ticket {ticket_id}")
            else:
                self.logger.warning(f"Failed to update ticket {ticket_id} - not found")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error updating ticket resolution: {str(e)}")
            return False
    
    def add_resolution_to_knowledge_base(self, issue_category: str, issue_description: str,
                                        resolution_steps: str, effectiveness_score: float = 0.8,
                                        ticket_id: str = None) -> bool:
        """
        Add a successful resolution to the knowledge base for future use
        
        Args:
            issue_category: Category of the issue
            issue_description: Description of the issue
            resolution_steps: Steps taken to resolve the issue
            effectiveness_score: How effective this resolution is (0-1)
            ticket_id: Optional ticket ID this resolution came from
            
        Returns:
            Success boolean
        """
        try:
            resolution_data = {
                'ticket_id': ticket_id,
                'issue_category': issue_category,
                'issue_description': issue_description,
                'resolution_steps': resolution_steps,
                'effectiveness_score': effectiveness_score,
                'resolution_type': 'user_added'
            }
            
            resolution_id = self.database.store_resolution(resolution_data)
            
            if resolution_id:
                self.logger.info(f"Added resolution {resolution_id} to knowledge base")
                return True
            else:
                return False
                
        except Exception as e:
            self.logger.error(f"Error adding resolution to knowledge base: {str(e)}")
            return False
    
    def get_historical_data(self, days: int = 30) -> Dict[str, Any]:
        """
        Get historical ticket data and statistics
        
        Args:
            days: Number of days to look back
            
        Returns:
            Historical data and statistics
        """
        try:
            # Get statistics from database
            stats = self.database.get_ticket_statistics(days)
            
            # Get recent tickets
            recent_tickets = self.database.get_recent_tickets(20)
            
            # Enhanced statistics
            historical_data = {
                'statistics': stats,
                'recent_tickets': recent_tickets,
                'trends': self._analyze_trends(stats),
                'recommendations': self._generate_historical_recommendations(stats)
            }
            
            return historical_data
            
        except Exception as e:
            self.logger.error(f"Error getting historical data: {str(e)}")
            return {}
    
    def search_ticket_database(self, query: str, category: str = None) -> Dict[str, Any]:
        """
        Search the persistent ticket database for similar tickets and resolutions
        
        Args:
            query: Search query
            category: Optional category filter
            
        Returns:
            Search results from ticket database
        """
        try:
            # Search similar tickets - if no category specified or no results, search all
            similar_tickets = self.database.search_similar_tickets(query, category, limit=10)
            
            # If no results and category was specified, try searching without category filter
            if not similar_tickets and category:
                similar_tickets = self.database.search_similar_tickets(query, None, limit=10)
            
            # Get resolutions for the category
            resolutions = []
            if category:
                resolutions = self.database.get_resolutions_by_category(category, limit=5)
                # If no resolutions for specific category, get some general ones
                if not resolutions:
                    resolutions = self.database.get_resolutions_by_category('General', limit=3)
            
            return {
                'query': query,
                'category': category,
                'similar_tickets': similar_tickets,
                'relevant_resolutions': resolutions,
                'total_results': len(similar_tickets) + len(resolutions),
                'search_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error searching knowledge base: {str(e)}")
            return {
                'query': query,
                'error': str(e),
                'similar_tickets': [],
                'relevant_resolutions': [],
                'total_results': 0
            }
    
    def export_historical_data(self, format: str = 'json') -> str:
        """
        Export all historical data
        
        Args:
            format: Export format ('json' or 'csv')
            
        Returns:
            Path to exported file
        """
        try:
            export_path = self.database.export_data(format)
            self.logger.info(f"Exported historical data to {export_path}")
            return export_path
            
        except Exception as e:
            self.logger.error(f"Error exporting historical data: {str(e)}")
            raise
    
    def _analyze_trends(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze trends from statistics data
        
        Args:
            stats: Statistics dictionary
            
        Returns:
            Trend analysis
        """
        trends = {}
        
        # Analyze category trends
        category_dist = stats.get('category_distribution', {})
        if category_dist:
            most_common_category = max(category_dist.items(), key=lambda x: x[1])
            trends['most_common_category'] = {
                'category': most_common_category[0],
                'count': most_common_category[1],
                'percentage': round(most_common_category[1] / sum(category_dist.values()) * 100, 1)
            }
        
        # Analyze urgency trends
        urgency_dist = stats.get('urgency_distribution', {})
        if urgency_dist:
            high_urgency_count = urgency_dist.get('high', 0) + urgency_dist.get('critical', 0)
            total_urgency = sum(urgency_dist.values())
            if total_urgency > 0:
                trends['high_urgency_percentage'] = round(high_urgency_count / total_urgency * 100, 1)
        
        # Resolution rate trend
        resolution_rate = stats.get('resolution_rate', 0)
        if resolution_rate >= 80:
            trends['resolution_trend'] = 'excellent'
        elif resolution_rate >= 60:
            trends['resolution_trend'] = 'good'
        else:
            trends['resolution_trend'] = 'needs_improvement'
        
        return trends
    
    def _generate_historical_recommendations(self, stats: Dict[str, Any]) -> List[str]:
        """
        Generate recommendations based on historical data
        
        Args:
            stats: Statistics dictionary
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        # Resolution rate recommendations
        resolution_rate = stats.get('resolution_rate', 0)
        if resolution_rate < 70:
            recommendations.append("Consider improving resolution processes - current rate is below 70%")
        
        # Satisfaction recommendations
        avg_satisfaction = stats.get('average_satisfaction', 0)
        if avg_satisfaction < 3.5:
            recommendations.append("Focus on improving customer satisfaction - current average is below 3.5")
        
        # Category-based recommendations
        category_dist = stats.get('category_distribution', {})
        if category_dist:
            top_category = max(category_dist.items(), key=lambda x: x[1])
            recommendations.append(f"Consider creating more resources for '{top_category[0]}' - highest volume category")
        
        # Urgency recommendations
        urgency_dist = stats.get('urgency_distribution', {})
        if urgency_dist:
            critical_count = urgency_dist.get('critical', 0)
            total_count = sum(urgency_dist.values())
            if total_count > 0 and critical_count / total_count > 0.1:
                recommendations.append("High number of critical tickets - review escalation procedures")
        
        return recommendations
    
    def _extract_content_sections(self, processed_data: Dict[str, Any], file_type: str) -> List[Dict[str, Any]]:
        """
        Extract content sections from processed data for database storage
        
        Args:
            processed_data: The processed file data
            file_type: Type of file ('csv', 'pdf', 'txt')
            
        Returns:
            List of content sections with metadata
        """
        content_sections = []
        
        try:
            if file_type == 'csv':
                # Extract CSV data sections
                data = processed_data.get('data', [])
                summary = processed_data.get('summary', {})
                
                # Add summary as first section
                if summary:
                    content_sections.append({
                        'content': f"CSV Summary: {summary.get('total_rows', 0)} rows, {summary.get('total_columns', 0)} columns. Columns: {', '.join(summary.get('columns', []))}",
                        'title': 'CSV File Summary',
                        'category': 'Data',
                        'tags': 'csv,summary,data'
                    })
                
                # Add sample data sections
                for i, row in enumerate(data[:10]):  # Limit to first 10 rows
                    row_content = ', '.join([f"{k}: {v}" for k, v in row.items() if v])
                    content_sections.append({
                        'content': row_content,
                        'title': f'Data Row {i+1}',
                        'category': 'Data',
                        'tags': 'csv,data,row'
                    })
                    
            elif file_type == 'pdf':
                # Extract PDF sections
                sections = processed_data.get('sections', [])
                summary = processed_data.get('summary', {})
                
                # Add document summary
                if summary:
                    content_sections.append({
                        'content': f"PDF Document: {summary.get('total_pages', 0)} pages, {summary.get('total_sections', 0)} sections. {summary.get('summary_text', '')}",
                        'title': 'PDF Document Summary',
                        'category': 'Document',
                        'tags': 'pdf,summary,document'
                    })
                
                # Add each section
                for i, section in enumerate(sections):
                    content_sections.append({
                        'content': section.get('content', ''),
                        'title': section.get('title', f'Section {i+1}'),
                        'category': 'Document',
                        'tags': 'pdf,section,document'
                    })
                    
            elif file_type in ['txt', 'text']:
                # Extract text conversations
                conversations = processed_data.get('conversations', [])
                summary = processed_data.get('summary', {})
                
                # Add conversation summary
                if summary:
                    content_sections.append({
                        'content': f"Text Conversations: {summary.get('total_conversations', 0)} conversations, {summary.get('total_messages', 0)} messages. Most common intents: {', '.join(summary.get('common_intents', []))}",
                        'title': 'Text Conversations Summary',
                        'category': 'Conversation',
                        'tags': 'text,conversation,summary'
                    })
                
                # Add individual conversations
                for i, conv in enumerate(conversations[:20]):  # Limit to first 20
                    # Extract content from messages within the conversation
                    conv_content = ""
                    messages = conv.get('messages', [])
                    if messages:
                        # Combine all messages in the conversation
                        conv_content = "\n".join([msg.get('content', '') for msg in messages])
                    
                    # Fallback to direct message field if available
                    if not conv_content:
                        conv_content = conv.get('message', '')
                    
                    content_sections.append({
                        'content': conv_content,
                        'title': f"Text Document Section {i+1}",
                        'category': conv.get('intent_category', 'Document'),
                        'tags': f"text,document,section"
                    })
            
            # Ensure all sections have required fields
            for section in content_sections:
                if 'content' not in section or not section['content']:
                    section['content'] = 'No content available'
                if 'title' not in section:
                    section['title'] = 'Untitled Section'
                if 'category' not in section:
                    section['category'] = 'General'
                if 'tags' not in section:
                    section['tags'] = file_type
                    
        except Exception as e:
            self.logger.error(f"Error extracting content sections: {str(e)}")
            # Return at least one section with error info
            content_sections = [{
                'content': f"Error processing {file_type} file: {str(e)}",
                'title': 'Processing Error',
                'category': 'Error',
                'tags': f'{file_type},error'
            }]
        
        return content_sections
    
    def generate_supervisor_insights(self, time_period: str = '24h') -> Dict[str, Any]:
        """
        Generate supervisor insights using agent tools
        
        Args:
            time_period: Time period for analysis
            
        Returns:
            Supervisor insights and recommendations
        """
        try:
            insights = self.agent_tools.generate_supervisor_insights(time_period)
            
            # Enhance with session data
            session_summary = self.session_manager.get_session_summary()
            insights['session_data'] = session_summary
            
            return insights
            
        except Exception as e:
            self.logger.error(f"Error generating supervisor insights: {str(e)}")
            return {'error': str(e), 'time_period': time_period}
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get current session summary"""
        try:
            return self.session_manager.get_session_summary()
        except Exception as e:
            self.logger.error(f"Error getting session summary: {str(e)}")
            return {'error': str(e)}
    
    def _check_refund_eligibility(self, ticket_text: str, sentiment_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check refund eligibility for refund requests
        
        Args:
            ticket_text: Customer message
            sentiment_analysis: Sentiment analysis results
            
        Returns:
            Refund calculation results
        """
        try:
            # Extract basic information from ticket (simplified for demo)
            # In production, this would integrate with order management system
            
            # Determine refund reason based on sentiment and keywords
            reason = RefundReason.CUSTOMER_CHANGED_MIND  # Default
            
            if any(word in ticket_text.lower() for word in ['broken', 'defective', 'damaged', 'faulty']):
                reason = RefundReason.DEFECTIVE
            elif any(word in ticket_text.lower() for word in ['wrong', 'different', 'not what i ordered']):
                reason = RefundReason.WRONG_ITEM
            elif any(word in ticket_text.lower() for word in ['late', 'delayed', 'never arrived']):
                reason = RefundReason.SHIPPING_DELAY
            elif any(word in ticket_text.lower() for word in ['charged', 'billing', 'double']):
                reason = RefundReason.BILLING_ERROR
            
            # Create mock purchase for demonstration
            # In production, this would lookup actual order data
            import re
            order_match = re.search(r'order\s*#?\s*(\w+)', ticket_text, re.IGNORECASE)
            order_id = order_match.group(1) if order_match else "DEMO-ORDER"
            
            # Determine product type from context
            product_type = ProductType.PHYSICAL  # Default
            if any(word in ticket_text.lower() for word in ['download', 'software', 'digital']):
                product_type = ProductType.DIGITAL
            elif any(word in ticket_text.lower() for word in ['subscription', 'monthly', 'plan']):
                product_type = ProductType.SUBSCRIPTION
            
            # Create mock purchase (in production, fetch from database)
            from datetime import timedelta
            purchase = Purchase(
                order_id=order_id,
                purchase_date=datetime.now() - timedelta(days=15),  # 15 days ago
                product_type=product_type,
                amount=99.99,  # Default amount
                customer_type="premium" if sentiment_analysis.get('customer_mood') == 'polite' else "standard"
            )
            
            # Calculate eligibility
            calculation = self.refund_calculator.calculate_refund_eligibility(purchase, reason)
            
            # Convert to dictionary for JSON serialization
            return {
                'eligible': calculation.eligible,
                'refund_amount': calculation.refund_amount,
                'processing_time_days': calculation.processing_time_days,
                'reason': calculation.reason,
                'conditions_met': calculation.conditions_met,
                'conditions_failed': calculation.conditions_failed,
                'policy_applied': calculation.policy_applied,
                'calculation_details': calculation.calculation_details,
                'warnings': calculation.warnings,
                'next_steps': calculation.next_steps,
                'refund_reason': reason.value,
                'order_id': order_id
            }
            
        except Exception as e:
            self.logger.error(f"Error checking refund eligibility: {str(e)}")
            return {
                'eligible': False,
                'error': str(e),
                'reason': 'Calculation error - manual review required'
            }
    
    def get_comprehensive_tools_status(self) -> Dict[str, Any]:
        """Get status of all integrated tools"""
        return {
            'sentiment_analyzer': 'active',
            'intent_classifier': 'active',
            'policy_matcher': f"{len(self.policy_matcher.policies)} policies loaded",
            'external_lookup': 'active',
            'refund_calculator': f"{len(self.refund_calculator.policies)} product types configured",
            'response_generator': f"{len(self.response_generator.response_templates)} response templates",
            'agent_tools': 'active',
            'database_manager': 'active',
            'session_manager': 'active',
            'total_tools': 9
        }
    
    def process_with_all_tools(self, ticket_text: str, ticket_id: str = None) -> Dict[str, Any]:
        """
        Process ticket using all available tools for comprehensive analysis
        This is an enhanced version that ensures all tools are utilized
        """
        return self.process_support_ticket(ticket_text, ticket_id)
    
    def _prepare_chunks_for_vector_store(self, content_sections: List[Dict[str, Any]], file_id: str, file_name: str, file_type: str) -> List[Dict[str, Any]]:
        """
        Prepare content chunks for vector store storage
        
        Args:
            content_sections: List of content sections from file processing
            file_id: Unique file identifier
            file_name: Name of the file
            file_type: Type of the file
            
        Returns:
            List of chunks formatted for vector store
        """
        chunks = []
        
        for i, section in enumerate(content_sections):
            chunk_id = f"{file_id}_chunk_{i}"
            
            # Extract text content
            text_content = ""
            if isinstance(section, dict):
                text_content = section.get('content', '') or section.get('text', '') or str(section)
            else:
                text_content = str(section)
            
            # Skip empty or very short content
            if len(text_content.strip()) < 10:
                continue
                
            chunk = {
                'chunk_id': chunk_id,
                'text': text_content,
                'chunk_type': 'file_section',
                'word_count': len(text_content.split()),
                'char_count': len(text_content),
                'metadata': {
                    'file_id': file_id,
                    'file_name': file_name,
                    'file_type': file_type,
                    'chunk_index': i,
                    'source_type': 'uploaded_file',
                    'created_at': datetime.now().isoformat()
                }
            }
            
            chunks.append(chunk)
        
        return chunks
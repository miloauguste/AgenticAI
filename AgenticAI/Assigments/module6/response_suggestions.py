#!/usr/bin/env python3
"""
Auto-Suggest Draft Response System for Agents
Provides intelligent response suggestions based on ticket analysis, historical data, and best practices
"""

import logging
import re
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import json

@dataclass
class ResponseSuggestion:
    """Data class for response suggestions"""
    id: str
    response_text: str
    confidence_score: float
    response_type: str  # 'template', 'historical', 'generated', 'policy'
    tone: str  # 'professional', 'empathetic', 'urgent', 'apologetic'
    estimated_resolution_time: str
    requires_escalation: bool
    suggested_actions: List[str]
    source_reference: str
    metadata: Dict[str, Any]

class ResponseSuggestionEngine:
    """
    Intelligent response suggestion engine for customer service agents
    """
    
    def __init__(self, persistent_store_integration=None):
        """
        Initialize the response suggestion engine
        
        Args:
            persistent_store_integration: Optional persistent store for historical data
        """
        self.logger = logging.getLogger(__name__)
        self.persistent_store = persistent_store_integration
        self.response_templates = self._load_response_templates()
        self.tone_guidelines = self._load_tone_guidelines()
        self.escalation_triggers = self._load_escalation_triggers()
        
        self.logger.info("Response suggestion engine initialized")
    
    def generate_response_suggestions(self, ticket_analysis: Dict[str, Any], 
                                    context: Dict[str, Any] = None,
                                    num_suggestions: int = 3) -> List[ResponseSuggestion]:
        """
        Generate multiple response suggestions for a ticket
        
        Args:
            ticket_analysis: Analysis results from support agent
            context: Additional context (customer history, current workload, etc.)
            num_suggestions: Number of suggestions to generate
            
        Returns:
            List of response suggestions ranked by confidence
        """
        try:
            self.logger.info(f"Generating {num_suggestions} response suggestions")
            
            suggestions = []
            
            # Extract key information
            category = ticket_analysis.get('intent_classification', {}).get('intent_category', 'general')
            sentiment = ticket_analysis.get('sentiment_analysis', {})
            urgency = self._determine_urgency_level(ticket_analysis, context)
            customer_tier = context.get('customer_tier', 'standard') if context else 'standard'
            
            # 1. Template-based suggestions
            template_suggestions = self._get_template_suggestions(category, sentiment, urgency)
            suggestions.extend(template_suggestions[:2])  # Top 2 templates
            
            # 2. Historical-based suggestions (if persistent store available)
            if self.persistent_store:
                historical_suggestions = self._get_historical_suggestions(ticket_analysis, context)
                suggestions.extend(historical_suggestions[:2])  # Top 2 historical
            
            # 3. AI-generated personalized suggestions
            generated_suggestions = self._generate_personalized_responses(ticket_analysis, context)
            suggestions.extend(generated_suggestions[:2])  # Top 2 generated
            
            # 4. Policy-based suggestions
            policy_suggestions = self._get_policy_based_suggestions(category, ticket_analysis)
            suggestions.extend(policy_suggestions[:1])  # Top 1 policy
            
            # Rank and filter suggestions
            ranked_suggestions = self._rank_suggestions(suggestions, ticket_analysis, context)
            
            # Return top N suggestions
            final_suggestions = ranked_suggestions[:num_suggestions]
            
            self.logger.info(f"Generated {len(final_suggestions)} response suggestions")
            return final_suggestions
            
        except Exception as e:
            self.logger.error(f"Error generating response suggestions: {str(e)}")
            return self._get_fallback_suggestions(ticket_analysis)
    
    def _load_response_templates(self) -> Dict[str, Dict[str, Any]]:
        """Load response templates for different categories and scenarios"""
        return {
            'refund': {
                'standard': {
                    'template': "Thank you for contacting us regarding your refund request. I understand your concern about {issue_description}. I'll be happy to process your refund request. To proceed, I'll need to verify a few details: {verification_details}. Once verified, your refund will be processed within {timeframe} business days.",
                    'tone': 'professional',
                    'estimated_time': '3-5 business days',
                    'actions': ['verify_order', 'process_refund', 'send_confirmation']
                },
                'urgent': {
                    'template': "I sincerely apologize for the inconvenience you've experienced with {issue_description}. I understand this is frustrating, and I want to resolve this immediately. I'm escalating your refund request to our priority queue and will personally ensure it's processed within {timeframe}.",
                    'tone': 'apologetic',
                    'estimated_time': '24-48 hours',
                    'actions': ['escalate_priority', 'expedite_refund', 'manager_review']
                }
            },
            'delivery': {
                'standard': {
                    'template': "Thank you for reaching out about your delivery. I can see your order {order_id} and I'm checking the current status. {delivery_update}. If you have any additional concerns, please don't hesitate to reach out.",
                    'tone': 'professional',
                    'estimated_time': '2-3 business days',
                    'actions': ['track_package', 'contact_carrier', 'update_customer']
                },
                'delayed': {
                    'template': "I apologize for the delay with your delivery. I understand this is inconvenient, especially when you're expecting your order. Let me investigate this immediately and provide you with an updated timeline. {compensation_offer}",
                    'tone': 'apologetic',
                    'estimated_time': '1-2 business days',
                    'actions': ['investigate_delay', 'contact_warehouse', 'offer_compensation']
                }
            },
            'billing': {
                'standard': {
                    'template': "Thank you for contacting us about your billing inquiry. I've reviewed your account and can see {billing_details}. Let me explain the charges and help resolve any discrepancies.",
                    'tone': 'professional',
                    'estimated_time': '1-2 business days',
                    'actions': ['review_account', 'explain_charges', 'process_adjustment']
                },
                'dispute': {
                    'template': "I understand your concern about the charges on your account. Billing disputes are taken very seriously, and I want to ensure we resolve this fairly. I'm initiating a formal review of {disputed_charges} and will have our billing specialist investigate.",
                    'tone': 'empathetic',
                    'estimated_time': '5-7 business days',
                    'actions': ['initiate_dispute', 'specialist_review', 'account_freeze']
                }
            },
            'technical': {
                'standard': {
                    'template': "Thank you for reporting this technical issue. I understand how frustrating technical problems can be. Let me walk you through some troubleshooting steps: {troubleshooting_steps}. If these don't resolve the issue, I'll escalate to our technical team.",
                    'tone': 'professional',
                    'estimated_time': '2-4 hours',
                    'actions': ['provide_troubleshooting', 'test_solution', 'escalate_if_needed']
                },
                'complex': {
                    'template': "I understand you're experiencing a complex technical issue. This requires specialized attention from our technical team. I'm creating a priority ticket and our senior technicians will investigate immediately. You'll receive updates every {update_frequency}.",
                    'tone': 'professional',
                    'estimated_time': '24-48 hours',
                    'actions': ['escalate_technical', 'assign_specialist', 'schedule_updates']
                }
            },
            'general': {
                'inquiry': {
                    'template': "Thank you for your inquiry. I'm happy to help you with {inquiry_topic}. Based on your question, here's the information you need: {information}. Is there anything else I can clarify for you?",
                    'tone': 'professional',
                    'estimated_time': '1-2 hours',
                    'actions': ['provide_information', 'confirm_understanding', 'offer_additional_help']
                }
            }
        }
    
    def _load_tone_guidelines(self) -> Dict[str, Dict[str, str]]:
        """Load tone guidelines for different response types"""
        return {
            'professional': {
                'greeting': 'Thank you for contacting us',
                'acknowledgment': 'I understand your concern',
                'action': 'I will',
                'closing': 'Please do not hesitate to reach out if you need further assistance'
            },
            'empathetic': {
                'greeting': 'Thank you for reaching out to us',
                'acknowledgment': 'I can understand how this situation must feel',
                'action': 'Let me personally ensure',
                'closing': 'We truly value your patience and loyalty'
            },
            'apologetic': {
                'greeting': 'Thank you for bringing this to our attention',
                'acknowledgment': 'I sincerely apologize for',
                'action': 'I will immediately',
                'closing': 'We appreciate your understanding and patience'
            },
            'urgent': {
                'greeting': 'I understand this is urgent',
                'acknowledgment': 'This requires immediate attention',
                'action': 'I am prioritizing',
                'closing': 'I will personally monitor this until resolved'
            }
        }
    
    def _load_escalation_triggers(self) -> Dict[str, List[str]]:
        """Load triggers that indicate a case should be escalated"""
        return {
            'keywords': ['legal action', 'lawyer', 'lawsuit', 'attorney', 'court', 'sue', 'media', 'news', 'social media complaint'],
            'sentiment_threshold': -0.8,  # Very negative sentiment
            'value_thresholds': {
                'refund_amount': 500.0,  # Refunds over $500
                'customer_lifetime_value': 1000.0  # High-value customers
            },
            'repeat_contact': 3,  # Customer contacted 3+ times about same issue
            'urgency_keywords': ['urgent', 'emergency', 'immediately', 'asap', 'critical'],
            'regulatory_keywords': ['hipaa', 'gdpr', 'privacy violation', 'data breach']
        }
    
    def _get_template_suggestions(self, category: str, sentiment: Dict[str, Any], 
                                urgency: str) -> List[ResponseSuggestion]:
        """Get template-based response suggestions"""
        suggestions = []
        
        templates = self.response_templates.get(category, self.response_templates.get('general', {}))
        
        # Determine which template variant to use
        sentiment_score = sentiment.get('confidence', 0.5)
        sentiment_label = sentiment.get('sentiment', 'neutral')
        
        if urgency == 'high' or sentiment_label == 'negative':
            template_key = 'urgent' if 'urgent' in templates else list(templates.keys())[0]
        else:
            template_key = 'standard' if 'standard' in templates else list(templates.keys())[0]
        
        template_data = templates.get(template_key, {})
        
        if template_data:
            suggestion = ResponseSuggestion(
                id=f"template_{category}_{template_key}",
                response_text=template_data['template'],
                confidence_score=0.85,  # High confidence for templates
                response_type='template',
                tone=template_data.get('tone', 'professional'),
                estimated_resolution_time=template_data.get('estimated_time', '2-3 business days'),
                requires_escalation=urgency == 'critical',
                suggested_actions=template_data.get('actions', []),
                source_reference=f"Template: {category}/{template_key}",
                metadata={
                    'template_category': category,
                    'template_variant': template_key,
                    'urgency_level': urgency
                }
            )
            suggestions.append(suggestion)
        
        return suggestions
    
    def _get_historical_suggestions(self, ticket_analysis: Dict[str, Any], 
                                  context: Dict[str, Any]) -> List[ResponseSuggestion]:
        """Get suggestions based on historical successful resolutions"""
        suggestions = []
        
        if not self.persistent_store:
            return suggestions
        
        try:
            # Get similar historical cases
            original_text = ticket_analysis.get('original_text', '')
            category = ticket_analysis.get('intent_classification', {}).get('intent_category', 'general')
            
            similar_cases = self.persistent_store.find_similar_complaints(original_text, category, 3)
            
            for case in similar_cases:
                # Get resolutions for this case
                resolutions = self.persistent_store.get_best_resolutions(
                    case.get('category', category),
                    case.get('description', ''),
                    2
                )
                
                for resolution in resolutions:
                    if resolution.get('effectiveness_score', 0) > 0.7:  # High effectiveness
                        suggestion = ResponseSuggestion(
                            id=f"historical_{resolution.get('id', 'unknown')}",
                            response_text=self._adapt_historical_response(resolution, ticket_analysis),
                            confidence_score=min(0.9, resolution.get('effectiveness_score', 0.7) + 0.1),
                            response_type='historical',
                            tone='professional',
                            estimated_resolution_time=f"{resolution.get('resolution_time_hours', 24):.0f} hours",
                            requires_escalation=False,
                            suggested_actions=resolution.get('solution_steps', []),
                            source_reference=f"Historical case: {case.get('id', 'unknown')}",
                            metadata={
                                'similar_case_id': case.get('id'),
                                'similarity_score': case.get('similarity_score', 0),
                                'historical_effectiveness': resolution.get('effectiveness_score', 0),
                                'usage_count': resolution.get('used_count', 0)
                            }
                        )
                        suggestions.append(suggestion)
        
        except Exception as e:
            self.logger.warning(f"Error getting historical suggestions: {str(e)}")
        
        return suggestions
    
    def _generate_personalized_responses(self, ticket_analysis: Dict[str, Any], 
                                       context: Dict[str, Any]) -> List[ResponseSuggestion]:
        """Generate AI-powered personalized responses"""
        suggestions = []
        
        try:
            # Extract key elements
            category = ticket_analysis.get('intent_classification', {}).get('intent_category', 'general')
            sentiment = ticket_analysis.get('sentiment_analysis', {})
            original_text = ticket_analysis.get('original_text', '')
            
            # Determine appropriate tone
            tone = self._determine_response_tone(sentiment, category)
            
            # Generate personalized response
            personalized_response = self._create_personalized_response(
                category, sentiment, original_text, context, tone
            )
            
            if personalized_response:
                suggestion = ResponseSuggestion(
                    id=f"generated_{category}_{datetime.now().strftime('%H%M%S')}",
                    response_text=personalized_response['text'],
                    confidence_score=personalized_response['confidence'],
                    response_type='generated',
                    tone=tone,
                    estimated_resolution_time=personalized_response['estimated_time'],
                    requires_escalation=personalized_response.get('escalation_needed', False),
                    suggested_actions=personalized_response.get('actions', []),
                    source_reference="AI-Generated Response",
                    metadata={
                        'generation_method': 'rule_based_ai',
                        'customer_context': context.get('customer_tier', 'standard') if context else 'standard',
                        'personalization_level': 'high'
                    }
                )
                suggestions.append(suggestion)
        
        except Exception as e:
            self.logger.warning(f"Error generating personalized responses: {str(e)}")
        
        return suggestions
    
    def _get_policy_based_suggestions(self, category: str, 
                                    ticket_analysis: Dict[str, Any]) -> List[ResponseSuggestion]:
        """Get suggestions based on company policies and procedures"""
        suggestions = []
        
        # Policy-based responses for compliance and consistency
        policy_responses = {
            'refund': {
                'text': "Based on our refund policy, I can confirm that your request qualifies for processing. Our standard refund timeframe is 5-7 business days from approval. I'll initiate this process now and send you a confirmation email with tracking details.",
                'actions': ['verify_policy_compliance', 'process_standard_refund', 'send_policy_reference'],
                'confidence': 0.9
            },
            'billing': {
                'text': "I've reviewed your account against our billing policies and procedures. To ensure accuracy and compliance, I'm having our billing compliance team review the charges. This review typically takes 3-5 business days and you'll receive a detailed explanation.",
                'actions': ['policy_compliance_check', 'escalate_billing_review', 'document_compliance'],
                'confidence': 0.85
            },
            'privacy': {
                'text': "Thank you for your privacy-related inquiry. In accordance with our privacy policy and applicable regulations, I'm connecting you with our Data Protection Officer who will address your request within the required timeframe.",
                'actions': ['escalate_privacy_officer', 'log_privacy_request', 'compliance_tracking'],
                'confidence': 0.95
            }
        }
        
        if category in policy_responses:
            policy_data = policy_responses[category]
            
            suggestion = ResponseSuggestion(
                id=f"policy_{category}",
                response_text=policy_data['text'],
                confidence_score=policy_data['confidence'],
                response_type='policy',
                tone='professional',
                estimated_resolution_time='3-5 business days',
                requires_escalation=category == 'privacy',
                suggested_actions=policy_data['actions'],
                source_reference=f"Company Policy: {category.title()}",
                metadata={
                    'policy_category': category,
                    'compliance_required': True,
                    'regulatory_consideration': category in ['privacy', 'billing']
                }
            )
            suggestions.append(suggestion)
        
        return suggestions
    
    def _determine_urgency_level(self, ticket_analysis: Dict[str, Any], 
                               context: Dict[str, Any]) -> str:
        """Determine urgency level based on ticket analysis and context"""
        original_text = ticket_analysis.get('original_text', '').lower()
        sentiment = ticket_analysis.get('sentiment_analysis', {})
        
        # Check for urgent keywords
        urgent_keywords = self.escalation_triggers['urgency_keywords']
        if any(keyword in original_text for keyword in urgent_keywords):
            return 'critical'
        
        # Check sentiment
        if sentiment.get('sentiment') == 'negative' and sentiment.get('confidence', 0) > 0.8:
            return 'high'
        
        # Check customer context
        if context:
            if context.get('customer_tier') == 'premium':
                return 'high'
            if context.get('previous_contacts', 0) >= 2:
                return 'high'
        
        return 'medium'
    
    def _determine_response_tone(self, sentiment: Dict[str, Any], category: str) -> str:
        """Determine appropriate response tone"""
        sentiment_score = sentiment.get('confidence', 0.5)
        sentiment_label = sentiment.get('sentiment', 'neutral')
        
        if sentiment_label == 'negative' and sentiment_score > 0.7:
            return 'apologetic'
        elif category in ['billing', 'refund'] and sentiment_label == 'negative':
            return 'empathetic'
        elif 'urgent' in sentiment.get('metadata', {}).get('keywords', []):
            return 'urgent'
        else:
            return 'professional'
    
    def _create_personalized_response(self, category: str, sentiment: Dict[str, Any], 
                                    original_text: str, context: Dict[str, Any], tone: str) -> Dict[str, Any]:
        """Create a personalized response using AI logic"""
        
        # Extract key entities and issues from the original text
        issues = self._extract_key_issues(original_text)
        customer_name = context.get('customer_name', 'valued customer') if context else 'valued customer'
        
        # Get tone elements
        tone_elements = self.tone_guidelines.get(tone, self.tone_guidelines['professional'])
        
        # Build personalized response
        response_parts = []
        
        # Greeting
        response_parts.append(f"{tone_elements['greeting']}, {customer_name}.")
        
        # Acknowledgment
        if issues:
            response_parts.append(f"{tone_elements['acknowledgment']} regarding {issues[0]}.")
        else:
            response_parts.append(f"{tone_elements['acknowledgment']} your concern.")
        
        # Action statement
        if category == 'refund':
            response_parts.append(f"{tone_elements['action']} review your refund request and process it according to our policy.")
        elif category == 'delivery':
            response_parts.append(f"{tone_elements['action']} track your order and provide you with an update immediately.")
        elif category == 'billing':
            response_parts.append(f"{tone_elements['action']} review your account and clarify any billing questions.")
        else:
            response_parts.append(f"{tone_elements['action']} investigate this matter and provide you with a resolution.")
        
        # Estimated timeline
        timeline = self._estimate_resolution_time(category, sentiment)
        response_parts.append(f"You can expect a resolution within {timeline}.")
        
        # Closing
        response_parts.append(tone_elements['closing'])
        
        response_text = " ".join(response_parts)
        
        return {
            'text': response_text,
            'confidence': 0.75,
            'estimated_time': timeline,
            'escalation_needed': sentiment.get('confidence', 0) > 0.8 and sentiment.get('sentiment') == 'negative',
            'actions': self._suggest_actions(category, sentiment)
        }
    
    def _extract_key_issues(self, text: str) -> List[str]:
        """Extract key issues from customer text"""
        # Simple keyword-based extraction (could be enhanced with NLP)
        issue_patterns = {
            'refund': r'(refund|money back|return|cancel)',
            'delivery': r'(delivery|shipping|package|order)',
            'billing': r'(charge|bill|payment|invoice)',
            'product': r'(defective|broken|damaged|quality)',
            'account': r'(account|login|password|access)'
        }
        
        issues = []
        text_lower = text.lower()
        
        for issue_type, pattern in issue_patterns.items():
            if re.search(pattern, text_lower):
                issues.append(f"your {issue_type} concern")
        
        return issues
    
    def _estimate_resolution_time(self, category: str, sentiment: Dict[str, Any]) -> str:
        """Estimate resolution time based on category and complexity"""
        base_times = {
            'refund': '3-5 business days',
            'delivery': '1-2 business days',
            'billing': '2-3 business days',
            'technical': '24-48 hours',
            'general': '1-2 business days'
        }
        
        # Adjust for negative sentiment (might need expedited handling)
        if sentiment.get('sentiment') == 'negative' and sentiment.get('confidence', 0) > 0.7:
            expedited_times = {
                'refund': '1-2 business days',
                'delivery': '24 hours',
                'billing': '1 business day',
                'technical': '4-8 hours',
                'general': '24 hours'
            }
            return expedited_times.get(category, '24 hours')
        
        return base_times.get(category, '2-3 business days')
    
    def _suggest_actions(self, category: str, sentiment: Dict[str, Any]) -> List[str]:
        """Suggest actions based on category and sentiment"""
        base_actions = {
            'refund': ['verify_purchase', 'check_policy_compliance', 'process_refund'],
            'delivery': ['track_package', 'contact_shipping', 'provide_update'],
            'billing': ['review_account', 'verify_charges', 'explain_billing'],
            'technical': ['gather_details', 'troubleshoot', 'escalate_if_needed'],
            'general': ['gather_information', 'research_solution', 'follow_up']
        }
        
        actions = base_actions.get(category, ['investigate', 'research', 'respond'])
        
        # Add escalation action for very negative sentiment
        if sentiment.get('sentiment') == 'negative' and sentiment.get('confidence', 0) > 0.8:
            actions.append('consider_escalation')
        
        return actions
    
    def _adapt_historical_response(self, resolution: Dict[str, Any], 
                                 ticket_analysis: Dict[str, Any]) -> str:
        """Adapt a historical resolution to current context"""
        base_response = resolution.get('description', '')
        solution_steps = resolution.get('solution_steps', [])
        
        if isinstance(solution_steps, list) and solution_steps:
            # Create a response from solution steps
            steps_text = " First, I'll " + ". Then, I'll ".join(solution_steps[:3]) + "."
            adapted_response = f"Based on similar cases we've successfully resolved, here's how I'll help you: {steps_text}"
        else:
            adapted_response = f"I can help you with this issue. {base_response}"
        
        return adapted_response
    
    def _rank_suggestions(self, suggestions: List[ResponseSuggestion], 
                         ticket_analysis: Dict[str, Any], 
                         context: Dict[str, Any]) -> List[ResponseSuggestion]:
        """Rank suggestions by relevance and confidence"""
        
        # Scoring factors
        def calculate_score(suggestion: ResponseSuggestion) -> float:
            score = suggestion.confidence_score
            
            # Boost template suggestions for consistency
            if suggestion.response_type == 'template':
                score += 0.1
            
            # Boost historical suggestions with high usage
            if suggestion.response_type == 'historical':
                usage_count = suggestion.metadata.get('usage_count', 0)
                score += min(0.15, usage_count * 0.01)  # Up to 0.15 boost
            
            # Boost policy suggestions for compliance categories
            if suggestion.response_type == 'policy':
                score += 0.05
            
            # Reduce score if escalation is required but suggestion doesn't address it
            sentiment = ticket_analysis.get('sentiment_analysis', {})
            if (sentiment.get('sentiment') == 'negative' and 
                sentiment.get('confidence', 0) > 0.7 and 
                not suggestion.requires_escalation):
                score -= 0.1
            
            return score
        
        # Sort by calculated score
        suggestions_with_scores = [(s, calculate_score(s)) for s in suggestions]
        suggestions_with_scores.sort(key=lambda x: x[1], reverse=True)
        
        return [s for s, score in suggestions_with_scores]
    
    def _get_fallback_suggestions(self, ticket_analysis: Dict[str, Any]) -> List[ResponseSuggestion]:
        """Provide fallback suggestions when main system fails"""
        fallback = ResponseSuggestion(
            id="fallback_general",
            response_text="Thank you for contacting us. I understand your concern and I'm here to help. Let me review your inquiry and get back to you with a solution as soon as possible.",
            confidence_score=0.5,
            response_type='template',
            tone='professional',
            estimated_resolution_time='24 hours',
            requires_escalation=False,
            suggested_actions=['review_inquiry', 'research_solution', 'respond_promptly'],
            source_reference="Fallback Template",
            metadata={'fallback': True}
        )
        
        return [fallback]

    def get_response_quality_score(self, response_text: str, ticket_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate the quality of a response
        
        Args:
            response_text: The response text to evaluate
            ticket_analysis: Original ticket analysis
            
        Returns:
            Quality score and feedback
        """
        try:
            quality_metrics = {
                'completeness': 0.0,
                'tone_appropriateness': 0.0,
                'clarity': 0.0,
                'empathy': 0.0,
                'actionability': 0.0
            }
            
            # Check completeness (addresses key points)
            original_text = ticket_analysis.get('original_text', '').lower()
            response_lower = response_text.lower()
            
            key_topics = ['refund', 'delivery', 'billing', 'technical', 'account']
            addressed_topics = sum(1 for topic in key_topics if topic in original_text and topic in response_lower)
            total_topics = sum(1 for topic in key_topics if topic in original_text)
            
            if total_topics > 0:
                quality_metrics['completeness'] = addressed_topics / total_topics
            else:
                quality_metrics['completeness'] = 0.8  # Default for general inquiries
            
            # Check tone appropriateness
            sentiment = ticket_analysis.get('sentiment_analysis', {})
            if sentiment.get('sentiment') == 'negative':
                # Negative sentiment should have empathetic/apologetic tone
                empathy_words = ['apologize', 'understand', 'sorry', 'inconvenience', 'frustration']
                empathy_count = sum(1 for word in empathy_words if word in response_lower)
                quality_metrics['tone_appropriateness'] = min(1.0, empathy_count * 0.3)
                quality_metrics['empathy'] = min(1.0, empathy_count * 0.25)
            else:
                # Neutral/positive can be professional
                professional_indicators = ['thank you', 'happy to help', 'assist', 'resolve']
                prof_count = sum(1 for phrase in professional_indicators if phrase in response_lower)
                quality_metrics['tone_appropriateness'] = min(1.0, prof_count * 0.3)
                quality_metrics['empathy'] = 0.7  # Default good score for neutral cases
            
            # Check clarity (sentence structure, length)
            sentences = response_text.split('.')
            avg_sentence_length = sum(len(s.split()) for s in sentences) / max(len(sentences), 1)
            
            # Optimal sentence length is 15-25 words
            if 15 <= avg_sentence_length <= 25:
                quality_metrics['clarity'] = 0.9
            elif 10 <= avg_sentence_length <= 30:
                quality_metrics['clarity'] = 0.7
            else:
                quality_metrics['clarity'] = 0.5
            
            # Check actionability (clear next steps)
            action_indicators = ['will', 'i\'ll', 'next step', 'process', 'review', 'contact', 'send']
            action_count = sum(1 for phrase in action_indicators if phrase in response_lower)
            quality_metrics['actionability'] = min(1.0, action_count * 0.2)
            
            # Calculate overall score
            overall_score = sum(quality_metrics.values()) / len(quality_metrics)
            
            # Generate feedback
            feedback = []
            if quality_metrics['completeness'] < 0.7:
                feedback.append("Consider addressing more aspects of the customer's concern")
            if quality_metrics['tone_appropriateness'] < 0.6:
                feedback.append("Adjust tone to better match customer sentiment")
            if quality_metrics['empathy'] < 0.6 and sentiment.get('sentiment') == 'negative':
                feedback.append("Add more empathetic language for negative sentiment")
            if quality_metrics['actionability'] < 0.6:
                feedback.append("Include clearer next steps or actions")
            
            return {
                'overall_score': overall_score,
                'metrics': quality_metrics,
                'feedback': feedback,
                'grade': self._score_to_grade(overall_score)
            }
            
        except Exception as e:
            self.logger.error(f"Error evaluating response quality: {str(e)}")
            return {
                'overall_score': 0.5,
                'metrics': {},
                'feedback': ['Unable to evaluate response quality'],
                'grade': 'C'
            }
    
    def _score_to_grade(self, score: float) -> str:
        """Convert numeric score to letter grade"""
        if score >= 0.9:
            return 'A+'
        elif score >= 0.8:
            return 'A'
        elif score >= 0.7:
            return 'B'
        elif score >= 0.6:
            return 'C'
        else:
            return 'D'
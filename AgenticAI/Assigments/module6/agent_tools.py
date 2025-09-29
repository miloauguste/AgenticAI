"""
Agent Tools Module
Provides auto-suggest responses, escalation highlighting, and supervisor insights
"""
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import logging
from collections import defaultdict, Counter

class AgentTools:
    """
    Tools for agents including auto-suggestions, escalation detection, and supervisor insights
    """
    
    def __init__(self, database_manager=None, session_manager=None):
        """
        Initialize agent tools
        
        Args:
            database_manager: Database manager instance for historical data
            session_manager: Session manager for current session context
        """
        self.logger = logging.getLogger(__name__)
        self.database = database_manager
        self.session = session_manager
        
        # Escalation triggers
        self.escalation_triggers = {
            'high_urgency_keywords': [
                'legal action', 'lawsuit', 'attorney', 'lawyer', 'sue', 'court',
                'better business bureau', 'bbb', 'consumer protection',
                'discrimination', 'harassment', 'abuse', 'threat'
            ],
            'frustration_indicators': [
                'never again', 'worst service', 'completely unacceptable',
                'absolutely ridiculous', 'total waste', 'fed up',
                'sick of this', 'done with you', 'cancel everything'
            ],
            'urgency_indicators': [
                'emergency', 'urgent', 'asap', 'immediately', 'right now',
                'critical', 'business critical', 'losing money',
                'deadline', 'time sensitive'
            ],
            'sentiment_thresholds': {
                'negative_sentiment': -0.6,
                'high_frustration': 0.8,
                'high_urgency': 0.7
            }
        }
        
        # Response templates for auto-suggestions
        self.response_templates = {
            'angry_customer': {
                'opening': [
                    "I sincerely apologize for the frustration this has caused you.",
                    "I understand how disappointing this experience has been for you.",
                    "I'm truly sorry that we've let you down."
                ],
                'acknowledgment': [
                    "Your concerns are completely valid and I take full responsibility.",
                    "I can see why you're upset, and I want to make this right immediately.",
                    "This is not the level of service you deserve from us."
                ],
                'action': [
                    "Let me personally handle this issue and provide you with a solution today.",
                    "I'm escalating this to my supervisor to ensure we resolve this quickly.",
                    "Here's exactly what I'm going to do to fix this problem:"
                ]
            },
            'billing_issue': {
                'opening': [
                    "I'll help you resolve this billing concern right away.",
                    "Let me investigate this billing discrepancy for you immediately."
                ],
                'action': [
                    "I'm reviewing your account now to identify the issue.",
                    "I can process a refund for any incorrect charges today.",
                    "Let me check our billing system to understand what happened."
                ]
            },
            'technical_issue': {
                'opening': [
                    "I'll help you troubleshoot this technical issue step by step.",
                    "Let's work together to resolve this technical problem."
                ],
                'action': [
                    "First, let's try these troubleshooting steps:",
                    "I'm going to connect you with our technical specialist.",
                    "Here's a workaround while we fix the underlying issue:"
                ]
            },
            'positive_feedback': {
                'opening': [
                    "Thank you so much for your kind words!",
                    "I'm delighted to hear about your positive experience."
                ],
                'action': [
                    "I'll share your feedback with the team - it means a lot to us.",
                    "Is there anything else I can help you with today?"
                ]
            }
        }
    
    def auto_suggest_response(self, ticket_text: str, sentiment_analysis: Dict[str, Any],
                             intent_analysis: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Auto-suggest draft responses based on ticket analysis
        
        Args:
            ticket_text: Original customer message
            sentiment_analysis: Sentiment analysis results
            intent_analysis: Intent classification results
            context: Additional context (session history, similar cases)
            
        Returns:
            Suggested response with variants and metadata
        """
        try:
            # Determine response category based on sentiment and intent
            sentiment_category = sentiment_analysis.get('sentiment_category', 'neutral')
            customer_mood = sentiment_analysis.get('customer_mood', 'neutral')
            intent_category = intent_analysis.get('intent_category', 'General')
            urgency_level = sentiment_analysis.get('urgency_level', 'low')
            
            # Select appropriate template
            template_key = self._select_response_template(sentiment_category, customer_mood, intent_category)
            template = self.response_templates.get(template_key, self.response_templates['technical_issue'])
            
            # Generate response variants
            suggestions = []
            
            # Generate 3 different response variants
            for i in range(3):
                response_parts = []
                
                # Opening
                if 'opening' in template:
                    opening_idx = i % len(template['opening'])
                    response_parts.append(template['opening'][opening_idx])
                
                # Acknowledgment (if customer is upset)
                if sentiment_category == 'negative' and 'acknowledgment' in template:
                    ack_idx = i % len(template['acknowledgment'])
                    response_parts.append(template['acknowledgment'][ack_idx])
                
                # Action
                if 'action' in template:
                    action_idx = i % len(template['action'])
                    response_parts.append(template['action'][action_idx])
                
                # Closing
                response_parts.append("Please let me know if you need any additional assistance.")
                
                suggested_response = "\n\n".join(response_parts)
                
                suggestions.append({
                    'response_text': suggested_response,
                    'tone': self._determine_response_tone(sentiment_category, urgency_level),
                    'confidence': self._calculate_suggestion_confidence(sentiment_analysis, intent_analysis),
                    'variant': f"Option {i+1}",
                    'estimated_length': len(suggested_response),
                    'personalization_level': 'medium'
                })
            
            # Add context-aware enhancements
            if context:
                suggestions = self._enhance_with_context(suggestions, context)
            
            # Get escalation recommendation
            escalation_rec = self.analyze_escalation_need(ticket_text, sentiment_analysis, intent_analysis)
            
            return {
                'suggestions': suggestions,
                'recommended_option': 1,  # Default to first option
                'escalation_recommendation': escalation_rec,
                'response_metadata': {
                    'template_used': template_key,
                    'sentiment_category': sentiment_category,
                    'customer_mood': customer_mood,
                    'urgency_level': urgency_level,
                    'estimated_response_time': self._estimate_response_time(urgency_level),
                    'follow_up_required': urgency_level in ['high', 'critical'],
                    'supervisor_review': escalation_rec['requires_escalation']
                },
                'generation_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error generating auto-suggestions: {str(e)}")
            return {
                'suggestions': [{
                    'response_text': "Thank you for contacting us. I'll look into your request and get back to you shortly.",
                    'tone': 'professional',
                    'confidence': 0.5,
                    'variant': 'Fallback',
                    'estimated_length': 88,
                    'personalization_level': 'low'
                }],
                'recommended_option': 1,
                'escalation_recommendation': {'requires_escalation': False, 'reason': 'fallback_mode'},
                'error': str(e)
            }
    
    def analyze_escalation_need(self, ticket_text: str, sentiment_analysis: Dict[str, Any],
                               intent_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze if ticket needs escalation and highlight reasons
        
        Args:
            ticket_text: Customer message
            sentiment_analysis: Sentiment analysis results
            intent_analysis: Intent classification results
            
        Returns:
            Escalation analysis with highlighting
        """
        escalation_score = 0.0
        escalation_reasons = []
        highlighted_phrases = []
        
        ticket_lower = ticket_text.lower()
        
        # Check for high urgency keywords
        for keyword in self.escalation_triggers['high_urgency_keywords']:
            if keyword in ticket_lower:
                escalation_score += 0.3
                escalation_reasons.append(f"Legal/threat indicator: '{keyword}'")
                highlighted_phrases.append(keyword)
        
        # Check for frustration indicators
        for indicator in self.escalation_triggers['frustration_indicators']:
            if indicator in ticket_lower:
                escalation_score += 0.2
                escalation_reasons.append(f"High frustration: '{indicator}'")
                highlighted_phrases.append(indicator)
        
        # Check for urgency indicators
        for indicator in self.escalation_triggers['urgency_indicators']:
            if indicator in ticket_lower:
                escalation_score += 0.15
                escalation_reasons.append(f"Urgency indicator: '{indicator}'")
                highlighted_phrases.append(indicator)
        
        # Check sentiment thresholds
        sentiment_score = sentiment_analysis.get('sentiment_score', 0)
        if sentiment_score <= self.escalation_triggers['sentiment_thresholds']['negative_sentiment']:
            escalation_score += 0.25
            escalation_reasons.append(f"Very negative sentiment: {sentiment_score:.2f}")
        
        # Check frustration and urgency scores
        frustration_score = sentiment_analysis.get('component_scores', {}).get('frustration', 0)
        urgency_score = sentiment_analysis.get('urgency_score', 0)
        
        if frustration_score >= self.escalation_triggers['sentiment_thresholds']['high_frustration']:
            escalation_score += 0.2
            escalation_reasons.append(f"High frustration score: {frustration_score:.2f}")
        
        if urgency_score >= self.escalation_triggers['sentiment_thresholds']['high_urgency']:
            escalation_score += 0.2
            escalation_reasons.append(f"High urgency score: {urgency_score:.2f}")
        
        # Determine escalation level
        requires_escalation = escalation_score >= 0.6
        escalation_level = self._determine_escalation_level(escalation_score)
        
        return {
            'requires_escalation': requires_escalation,
            'escalation_score': round(escalation_score, 2),
            'escalation_level': escalation_level,
            'escalation_reasons': escalation_reasons,
            'highlighted_phrases': list(set(highlighted_phrases)),
            'recommended_actions': self._get_escalation_actions(escalation_level),
            'priority': self._get_escalation_priority(escalation_score),
            'estimated_resolution_sla': self._get_escalation_sla(escalation_level)
        }
    
    def generate_supervisor_insights(self, time_period: str = '24h') -> Dict[str, Any]:
        """
        Generate supervisor insights including issue spikes and SLA violations
        
        Args:
            time_period: Time period for analysis ('24h', '7d', '30d')
            
        Returns:
            Comprehensive supervisor insights
        """
        try:
            if not self.database:
                return {'error': 'Database not available for insights'}
            
            # Parse time period
            hours_back = self._parse_time_period(time_period)
            cutoff_time = datetime.now() - timedelta(hours=hours_back)
            
            # Get statistics from database
            stats = self.database.get_ticket_statistics(hours_back // 24)
            
            # Get recent tickets for detailed analysis
            recent_tickets = self.database.get_recent_tickets(100)  # Last 100 tickets
            
            # Analyze issue spikes
            issue_spikes = self._detect_issue_spikes(recent_tickets, time_period)
            
            # Analyze SLA violations
            sla_analysis = self._analyze_sla_performance(recent_tickets)
            
            # Analyze escalation trends
            escalation_trends = self._analyze_escalation_trends(recent_tickets)
            
            # Generate recommendations
            recommendations = self._generate_supervisor_recommendations(
                stats, issue_spikes, sla_analysis, escalation_trends
            )
            
            # Risk alerts
            risk_alerts = self._identify_risk_alerts(issue_spikes, sla_analysis, escalation_trends)
            
            return {
                'time_period': time_period,
                'generated_at': datetime.now().isoformat(),
                'overview': {
                    'total_tickets': stats.get('total_tickets', 0),
                    'resolution_rate': stats.get('resolution_rate', 0),
                    'average_satisfaction': stats.get('average_satisfaction', 0),
                    'escalation_rate': escalation_trends.get('escalation_rate', 0)
                },
                'issue_spikes': issue_spikes,
                'sla_analysis': sla_analysis,
                'escalation_trends': escalation_trends,
                'risk_alerts': risk_alerts,
                'recommendations': recommendations,
                'category_insights': self._analyze_category_performance(stats),
                'team_performance': self._analyze_team_performance(recent_tickets),
                'trend_analysis': self._analyze_trends(stats, time_period)
            }
            
        except Exception as e:
            self.logger.error(f"Error generating supervisor insights: {str(e)}")
            return {'error': str(e), 'generated_at': datetime.now().isoformat()}
    
    def _select_response_template(self, sentiment: str, mood: str, intent: str) -> str:
        """Select appropriate response template based on analysis"""
        if sentiment == 'negative' or mood in ['frustrated', 'angry']:
            return 'angry_customer'
        elif intent == 'Billing':
            return 'billing_issue'
        elif intent == 'Technical':
            return 'technical_issue'
        elif sentiment == 'positive':
            return 'positive_feedback'
        else:
            return 'technical_issue'  # Default template
    
    def _determine_response_tone(self, sentiment: str, urgency: str) -> str:
        """Determine appropriate response tone"""
        if sentiment == 'negative' and urgency in ['high', 'critical']:
            return 'empathetic_urgent'
        elif sentiment == 'negative':
            return 'empathetic_professional'
        elif sentiment == 'positive':
            return 'appreciative'
        else:
            return 'professional'
    
    def _calculate_suggestion_confidence(self, sentiment: Dict, intent: Dict) -> float:
        """Calculate confidence in suggestion quality"""
        base_confidence = 0.7
        
        # Higher confidence for clear sentiment
        sentiment_confidence = sentiment.get('analysis_confidence', 0.5)
        intent_confidence = intent.get('confidence', 0.5)
        
        combined_confidence = (base_confidence + sentiment_confidence + intent_confidence) / 3
        return round(min(combined_confidence, 1.0), 2)
    
    def _enhance_with_context(self, suggestions: List[Dict], context: Dict) -> List[Dict]:
        """Enhance suggestions with session context"""
        # Add context-aware modifications (could be expanded)
        for suggestion in suggestions:
            if context.get('recent_issues'):
                suggestion['context_aware'] = True
                suggestion['personalization_level'] = 'high'
        
        return suggestions
    
    def _estimate_response_time(self, urgency: str) -> str:
        """Estimate response time based on urgency"""
        time_estimates = {
            'critical': '15 minutes',
            'high': '2 hours',
            'medium': '24 hours',
            'low': '48 hours'
        }
        return time_estimates.get(urgency, '24 hours')
    
    def _determine_escalation_level(self, score: float) -> str:
        """Determine escalation level based on score"""
        if score >= 0.8:
            return 'immediate'
        elif score >= 0.6:
            return 'high_priority'
        elif score >= 0.4:
            return 'monitor'
        else:
            return 'standard'
    
    def _get_escalation_actions(self, level: str) -> List[str]:
        """Get recommended actions for escalation level"""
        actions = {
            'immediate': [
                'Notify supervisor immediately',
                'Prepare incident report',
                'Consider executive escalation',
                'Document all interactions'
            ],
            'high_priority': [
                'Alert team lead within 1 hour',
                'Prioritize for senior agent',
                'Monitor closely for resolution',
                'Prepare detailed notes'
            ],
            'monitor': [
                'Flag for supervisor review',
                'Track resolution progress',
                'Consider proactive follow-up'
            ],
            'standard': [
                'Follow standard procedures',
                'Monitor customer satisfaction'
            ]
        }
        return actions.get(level, actions['standard'])
    
    def _get_escalation_priority(self, score: float) -> str:
        """Get priority level for escalation"""
        if score >= 0.8:
            return 'P1 - Critical'
        elif score >= 0.6:
            return 'P2 - High'
        elif score >= 0.4:
            return 'P3 - Medium'
        else:
            return 'P4 - Low'
    
    def _get_escalation_sla(self, level: str) -> str:
        """Get SLA for escalation level"""
        sla_times = {
            'immediate': '15 minutes',
            'high_priority': '2 hours',
            'monitor': '24 hours',
            'standard': '48 hours'
        }
        return sla_times.get(level, '48 hours')
    
    def _parse_time_period(self, period: str) -> int:
        """Parse time period string to hours"""
        if period == '24h':
            return 24
        elif period == '7d':
            return 168
        elif period == '30d':
            return 720
        else:
            return 24  # Default to 24 hours
    
    def _detect_issue_spikes(self, tickets: List[Dict], time_period: str) -> Dict[str, Any]:
        """Detect unusual spikes in specific issue types"""
        # Group tickets by category and time
        category_counts = defaultdict(list)
        
        for ticket in tickets:
            category = ticket.get('category', 'Unknown')
            created_at = ticket.get('created_at', '')
            category_counts[category].append(created_at)
        
        # Simple spike detection (can be enhanced with statistical analysis)
        spikes = []
        for category, timestamps in category_counts.items():
            if len(timestamps) > 5:  # More than 5 tickets in period
                spikes.append({
                    'category': category,
                    'count': len(timestamps),
                    'severity': 'high' if len(timestamps) > 15 else 'medium'
                })
        
        return {
            'detected_spikes': spikes,
            'total_categories_with_spikes': len(spikes),
            'highest_spike_category': max(spikes, key=lambda x: x['count'])['category'] if spikes else None
        }
    
    def _analyze_sla_performance(self, tickets: List[Dict]) -> Dict[str, Any]:
        """Analyze SLA performance and violations"""
        # Mock SLA analysis (would need actual resolution times)
        total_tickets = len(tickets)
        
        # Simulate some SLA metrics
        violations = []
        critical_violations = 0
        
        for ticket in tickets:
            urgency = ticket.get('urgency_level', 'medium')
            if urgency == 'critical':
                # Assume some critical tickets violate SLA
                if len(violations) < total_tickets * 0.1:  # 10% violation rate
                    violations.append({
                        'ticket_id': ticket.get('ticket_id'),
                        'urgency': urgency,
                        'violation_type': 'response_time'
                    })
                    critical_violations += 1
        
        return {
            'total_tickets': total_tickets,
            'sla_violations': len(violations),
            'violation_rate': round(len(violations) / max(total_tickets, 1) * 100, 1),
            'critical_violations': critical_violations,
            'violations_by_type': {
                'response_time': len(violations),
                'resolution_time': 0
            },
            'performance_status': 'warning' if len(violations) > total_tickets * 0.05 else 'good'
        }
    
    def _analyze_escalation_trends(self, tickets: List[Dict]) -> Dict[str, Any]:
        """Analyze escalation patterns and trends"""
        total_tickets = len(tickets)
        # Simulate escalation analysis
        escalated_count = int(total_tickets * 0.08)  # 8% escalation rate
        
        return {
            'total_escalations': escalated_count,
            'escalation_rate': round(escalated_count / max(total_tickets, 1) * 100, 1),
            'trend': 'stable',  # Could be 'increasing', 'decreasing', 'stable'
            'most_escalated_category': 'Billing',
            'escalation_reasons': {
                'high_frustration': escalated_count * 0.4,
                'technical_complexity': escalated_count * 0.3,
                'policy_exceptions': escalated_count * 0.3
            }
        }
    
    def _generate_supervisor_recommendations(self, stats: Dict, spikes: Dict, 
                                           sla: Dict, escalations: Dict) -> List[str]:
        """Generate actionable recommendations for supervisors"""
        recommendations = []
        
        # SLA-based recommendations
        if sla.get('violation_rate', 0) > 5:
            recommendations.append(f"🚨 SLA violation rate is {sla['violation_rate']}% - consider additional staffing")
        
        # Spike-based recommendations
        if spikes.get('total_categories_with_spikes', 0) > 0:
            recommendations.append(f"📈 Issue spike detected in {spikes['highest_spike_category']} - investigate root cause")
        
        # Escalation-based recommendations
        if escalations.get('escalation_rate', 0) > 10:
            recommendations.append(f"⚠️ High escalation rate ({escalations['escalation_rate']}%) - review agent training")
        
        # Satisfaction-based recommendations
        if stats.get('average_satisfaction', 5) < 3.5:
            recommendations.append("😞 Low customer satisfaction - implement quality improvement measures")
        
        if not recommendations:
            recommendations.append("✅ System performance is within normal parameters")
        
        return recommendations
    
    def _identify_risk_alerts(self, spikes: Dict, sla: Dict, escalations: Dict) -> List[Dict[str, Any]]:
        """Identify high-risk situations requiring immediate attention"""
        alerts = []
        
        # Critical SLA violations
        if sla.get('critical_violations', 0) > 0:
            alerts.append({
                'type': 'critical_sla',
                'severity': 'high',
                'message': f"{sla['critical_violations']} critical tickets violating SLA",
                'action_required': 'immediate_review'
            })
        
        # Severe issue spikes
        severe_spikes = [s for s in spikes.get('detected_spikes', []) if s.get('severity') == 'high']
        if severe_spikes:
            alerts.append({
                'type': 'issue_spike',
                'severity': 'medium',
                'message': f"High volume spike in {len(severe_spikes)} categories",
                'action_required': 'investigate_cause'
            })
        
        return alerts
    
    def _analyze_category_performance(self, stats: Dict) -> Dict[str, Any]:
        """Analyze performance by category"""
        category_dist = stats.get('category_distribution', {})
        
        return {
            'top_categories': dict(sorted(category_dist.items(), key=lambda x: x[1], reverse=True)[:5]),
            'category_insights': {
                'most_volume': max(category_dist.items(), key=lambda x: x[1])[0] if category_dist else 'None',
                'total_categories': len(category_dist)
            }
        }
    
    def _analyze_team_performance(self, tickets: List[Dict]) -> Dict[str, Any]:
        """Analyze team performance metrics"""
        # Mock team performance analysis
        return {
            'agent_workload': 'balanced',
            'response_time_average': '2.3 hours',
            'quality_score': 4.2,
            'training_recommendations': ['escalation_handling', 'customer_empathy']
        }
    
    def _analyze_trends(self, stats: Dict, time_period: str) -> Dict[str, Any]:
        """Analyze trends over the time period"""
        return {
            'ticket_volume_trend': 'stable',
            'satisfaction_trend': 'improving',
            'resolution_rate_trend': 'stable',
            'period_comparison': {
                'current_period': time_period,
                'vs_previous': 'similar_volume'
            }
        }
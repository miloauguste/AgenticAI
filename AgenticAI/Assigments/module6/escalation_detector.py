#!/usr/bin/env python3
"""
Escalation Detection and Highlighting System
Automatically identifies cases that need supervisor attention or escalation
"""

import logging
import re
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import json

class EscalationLevel(Enum):
    """Escalation priority levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    IMMEDIATE = "immediate"

class EscalationReason(Enum):
    """Reasons for escalation"""
    LEGAL_THREAT = "legal_threat"
    HIGH_VALUE_CUSTOMER = "high_value_customer"
    REPEATED_CONTACT = "repeated_contact"
    SEVERE_NEGATIVE_SENTIMENT = "severe_negative_sentiment"
    REGULATORY_COMPLIANCE = "regulatory_compliance"
    SOCIAL_MEDIA_THREAT = "social_media_threat"
    COMPLEX_TECHNICAL_ISSUE = "complex_technical_issue"
    HIGH_FINANCIAL_IMPACT = "high_financial_impact"
    VIP_CUSTOMER = "vip_customer"
    SAFETY_CONCERN = "safety_concern"
    DATA_PRIVACY_ISSUE = "data_privacy_issue"
    COMPETITOR_MENTION = "competitor_mention"

@dataclass
class EscalationAlert:
    """Data class for escalation alerts"""
    id: str
    ticket_id: str
    level: EscalationLevel
    reasons: List[EscalationReason]
    confidence_score: float
    priority_score: int  # 1-10 scale
    recommended_action: str
    suggested_assignee: str
    timeframe: str
    risk_factors: List[str]
    customer_context: Dict[str, Any]
    extracted_indicators: List[str]
    metadata: Dict[str, Any]
    created_at: datetime

class EscalationDetector:
    """
    Advanced escalation detection system for customer service
    """
    
    def __init__(self, persistent_store_integration=None):
        """
        Initialize escalation detector
        
        Args:
            persistent_store_integration: Optional persistent store for historical analysis
        """
        self.logger = logging.getLogger(__name__)
        self.persistent_store = persistent_store_integration
        
        # Load detection rules and patterns
        self.legal_patterns = self._load_legal_patterns()
        self.sentiment_thresholds = self._load_sentiment_thresholds()
        self.value_thresholds = self._load_value_thresholds()
        self.technical_complexity_indicators = self._load_technical_indicators()
        self.regulatory_patterns = self._load_regulatory_patterns()
        self.social_media_patterns = self._load_social_media_patterns()
        
        self.logger.info("Escalation detector initialized with comprehensive rule sets")
    
    def analyze_escalation_need(self, ticket_analysis: Dict[str, Any], 
                              customer_context: Dict[str, Any] = None,
                              interaction_history: List[Dict[str, Any]] = None) -> EscalationAlert:
        """
        Analyze if a ticket needs escalation and generate alert
        
        Args:
            ticket_analysis: Analysis results from support agent
            customer_context: Customer information and history
            interaction_history: Previous interactions with this customer
            
        Returns:
            EscalationAlert with recommendations
        """
        try:
            self.logger.info(f"Analyzing escalation need for ticket {ticket_analysis.get('ticket_id', 'unknown')}")
            
            escalation_indicators = []
            risk_factors = []
            confidence_scores = []
            
            # 1. Legal threat detection
            legal_score, legal_indicators = self._detect_legal_threats(ticket_analysis)
            if legal_score > 0.7:
                escalation_indicators.append(EscalationReason.LEGAL_THREAT)
                risk_factors.extend(legal_indicators)
                confidence_scores.append(legal_score)
            
            # 2. Sentiment analysis for severe negativity
            sentiment_score, sentiment_risk = self._analyze_severe_sentiment(ticket_analysis)
            if sentiment_score > 0.8:
                escalation_indicators.append(EscalationReason.SEVERE_NEGATIVE_SENTIMENT)
                risk_factors.extend(sentiment_risk)
                confidence_scores.append(sentiment_score)
            
            # 3. Customer value and tier analysis
            value_score, value_indicators = self._analyze_customer_value(customer_context)
            if value_score > 0.7:
                if customer_context and customer_context.get('tier') == 'VIP':
                    escalation_indicators.append(EscalationReason.VIP_CUSTOMER)
                else:
                    escalation_indicators.append(EscalationReason.HIGH_VALUE_CUSTOMER)
                risk_factors.extend(value_indicators)
                confidence_scores.append(value_score)
            
            # 4. Repeated contact pattern
            repeat_score, repeat_indicators = self._analyze_repeat_contacts(interaction_history, customer_context)
            if repeat_score > 0.6:
                escalation_indicators.append(EscalationReason.REPEATED_CONTACT)
                risk_factors.extend(repeat_indicators)
                confidence_scores.append(repeat_score)
            
            # 5. Regulatory and compliance issues
            regulatory_score, regulatory_indicators = self._detect_regulatory_issues(ticket_analysis)
            if regulatory_score > 0.8:
                escalation_indicators.append(EscalationReason.REGULATORY_COMPLIANCE)
                risk_factors.extend(regulatory_indicators)
                confidence_scores.append(regulatory_score)
            
            # 6. Social media threat detection
            social_score, social_indicators = self._detect_social_media_threats(ticket_analysis)
            if social_score > 0.6:
                escalation_indicators.append(EscalationReason.SOCIAL_MEDIA_THREAT)
                risk_factors.extend(social_indicators)
                confidence_scores.append(social_score)
            
            # 7. Technical complexity assessment
            tech_score, tech_indicators = self._assess_technical_complexity(ticket_analysis)
            if tech_score > 0.7:
                escalation_indicators.append(EscalationReason.COMPLEX_TECHNICAL_ISSUE)
                risk_factors.extend(tech_indicators)
                confidence_scores.append(tech_score)
            
            # 8. Financial impact assessment
            financial_score, financial_indicators = self._assess_financial_impact(ticket_analysis, customer_context)
            if financial_score > 0.7:
                escalation_indicators.append(EscalationReason.HIGH_FINANCIAL_IMPACT)
                risk_factors.extend(financial_indicators)
                confidence_scores.append(financial_score)
            
            # 9. Safety and security concerns
            safety_score, safety_indicators = self._detect_safety_concerns(ticket_analysis)
            if safety_score > 0.8:
                escalation_indicators.append(EscalationReason.SAFETY_CONCERN)
                risk_factors.extend(safety_indicators)
                confidence_scores.append(safety_score)
            
            # 10. Data privacy issues
            privacy_score, privacy_indicators = self._detect_privacy_issues(ticket_analysis)
            if privacy_score > 0.7:
                escalation_indicators.append(EscalationReason.DATA_PRIVACY_ISSUE)
                risk_factors.extend(privacy_indicators)
                confidence_scores.append(privacy_score)
            
            # Calculate overall escalation metrics
            overall_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
            escalation_level = self._determine_escalation_level(escalation_indicators, confidence_scores)
            priority_score = self._calculate_priority_score(escalation_indicators, confidence_scores, customer_context)
            
            # Generate recommendations
            recommended_action = self._generate_recommended_action(escalation_indicators, escalation_level)
            suggested_assignee = self._suggest_assignee(escalation_indicators, escalation_level)
            timeframe = self._determine_timeframe(escalation_level, escalation_indicators)
            
            # Create escalation alert
            alert = EscalationAlert(
                id=f"escalation_{ticket_analysis.get('ticket_id', 'unknown')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                ticket_id=ticket_analysis.get('ticket_id', 'unknown'),
                level=escalation_level,
                reasons=escalation_indicators,
                confidence_score=overall_confidence,
                priority_score=priority_score,
                recommended_action=recommended_action,
                suggested_assignee=suggested_assignee,
                timeframe=timeframe,
                risk_factors=list(set(risk_factors)),  # Remove duplicates
                customer_context=customer_context or {},
                extracted_indicators=self._extract_key_indicators(ticket_analysis, escalation_indicators),
                metadata={
                    'analysis_timestamp': datetime.now().isoformat(),
                    'detection_confidence_scores': confidence_scores,
                    'escalation_trigger_count': len(escalation_indicators),
                    'risk_factor_count': len(set(risk_factors))
                },
                created_at=datetime.now()
            )
            
            self.logger.info(f"Escalation analysis complete: Level {escalation_level.value}, {len(escalation_indicators)} reasons")
            return alert
            
        except Exception as e:
            self.logger.error(f"Error analyzing escalation need: {str(e)}")
            return self._create_default_alert(ticket_analysis)
    
    def get_escalation_insights(self, days: int = 30) -> Dict[str, Any]:
        """
        Get insights about escalation patterns and trends
        
        Args:
            days: Number of days to analyze
            
        Returns:
            Escalation insights and analytics
        """
        try:
            if not self.persistent_store:
                return {'error': 'Persistent store not available for historical analysis'}
            
            # This would query historical escalations from the persistent store
            insights = {
                'escalation_trends': self._analyze_escalation_trends(days),
                'top_escalation_reasons': self._get_top_escalation_reasons(days),
                'escalation_by_category': self._get_escalations_by_category(days),
                'resolution_effectiveness': self._analyze_escalation_resolution_effectiveness(days),
                'prevention_recommendations': self._generate_prevention_recommendations(),
                'risk_alerts': self._identify_emerging_risks()
            }
            
            return insights
            
        except Exception as e:
            self.logger.error(f"Error getting escalation insights: {str(e)}")
            return {'error': str(e)}
    
    def _load_legal_patterns(self) -> List[Dict[str, Any]]:
        """Load patterns that indicate legal threats"""
        return [
            {
                'pattern': r'\b(lawyer|attorney|legal action|lawsuit|sue|court|litigation)\b',
                'weight': 0.9,
                'description': 'Direct legal threat language'
            },
            {
                'pattern': r'\b(better business bureau|bbb|report|complaint|authorities)\b',
                'weight': 0.7,
                'description': 'Regulatory reporting threat'
            },
            {
                'pattern': r'\b(contact.*media|news|public|expose|viral)\b',
                'weight': 0.6,
                'description': 'Media/publicity threat'
            },
            {
                'pattern': r'\b(fraud|scam|illegal|breach of contract)\b',
                'weight': 0.8,
                'description': 'Legal violation claims'
            }
        ]
    
    def _load_sentiment_thresholds(self) -> Dict[str, float]:
        """Load sentiment analysis thresholds for escalation"""
        return {
            'severe_negative': -0.8,
            'high_negative': -0.6,
            'confidence_threshold': 0.7,
            'anger_indicators': 0.8,
            'frustration_threshold': 0.7
        }
    
    def _load_value_thresholds(self) -> Dict[str, Any]:
        """Load customer value thresholds"""
        return {
            'high_value_customer': {
                'lifetime_value': 10000,
                'annual_spend': 5000,
                'tenure_years': 5
            },
            'vip_indicators': ['premium', 'enterprise', 'vip', 'platinum'],
            'high_impact_amounts': {
                'refund_threshold': 1000,
                'order_value_threshold': 2000
            }
        }
    
    def _load_technical_indicators(self) -> List[str]:
        """Load indicators of complex technical issues"""
        return [
            'system crash', 'data loss', 'security breach', 'integration failure',
            'api error', 'database corruption', 'server down', 'network issue',
            'multiple systems affected', 'escalate to engineering', 'critical bug',
            'production outage', 'service degradation'
        ]
    
    def _load_regulatory_patterns(self) -> List[Dict[str, Any]]:
        """Load patterns for regulatory compliance issues"""
        return [
            {
                'pattern': r'\b(gdpr|data protection|privacy violation|personal data)\b',
                'regulation': 'GDPR',
                'weight': 0.9
            },
            {
                'pattern': r'\b(hipaa|health information|medical records|phi)\b',
                'regulation': 'HIPAA',
                'weight': 0.95
            },
            {
                'pattern': r'\b(pci|payment card|credit card security|cardholder data)\b',
                'regulation': 'PCI DSS',
                'weight': 0.9
            },
            {
                'pattern': r'\b(sox|sarbanes oxley|financial reporting|audit)\b',
                'regulation': 'SOX',
                'weight': 0.8
            }
        ]
    
    def _load_social_media_patterns(self) -> List[str]:
        """Load patterns indicating social media threats"""
        return [
            'twitter', 'facebook', 'instagram', 'linkedin', 'social media',
            'post online', 'share experience', 'tell everyone', 'viral',
            'public review', 'glassdoor', 'yelp', 'google reviews'
        ]
    
    def _detect_legal_threats(self, ticket_analysis: Dict[str, Any]) -> Tuple[float, List[str]]:
        """Detect legal threat indicators"""
        text = ticket_analysis.get('original_text', '').lower()
        indicators = []
        scores = []
        
        for pattern_data in self.legal_patterns:
            pattern = pattern_data['pattern']
            weight = pattern_data['weight']
            description = pattern_data['description']
            
            if re.search(pattern, text):
                indicators.append(description)
                scores.append(weight)
        
        overall_score = max(scores) if scores else 0.0
        return overall_score, indicators
    
    def _analyze_severe_sentiment(self, ticket_analysis: Dict[str, Any]) -> Tuple[float, List[str]]:
        """Analyze for severe negative sentiment"""
        sentiment = ticket_analysis.get('sentiment_analysis', {})
        sentiment_score = sentiment.get('confidence', 0.0)
        sentiment_label = sentiment.get('sentiment', 'neutral')
        
        risk_indicators = []
        
        if sentiment_label == 'negative':
            if sentiment_score > self.sentiment_thresholds['severe_negative']:
                risk_indicators.append(f"Severe negative sentiment (confidence: {sentiment_score:.2f})")
                return sentiment_score, risk_indicators
            elif sentiment_score > self.sentiment_thresholds['high_negative']:
                risk_indicators.append(f"High negative sentiment (confidence: {sentiment_score:.2f})")
                return sentiment_score * 0.7, risk_indicators
        
        # Check for anger/frustration keywords
        text = ticket_analysis.get('original_text', '').lower()
        anger_keywords = ['furious', 'outraged', 'disgusted', 'appalled', 'livid', 'infuriated']
        frustration_keywords = ['frustrated', 'annoyed', 'irritated', 'fed up', 'sick of']
        
        anger_count = sum(1 for word in anger_keywords if word in text)
        frustration_count = sum(1 for word in frustration_keywords if word in text)
        
        if anger_count > 0:
            risk_indicators.append(f"Anger indicators detected ({anger_count} instances)")
            return 0.8, risk_indicators
        elif frustration_count > 1:
            risk_indicators.append(f"Multiple frustration indicators ({frustration_count} instances)")
            return 0.6, risk_indicators
        
        return 0.0, risk_indicators
    
    def _analyze_customer_value(self, customer_context: Dict[str, Any]) -> Tuple[float, List[str]]:
        """Analyze customer value and tier"""
        if not customer_context:
            return 0.0, []
        
        indicators = []
        score = 0.0
        
        # Check customer tier
        tier = customer_context.get('tier', '').lower()
        if tier in ['vip', 'premium', 'enterprise', 'platinum']:
            indicators.append(f"High-tier customer: {tier}")
            score = max(score, 0.9)
        
        # Check lifetime value
        lifetime_value = customer_context.get('lifetime_value', 0)
        if lifetime_value > self.value_thresholds['high_value_customer']['lifetime_value']:
            indicators.append(f"High lifetime value: ${lifetime_value:,.2f}")
            score = max(score, 0.8)
        
        # Check annual spend
        annual_spend = customer_context.get('annual_spend', 0)
        if annual_spend > self.value_thresholds['high_value_customer']['annual_spend']:
            indicators.append(f"High annual spend: ${annual_spend:,.2f}")
            score = max(score, 0.7)
        
        # Check tenure
        tenure_years = customer_context.get('tenure_years', 0)
        if tenure_years > self.value_thresholds['high_value_customer']['tenure_years']:
            indicators.append(f"Long-term customer: {tenure_years} years")
            score = max(score, 0.6)
        
        return score, indicators
    
    def _analyze_repeat_contacts(self, interaction_history: List[Dict[str, Any]], 
                               customer_context: Dict[str, Any]) -> Tuple[float, List[str]]:
        """Analyze for repeated contacts about same issue"""
        indicators = []
        
        if not interaction_history:
            # Check session context for repeat indicators
            if customer_context and customer_context.get('previous_contacts', 0) > 2:
                indicators.append(f"Customer contacted {customer_context['previous_contacts']} times recently")
                return 0.7, indicators
            return 0.0, indicators
        
        # Analyze interaction patterns
        recent_interactions = [
            i for i in interaction_history 
            if (datetime.now() - datetime.fromisoformat(i.get('created_at', '2020-01-01'))).days <= 7
        ]
        
        if len(recent_interactions) >= 3:
            indicators.append(f"{len(recent_interactions)} contacts in past 7 days")
            return 0.8, indicators
        elif len(recent_interactions) >= 2:
            indicators.append(f"{len(recent_interactions)} recent contacts")
            return 0.6, indicators
        
        return 0.0, indicators
    
    def _detect_regulatory_issues(self, ticket_analysis: Dict[str, Any]) -> Tuple[float, List[str]]:
        """Detect regulatory compliance issues"""
        text = ticket_analysis.get('original_text', '').lower()
        indicators = []
        scores = []
        
        for pattern_data in self.regulatory_patterns:
            pattern = pattern_data['pattern']
            regulation = pattern_data['regulation']
            weight = pattern_data['weight']
            
            if re.search(pattern, text):
                indicators.append(f"{regulation} compliance issue detected")
                scores.append(weight)
        
        overall_score = max(scores) if scores else 0.0
        return overall_score, indicators
    
    def _detect_social_media_threats(self, ticket_analysis: Dict[str, Any]) -> Tuple[float, List[str]]:
        """Detect social media threat indicators"""
        text = ticket_analysis.get('original_text', '').lower()
        indicators = []
        
        threat_count = sum(1 for pattern in self.social_media_patterns if pattern in text)
        
        if threat_count > 0:
            indicators.append(f"Social media threat indicators ({threat_count} found)")
            # Higher threat score for multiple indicators
            score = min(0.9, 0.3 + (threat_count * 0.2))
            return score, indicators
        
        return 0.0, indicators
    
    def _assess_technical_complexity(self, ticket_analysis: Dict[str, Any]) -> Tuple[float, List[str]]:
        """Assess technical complexity of the issue"""
        text = ticket_analysis.get('original_text', '').lower()
        indicators = []
        
        complexity_indicators = sum(1 for indicator in self.technical_complexity_indicators if indicator in text)
        
        if complexity_indicators > 0:
            indicators.append(f"Complex technical issue indicators ({complexity_indicators} found)")
            score = min(0.9, 0.4 + (complexity_indicators * 0.2))
            return score, indicators
        
        # Check for multiple system mentions
        system_keywords = ['system', 'platform', 'integration', 'api', 'database', 'server']
        system_mentions = sum(1 for keyword in system_keywords if keyword in text)
        
        if system_mentions >= 3:
            indicators.append(f"Multiple system components involved ({system_mentions} mentioned)")
            return 0.6, indicators
        
        return 0.0, indicators
    
    def _assess_financial_impact(self, ticket_analysis: Dict[str, Any], 
                               customer_context: Dict[str, Any]) -> Tuple[float, List[str]]:
        """Assess financial impact of the issue"""
        text = ticket_analysis.get('original_text', '').lower()
        indicators = []
        
        # Extract monetary amounts
        import re
        money_pattern = r'\$([0-9,]+\.?[0-9]*)'
        amounts = re.findall(money_pattern, text)
        
        if amounts:
            max_amount = max(float(amount.replace(',', '')) for amount in amounts)
            if max_amount > self.value_thresholds['high_impact_amounts']['refund_threshold']:
                indicators.append(f"High financial impact: ${max_amount:,.2f}")
                return 0.8, indicators
        
        # Check for financial keywords
        financial_keywords = ['revenue', 'loss', 'cost', 'expensive', 'investment', 'budget']
        financial_mentions = sum(1 for keyword in financial_keywords if keyword in text)
        
        if financial_mentions >= 2:
            indicators.append(f"Financial impact keywords detected ({financial_mentions} found)")
            return 0.6, indicators
        
        return 0.0, indicators
    
    def _detect_safety_concerns(self, ticket_analysis: Dict[str, Any]) -> Tuple[float, List[str]]:
        """Detect safety and security concerns"""
        text = ticket_analysis.get('original_text', '').lower()
        indicators = []
        
        safety_keywords = ['safety', 'dangerous', 'hazard', 'injury', 'harm', 'risk', 'emergency']
        security_keywords = ['security', 'breach', 'hack', 'vulnerability', 'unauthorized', 'compromised']
        
        safety_count = sum(1 for keyword in safety_keywords if keyword in text)
        security_count = sum(1 for keyword in security_keywords if keyword in text)
        
        if safety_count > 0:
            indicators.append(f"Safety concerns detected ({safety_count} indicators)")
            return 0.9, indicators
        
        if security_count > 0:
            indicators.append(f"Security concerns detected ({security_count} indicators)")
            return 0.8, indicators
        
        return 0.0, indicators
    
    def _detect_privacy_issues(self, ticket_analysis: Dict[str, Any]) -> Tuple[float, List[str]]:
        """Detect data privacy issues"""
        text = ticket_analysis.get('original_text', '').lower()
        indicators = []
        
        privacy_keywords = [
            'privacy', 'personal data', 'private information', 'confidential',
            'data protection', 'data breach', 'unauthorized access', 'identity theft'
        ]
        
        privacy_count = sum(1 for keyword in privacy_keywords if keyword in text)
        
        if privacy_count > 0:
            indicators.append(f"Privacy/data protection concerns ({privacy_count} indicators)")
            return 0.8, indicators
        
        return 0.0, indicators
    
    def _determine_escalation_level(self, reasons: List[EscalationReason], 
                                  confidence_scores: List[float]) -> EscalationLevel:
        """Determine escalation level based on reasons and confidence"""
        if not reasons:
            return EscalationLevel.LOW
        
        critical_reasons = {
            EscalationReason.LEGAL_THREAT,
            EscalationReason.SAFETY_CONCERN,
            EscalationReason.REGULATORY_COMPLIANCE
        }
        
        high_priority_reasons = {
            EscalationReason.VIP_CUSTOMER,
            EscalationReason.SEVERE_NEGATIVE_SENTIMENT,
            EscalationReason.HIGH_FINANCIAL_IMPACT,
            EscalationReason.DATA_PRIVACY_ISSUE
        }
        
        # Check for critical reasons
        if any(reason in critical_reasons for reason in reasons):
            return EscalationLevel.CRITICAL
        
        # Check for immediate action needed
        if (EscalationReason.LEGAL_THREAT in reasons or 
            EscalationReason.SAFETY_CONCERN in reasons):
            return EscalationLevel.IMMEDIATE
        
        # Check for high priority
        if any(reason in high_priority_reasons for reason in reasons):
            return EscalationLevel.HIGH
        
        # Check confidence scores
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
        
        if avg_confidence > 0.8:
            return EscalationLevel.HIGH
        elif avg_confidence > 0.6:
            return EscalationLevel.MEDIUM
        else:
            return EscalationLevel.LOW
    
    def _calculate_priority_score(self, reasons: List[EscalationReason], 
                                confidence_scores: List[float],
                                customer_context: Dict[str, Any]) -> int:
        """Calculate priority score (1-10)"""
        base_score = len(reasons) * 2  # 2 points per reason
        
        # Boost for critical reasons
        critical_boost = sum(3 for reason in reasons if reason in {
            EscalationReason.LEGAL_THREAT,
            EscalationReason.SAFETY_CONCERN,
            EscalationReason.REGULATORY_COMPLIANCE
        })
        
        # Boost for high confidence
        confidence_boost = sum(1 for score in confidence_scores if score > 0.8)
        
        # Customer tier boost
        tier_boost = 0
        if customer_context:
            tier = customer_context.get('tier', '').lower()
            if tier in ['vip', 'enterprise']:
                tier_boost = 2
            elif tier in ['premium', 'platinum']:
                tier_boost = 1
        
        total_score = base_score + critical_boost + confidence_boost + tier_boost
        return min(10, max(1, total_score))  # Clamp to 1-10 range
    
    def _generate_recommended_action(self, reasons: List[EscalationReason], 
                                   level: EscalationLevel) -> str:
        """Generate recommended action based on escalation reasons"""
        if EscalationReason.LEGAL_THREAT in reasons:
            return "Immediately escalate to legal team and senior management. Document all communications."
        
        if EscalationReason.SAFETY_CONCERN in reasons:
            return "Urgent escalation to safety team. Initiate safety protocol review."
        
        if EscalationReason.REGULATORY_COMPLIANCE in reasons:
            return "Escalate to compliance officer immediately. Begin regulatory response procedure."
        
        if EscalationReason.VIP_CUSTOMER in reasons:
            return "Assign to senior customer success manager. Expedite resolution with executive oversight."
        
        if level == EscalationLevel.CRITICAL:
            return "Critical escalation: Notify supervisor immediately and begin crisis management protocol."
        elif level == EscalationLevel.HIGH:
            return "High priority escalation: Assign to experienced agent with supervisor oversight."
        elif level == EscalationLevel.MEDIUM:
            return "Medium escalation: Route to team lead with timeline monitoring."
        else:
            return "Standard escalation: Monitor closely and follow standard procedures."
    
    def _suggest_assignee(self, reasons: List[EscalationReason], level: EscalationLevel) -> str:
        """Suggest appropriate assignee based on escalation type"""
        if EscalationReason.LEGAL_THREAT in reasons:
            return "Legal Team + Senior Customer Relations Manager"
        
        if EscalationReason.REGULATORY_COMPLIANCE in reasons:
            return "Compliance Officer + Legal Team"
        
        if EscalationReason.COMPLEX_TECHNICAL_ISSUE in reasons:
            return "Senior Technical Support + Engineering Team"
        
        if EscalationReason.VIP_CUSTOMER in reasons:
            return "Senior Customer Success Manager"
        
        if level == EscalationLevel.CRITICAL:
            return "Department Manager + Senior Agent"
        elif level == EscalationLevel.HIGH:
            return "Team Lead + Senior Agent"
        else:
            return "Experienced Customer Support Agent"
    
    def _determine_timeframe(self, level: EscalationLevel, reasons: List[EscalationReason]) -> str:
        """Determine response timeframe"""
        if EscalationReason.LEGAL_THREAT in reasons or EscalationReason.SAFETY_CONCERN in reasons:
            return "Immediate (within 1 hour)"
        
        if level == EscalationLevel.CRITICAL:
            return "Critical (within 2 hours)"
        elif level == EscalationLevel.HIGH:
            return "High priority (within 4 hours)"
        elif level == EscalationLevel.MEDIUM:
            return "Medium priority (within 8 hours)"
        else:
            return "Standard (within 24 hours)"
    
    def _extract_key_indicators(self, ticket_analysis: Dict[str, Any], 
                              reasons: List[EscalationReason]) -> List[str]:
        """Extract key text indicators that triggered escalation"""
        text = ticket_analysis.get('original_text', '')
        indicators = []
        
        # Extract sentences that contain escalation triggers
        sentences = text.split('.')
        
        for sentence in sentences:
            sentence_lower = sentence.lower().strip()
            if any(pattern['pattern'].replace(r'\b', '').replace('(', '').replace(')', '') 
                  in sentence_lower for pattern in self.legal_patterns):
                indicators.append(sentence.strip())
            
            # Add other pattern checks as needed
        
        return indicators[:5]  # Return top 5 indicators
    
    def _create_default_alert(self, ticket_analysis: Dict[str, Any]) -> EscalationAlert:
        """Create default alert when analysis fails"""
        return EscalationAlert(
            id=f"default_{ticket_analysis.get('ticket_id', 'unknown')}",
            ticket_id=ticket_analysis.get('ticket_id', 'unknown'),
            level=EscalationLevel.LOW,
            reasons=[],
            confidence_score=0.0,
            priority_score=1,
            recommended_action="Review ticket manually for escalation needs",
            suggested_assignee="Standard Support Agent",
            timeframe="Standard (within 24 hours)",
            risk_factors=[],
            customer_context={},
            extracted_indicators=[],
            metadata={'error': 'Analysis failed'},
            created_at=datetime.now()
        )
    
    # Placeholder methods for historical analysis (would be implemented with real data)
    def _analyze_escalation_trends(self, days: int) -> Dict[str, Any]:
        """Analyze escalation trends over time"""
        return {
            'trend_direction': 'stable',
            'volume_change': 5,
            'average_per_day': 12,
            'peak_hours': [10, 14, 16]
        }
    
    def _get_top_escalation_reasons(self, days: int) -> List[Dict[str, Any]]:
        """Get top escalation reasons"""
        return [
            {'reason': 'Severe Negative Sentiment', 'count': 45, 'percentage': 35},
            {'reason': 'Repeated Contact', 'count': 32, 'percentage': 25},
            {'reason': 'High Value Customer', 'count': 28, 'percentage': 22},
            {'reason': 'Complex Technical Issue', 'count': 23, 'percentage': 18}
        ]
    
    def _get_escalations_by_category(self, days: int) -> Dict[str, int]:
        """Get escalations by issue category"""
        return {
            'billing': 28,
            'technical': 25,
            'refund': 22,
            'delivery': 18,
            'account': 12
        }
    
    def _analyze_escalation_resolution_effectiveness(self, days: int) -> Dict[str, Any]:
        """Analyze how effectively escalations are resolved"""
        return {
            'average_resolution_time': 4.2,
            'resolution_rate': 0.94,
            'customer_satisfaction': 4.1,
            'repeat_escalation_rate': 0.08
        }
    
    def _generate_prevention_recommendations(self) -> List[str]:
        """Generate recommendations to prevent escalations"""
        return [
            "Implement proactive customer outreach for high-value customers",
            "Improve first-contact resolution training for agents",
            "Create early warning system for repeat contacts",
            "Enhance technical documentation for complex issues",
            "Develop customer sentiment monitoring in real-time"
        ]
    
    def _identify_emerging_risks(self) -> List[Dict[str, Any]]:
        """Identify emerging risk patterns"""
        return [
            {
                'risk': 'Increasing social media threats',
                'trend': 'upward',
                'impact': 'medium',
                'recommendation': 'Monitor social media mentions more closely'
            },
            {
                'risk': 'Complex technical issues rising',
                'trend': 'upward',
                'impact': 'high',
                'recommendation': 'Increase technical training for agents'
            }
        ]
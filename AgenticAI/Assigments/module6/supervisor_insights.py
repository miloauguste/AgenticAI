#!/usr/bin/env python3
"""
Supervisor Insights Dashboard
Provides comprehensive insights for supervisors including issue spikes, SLA violations, and team performance
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import json
import statistics

class SLAStatus(Enum):
    """SLA compliance status"""
    COMPLIANT = "compliant"
    AT_RISK = "at_risk"
    VIOLATED = "violated"
    CRITICAL = "critical"

class AlertSeverity(Enum):
    """Alert severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class SLAMetrics:
    """SLA performance metrics"""
    metric_name: str
    target_value: float
    current_value: float
    status: SLAStatus
    compliance_percentage: float
    time_to_violation: Optional[timedelta]
    trend: str  # "improving", "stable", "declining"
    violations_count: int
    last_violation: Optional[datetime]

@dataclass
class IssueSpike:
    """Issue spike detection"""
    spike_id: str
    category: str
    spike_start: datetime
    spike_duration: timedelta
    volume_increase: float  # percentage increase
    current_volume: int
    baseline_volume: int
    severity: AlertSeverity
    probable_causes: List[str]
    recommended_actions: List[str]
    affected_metrics: List[str]

@dataclass
class SupervisorAlert:
    """Supervisor alert notification"""
    alert_id: str
    title: str
    description: str
    severity: AlertSeverity
    category: str
    created_at: datetime
    requires_action: bool
    recommended_actions: List[str]
    affected_agents: List[str]
    metrics_impacted: List[str]
    estimated_impact: str

class SupervisorInsights:
    """
    Comprehensive supervisor insights and monitoring system
    """
    
    def __init__(self, persistent_store_integration=None, escalation_detector=None):
        """
        Initialize supervisor insights system
        
        Args:
            persistent_store_integration: For historical data analysis
            escalation_detector: For escalation trend analysis
        """
        self.logger = logging.getLogger(__name__)
        self.persistent_store = persistent_store_integration
        self.escalation_detector = escalation_detector
        
        # Load SLA configurations
        self.sla_config = self._load_sla_configuration()
        self.baseline_metrics = self._load_baseline_metrics()
        self.alert_thresholds = self._load_alert_thresholds()
        
        self.logger.info("Supervisor insights system initialized")
    
    def get_comprehensive_dashboard(self, timeframe_hours: int = 24) -> Dict[str, Any]:
        """
        Get comprehensive supervisor dashboard data
        
        Args:
            timeframe_hours: Timeframe for analysis in hours
            
        Returns:
            Complete dashboard data
        """
        try:
            self.logger.info(f"Generating supervisor dashboard for {timeframe_hours}h timeframe")
            
            dashboard = {
                'overview': self._get_overview_metrics(timeframe_hours),
                'sla_performance': self._get_sla_performance(timeframe_hours),
                'issue_spikes': self._detect_issue_spikes(timeframe_hours),
                'escalation_insights': self._get_escalation_insights(timeframe_hours),
                'team_performance': self._get_team_performance(timeframe_hours),
                'alerts': self._get_active_alerts(),
                'trends': self._get_trend_analysis(timeframe_hours),
                'recommendations': self._generate_supervisor_recommendations(),
                'resource_allocation': self._analyze_resource_needs(timeframe_hours),
                'quality_metrics': self._get_quality_metrics(timeframe_hours),
                'customer_satisfaction': self._get_satisfaction_trends(timeframe_hours),
                'operational_efficiency': self._get_efficiency_metrics(timeframe_hours)
            }
            
            self.logger.info("Supervisor dashboard generated successfully")
            return dashboard
            
        except Exception as e:
            self.logger.error(f"Error generating supervisor dashboard: {str(e)}")
            return self._get_fallback_dashboard()
    
    def monitor_sla_compliance(self, timeframe_hours: int = 24) -> List[SLAMetrics]:
        """
        Monitor SLA compliance across all metrics
        
        Args:
            timeframe_hours: Timeframe for analysis
            
        Returns:
            List of SLA metrics with compliance status
        """
        try:
            sla_metrics = []
            
            # Response time SLA
            response_sla = self._monitor_response_time_sla(timeframe_hours)
            sla_metrics.append(response_sla)
            
            # Resolution time SLA
            resolution_sla = self._monitor_resolution_time_sla(timeframe_hours)
            sla_metrics.append(resolution_sla)
            
            # Customer satisfaction SLA
            satisfaction_sla = self._monitor_satisfaction_sla(timeframe_hours)
            sla_metrics.append(satisfaction_sla)
            
            # First contact resolution SLA
            fcr_sla = self._monitor_fcr_sla(timeframe_hours)
            sla_metrics.append(fcr_sla)
            
            # Escalation rate SLA
            escalation_sla = self._monitor_escalation_rate_sla(timeframe_hours)
            sla_metrics.append(escalation_sla)
            
            self.logger.info(f"Monitored {len(sla_metrics)} SLA metrics")
            return sla_metrics
            
        except Exception as e:
            self.logger.error(f"Error monitoring SLA compliance: {str(e)}")
            return []
    
    def detect_and_analyze_spikes(self, timeframe_hours: int = 24) -> List[IssueSpike]:
        """
        Detect and analyze issue volume spikes
        
        Args:
            timeframe_hours: Timeframe for spike detection
            
        Returns:
            List of detected issue spikes
        """
        try:
            spikes = []
            
            # Analyze spikes by category
            categories = ['billing', 'technical', 'delivery', 'refund', 'account']
            
            for category in categories:
                spike = self._detect_category_spike(category, timeframe_hours)
                if spike:
                    spikes.append(spike)
            
            # Analyze overall volume spike
            overall_spike = self._detect_overall_volume_spike(timeframe_hours)
            if overall_spike:
                spikes.append(overall_spike)
            
            # Analyze sentiment spikes (negative sentiment increase)
            sentiment_spike = self._detect_sentiment_spike(timeframe_hours)
            if sentiment_spike:
                spikes.append(sentiment_spike)
            
            # Analyze escalation spikes
            escalation_spike = self._detect_escalation_spike(timeframe_hours)
            if escalation_spike:
                spikes.append(escalation_spike)
            
            self.logger.info(f"Detected {len(spikes)} issue spikes")
            return spikes
            
        except Exception as e:
            self.logger.error(f"Error detecting issue spikes: {str(e)}")
            return []
    
    def generate_supervisor_alerts(self, priority_threshold: AlertSeverity = AlertSeverity.MEDIUM) -> List[SupervisorAlert]:
        """
        Generate supervisor alerts based on current conditions
        
        Args:
            priority_threshold: Minimum severity for alerts
            
        Returns:
            List of supervisor alerts
        """
        try:
            alerts = []
            
            # SLA violation alerts
            sla_alerts = self._generate_sla_alerts()
            alerts.extend(sla_alerts)
            
            # Issue spike alerts
            spike_alerts = self._generate_spike_alerts()
            alerts.extend(spike_alerts)
            
            # Team performance alerts
            performance_alerts = self._generate_performance_alerts()
            alerts.extend(performance_alerts)
            
            # Quality alerts
            quality_alerts = self._generate_quality_alerts()
            alerts.extend(quality_alerts)
            
            # Resource capacity alerts
            capacity_alerts = self._generate_capacity_alerts()
            alerts.extend(capacity_alerts)
            
            # Filter by priority threshold
            filtered_alerts = [alert for alert in alerts if 
                             self._alert_severity_value(alert.severity) >= 
                             self._alert_severity_value(priority_threshold)]
            
            # Sort by severity (highest first)
            filtered_alerts.sort(key=lambda x: self._alert_severity_value(x.severity), reverse=True)
            
            self.logger.info(f"Generated {len(filtered_alerts)} supervisor alerts")
            return filtered_alerts
            
        except Exception as e:
            self.logger.error(f"Error generating supervisor alerts: {str(e)}")
            return []
    
    def _load_sla_configuration(self) -> Dict[str, Any]:
        """Load SLA configuration and targets"""
        return {
            'response_time': {
                'target_minutes': 15,
                'critical_threshold': 30,
                'measurement_window': 24  # hours
            },
            'resolution_time': {
                'target_hours': 24,
                'critical_threshold': 48,
                'by_priority': {
                    'critical': 4,
                    'high': 8,
                    'medium': 24,
                    'low': 72
                }
            },
            'customer_satisfaction': {
                'target_score': 4.0,
                'minimum_acceptable': 3.5,
                'survey_response_rate': 0.3
            },
            'first_contact_resolution': {
                'target_rate': 0.85,
                'minimum_acceptable': 0.75
            },
            'escalation_rate': {
                'target_max': 0.10,
                'warning_threshold': 0.15,
                'critical_threshold': 0.20
            }
        }
    
    def _load_baseline_metrics(self) -> Dict[str, float]:
        """Load baseline metrics for comparison"""
        return {
            'average_daily_volume': 150,
            'average_hourly_volume': 6.25,
            'baseline_categories': {
                'billing': 25,
                'technical': 20,
                'delivery': 18,
                'refund': 15,
                'account': 12,
                'general': 10
            },
            'baseline_sentiment_distribution': {
                'positive': 0.3,
                'neutral': 0.5,
                'negative': 0.2
            },
            'baseline_escalation_rate': 0.08
        }
    
    def _load_alert_thresholds(self) -> Dict[str, Any]:
        """Load alert threshold configurations"""
        return {
            'volume_spike': {
                'medium': 1.5,  # 50% increase
                'high': 2.0,    # 100% increase
                'critical': 3.0  # 200% increase
            },
            'response_time_degradation': {
                'medium': 1.2,  # 20% slower
                'high': 1.5,    # 50% slower
                'critical': 2.0  # 100% slower
            },
            'satisfaction_drop': {
                'medium': -0.2,  # 0.2 point drop
                'high': -0.5,    # 0.5 point drop
                'critical': -1.0  # 1.0 point drop
            }
        }
    
    def _get_overview_metrics(self, timeframe_hours: int) -> Dict[str, Any]:
        """Get high-level overview metrics"""
        # This would query real data from persistent store
        return {
            'total_tickets': self._calculate_total_tickets(timeframe_hours),
            'tickets_by_status': self._get_tickets_by_status(timeframe_hours),
            'average_response_time': self._calculate_avg_response_time(timeframe_hours),
            'average_resolution_time': self._calculate_avg_resolution_time(timeframe_hours),
            'customer_satisfaction': self._calculate_avg_satisfaction(timeframe_hours),
            'escalation_rate': self._calculate_escalation_rate(timeframe_hours),
            'active_agents': self._count_active_agents(timeframe_hours),
            'tickets_per_agent': self._calculate_tickets_per_agent(timeframe_hours),
            'peak_hours': self._identify_peak_hours(timeframe_hours),
            'sla_compliance_summary': self._get_sla_summary(timeframe_hours)
        }
    
    def _get_sla_performance(self, timeframe_hours: int) -> Dict[str, Any]:
        """Get detailed SLA performance metrics"""
        sla_metrics = self.monitor_sla_compliance(timeframe_hours)
        
        return {
            'metrics': [
                {
                    'name': metric.metric_name,
                    'target': metric.target_value,
                    'current': metric.current_value,
                    'status': metric.status.value,
                    'compliance': metric.compliance_percentage,
                    'trend': metric.trend,
                    'violations': metric.violations_count
                }
                for metric in sla_metrics
            ],
            'overall_compliance': sum(m.compliance_percentage for m in sla_metrics) / len(sla_metrics) if sla_metrics else 0,
            'critical_violations': len([m for m in sla_metrics if m.status == SLAStatus.VIOLATED]),
            'at_risk_count': len([m for m in sla_metrics if m.status == SLAStatus.AT_RISK])
        }
    
    def _detect_issue_spikes(self, timeframe_hours: int) -> List[Dict[str, Any]]:
        """Detect and format issue spikes"""
        spikes = self.detect_and_analyze_spikes(timeframe_hours)
        
        return [
            {
                'category': spike.category,
                'start_time': spike.spike_start.isoformat(),
                'duration_hours': spike.spike_duration.total_seconds() / 3600,
                'volume_increase': spike.volume_increase,
                'current_volume': spike.current_volume,
                'baseline_volume': spike.baseline_volume,
                'severity': spike.severity.value,
                'probable_causes': spike.probable_causes,
                'recommended_actions': spike.recommended_actions
            }
            for spike in spikes
        ]
    
    def _get_escalation_insights(self, timeframe_hours: int) -> Dict[str, Any]:
        """Get escalation insights and trends"""
        if self.escalation_detector:
            # Get escalation insights from detector
            insights = self.escalation_detector.get_escalation_insights(
                days=max(1, timeframe_hours // 24)
            )
            return insights
        else:
            # Return mock data
            return {
                'total_escalations': 12,
                'escalation_rate': 0.08,
                'top_reasons': [
                    {'reason': 'Severe Negative Sentiment', 'count': 5},
                    {'reason': 'High Value Customer', 'count': 3},
                    {'reason': 'Complex Technical Issue', 'count': 2}
                ],
                'resolution_time': 4.2,
                'success_rate': 0.94
            }
    
    def _get_team_performance(self, timeframe_hours: int) -> Dict[str, Any]:
        """Get team performance metrics"""
        return {
            'agent_performance': self._get_agent_performance_metrics(timeframe_hours),
            'team_metrics': {
                'average_tickets_per_agent': 12.3,
                'average_resolution_time': 2.1,
                'team_satisfaction_score': 4.2,
                'team_escalation_rate': 0.06
            },
            'performance_distribution': {
                'top_performers': 3,
                'meeting_expectations': 8,
                'needs_improvement': 2
            },
            'skill_gaps': self._identify_skill_gaps(),
            'training_recommendations': self._get_training_recommendations()
        }
    
    def _get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get current active alerts"""
        alerts = self.generate_supervisor_alerts()
        
        return [
            {
                'id': alert.alert_id,
                'title': alert.title,
                'description': alert.description,
                'severity': alert.severity.value,
                'category': alert.category,
                'created_at': alert.created_at.isoformat(),
                'requires_action': alert.requires_action,
                'recommended_actions': alert.recommended_actions
            }
            for alert in alerts
        ]
    
    def _monitor_response_time_sla(self, timeframe_hours: int) -> SLAMetrics:
        """Monitor response time SLA"""
        # Mock implementation - would query real data
        target = self.sla_config['response_time']['target_minutes']
        current = 18.5  # Current average response time in minutes
        
        compliance = max(0, min(100, (target / current) * 100)) if current > 0 else 100
        status = SLAStatus.COMPLIANT if current <= target else SLAStatus.VIOLATED
        
        return SLAMetrics(
            metric_name="Response Time",
            target_value=target,
            current_value=current,
            status=status,
            compliance_percentage=compliance,
            time_to_violation=None,
            trend="stable",
            violations_count=2,
            last_violation=datetime.now() - timedelta(hours=3)
        )
    
    def _monitor_resolution_time_sla(self, timeframe_hours: int) -> SLAMetrics:
        """Monitor resolution time SLA"""
        target = self.sla_config['resolution_time']['target_hours']
        current = 21.3  # Current average resolution time in hours
        
        compliance = max(0, min(100, (target / current) * 100)) if current > 0 else 100
        status = SLAStatus.COMPLIANT if current <= target else SLAStatus.AT_RISK
        
        return SLAMetrics(
            metric_name="Resolution Time",
            target_value=target,
            current_value=current,
            status=status,
            compliance_percentage=compliance,
            time_to_violation=timedelta(hours=2.7),
            trend="improving",
            violations_count=0,
            last_violation=None
        )
    
    def _monitor_satisfaction_sla(self, timeframe_hours: int) -> SLAMetrics:
        """Monitor customer satisfaction SLA"""
        target = self.sla_config['customer_satisfaction']['target_score']
        current = 4.1  # Current average satisfaction score
        
        compliance = min(100, (current / target) * 100) if target > 0 else 100
        status = SLAStatus.COMPLIANT if current >= target else SLAStatus.AT_RISK
        
        return SLAMetrics(
            metric_name="Customer Satisfaction",
            target_value=target,
            current_value=current,
            status=status,
            compliance_percentage=compliance,
            time_to_violation=None,
            trend="stable",
            violations_count=0,
            last_violation=None
        )
    
    def _monitor_fcr_sla(self, timeframe_hours: int) -> SLAMetrics:
        """Monitor first contact resolution SLA"""
        target = self.sla_config['first_contact_resolution']['target_rate']
        current = 0.82  # Current FCR rate
        
        compliance = min(100, (current / target) * 100) if target > 0 else 100
        status = SLAStatus.COMPLIANT if current >= target else SLAStatus.AT_RISK
        
        return SLAMetrics(
            metric_name="First Contact Resolution",
            target_value=target,
            current_value=current,
            status=status,
            compliance_percentage=compliance,
            time_to_violation=None,
            trend="declining",
            violations_count=1,
            last_violation=datetime.now() - timedelta(days=2)
        )
    
    def _monitor_escalation_rate_sla(self, timeframe_hours: int) -> SLAMetrics:
        """Monitor escalation rate SLA"""
        target = self.sla_config['escalation_rate']['target_max']
        current = 0.09  # Current escalation rate
        
        compliance = max(0, min(100, (target / current) * 100)) if current > 0 else 100
        status = SLAStatus.COMPLIANT if current <= target else SLAStatus.AT_RISK
        
        return SLAMetrics(
            metric_name="Escalation Rate",
            target_value=target,
            current_value=current,
            status=status,
            compliance_percentage=compliance,
            time_to_violation=None,
            trend="stable",
            violations_count=0,
            last_violation=None
        )
    
    def _detect_category_spike(self, category: str, timeframe_hours: int) -> Optional[IssueSpike]:
        """Detect spike in specific category"""
        # Mock implementation
        baseline = self.baseline_metrics['baseline_categories'].get(category, 10)
        current = baseline * 1.8  # 80% increase
        
        if current > baseline * 1.5:  # 50% spike threshold
            return IssueSpike(
                spike_id=f"spike_{category}_{datetime.now().strftime('%Y%m%d_%H%M')}",
                category=category,
                spike_start=datetime.now() - timedelta(hours=2),
                spike_duration=timedelta(hours=2),
                volume_increase=80.0,
                current_volume=int(current),
                baseline_volume=baseline,
                severity=AlertSeverity.HIGH,
                probable_causes=[f"System outage affecting {category} services", "Seasonal increase", "Product issue"],
                recommended_actions=[
                    f"Investigate {category} system status",
                    "Allocate additional agents to category",
                    "Prepare proactive customer communication"
                ],
                affected_metrics=["response_time", "customer_satisfaction"]
            )
        
        return None
    
    def _detect_overall_volume_spike(self, timeframe_hours: int) -> Optional[IssueSpike]:
        """Detect overall volume spike"""
        baseline = self.baseline_metrics['average_hourly_volume'] * timeframe_hours
        current = baseline * 2.2  # 120% increase
        
        if current > baseline * 1.5:
            return IssueSpike(
                spike_id=f"spike_overall_{datetime.now().strftime('%Y%m%d_%H%M')}",
                category="overall",
                spike_start=datetime.now() - timedelta(hours=1),
                spike_duration=timedelta(hours=1),
                volume_increase=120.0,
                current_volume=int(current),
                baseline_volume=int(baseline),
                severity=AlertSeverity.CRITICAL,
                probable_causes=["Major system outage", "Service disruption", "Marketing campaign"],
                recommended_actions=[
                    "Activate emergency response plan",
                    "Deploy all available agents",
                    "Implement queue management",
                    "Send proactive customer notifications"
                ],
                affected_metrics=["response_time", "resolution_time", "customer_satisfaction"]
            )
        
        return None
    
    def _detect_sentiment_spike(self, timeframe_hours: int) -> Optional[IssueSpike]:
        """Detect negative sentiment spike"""
        # Mock implementation
        baseline_negative = self.baseline_metrics['baseline_sentiment_distribution']['negative']
        current_negative = 0.4  # 40% negative sentiment
        
        if current_negative > baseline_negative * 1.5:
            return IssueSpike(
                spike_id=f"spike_sentiment_{datetime.now().strftime('%Y%m%d_%H%M')}",
                category="sentiment",
                spike_start=datetime.now() - timedelta(hours=3),
                spike_duration=timedelta(hours=3),
                volume_increase=(current_negative / baseline_negative - 1) * 100,
                current_volume=int(current_negative * 100),
                baseline_volume=int(baseline_negative * 100),
                severity=AlertSeverity.HIGH,
                probable_causes=["Service quality issue", "Communication problem", "System outage"],
                recommended_actions=[
                    "Review recent customer interactions",
                    "Investigate service quality issues",
                    "Prepare recovery communication plan"
                ],
                affected_metrics=["customer_satisfaction", "escalation_rate"]
            )
        
        return None
    
    def _detect_escalation_spike(self, timeframe_hours: int) -> Optional[IssueSpike]:
        """Detect escalation volume spike"""
        baseline_rate = self.baseline_metrics['baseline_escalation_rate']
        current_rate = 0.15  # 15% escalation rate
        
        if current_rate > baseline_rate * 1.5:
            return IssueSpike(
                spike_id=f"spike_escalation_{datetime.now().strftime('%Y%m%d_%H%M')}",
                category="escalation",
                spike_start=datetime.now() - timedelta(hours=4),
                spike_duration=timedelta(hours=4),
                volume_increase=(current_rate / baseline_rate - 1) * 100,
                current_volume=int(current_rate * 100),
                baseline_volume=int(baseline_rate * 100),
                severity=AlertSeverity.HIGH,
                probable_causes=["Complex issues increase", "Agent skill gaps", "Process breakdown"],
                recommended_actions=[
                    "Review escalation cases",
                    "Provide additional agent training",
                    "Deploy senior agents"
                ],
                affected_metrics=["resolution_time", "customer_satisfaction"]
            )
        
        return None
    
    # Mock implementations for other methods
    def _calculate_total_tickets(self, timeframe_hours: int) -> int:
        return int(self.baseline_metrics['average_hourly_volume'] * timeframe_hours * 1.2)
    
    def _get_tickets_by_status(self, timeframe_hours: int) -> Dict[str, int]:
        total = self._calculate_total_tickets(timeframe_hours)
        return {
            'open': int(total * 0.15),
            'in_progress': int(total * 0.25),
            'resolved': int(total * 0.55),
            'escalated': int(total * 0.05)
        }
    
    def _calculate_avg_response_time(self, timeframe_hours: int) -> float:
        return 18.5  # minutes
    
    def _calculate_avg_resolution_time(self, timeframe_hours: int) -> float:
        return 21.3  # hours
    
    def _calculate_avg_satisfaction(self, timeframe_hours: int) -> float:
        return 4.1  # out of 5
    
    def _calculate_escalation_rate(self, timeframe_hours: int) -> float:
        return 0.09  # 9%
    
    def _count_active_agents(self, timeframe_hours: int) -> int:
        return 13
    
    def _calculate_tickets_per_agent(self, timeframe_hours: int) -> float:
        return self._calculate_total_tickets(timeframe_hours) / self._count_active_agents(timeframe_hours)
    
    def _identify_peak_hours(self, timeframe_hours: int) -> List[int]:
        return [10, 14, 16]  # Peak hours of the day
    
    def _get_sla_summary(self, timeframe_hours: int) -> Dict[str, Any]:
        return {
            'overall_compliance': 87.5,
            'violations': 3,
            'at_risk': 2
        }
    
    def _get_agent_performance_metrics(self, timeframe_hours: int) -> List[Dict[str, Any]]:
        return [
            {
                'agent_id': 'agent_001',
                'name': 'Alice Johnson',
                'tickets_handled': 15,
                'avg_resolution_time': 1.8,
                'customer_satisfaction': 4.3,
                'escalation_rate': 0.05,
                'performance_score': 92
            },
            {
                'agent_id': 'agent_002', 
                'name': 'Bob Smith',
                'tickets_handled': 12,
                'avg_resolution_time': 2.2,
                'customer_satisfaction': 4.0,
                'escalation_rate': 0.08,
                'performance_score': 85
            }
        ]
    
    def _identify_skill_gaps(self) -> List[str]:
        return [
            "Technical troubleshooting for complex integrations",
            "De-escalation techniques for angry customers", 
            "Advanced billing system knowledge"
        ]
    
    def _get_training_recommendations(self) -> List[str]:
        return [
            "Schedule technical training session on API integrations",
            "Implement role-playing exercises for difficult customers",
            "Create billing system deep-dive workshop"
        ]
    
    def _get_trend_analysis(self, timeframe_hours: int) -> Dict[str, Any]:
        return {
            'volume_trend': 'increasing',
            'satisfaction_trend': 'stable',
            'response_time_trend': 'improving',
            'escalation_trend': 'stable'
        }
    
    def _generate_supervisor_recommendations(self) -> List[str]:
        return [
            "Consider increasing staffing during peak hours (10 AM, 2 PM, 4 PM)",
            "Focus training on technical issues which show higher escalation rates",
            "Implement proactive customer communication for known service issues",
            "Review and update response templates for billing category"
        ]
    
    def _analyze_resource_needs(self, timeframe_hours: int) -> Dict[str, Any]:
        return {
            'recommended_staffing': {
                'current_shift': 13,
                'optimal_staffing': 15,
                'peak_hour_needs': 18
            },
            'skill_requirements': {
                'technical_specialists': 3,
                'billing_experts': 2,
                'escalation_handlers': 2
            },
            'capacity_utilization': 87.5
        }
    
    def _get_quality_metrics(self, timeframe_hours: int) -> Dict[str, Any]:
        return {
            'response_quality_score': 4.2,
            'resolution_accuracy': 0.94,
            'customer_effort_score': 3.1,
            'agent_adherence_to_process': 0.91
        }
    
    def _get_satisfaction_trends(self, timeframe_hours: int) -> Dict[str, Any]:
        return {
            'current_score': 4.1,
            'trend': 'stable',
            'by_category': {
                'billing': 3.8,
                'technical': 4.2,
                'delivery': 4.3,
                'refund': 4.0
            },
            'response_rate': 0.32
        }
    
    def _get_efficiency_metrics(self, timeframe_hours: int) -> Dict[str, Any]:
        return {
            'first_contact_resolution': 0.82,
            'average_handle_time': 12.5,
            'agent_utilization': 0.87,
            'cost_per_resolution': 25.50
        }
    
    def _generate_sla_alerts(self) -> List[SupervisorAlert]:
        return [
            SupervisorAlert(
                alert_id="sla_001",
                title="Response Time SLA Violation",
                description="Average response time exceeded 15-minute target",
                severity=AlertSeverity.HIGH,
                category="sla",
                created_at=datetime.now(),
                requires_action=True,
                recommended_actions=["Review agent availability", "Implement queue prioritization"],
                affected_agents=["team_all"],
                metrics_impacted=["response_time", "customer_satisfaction"],
                estimated_impact="Medium impact on customer satisfaction"
            )
        ]
    
    def _generate_spike_alerts(self) -> List[SupervisorAlert]:
        return [
            SupervisorAlert(
                alert_id="spike_001",
                title="Billing Category Volume Spike",
                description="80% increase in billing-related tickets",
                severity=AlertSeverity.HIGH,
                category="volume_spike",
                created_at=datetime.now(),
                requires_action=True,
                recommended_actions=["Investigate billing system", "Allocate additional agents"],
                affected_agents=["billing_team"],
                metrics_impacted=["response_time", "resolution_time"],
                estimated_impact="High impact on response times"
            )
        ]
    
    def _generate_performance_alerts(self) -> List[SupervisorAlert]:
        return []
    
    def _generate_quality_alerts(self) -> List[SupervisorAlert]:
        return []
    
    def _generate_capacity_alerts(self) -> List[SupervisorAlert]:
        return []
    
    def _alert_severity_value(self, severity: AlertSeverity) -> int:
        """Convert alert severity to numeric value for comparison"""
        severity_values = {
            AlertSeverity.LOW: 1,
            AlertSeverity.MEDIUM: 2,
            AlertSeverity.HIGH: 3,
            AlertSeverity.CRITICAL: 4
        }
        return severity_values.get(severity, 1)
    
    def _get_fallback_dashboard(self) -> Dict[str, Any]:
        """Return minimal dashboard when main generation fails"""
        return {
            'overview': {'total_tickets': 0, 'status': 'error'},
            'sla_performance': {'overall_compliance': 0},
            'issue_spikes': [],
            'alerts': [],
            'error': 'Dashboard generation failed'
        }
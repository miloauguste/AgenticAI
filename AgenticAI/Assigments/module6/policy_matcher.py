#!/usr/bin/env python3
"""
Policy Reference Matching System
Matches customer queries with relevant policy sections and company procedures
"""

import re
import json
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime
import logging
from pathlib import Path

@dataclass
class PolicySection:
    """Represents a policy section with metadata"""
    id: str
    title: str
    content: str
    category: str
    subcategory: str = ""
    keywords: List[str] = field(default_factory=list)
    applies_to: List[str] = field(default_factory=list)  # customer types, regions, etc.
    last_updated: str = ""
    version: str = "1.0"
    priority: int = 1  # 1=high, 5=low

@dataclass
class PolicyMatch:
    """Represents a match between query and policy"""
    policy_section: PolicySection
    relevance_score: float
    matched_keywords: List[str]
    match_type: str  # exact, partial, semantic
    context_applicable: bool = True

class PolicyMatcher:
    """
    Intelligent policy matching system for customer service
    
    Features:
    - Keyword-based matching
    - Category-based filtering
    - Context-aware relevance scoring
    - Multi-level policy hierarchy
    """
    
    def __init__(self, policy_file_path: str = None):
        """
        Initialize the policy matcher
        
        Args:
            policy_file_path: Path to policy JSON file (optional)
        """
        self.logger = logging.getLogger(__name__)
        self.policies: Dict[str, PolicySection] = {}
        self.category_index: Dict[str, List[str]] = {}
        self.keyword_index: Dict[str, List[str]] = {}
        
        # Load default policies if no file specified
        if policy_file_path and Path(policy_file_path).exists():
            self.load_policies_from_file(policy_file_path)
        else:
            self._load_default_policies()
            
        self._build_indexes()
    
    def _load_default_policies(self):
        """Load default customer service policies"""
        default_policies = [
            # Refund Policies
            PolicySection(
                id="refund_001",
                title="Standard Refund Policy",
                content="Customers can request refunds within 30 days of purchase for unused items in original packaging. Digital products are eligible for refund within 14 days. Processing time is 5-10 business days.",
                category="refunds",
                subcategory="standard",
                keywords=["refund", "return", "money back", "30 days", "unused", "original packaging"],
                applies_to=["all_customers"],
                priority=1
            ),
            PolicySection(
                id="refund_002", 
                title="Defective Product Refund",
                content="Defective or damaged products can be returned for full refund or replacement regardless of purchase date. Customer must provide photos or description of defect. Expedited processing within 3 business days.",
                category="refunds",
                subcategory="defective",
                keywords=["defective", "damaged", "broken", "faulty", "malfunction", "replacement"],
                applies_to=["all_customers"],
                priority=1
            ),
            PolicySection(
                id="refund_003",
                title="Digital Product Refund",
                content="Digital downloads and software can be refunded within 14 days if not downloaded or activated. Once downloaded/activated, refunds are only available for technical issues that cannot be resolved.",
                category="refunds",
                subcategory="digital",
                keywords=["digital", "download", "software", "activated", "technical issues"],
                applies_to=["digital_customers"],
                priority=2
            ),
            
            # Shipping Policies
            PolicySection(
                id="shipping_001",
                title="Standard Shipping Policy",
                content="Standard shipping takes 5-7 business days. Express shipping (2-3 days) and overnight shipping available. Free shipping on orders over $50. Tracking information provided for all shipments.",
                category="shipping",
                subcategory="standard",
                keywords=["shipping", "delivery", "5-7 days", "express", "overnight", "free shipping", "tracking"],
                applies_to=["all_customers"],
                priority=1
            ),
            PolicySection(
                id="shipping_002",
                title="Lost Package Policy",
                content="If package is marked delivered but not received, customer should wait 24 hours then contact us. We will investigate with carrier and provide replacement or refund if package confirmed lost.",
                category="shipping",
                subcategory="lost_packages",
                keywords=["lost package", "not received", "marked delivered", "carrier", "investigation"],
                applies_to=["all_customers"],
                priority=1
            ),
            PolicySection(
                id="shipping_003",
                title="International Shipping",
                content="International shipping available to most countries. Delivery time 10-21 business days. Customer responsible for customs fees and import duties. Some restrictions apply.",
                category="shipping",
                subcategory="international",
                keywords=["international", "customs", "import duties", "10-21 days", "restrictions"],
                applies_to=["international_customers"],
                priority=2
            ),
            
            # Account Policies
            PolicySection(
                id="account_001",
                title="Account Security Policy",
                content="Accounts are protected with secure password requirements and optional two-factor authentication. Users must verify email address. Account lockout after 5 failed login attempts.",
                category="account",
                subcategory="security",
                keywords=["password", "two-factor", "authentication", "verify email", "lockout", "login"],
                applies_to=["all_customers"],
                priority=1
            ),
            PolicySection(
                id="account_002",
                title="Password Reset Procedure",
                content="Password reset requires email verification. Reset link valid for 24 hours. If email not received, check spam folder. Alternative verification through customer service with identity confirmation.",
                category="account", 
                subcategory="password_reset",
                keywords=["password reset", "email verification", "24 hours", "spam folder", "identity confirmation"],
                applies_to=["all_customers"],
                priority=1
            ),
            PolicySection(
                id="account_003",
                title="Account Closure Policy",
                content="Customers can close accounts at any time through account settings or by contacting customer service. Data retention according to legal requirements. Active subscriptions must be cancelled separately.",
                category="account",
                subcategory="closure",
                keywords=["close account", "account settings", "data retention", "subscriptions", "cancel"],
                applies_to=["all_customers"],
                priority=2
            ),
            
            # Billing Policies
            PolicySection(
                id="billing_001",
                title="Payment Methods",
                content="Accepted payment methods include major credit cards, PayPal, bank transfers. Auto-renewal for subscriptions. Payment processing by secure third-party providers. PCI compliance maintained.",
                category="billing",
                subcategory="payment_methods",
                keywords=["credit cards", "paypal", "bank transfer", "auto-renewal", "subscriptions", "secure payment"],
                applies_to=["all_customers"],
                priority=1
            ),
            PolicySection(
                id="billing_002",
                title="Billing Disputes",
                content="Billing disputes must be reported within 60 days. Provide transaction details and reason for dispute. Investigation takes 5-10 business days. Temporary credit may be issued during investigation.",
                category="billing",
                subcategory="disputes",
                keywords=["billing dispute", "60 days", "transaction details", "investigation", "temporary credit"],
                applies_to=["all_customers"],
                priority=1
            ),
            PolicySection(
                id="billing_003",
                title="Subscription Cancellation",
                content="Subscriptions can be cancelled anytime before next billing cycle. Cancellation takes effect at end of current billing period. No partial refunds for unused subscription time unless required by law.",
                category="billing",
                subcategory="cancellation", 
                keywords=["subscription", "cancel", "billing cycle", "partial refunds", "unused time"],
                applies_to=["subscription_customers"],
                priority=1
            ),
            
            # Privacy Policies
            PolicySection(
                id="privacy_001",
                title="Data Collection Policy",
                content="We collect minimal necessary data to provide services. Personal data encrypted and stored securely. Data shared only with explicit consent or legal requirement. Users can request data deletion.",
                category="privacy",
                subcategory="data_collection",
                keywords=["data collection", "personal data", "encrypted", "consent", "data deletion"],
                applies_to=["all_customers"],
                priority=1
            ),
            PolicySection(
                id="privacy_002",
                title="Cookie Policy",
                content="Website uses necessary cookies for functionality and optional cookies for analytics with user consent. Cookie preferences can be managed in browser settings. Third-party cookies from trusted partners only.",
                category="privacy",
                subcategory="cookies",
                keywords=["cookies", "analytics", "consent", "browser settings", "third-party"],
                applies_to=["website_users"],
                priority=3
            ),
            
            # Technical Support Policies
            PolicySection(
                id="support_001",
                title="Technical Support Hours",
                content="Technical support available 24/7 for critical issues, 9 AM - 6 PM EST for general support. Response time: 1 hour for critical, 24 hours for standard. Premium support available for enterprise customers.",
                category="support",
                subcategory="hours",
                keywords=["24/7", "critical issues", "9 AM - 6 PM", "response time", "premium support"],
                applies_to=["all_customers"],
                priority=1
            ),
            PolicySection(
                id="support_002",
                title="Remote Support Policy",
                content="Remote desktop support available with customer consent for complex technical issues. Session recorded for quality purposes. Customer maintains full control and can end session anytime.",
                category="support", 
                subcategory="remote_support",
                keywords=["remote desktop", "consent", "complex issues", "recorded", "customer control"],
                applies_to=["technical_customers"],
                priority=2
            )
        ]
        
        # Convert to dictionary
        for policy in default_policies:
            self.policies[policy.id] = policy
            
        self.logger.info(f"Loaded {len(default_policies)} default policies")
    
    def _build_indexes(self):
        """Build search indexes for efficient policy matching"""
        self.category_index = {}
        self.keyword_index = {}
        
        for policy_id, policy in self.policies.items():
            # Category index
            category = policy.category
            if category not in self.category_index:
                self.category_index[category] = []
            self.category_index[category].append(policy_id)
            
            # Keyword index
            for keyword in policy.keywords:
                keyword_lower = keyword.lower()
                if keyword_lower not in self.keyword_index:
                    self.keyword_index[keyword_lower] = []
                self.keyword_index[keyword_lower].append(policy_id)
        
        self.logger.info(f"Built indexes: {len(self.category_index)} categories, {len(self.keyword_index)} keywords")
    
    def find_relevant_policies(self, query: str, intent_category: str = None, 
                             customer_type: str = "all_customers", 
                             max_results: int = 5) -> List[PolicyMatch]:
        """
        Find policies relevant to customer query
        
        Args:
            query: Customer query text
            intent_category: Classified intent category (optional)
            customer_type: Type of customer (default: all_customers)
            max_results: Maximum number of results to return
            
        Returns:
            List of PolicyMatch objects sorted by relevance
        """
        try:
            query_lower = query.lower()
            matches = []
            
            # Extract keywords from query
            query_keywords = self._extract_keywords(query_lower)
            
            # Search by intent category first
            category_matches = self._search_by_category(intent_category, query_keywords) if intent_category else []
            
            # Search by keywords
            keyword_matches = self._search_by_keywords(query_keywords)
            
            # Combine and deduplicate matches
            all_policy_ids = set(category_matches + keyword_matches)
            
            # Score each policy
            for policy_id in all_policy_ids:
                policy = self.policies[policy_id]
                
                # Check if policy applies to customer type
                if not self._policy_applies_to_customer(policy, customer_type):
                    continue
                
                # Calculate relevance score
                relevance_score, matched_keywords, match_type = self._calculate_relevance(
                    policy, query_lower, query_keywords, intent_category
                )
                
                if relevance_score > 0.1:  # Minimum relevance threshold
                    matches.append(PolicyMatch(
                        policy_section=policy,
                        relevance_score=relevance_score,
                        matched_keywords=matched_keywords,
                        match_type=match_type,
                        context_applicable=True
                    ))
            
            # Sort by relevance score (descending) and priority (ascending)
            matches.sort(key=lambda x: (-x.relevance_score, x.policy_section.priority))
            
            # Return top matches
            return matches[:max_results]
            
        except Exception as e:
            self.logger.error(f"Error finding relevant policies: {str(e)}")
            return []
    
    def get_policy_summary(self, policy_match: PolicyMatch) -> Dict[str, Any]:
        """
        Generate a summary of a policy match for customer service agent
        
        Args:
            policy_match: PolicyMatch object
            
        Returns:
            Dictionary with policy summary and guidance
        """
        policy = policy_match.policy_section
        
        return {
            'policy_id': policy.id,
            'title': policy.title,
            'category': policy.category,
            'subcategory': policy.subcategory,
            'relevance_score': round(policy_match.relevance_score, 3),
            'match_type': policy_match.match_type,
            'matched_keywords': policy_match.matched_keywords,
            'content_summary': self._summarize_content(policy.content),
            'full_content': policy.content,
            'key_points': self._extract_key_points(policy.content),
            'agent_guidance': self._generate_agent_guidance(policy, policy_match),
            'customer_impact': self._assess_customer_impact(policy),
            'last_updated': policy.last_updated or "Not specified",
            'priority': policy.priority
        }
    
    def search_policies_by_category(self, category: str) -> List[PolicySection]:
        """
        Get all policies in a specific category
        
        Args:
            category: Policy category
            
        Returns:
            List of PolicySection objects
        """
        category_lower = category.lower()
        policy_ids = self.category_index.get(category_lower, [])
        return [self.policies[pid] for pid in policy_ids]
    
    def get_policy_by_id(self, policy_id: str) -> Optional[PolicySection]:
        """Get specific policy by ID"""
        return self.policies.get(policy_id)
    
    def add_policy(self, policy: PolicySection) -> bool:
        """
        Add a new policy to the system
        
        Args:
            policy: PolicySection object
            
        Returns:
            Success boolean
        """
        try:
            self.policies[policy.id] = policy
            self._build_indexes()  # Rebuild indexes
            self.logger.info(f"Added policy: {policy.id}")
            return True
        except Exception as e:
            self.logger.error(f"Error adding policy: {str(e)}")
            return False
    
    def update_policy(self, policy_id: str, updated_policy: PolicySection) -> bool:
        """
        Update existing policy
        
        Args:
            policy_id: ID of policy to update
            updated_policy: Updated PolicySection object
            
        Returns:
            Success boolean
        """
        try:
            if policy_id in self.policies:
                self.policies[policy_id] = updated_policy
                self._build_indexes()  # Rebuild indexes
                self.logger.info(f"Updated policy: {policy_id}")
                return True
            else:
                self.logger.warning(f"Policy not found for update: {policy_id}")
                return False
        except Exception as e:
            self.logger.error(f"Error updating policy: {str(e)}")
            return False
    
    def load_policies_from_file(self, file_path: str) -> bool:
        """
        Load policies from JSON file
        
        Args:
            file_path: Path to JSON file
            
        Returns:
            Success boolean
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for policy_data in data.get('policies', []):
                policy = PolicySection(**policy_data)
                self.policies[policy.id] = policy
            
            self._build_indexes()
            self.logger.info(f"Loaded {len(data.get('policies', []))} policies from file")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading policies from file: {str(e)}")
            return False
    
    def save_policies_to_file(self, file_path: str) -> bool:
        """
        Save policies to JSON file
        
        Args:
            file_path: Path to save JSON file
            
        Returns:
            Success boolean
        """
        try:
            policies_data = []
            for policy in self.policies.values():
                policy_dict = {
                    'id': policy.id,
                    'title': policy.title,
                    'content': policy.content,
                    'category': policy.category,
                    'subcategory': policy.subcategory,
                    'keywords': policy.keywords,
                    'applies_to': policy.applies_to,
                    'last_updated': policy.last_updated,
                    'version': policy.version,
                    'priority': policy.priority
                }
                policies_data.append(policy_dict)
            
            output_data = {
                'policies': policies_data,
                'exported_at': datetime.now().isoformat(),
                'total_policies': len(policies_data)
            }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Saved {len(policies_data)} policies to file")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving policies to file: {str(e)}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the policy database"""
        category_counts = {}
        for policy in self.policies.values():
            category = policy.category
            category_counts[category] = category_counts.get(category, 0) + 1
        
        return {
            'total_policies': len(self.policies),
            'total_categories': len(self.category_index),
            'total_keywords': len(self.keyword_index),
            'category_distribution': category_counts,
            'average_keywords_per_policy': sum(len(p.keywords) for p in self.policies.values()) / len(self.policies) if self.policies else 0
        }
    
    # Private helper methods
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract relevant keywords from text"""
        # Remove common stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'i', 'you', 'he', 'she', 'it', 'we', 'they', 'is', 'are', 'was', 'were', 'be', 'been',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'can',
            'my', 'your', 'his', 'her', 'its', 'our', 'their', 'this', 'that', 'these', 'those'
        }
        
        # Split text and filter
        words = re.findall(r'\b\w+\b', text.lower())
        keywords = [word for word in words if len(word) > 2 and word not in stop_words]
        
        return keywords
    
    def _search_by_category(self, category: str, query_keywords: List[str]) -> List[str]:
        """Search policies by category"""
        if not category:
            return []
        
        category_lower = category.lower()
        
        # Direct category match
        if category_lower in self.category_index:
            return self.category_index[category_lower]
        
        # Try to map common category names
        category_mapping = {
            'refund': 'refunds',
            'return': 'refunds', 
            'delivery': 'shipping',
            'billing': 'billing',
            'payment': 'billing',
            'account': 'account',
            'technical': 'support',
            'privacy': 'privacy'
        }
        
        mapped_category = category_mapping.get(category_lower)
        if mapped_category and mapped_category in self.category_index:
            return self.category_index[mapped_category]
        
        return []
    
    def _search_by_keywords(self, keywords: List[str]) -> List[str]:
        """Search policies by keywords"""
        policy_ids = []
        
        for keyword in keywords:
            if keyword in self.keyword_index:
                policy_ids.extend(self.keyword_index[keyword])
        
        return policy_ids
    
    def _policy_applies_to_customer(self, policy: PolicySection, customer_type: str) -> bool:
        """Check if policy applies to customer type"""
        return (not policy.applies_to or 
                'all_customers' in policy.applies_to or 
                customer_type in policy.applies_to)
    
    def _calculate_relevance(self, policy: PolicySection, query: str, 
                           query_keywords: List[str], intent_category: str = None) -> Tuple[float, List[str], str]:
        """Calculate relevance score for policy"""
        score = 0.0
        matched_keywords = []
        match_type = "partial"
        
        # Category match bonus
        if intent_category and policy.category == intent_category.lower():
            score += 0.3
            match_type = "category"
        
        # Keyword matching
        policy_keywords_lower = [kw.lower() for kw in policy.keywords]
        
        for keyword in query_keywords:
            if keyword in policy_keywords_lower:
                score += 0.2
                matched_keywords.append(keyword)
                match_type = "exact" if score > 0.5 else "partial"
        
        # Content matching (basic text search)
        content_lower = policy.content.lower()
        for keyword in query_keywords:
            if keyword in content_lower:
                score += 0.1
                if keyword not in matched_keywords:
                    matched_keywords.append(keyword)
        
        # Priority bonus (higher priority = lower number)
        score += (5 - policy.priority) * 0.05
        
        # Title matching
        title_lower = policy.title.lower()
        for keyword in query_keywords:
            if keyword in title_lower:
                score += 0.15
                if keyword not in matched_keywords:
                    matched_keywords.append(keyword)
        
        return min(score, 1.0), matched_keywords, match_type
    
    def _summarize_content(self, content: str) -> str:
        """Create a brief summary of policy content"""
        sentences = content.split('. ')
        if len(sentences) <= 2:
            return content
        
        # Return first two sentences
        return '. '.join(sentences[:2]) + '.'
    
    def _extract_key_points(self, content: str) -> List[str]:
        """Extract key points from policy content"""
        key_points = []
        
        # Look for numbered items or bullet points
        numbered_items = re.findall(r'(\d+\.?\s+[^.]+\.)', content)
        if numbered_items:
            key_points.extend(numbered_items)
        
        # Look for time references
        time_refs = re.findall(r'(\d+\s+(?:days?|hours?|minutes?|business\s+days?))', content)
        if time_refs:
            key_points.extend([f"Timeline: {ref}" for ref in time_refs])
        
        # Look for important phrases
        important_phrases = re.findall(r'(must\s+[^.]+\.)', content, re.IGNORECASE)
        key_points.extend(important_phrases[:3])  # Limit to 3
        
        return key_points[:5]  # Limit to 5 key points
    
    def _generate_agent_guidance(self, policy: PolicySection, match: PolicyMatch) -> Dict[str, Any]:
        """Generate guidance for customer service agent"""
        guidance = {
            'recommended_action': 'Review policy details with customer',
            'key_message': 'Follow standard procedure',
            'escalation_needed': False,
            'follow_up_required': False
        }
        
        # Category-specific guidance
        if policy.category == 'refunds':
            guidance['recommended_action'] = 'Verify purchase date and product condition'
            guidance['key_message'] = 'Explain refund timeline and requirements'
            guidance['follow_up_required'] = True
            
        elif policy.category == 'shipping':
            guidance['recommended_action'] = 'Check tracking information and delivery status'
            guidance['key_message'] = 'Provide clear delivery expectations'
            
        elif policy.category == 'billing':
            guidance['recommended_action'] = 'Review account charges and payment history'
            guidance['key_message'] = 'Explain billing cycle and payment methods'
            guidance['escalation_needed'] = 'dispute' in ' '.join(match.matched_keywords)
            
        elif policy.category == 'account':
            guidance['recommended_action'] = 'Verify customer identity before account changes'
            guidance['key_message'] = 'Prioritize account security'
            
        return guidance
    
    def _assess_customer_impact(self, policy: PolicySection) -> Dict[str, Any]:
        """Assess potential customer impact of policy"""
        impact = {
            'satisfaction_risk': 'low',
            'complexity': 'simple',
            'time_requirement': 'minimal'
        }
        
        # Assess based on category and content
        if policy.category == 'refunds':
            impact['satisfaction_risk'] = 'medium'
            impact['time_requirement'] = 'moderate'
            
        elif policy.category == 'billing':
            impact['satisfaction_risk'] = 'high'
            impact['complexity'] = 'complex'
            
        elif policy.category == 'privacy':
            impact['complexity'] = 'complex'
            
        return impact


def main():
    """Demonstration of the policy matcher"""
    print("Policy Reference Matching System Demo")
    print("=" * 50)
    
    # Initialize matcher
    matcher = PolicyMatcher()
    
    # Test queries
    test_queries = [
        ("I want to return this broken laptop", "refund_request"),
        ("Where is my package? It was supposed to arrive yesterday", "delivery_issue"),
        ("I can't log into my account", "account_issue"),
        ("Why was I charged twice for the same order?", "billing_question"),
        ("Can I cancel my subscription?", "cancellation_request"),
        ("How long do refunds take?", "general_inquiry")
    ]
    
    print("Testing policy matching:")
    print("-" * 30)
    
    for query, intent in test_queries:
        print(f"\nQuery: '{query}'")
        print(f"Intent: {intent}")
        
        matches = matcher.find_relevant_policies(query, intent)
        
        if matches:
            print(f"Found {len(matches)} relevant policies:")
            for i, match in enumerate(matches, 1):
                summary = matcher.get_policy_summary(match)
                print(f"  {i}. {summary['title']} (Score: {summary['relevance_score']})")
                print(f"     Category: {summary['category']}")
                print(f"     Key Points: {summary['key_points'][:2]}")
        else:
            print("  No relevant policies found")
        
        print("-" * 30)
    
    # Show statistics
    stats = matcher.get_statistics()
    print("\nPolicy Database Statistics:")
    print(f"Total Policies: {stats['total_policies']}")
    print(f"Categories: {stats['total_categories']}")
    print(f"Category Distribution: {stats['category_distribution']}")


if __name__ == "__main__":
    main()
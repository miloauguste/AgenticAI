"""
Simplified Intent classification tool for customer support tickets
"""
import re
from typing import Dict, Any, List, Optional
import logging

class IntentClassifier:
    """Classify customer support ticket intents and extract relevant entities"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Intent patterns with keywords
        self.intent_patterns = {
            'refund': {
                'keywords': ['refund', 'money back', 'return', 'reimburse', 'compensation'],
                'patterns': [r'\b(refund|money\s+back|return.*money|reimburse)\b'],
                'department': 'billing'
            },
            'delivery': {
                'keywords': ['shipping', 'delivery', 'tracking', 'delayed', 'package'],
                'patterns': [r'\b(shipping|delivery|package|tracking|shipment)\b'],
                'department': 'logistics'
            },
            'product_issue': {
                'keywords': ['defective', 'broken', 'damaged', 'not working', 'quality'],
                'patterns': [r'\b(defective|broken|damaged|not\s+working|malfunctioning)\b'],
                'department': 'quality_assurance'
            },
            'account': {
                'keywords': ['account', 'login', 'password', 'profile', 'access'],
                'patterns': [r'\b(account|login|password|profile|settings)\b'],
                'department': 'technical_support'
            },
            'billing': {
                'keywords': ['charge', 'payment', 'invoice', 'billing', 'transaction'],
                'patterns': [r'\b(charge|payment|billing|invoice|transaction)\b'],
                'department': 'billing'
            },
            'complaint': {
                'keywords': ['complaint', 'unsatisfied', 'disappointed', 'poor service'],
                'patterns': [r'\b(complaint|complain|unsatisfied|disappointed)\b'],
                'department': 'customer_service'
            },
            'information': {
                'keywords': ['question', 'how to', 'information', 'help', 'support'],
                'patterns': [r'\b(question|how\s+to|information|help|support)\b'],
                'department': 'customer_service'
            }
        }
        
        # Entity extraction patterns
        self.entity_patterns = {
            'order_id': [r'\border\s*#?\s*([A-Z0-9]{6,})\b'],
            'tracking_number': [r'\btracking\s*#?\s*([A-Z0-9]{10,})\b'],
            'amount': [r'\$(\d+(?:\.\d{2})?)\b'],
            'email': [r'\b([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})\b'],
            'phone': [r'\b(\+?\d{1,3}[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4})\b']
        }
    
    def classify_intent(self, text: str, confidence_threshold: float = 0.3) -> Dict[str, Any]:
        """
        Classify the intent of a customer support text
        """
        try:
            text_lower = text.lower()
            intent_scores = {}
            
            # Calculate scores for each intent
            for intent, config in self.intent_patterns.items():
                score = self._calculate_intent_score(text_lower, config)
                intent_scores[intent] = score
            
            # Find the best matching intent
            best_intent = max(intent_scores, key=intent_scores.get)
            best_score = intent_scores[best_intent]
            
            # Extract entities
            entities = self._extract_entities(text)
            
            # Determine confidence and final intent
            if best_score >= confidence_threshold:
                final_intent = best_intent
                confidence = best_score
            else:
                final_intent = 'general'
                
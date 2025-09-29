#!/usr/bin/env python3
"""
Intent Classifier for Text Analysis

This module provides a simple yet effective intent classification system
that can be used for chatbots, customer service, or any text understanding task.
"""

import re
import json
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import Counter


@dataclass
class Intent:
    """Represents an intent with its name, patterns, and confidence threshold."""
    name: str
    patterns: List[str]
    keywords: List[str]
    confidence_threshold: float = 0.3


class IntentClassifier:
    """
    A rule-based intent classifier that uses pattern matching and keyword scoring.
    
    This classifier is designed to be:
    - Easy to understand and modify
    - Fast for real-time applications
    - Transparent in its decision-making
    """
    
    def __init__(self):
        self.intents = []
        self.default_intent = "unknown"
        
    def add_intent(self, name: str, patterns: List[str], keywords: List[str], 
                   confidence_threshold: float = 0.3) -> None:
        """
        Add a new intent to the classifier.
        
        Args:
            name: The intent name (e.g., "greeting", "booking", "complaint")
            patterns: List of regex patterns that match this intent
            keywords: List of important keywords for this intent
            confidence_threshold: Minimum confidence score to classify as this intent
        """
        intent = Intent(name, patterns, keywords, confidence_threshold)
        self.intents.append(intent)
        
    def preprocess_text(self, text: str) -> str:
        """Clean and normalize input text."""
        # Convert to lowercase
        text = text.lower().strip()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\?\!\.]', '', text)
        
        return text
    
    def calculate_keyword_score(self, text: str, keywords: List[str]) -> float:
        """Calculate how well the text matches the intent keywords."""
        if not keywords:
            return 0.0
            
        words = text.split()
        word_count = len(words)
        
        if word_count == 0:
            return 0.0
        
        # Count keyword matches
        matches = sum(1 for word in words if word in keywords)
        
        # Calculate score as percentage of matched keywords
        score = matches / len(keywords)
        
        # Boost score if text is short and has high keyword density
        if word_count <= 10 and matches > 0:
            score *= 1.2
            
        return min(score, 1.0)  # Cap at 1.0
    
    def check_patterns(self, text: str, patterns: List[str]) -> float:
        """Check if text matches any of the given patterns."""
        if not patterns:
            return 0.0
            
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return 1.0
                
        return 0.0
    
    def classify(self, text: str) -> Tuple[str, float, Dict]:
        """
        Classify the intent of the given text.
        
        Args:
            text: The input text to classify
            
        Returns:
            Tuple of (intent_name, confidence_score, debug_info)
        """
        if not text.strip():
            return self.default_intent, 0.0, {"error": "Empty input"}
        
        processed_text = self.preprocess_text(text)
        
        best_intent = self.default_intent
        best_score = 0.0
        debug_info = {"processed_text": processed_text, "scores": {}}
        
        for intent in self.intents:
            # Calculate pattern match score
            pattern_score = self.check_patterns(processed_text, intent.patterns)
            
            # Calculate keyword match score
            keyword_score = self.calculate_keyword_score(processed_text, intent.keywords)
            
            # Combine scores (pattern matching is weighted higher)
            combined_score = (pattern_score * 0.7) + (keyword_score * 0.3)
            
            debug_info["scores"][intent.name] = {
                "pattern_score": pattern_score,
                "keyword_score": keyword_score,
                "combined_score": combined_score,
                "threshold": intent.confidence_threshold
            }
            
            # Update best match if this intent scores higher and meets threshold
            if (combined_score > best_score and 
                combined_score >= intent.confidence_threshold):
                best_intent = intent.name
                best_score = combined_score
        
        debug_info["best_match"] = {"intent": best_intent, "score": best_score}
        return best_intent, best_score, debug_info
    
    def batch_classify(self, texts: List[str]) -> List[Tuple[str, float, Dict]]:
        """Classify multiple texts at once."""
        return [self.classify(text) for text in texts]
    
    def get_intent_stats(self, texts: List[str]) -> Dict:
        """Get statistics about intent distribution in a list of texts."""
        results = self.batch_classify(texts)
        intent_counts = Counter(result[0] for result in results)
        
        total = len(texts)
        stats = {
            "total_texts": total,
            "intent_distribution": dict(intent_counts),
            "intent_percentages": {
                intent: (count / total) * 100 
                for intent, count in intent_counts.items()
            }
        }
        
        return stats
    
    def save_model(self, filepath: str) -> None:
        """Save the classifier configuration to a JSON file."""
        model_data = {
            "intents": [
                {
                    "name": intent.name,
                    "patterns": intent.patterns,
                    "keywords": intent.keywords,
                    "confidence_threshold": intent.confidence_threshold
                }
                for intent in self.intents
            ],
            "default_intent": self.default_intent
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(model_data, f, indent=2, ensure_ascii=False)
    
    def load_model(self, filepath: str) -> None:
        """Load classifier configuration from a JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            model_data = json.load(f)
        
        self.intents = []
        for intent_data in model_data["intents"]:
            self.add_intent(
                intent_data["name"],
                intent_data["patterns"],
                intent_data["keywords"],
                intent_data.get("confidence_threshold", 0.3)
            )
        
        self.default_intent = model_data.get("default_intent", "unknown")


def create_sample_classifier() -> IntentClassifier:
    """Create a comprehensive customer service classifier with all relevant intents."""
    classifier = IntentClassifier()
    
    # Refund Request
    classifier.add_intent(
        name="refund_request",
        patterns=[
            r"\b(refund|money back|return|reimburse|charge back|chargeback)\b",
            r"\b(want my money|get my money|pay me back)\b"
        ],
        keywords=["refund", "money", "back", "return", "reimburse", "charge", "payment", "cancel"],
        confidence_threshold=0.4
    )
    
    # Delivery Issues
    classifier.add_intent(
        name="delivery_issue",
        patterns=[
            r"\b(delivery|shipping|package|order|hasn't arrived|not received|late|delayed)\b",
            r"\b(where is my|track my|haven't got|still waiting)\b"
        ],
        keywords=["delivery", "shipping", "package", "order", "arrived", "received", "late", "track", "waiting"],
        confidence_threshold=0.3
    )
    
    # Account Issues
    classifier.add_intent(
        name="account_issue",
        patterns=[
            r"\b(account|login|password|reset|access|locked|blocked|sign in)\b",
            r"\b(can't access|unable to login|forgot password)\b"
        ],
        keywords=["account", "login", "password", "reset", "access", "locked", "blocked", "signin", "forgot"],
        confidence_threshold=0.4
    )
    
    # Billing Questions
    classifier.add_intent(
        name="billing_question",
        patterns=[
            r"\b(bill|billing|invoice|charge|payment|cost|price|fee)\b",
            r"\b(how much|what does it cost|charged me|unexpected charge)\b"
        ],
        keywords=["bill", "billing", "invoice", "charge", "payment", "cost", "price", "fee", "amount"],
        confidence_threshold=0.3
    )
    
    # Technical Issues
    classifier.add_intent(
        name="technical_issue",
        patterns=[
            r"\b(bug|error|crash|freeze|not working|broken|glitch|malfunction)\b",
            r"\b(technical|website|app|software|system)\b"
        ],
        keywords=["bug", "error", "crash", "freeze", "working", "broken", "technical", "website", "app"],
        confidence_threshold=0.3
    )
    
    # Product Information
    classifier.add_intent(
        name="product_inquiry",
        patterns=[
            r"\b(product|item|details|specifications|features|size|color|availability)\b",
            r"\b(tell me about|information about|learn more)\b"
        ],
        keywords=["product", "item", "details", "specifications", "features", "size", "color", "available"],
        confidence_threshold=0.3
    )
    
    # Complaint
    classifier.add_intent(
        name="complaint",
        patterns=[
            r"\b(terrible|awful|horrible|disgusting|worst|pathetic|unacceptable)\b",
            r"\b(disappointed|frustrated|angry|upset|complain|complaint)\b"
        ],
        keywords=["terrible", "awful", "horrible", "worst", "disappointed", "frustrated", "angry", "complaint"],
        confidence_threshold=0.3
    )
    
    # Compliment/Praise
    classifier.add_intent(
        name="compliment",
        patterns=[
            r"\b(great|excellent|amazing|wonderful|fantastic|awesome|perfect|love)\b",
            r"\b(thank you|appreciate|satisfied|happy|pleased)\b"
        ],
        keywords=["great", "excellent", "amazing", "wonderful", "fantastic", "awesome", "perfect", "love", "thank"],
        confidence_threshold=0.4
    )
    
    # General Inquiry
    classifier.add_intent(
        name="general_inquiry",
        patterns=[
            r"\b(question|help|information|assistance|support)\b",
            r"\b(can you|could you|would you|please|how do|how can)\b"
        ],
        keywords=["question", "help", "information", "assistance", "support", "can", "could", "would", "please"],
        confidence_threshold=0.2
    )
    
    # Order Status
    classifier.add_intent(
        name="order_status",
        patterns=[
            r"\b(order|status|tracking|shipped|processing|confirmed)\b",
            r"\b(order number|track order|where is|when will)\b"
        ],
        keywords=["order", "status", "tracking", "shipped", "processing", "confirmed", "number", "track"],
        confidence_threshold=0.4
    )
    
    # Cancellation Request
    classifier.add_intent(
        name="cancellation_request",
        patterns=[
            r"\b(cancel|cancellation|stop|terminate|end|discontinue)\b",
            r"\b(don't want|no longer need|changed my mind)\b"
        ],
        keywords=["cancel", "cancellation", "stop", "terminate", "end", "discontinue", "want", "need", "mind"],
        confidence_threshold=0.4
    )
    
    # Password Reset
    classifier.add_intent(
        name="password_reset",
        patterns=[
            r"\b(password|reset|forgot|change password|new password)\b",
            r"\b(can't remember|lost password|update password)\b"
        ],
        keywords=["password", "reset", "forgot", "change", "remember", "lost", "update", "new"],
        confidence_threshold=0.5
    )
    
    return classifier


def main():
    """Demonstration of the intent classifier."""
    print("Intent Classifier Demo")
    print("=" * 50)
    
    # Create sample classifier
    classifier = create_sample_classifier()
    
    # Test texts
    test_texts = [
        "Hello there!",
        "Hi, how are you doing?",
        "What time do you open?",
        "I'd like to book a table for tonight",
        "This service is terrible",
        "You guys are amazing!",
        "Can you help me with my order?",
        "The food was cold and awful",
        "I need to make a reservation",
        "Random text that doesn't fit anywhere"
    ]
    
    print("Testing individual classifications:")
    print("-" * 50)
    
    for text in test_texts:
        intent, confidence, debug = classifier.classify(text)
        print(f"Text: '{text}'")
        print(f"Intent: {intent} (confidence: {confidence:.3f})")
        print(f"Top scores: {debug['scores']}")
        print()
    
    # Show statistics
    print("Intent Distribution:")
    print("-" * 50)
    stats = classifier.get_intent_stats(test_texts)
    for intent, percentage in stats["intent_percentages"].items():
        print(f"{intent}: {percentage:.1f}% ({stats['intent_distribution'][intent]} texts)")
    
    # Save model
    classifier.save_model("sample_intent_model.json")
    print("\nModel saved to 'sample_intent_model.json'")


if __name__ == "__main__":
    main()
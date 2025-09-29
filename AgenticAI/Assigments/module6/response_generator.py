#!/usr/bin/env python3
"""
Response Generator for Intent-Based Conversations

This module generates contextual responses based on classified intents,
demonstrating key concepts in conversational AI and natural language generation.
"""

import json
import random
import re
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import time


@dataclass
class ResponseTemplate:
    """Represents a response template with variables and conditions."""
    text: str
    variables: Dict[str, Any] = field(default_factory=dict)
    conditions: Dict[str, Any] = field(default_factory=dict)
    follow_up_questions: List[str] = field(default_factory=list)
    actions: List[str] = field(default_factory=list)


@dataclass
class ConversationContext:
    """Maintains conversation state and user information."""
    user_name: Optional[str] = None
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    conversation_history: List[Dict] = field(default_factory=list)
    current_topic: Optional[str] = None
    session_start: datetime = field(default_factory=datetime.now)
    interaction_count: int = 0
    last_intent: Optional[str] = None
    pending_actions: List[str] = field(default_factory=list)


class ResponseGenerator:
    """
    Generates contextual responses based on intents and conversation state.
    
    This generator demonstrates:
    - Template-based response generation
    - Context awareness and personalization
    - Dynamic content insertion
    - Conversation flow management
    """
    
    def __init__(self):
        self.response_templates: Dict[str, List[ResponseTemplate]] = {}
        self.context = ConversationContext()
        self.fallback_responses = [
            "I'm not sure I understand. Could you please rephrase that?",
            "That's interesting. Can you tell me more about what you need?",
            "I'd like to help you with that. Could you provide more details?",
            "I'm still learning about that topic. What specifically are you looking for?"
        ]
        
    def add_response_template(self, intent: str, template: ResponseTemplate) -> None:
        """Add a response template for a specific intent."""
        if intent not in self.response_templates:
            self.response_templates[intent] = []
        self.response_templates[intent].append(template)
    
    def add_simple_responses(self, intent: str, responses: List[str]) -> None:
        """Add simple text responses for an intent (convenience method)."""
        for response_text in responses:
            template = ResponseTemplate(text=response_text)
            self.add_response_template(intent, template)
    
    def extract_entities(self, text: str) -> Dict[str, Any]:
        """Extract entities from user input (basic implementation)."""
        entities = {}
        
        # Extract names (capitalized words)
        names = re.findall(r'\b[A-Z][a-z]+\b', text)
        if names:
            entities['names'] = names
        
        # Extract numbers
        numbers = re.findall(r'\b\d+\b', text)
        if numbers:
            entities['numbers'] = [int(n) for n in numbers]
        
        # Extract time expressions
        time_patterns = [
            r'\b\d{1,2}:\d{2}\b',  # 10:30
            r'\b\d{1,2}\s*(am|pm)\b',  # 2pm
            r'\b(morning|afternoon|evening|night)\b',
            r'\b(today|tomorrow|yesterday)\b',
            r'\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b'
        ]
        
        times = []
        for pattern in time_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            times.extend(matches)
        
        if times:
            entities['time_references'] = times
        
        # Extract common booking-related terms
        booking_terms = re.findall(
            r'\b(table|room|appointment|reservation|booking)\b', 
            text, re.IGNORECASE
        )
        if booking_terms:
            entities['booking_type'] = booking_terms[0].lower()
        
        return entities
    
    def update_context(self, user_input: str, intent: str, confidence: float, 
                      entities: Dict[str, Any]) -> None:
        """Update conversation context with new information."""
        self.context.interaction_count += 1
        self.context.last_intent = intent
        
        # Add to conversation history
        self.context.conversation_history.append({
            'timestamp': datetime.now(),
            'user_input': user_input,
            'intent': intent,
            'confidence': confidence,
            'entities': entities
        })
        
        # Update user name if found
        if 'names' in entities and not self.context.user_name:
            # Simple heuristic: if user says "I'm [Name]" or "My name is [Name]"
            if re.search(r"(i'm|i am|my name is|call me)", user_input, re.IGNORECASE):
                self.context.user_name = entities['names'][0]
        
        # Update current topic based on intent
        if intent != 'unknown':
            self.context.current_topic = intent
        
        # Keep conversation history manageable
        if len(self.context.conversation_history) > 20:
            self.context.conversation_history = self.context.conversation_history[-15:]
    
    def get_time_appropriate_greeting(self) -> str:
        """Generate time-appropriate greeting."""
        current_hour = datetime.now().hour
        
        if 5 <= current_hour < 12:
            return "Good morning"
        elif 12 <= current_hour < 17:
            return "Good afternoon"
        elif 17 <= current_hour < 22:
            return "Good evening"
        else:
            return "Hello"
    
    def personalize_response(self, response: str) -> str:
        """Add personalization to responses."""
        # Add user name if available
        if self.context.user_name and "{name}" not in response:
            if random.random() < 0.3:  # 30% chance to add name
                response = f"{response.rstrip('.')} {self.context.user_name}!"
        
        # Replace placeholders
        replacements = {
            "{name}": self.context.user_name or "there",
            "{greeting}": self.get_time_appropriate_greeting(),
            "{interaction_count}": str(self.context.interaction_count)
        }
        
        for placeholder, value in replacements.items():
            response = response.replace(placeholder, value)
        
        return response
    
    def select_response_template(self, intent: str, entities: Dict[str, Any]) -> ResponseTemplate:
        """Select the most appropriate response template."""
        if intent not in self.response_templates:
            # Return fallback response
            return ResponseTemplate(text=random.choice(self.fallback_responses))
        
        templates = self.response_templates[intent]
        
        # If only one template, return it
        if len(templates) == 1:
            return templates[0]
        
        # Score templates based on context and conditions
        scored_templates = []
        
        for template in templates:
            score = 1.0
            
            # Check conditions
            if template.conditions:
                for condition, expected in template.conditions.items():
                    if condition == "has_name" and expected:
                        score += 0.5 if self.context.user_name else -1.0
                    elif condition == "repeat_visitor" and expected:
                        score += 0.3 if self.context.interaction_count > 1 else -0.3
                    elif condition == "entities_required":
                        if any(entity in entities for entity in expected):
                            score += 0.4
                        else:
                            score -= 0.5
            
            # Avoid repeating recent responses
            recent_responses = [
                entry.get('response', '') 
                for entry in self.context.conversation_history[-3:]
            ]
            
            if any(template.text in recent for recent in recent_responses):
                score -= 0.8
            
            scored_templates.append((template, max(score, 0.1)))
        
        # Select template using weighted random choice
        if not scored_templates:
            return ResponseTemplate(text=random.choice(self.fallback_responses))
        
        # Simple weighted selection
        templates, weights = zip(*scored_templates)
        total_weight = sum(weights)
        r = random.uniform(0, total_weight)
        
        cumulative = 0
        for template, weight in scored_templates:
            cumulative += weight
            if r <= cumulative:
                return template
        
        return scored_templates[0][0]  # Fallback to first template
    
    def apply_template_variables(self, template: ResponseTemplate, 
                                entities: Dict[str, Any]) -> str:
        """Apply variables and entity values to template text."""
        response = template.text
        
        # Apply template variables
        for var_name, var_value in template.variables.items():
            placeholder = f"{{{var_name}}}"
            response = response.replace(placeholder, str(var_value))
        
        # Apply entity values
        entity_replacements = {
            "{booking_type}": entities.get('booking_type', 'reservation'),
            "{time}": ', '.join(entities.get('time_references', [])) or 'your preferred time',
            "{number}": str(entities.get('numbers', [1])[0]) if entities.get('numbers') else '1'
        }
        
        for placeholder, value in entity_replacements.items():
            response = response.replace(placeholder, value)
        
        return response
    
    def add_follow_up(self, response: str, template: ResponseTemplate, 
                     entities: Dict[str, Any]) -> str:
        """Add appropriate follow-up questions."""
        if not template.follow_up_questions:
            return response
        
        # Select follow-up based on context
        follow_up = random.choice(template.follow_up_questions)
        
        # Personalize follow-up
        follow_up = self.personalize_response(follow_up)
        
        return f"{response} {follow_up}"
    
    def generate_response(self, user_input: str, intent: str, 
                         confidence: float, debug_info: Dict = None) -> Dict[str, Any]:
        """
        Generate a contextual response based on intent and conversation state.
        
        Returns a dictionary containing:
        - response: The generated response text
        - actions: Any actions that should be taken
        - confidence: Response generation confidence
        - context_used: Information about what context was used
        """
        # Extract entities from user input
        entities = self.extract_entities(user_input)
        
        # Update conversation context
        self.update_context(user_input, intent, confidence, entities)
        
        # Select appropriate response template
        template = self.select_response_template(intent, entities)
        
        # Generate base response
        response = self.apply_template_variables(template, entities)
        
        # Personalize the response
        response = self.personalize_response(response)
        
        # Add follow-up questions
        response = self.add_follow_up(response, template, entities)
        
        # Prepare response metadata
        response_data = {
            'response': response,
            'intent': intent,
            'confidence': confidence,
            'entities': entities,
            'actions': template.actions.copy(),
            'template_used': template.text,
            'context_used': {
                'user_name': self.context.user_name,
                'interaction_count': self.context.interaction_count,
                'current_topic': self.context.current_topic,
                'session_duration': (datetime.now() - self.context.session_start).seconds
            },
            'timestamp': datetime.now().isoformat()
        }
        
        # Add to conversation history
        self.context.conversation_history[-1]['response'] = response
        self.context.conversation_history[-1]['response_data'] = response_data
        
        return response_data
    
    def reset_context(self) -> None:
        """Reset conversation context for new session."""
        self.context = ConversationContext()
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get summary of current conversation."""
        if not self.context.conversation_history:
            return {"status": "no_conversation"}
        
        intents = [entry['intent'] for entry in self.context.conversation_history]
        intent_counts = {}
        for intent in intents:
            intent_counts[intent] = intent_counts.get(intent, 0) + 1
        
        return {
            "session_duration": (datetime.now() - self.context.session_start).seconds,
            "total_interactions": self.context.interaction_count,
            "user_name": self.context.user_name,
            "current_topic": self.context.current_topic,
            "intent_distribution": intent_counts,
            "last_intent": self.context.last_intent,
            "entities_mentioned": list(set(
                key for entry in self.context.conversation_history 
                for key in entry.get('entities', {}).keys()
            ))
        }
    
    def export_conversation(self, filepath: str) -> None:
        """Export conversation history to JSON file."""
        conversation_data = {
            "context": {
                "user_name": self.context.user_name,
                "session_start": self.context.session_start.isoformat(),
                "interaction_count": self.context.interaction_count,
                "current_topic": self.context.current_topic
            },
            "history": [
                {
                    "timestamp": entry["timestamp"].isoformat(),
                    "user_input": entry["user_input"],
                    "intent": entry["intent"],
                    "confidence": entry["confidence"],
                    "entities": entry["entities"],
                    "response": entry.get("response", "")
                }
                for entry in self.context.conversation_history
            ],
            "summary": self.get_conversation_summary()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(conversation_data, f, indent=2, ensure_ascii=False, default=str)


def create_sample_response_generator() -> ResponseGenerator:
    """Create a comprehensive customer service response generator."""
    generator = ResponseGenerator()
    
    # Refund Request responses
    refund_templates = [
        ResponseTemplate(
            text="I understand you're looking for a refund. I'll be happy to help you with that process.",
            follow_up_questions=[
                "Could you please provide your order number?",
                "What was the reason for the refund request?"
            ],
            actions=["initiate_refund_process", "gather_order_details"]
        ),
        ResponseTemplate(
            text="I sincerely apologize that you need to request a refund. Let me check our refund policy for your specific case.",
            follow_up_questions=["When was your purchase made?"],
            actions=["check_refund_eligibility", "create_refund_ticket"]
        )
    ]
    
    for template in refund_templates:
        generator.add_response_template("refund_request", template)
    
    # Delivery Issue responses
    delivery_templates = [
        ResponseTemplate(
            text="I'm sorry to hear about the delivery issue. Let me help track down your package right away.",
            follow_up_questions=[
                "Could you provide your order number?",
                "What was the expected delivery date?"
            ],
            actions=["track_package", "check_delivery_status"]
        ),
        ResponseTemplate(
            text="I understand how frustrating delivery delays can be. Let me investigate this for you immediately.",
            follow_up_questions=["What is your delivery address?"],
            actions=["escalate_delivery_issue", "provide_compensation"]
        )
    ]
    
    for template in delivery_templates:
        generator.add_response_template("delivery_issue", template)
    
    # Account Issue responses
    account_templates = [
        ResponseTemplate(
            text="I'll help you resolve this account issue right away. Account security is our top priority.",
            follow_up_questions=[
                "What specific issue are you experiencing?",
                "Can you verify your email address associated with the account?"
            ],
            actions=["verify_account", "initiate_account_recovery"]
        ),
        ResponseTemplate(
            text="Let me assist you with your account access. I'll guide you through the recovery process.",
            follow_up_questions=["When did you last successfully access your account?"],
            actions=["send_password_reset", "verify_identity"]
        )
    ]
    
    for template in account_templates:
        generator.add_response_template("account_issue", template)
    
    # Billing Question responses
    billing_templates = [
        ResponseTemplate(
            text="I'd be happy to help clarify any billing questions you have. Let me review your account.",
            follow_up_questions=[
                "Which charge are you asking about?",
                "What is your account number or email address?"
            ],
            actions=["review_billing", "generate_invoice_explanation"]
        ),
        ResponseTemplate(
            text="I understand billing questions can be confusing. Let me break down the charges for you.",
            follow_up_questions=["Are you seeing an unexpected charge?"],
            actions=["investigate_charges", "provide_billing_breakdown"]
        )
    ]
    
    for template in billing_templates:
        generator.add_response_template("billing_question", template)
    
    # Technical Issue responses
    technical_templates = [
        ResponseTemplate(
            text="I'm sorry you're experiencing technical difficulties. Let me help troubleshoot this issue.",
            follow_up_questions=[
                "What device and browser are you using?",
                "When did this issue first occur?"
            ],
            actions=["initiate_troubleshooting", "gather_technical_details"]
        ),
        ResponseTemplate(
            text="Technical issues can be very frustrating. I'll work with you to resolve this step by step.",
            follow_up_questions=["Can you describe exactly what happens when the issue occurs?"],
            actions=["escalate_to_technical_team", "provide_workaround"]
        )
    ]
    
    for template in technical_templates:
        generator.add_response_template("technical_issue", template)
    
    # Product Inquiry responses
    product_templates = [
        ResponseTemplate(
            text="I'd be delighted to help you learn more about our products!",
            follow_up_questions=[
                "Which specific product are you interested in?",
                "What particular features are you looking for?"
            ],
            actions=["provide_product_info", "search_product_catalog"]
        ),
        ResponseTemplate(
            text="Great question about our products! Let me get you the most current information.",
            follow_up_questions=["Are you comparing different options?"],
            actions=["generate_product_comparison", "check_availability"]
        )
    ]
    
    for template in product_templates:
        generator.add_response_template("product_inquiry", template)
    
    # Complaint responses (enhanced)
    complaint_templates = [
        ResponseTemplate(
            text="I sincerely apologize for this experience. This is not the level of service we strive for, and I'm going to make this right.",
            follow_up_questions=[
                "Can you tell me exactly what happened?",
                "What would be the best way to resolve this for you?"
            ],
            actions=["escalate_complaint", "create_priority_ticket", "offer_compensation"]
        ),
        ResponseTemplate(
            text="I'm truly sorry that we've fallen short of your expectations. Your feedback is invaluable to us.",
            follow_up_questions=["How can I personally ensure this doesn't happen again?"],
            actions=["document_feedback", "schedule_follow_up", "manager_review"]
        )
    ]
    
    for template in complaint_templates:
        generator.add_response_template("complaint", template)
    
    # Compliment responses (enhanced)
    compliment_templates = [
        ResponseTemplate(
            text="Thank you so much for taking the time to share this positive feedback! It truly means a lot to our team.",
            follow_up_questions=["Is there anything else we can help you with today?"]
        ),
        ResponseTemplate(
            text="We're absolutely thrilled to hear about your positive experience! I'll make sure to share your kind words with the team.",
            follow_up_questions=["Would you mind leaving a review to help other customers?"]
        )
    ]
    
    for template in compliment_templates:
        generator.add_response_template("compliment", template)
    
    # General Inquiry responses
    general_templates = [
        ResponseTemplate(
            text="I'm here to help! What can I assist you with today?",
            follow_up_questions=["Could you provide more details about what you're looking for?"]
        ),
        ResponseTemplate(
            text="Thanks for reaching out! I'd be happy to help you find the information you need.",
            follow_up_questions=["What specific area can I help you with?"]
        )
    ]
    
    for template in general_templates:
        generator.add_response_template("general_inquiry", template)
    
    # Order Status responses
    order_templates = [
        ResponseTemplate(
            text="I'll check your order status right away! Let me pull up the latest information for you.",
            follow_up_questions=["Could you provide your order number?"],
            actions=["track_order", "provide_status_update"]
        ),
        ResponseTemplate(
            text="I'd be happy to give you an update on your order. Let me look that up for you now.",
            follow_up_questions=["What email address did you use for the order?"],
            actions=["lookup_order_by_email", "send_tracking_info"]
        )
    ]
    
    for template in order_templates:
        generator.add_response_template("order_status", template)
    
    # Cancellation Request responses
    cancellation_templates = [
        ResponseTemplate(
            text="I understand you'd like to cancel. I'm sorry to see you go, but I'll help you with the cancellation process.",
            follow_up_questions=[
                "Could you tell me what order or service you'd like to cancel?",
                "Is there anything we could do to address your concerns instead?"
            ],
            actions=["initiate_cancellation", "offer_alternatives"]
        ),
        ResponseTemplate(
            text="I'll be happy to help you with the cancellation. Let me check what options are available for your specific situation.",
            follow_up_questions=["When would you like the cancellation to take effect?"],
            actions=["check_cancellation_policy", "process_cancellation"]
        )
    ]
    
    for template in cancellation_templates:
        generator.add_response_template("cancellation_request", template)
    
    # Password Reset responses
    password_templates = [
        ResponseTemplate(
            text="I'll help you reset your password right away. Account security is important to us.",
            follow_up_questions=["What email address is associated with your account?"],
            actions=["send_password_reset_email", "verify_account_ownership"]
        ),
        ResponseTemplate(
            text="No problem! I'll guide you through the password reset process step by step.",
            follow_up_questions=["Do you have access to the email address on your account?"],
            actions=["initiate_password_reset", "provide_reset_instructions"]
        )
    ]
    
    for template in password_templates:
        generator.add_response_template("password_reset", template)
    
    return generator


def demo_conversation():
    """Demonstrate the response generator with a sample conversation."""
    print("Response Generator Demo")
    print("=" * 50)
    
    # Import the intent classifier from the previous file
    try:
        from intent_classifier import create_sample_classifier
        classifier = create_sample_classifier()
    except ImportError:
        print("Note: This demo works best with the intent_classifier.py file")
        # Create a mock classifier for demo purposes
        class MockClassifier:
            def classify(self, text):
                # Simple mock classification
                if any(word in text.lower() for word in ['hi', 'hello', 'hey']):
                    return 'greeting', 0.9, {}
                elif '?' in text:
                    return 'question', 0.8, {}
                elif any(word in text.lower() for word in ['book', 'reserve']):
                    return 'booking', 0.85, {}
                elif any(word in text.lower() for word in ['terrible', 'awful', 'problem']):
                    return 'complaint', 0.75, {}
                elif any(word in text.lower() for word in ['great', 'amazing', 'love']):
                    return 'compliment', 0.8, {}
                else:
                    return 'unknown', 0.1, {}
        
        classifier = MockClassifier()
    
    generator = create_sample_response_generator()
    
    # Sample conversation
    conversation_inputs = [
        "Hello there!",
        "My name is Alice, nice to meet you",
        "I'd like to book a table for dinner",
        "Can we make it for 7pm tonight?",
        "That sounds perfect, thank you!",
        "What's your return policy?",
        "The service here is absolutely amazing!",
        "Thanks for all your help today"
    ]
    
    print("Sample Conversation:")
    print("-" * 30)
    
    for i, user_input in enumerate(conversation_inputs, 1):
        print(f"\nTurn {i}:")
        print(f"User: {user_input}")
        
        # Classify intent
        intent, confidence, debug = classifier.classify(user_input)
        
        # Generate response
        response_data = generator.generate_response(user_input, intent, confidence)
        
        print(f"Bot: {response_data['response']}")
        print(f"(Intent: {intent}, Confidence: {confidence:.2f})")
        
        if response_data['actions']:
            print(f"Actions: {', '.join(response_data['actions'])}")
        
        # Add a small delay for realism
        time.sleep(0.5)
    
    # Show conversation summary
    print("\n" + "=" * 50)
    print("Conversation Summary:")
    summary = generator.get_conversation_summary()
    print(f"Session Duration: {summary['session_duration']} seconds")
    print(f"Total Interactions: {summary['total_interactions']}")
    print(f"User Name: {summary['user_name']}")
    print(f"Current Topic: {summary['current_topic']}")
    print(f"Intent Distribution: {summary['intent_distribution']}")
    
    # Export conversation
    generator.export_conversation("sample_conversation.json")
    print("\nConversation exported to 'sample_conversation.json'")


if __name__ == "__main__":
    demo_conversation()
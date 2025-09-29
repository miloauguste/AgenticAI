"""
Sentiment analysis tool for customer support tickets
"""
import re
from typing import Dict, Any, List, Optional
import logging
from config import settings

# Try to import TextBlob, provide fallback if not available
try:
    from textblob import TextBlob
    HAS_TEXTBLOB = True
except ImportError:
    HAS_TEXTBLOB = False
    print("Warning: TextBlob not installed. Using pattern-based sentiment analysis only.")

class SentimentAnalyzer:
    """Analyze sentiment and urgency in customer support texts"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Predefined sentiment patterns
        self.positive_patterns = [
            r'\b(thank\w*|appreciate\w*|great|excellent|amazing|wonderful|fantastic|pleased|satisfied|happy)\b',
            r'\b(love|like|enjoy|impressed|glad|grateful|awesome|perfect|outstanding)\b',
            r'\b(good\s+job|well\s+done|keep\s+it\s+up|brilliant|superb|marvelous)\b'
        ]
        
        self.negative_patterns = [
            r'\b(terrible|awful|horrible|disgusting|hate|angry|furious|frustrated|disappointed|upset)\b',
            r'\b(worst|pathetic|useless|ridiculous|unacceptable|outrageous|appalling)\b',
            r'\b(refund|cancel|complaint|problem|issue|broken|defective|failed|damaged)\b',
            r'\b(disaster|nightmare|catastrophe|fed\s+up|sick\s+of|can\'?t\s+stand)\b'
        ]
        
        self.urgency_patterns = [
            r'\b(urgent|emergency|asap|immediately|critical|escalate|priority)\b',
            r'\b(need\s+help\s+now|right\s+away|can\'?t\s+wait|time\s+sensitive)\b',
            r'\b(deadline|running\s+out\s+of\s+time|by\s+tomorrow|today|tonight)\b',
            r'\b(losing\s+money|business\s+critical|mission\s+critical)\b'
        ]
        
        self.frustration_indicators = [
            r'\b(frustrated|annoyed|irritated|fed\s+up|sick\s+of|tired\s+of)\b',
            r'\b(this\s+is\s+ridiculous|what\s+the\s+hell|are\s+you\s+kidding|seriously)\b',
            r'[!]{2,}',  # Multiple exclamation marks
            r'[A-Z]{4,}',  # All caps words (4+ characters)
            r'\b(why\s+is\s+this\s+so\s+hard|this\s+shouldn\'?t\s+be\s+this\s+difficult)\b',
            r'\b(waste\s+of\s+time|complete\s+joke|total\s+mess|absolute\s+disaster)\b'
        ]
        
        self.polite_indicators = [
            r'\b(please|thank\s+you|thanks|appreciate|kindly|would\s+you\s+mind)\b',
            r'\b(sorry\s+to\s+bother|excuse\s+me|if\s+possible|when\s+you\s+have\s+time)\b',
            r'\b(could\s+you|would\s+you|may\s+I|if\s+you\s+could|hope\s+you\s+can)\b',
            r'\b(respectfully|with\s+respect|humbly|gratefully)\b'
        ]
        
        # Emotion-specific patterns
        self.emotion_patterns = {
            'anger': [
                r'\b(angry|mad|furious|rage|pissed|livid|enraged|irate)\b',
                r'\b(damn|hell|stupid|idiotic|moronic)\b'
            ],
            'sadness': [
                r'\b(sad|disappointed|heartbroken|depressed|upset|devastated)\b',
                r'\b(crying|tears|hopeless|miserable|down)\b'
            ],
            'fear': [
                r'\b(scared|afraid|worried|anxious|concerned|nervous|terrified)\b',
                r'\b(panic|frightened|alarmed|apprehensive)\b'
            ],
            'joy': [
                r'\b(happy|joyful|excited|thrilled|delighted|ecstatic|elated)\b',
                r'\b(cheerful|upbeat|optimistic|pleased|content)\b'
            ],
            'surprise': [
                r'\b(surprised|shocked|amazed|astonished|stunned|bewildered)\b',
                r'\b(can\'?t\s+believe|wow|incredible|unbelievable)\b'
            ],
            'disgust': [
                r'\b(disgusted|repulsed|appalled|revolted|sickened)\b',
                r'\b(gross|nasty|vile|repugnant)\b'
            ],
            'trust': [
                r'\b(trust|confident|reliable|dependable|faith|believe)\b',
                r'\b(counting\s+on|rely\s+on|have\s+faith)\b'
            ],
            'anticipation': [
                r'\b(excited|anticipating|looking\s+forward|eager|hopeful)\b',
                r'\b(can\'?t\s+wait|expecting|awaiting)\b'
            ]
        }
        
        # Intensity modifiers
        self.intensity_amplifiers = [
            r'\b(very|extremely|incredibly|absolutely|completely|totally|utterly)\b',
            r'\b(super|really|quite|rather|pretty|fairly|somewhat)\b',
            r'\b(so\s+much|way\s+too|far\s+too|much\s+too)\b'
        ]
        
        self.intensity_diminishers = [
            r'\b(slightly|somewhat|kind\s+of|sort\s+of|a\s+bit|a\s+little)\b',
            r'\b(not\s+very|not\s+really|barely|hardly|scarcely)\b'
        ]
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Comprehensive sentiment analysis of text
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with sentiment analysis results
        """
        try:
            # Basic sentiment analysis
            if HAS_TEXTBLOB:
                blob = TextBlob(text)
                polarity = blob.sentiment.polarity  # -1 (negative) to 1 (positive)
                subjectivity = blob.sentiment.subjectivity  # 0 (objective) to 1 (subjective)
            else:
                # Fallback pattern-based analysis
                polarity = self._calculate_pattern_polarity(text)
                subjectivity = 0.5  # Default subjectivity
            
            # Pattern-based analysis
            positive_score = self._calculate_pattern_score(text, self.positive_patterns)
            negative_score = self._calculate_pattern_score(text, self.negative_patterns)
            urgency_score = self._calculate_pattern_score(text, self.urgency_patterns)
            frustration_score = self._calculate_pattern_score(text, self.frustration_indicators)
            politeness_score = self._calculate_pattern_score(text, self.polite_indicators)
            
            # Calculate intensity adjustments
            intensity_multiplier = self._calculate_intensity_multiplier(text)
            
            # Apply intensity adjustments
            adjusted_positive = positive_score * intensity_multiplier
            adjusted_negative = negative_score * intensity_multiplier
            adjusted_urgency = urgency_score * intensity_multiplier
            adjusted_frustration = frustration_score * intensity_multiplier
            
            # Combine scores for final sentiment
            combined_sentiment = self._combine_sentiment_scores(
                polarity, adjusted_positive, adjusted_negative
            )
            
            # Determine sentiment category
            sentiment_category = self._categorize_sentiment(combined_sentiment)
            
            # Calculate urgency level
            urgency_level = self._calculate_urgency_level(adjusted_urgency, adjusted_frustration, adjusted_negative)
            
            # Determine customer mood
            customer_mood = self._determine_customer_mood(
                combined_sentiment, adjusted_frustration, politeness_score, adjusted_urgency
            )
            
            # Detect specific emotions
            detected_emotions = self._detect_specific_emotions(text)
            
            # Calculate confidence
            analysis_confidence = self._calculate_confidence(text, combined_sentiment, detected_emotions)
            
            # Determine response priority
            response_priority = self._determine_response_priority(urgency_level, sentiment_category)
            
            # Get recommended approach
            recommended_approach = self._recommend_response_approach(customer_mood, sentiment_category)
            
            # Extract keywords related to sentiment
            sentiment_keywords = self._extract_sentiment_keywords(text, sentiment_category)
            
            result = {
                'sentiment_score': round(combined_sentiment, 3),
                'sentiment_category': sentiment_category,
                'urgency_score': round(adjusted_urgency, 3),
                'urgency_level': urgency_level,
                'customer_mood': customer_mood,
                'polarity': round(polarity, 3),
                'subjectivity': round(subjectivity, 3),
                'intensity_multiplier': round(intensity_multiplier, 3),
                'component_scores': {
                    'positive': round(adjusted_positive, 3),
                    'negative': round(adjusted_negative, 3),
                    'frustration': round(adjusted_frustration, 3),
                    'politeness': round(politeness_score, 3),
                    'raw_positive': round(positive_score, 3),
                    'raw_negative': round(negative_score, 3),
                    'raw_urgency': round(urgency_score, 3),
                    'raw_frustration': round(frustration_score, 3)
                },
                'analysis_confidence': analysis_confidence,
                'detected_emotions': detected_emotions,
                'response_priority': response_priority,
                'recommended_approach': recommended_approach,
                'sentiment_keywords': sentiment_keywords,
                'text_statistics': self._get_text_statistics(text),
                'escalation_indicators': self._get_escalation_indicators(text, adjusted_frustration, adjusted_urgency),
                'customer_satisfaction_prediction': self._predict_customer_satisfaction(combined_sentiment, customer_mood)
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error analyzing sentiment: {str(e)}")
            return self._get_default_sentiment_result()
    
    def _calculate_pattern_polarity(self, text: str) -> float:
        """Calculate polarity using pattern matching (fallback for TextBlob)"""
        positive_words = [
            'good', 'great', 'excellent', 'amazing', 'love', 'like', 'happy', 
            'satisfied', 'pleased', 'wonderful', 'fantastic', 'awesome', 'perfect'
        ]
        
        negative_words = [
            'bad', 'terrible', 'awful', 'hate', 'angry', 'frustrated', 
            'disappointed', 'upset', 'horrible', 'disgusting', 'worst', 'pathetic'
        ]
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        total_words = len(text.split())
        if total_words == 0:
            return 0.0
        
        # Normalize scores
        positive_norm = positive_count / total_words
        negative_norm = negative_count / total_words
        
        # Return polarity score
        polarity = positive_norm - negative_norm
        return max(-1.0, min(1.0, polarity * 10))  # Scale up and clamp
    
    def _calculate_pattern_score(self, text: str, patterns: List[str]) -> float:
        """Calculate score based on pattern matching"""
        text_lower = text.lower()
        total_matches = 0
        
        for pattern in patterns:
            matches = len(re.findall(pattern, text_lower, re.IGNORECASE))
            total_matches += matches
        
        # Normalize by text length (per 100 words)
        word_count = len(text.split())
        if word_count == 0:
            return 0.0
        
        normalized_score = (total_matches * 100) / word_count
        return min(normalized_score, 1.0)  # Cap at 1.0
    
    def _calculate_intensity_multiplier(self, text: str) -> float:
        """Calculate intensity multiplier based on amplifiers and diminishers"""
        text_lower = text.lower()
        
        amplifier_count = 0
        for pattern in self.intensity_amplifiers:
            amplifier_count += len(re.findall(pattern, text_lower, re.IGNORECASE))
        
        diminisher_count = 0
        for pattern in self.intensity_diminishers:
            diminisher_count += len(re.findall(pattern, text_lower, re.IGNORECASE))
        
        # Calculate multiplier (base 1.0)
        multiplier = 1.0 + (amplifier_count * 0.2) - (diminisher_count * 0.2)
        
        # Clamp between 0.5 and 2.0
        return max(0.5, min(2.0, multiplier))
    
    def _combine_sentiment_scores(self, polarity: float, positive_score: float, negative_score: float) -> float:
        """Combine TextBlob polarity with pattern-based scores"""
        # Weight TextBlob polarity (60%) and pattern scores (40%)
        pattern_sentiment = positive_score - negative_score
        combined = (0.6 * polarity) + (0.4 * pattern_sentiment)
        
        # Ensure result is in range [-1, 1]
        return max(-1.0, min(1.0, combined))
    
    def _categorize_sentiment(self, sentiment_score: float) -> str:
        """Categorize sentiment based on score"""
        if sentiment_score >= 0.3:
            return "positive"
        elif sentiment_score <= -0.3:
            return "negative"
        else:
            return "neutral"
    
    def _calculate_urgency_level(self, urgency_score: float, frustration_score: float, negative_score: float) -> str:
        """Calculate urgency level based on multiple factors"""
        combined_urgency = urgency_score + (0.5 * frustration_score) + (0.3 * negative_score)
        
        if combined_urgency >= 0.7:
            return "critical"
        elif combined_urgency >= 0.4:
            return "high"
        elif combined_urgency >= 0.2:
            return "medium"
        else:
            return "low"
    
    def _determine_customer_mood(self, sentiment: float, frustration: float, politeness: float, urgency: float) -> str:
        """Determine overall customer mood"""
        # Priority order for mood determination
        if frustration >= 0.5:
            return "frustrated"
        elif urgency >= 0.6 and sentiment < 0:
            return "anxious"
        elif sentiment >= 0.4:
            return "satisfied" if politeness >= 0.3 else "pleased"
        elif sentiment <= -0.4:
            return "dissatisfied"
        elif politeness >= 0.4:
            return "polite"
        elif urgency >= 0.4:
            return "concerned"
        else:
            return "neutral"
    
    def _calculate_confidence(self, text: str, sentiment_score: float, emotions: List[str]) -> float:
        """Calculate confidence in the sentiment analysis"""
        # Base confidence on multiple factors
        word_count = len(text.split())
        
        # More words generally mean higher confidence
        length_confidence = min(word_count / 50, 1.0)  # Cap at 50 words
        
        # Stronger sentiment scores indicate higher confidence
        sentiment_strength = abs(sentiment_score)
        
        # More detected emotions indicate higher confidence
        emotion_confidence = min(len(emotions) / 3, 1.0)  # Cap at 3 emotions
        
        # TextBlob availability increases confidence
        textblob_bonus = 0.1 if HAS_TEXTBLOB else 0.0
        
        # Combine factors
        confidence = (0.4 * length_confidence) + (0.3 * sentiment_strength) + (0.2 * emotion_confidence) + textblob_bonus
        
        return round(max(0.1, min(1.0, confidence)), 3)
    
    def _detect_specific_emotions(self, text: str) -> List[str]:
        """Detect specific emotions in the text"""
        emotions = []
        text_lower = text.lower()
        
        for emotion, patterns in self.emotion_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower, re.IGNORECASE):
                    emotions.append(emotion)
                    break  # Only add each emotion once
        
        return emotions
    
    def _determine_response_priority(self, urgency_level: str, sentiment_category: str) -> str:
        """Determine response priority based on urgency and sentiment"""
        priority_matrix = {
            ('critical', 'negative'): 'immediate',
            ('critical', 'neutral'): 'immediate',
            ('critical', 'positive'): 'high',
            ('high', 'negative'): 'high',
            ('high', 'neutral'): 'high',
            ('high', 'positive'): 'medium',
            ('medium', 'negative'): 'medium',
            ('medium', 'neutral'): 'medium',
            ('medium', 'positive'): 'low',
            ('low', 'negative'): 'low',
            ('low', 'neutral'): 'low',
            ('low', 'positive'): 'low'
        }
        
        return priority_matrix.get((urgency_level, sentiment_category), 'medium')
    
    def _recommend_response_approach(self, customer_mood: str, sentiment_category: str) -> Dict[str, Any]:
        """Recommend response approach based on customer analysis"""
        approaches = {
            'frustrated': {
                'tone': 'empathetic_apologetic',
                'priority': 'acknowledge_frustration',
                'key_phrases': ['I sincerely apologize', 'I understand your frustration', 'Let me make this right'],
                'avoid_phrases': ['I understand your concern', 'Thank you for your patience'],
                'suggestions': [
                    'Acknowledge the frustration immediately',
                    'Take personal ownership of the issue',
                    'Offer direct solutions with timeline',
                    'Provide escalation path if needed'
                ]
            },
            'anxious': {
                'tone': 'reassuring_calm',
                'priority': 'provide_clarity',
                'key_phrases': ['Let me help you right away', 'I\'ll provide a clear update', 'You can count on me'],
                'avoid_phrases': ['Please be patient', 'We\'ll get back to you'],
                'suggestions': [
                    'Provide clear, immediate timeline',
                    'Explain exact next steps',
                    'Offer proactive updates',
                    'Give direct contact information'
                ]
            },
            'dissatisfied': {
                'tone': 'professional_solution_focused',
                'priority': 'resolve_issue',
                'key_phrases': ['I\'m committed to resolving this', 'Let\'s fix this together', 'You deserve better'],
                'avoid_phrases': ['Thank you for bringing this to our attention'],
                'suggestions': [
                    'Focus on concrete solutions',
                    'Offer compensation if appropriate',
                    'Provide multiple resolution options',
                    'Schedule follow-up to ensure satisfaction'
                ]
            },
            'satisfied': {
                'tone': 'appreciative_helpful',
                'priority': 'maintain_satisfaction',
                'key_phrases': ['Thank you for your kind words', 'I\'m glad we could help', 'We appreciate your business'],
                'avoid_phrases': ['Sorry for any inconvenience'],
                'suggestions': [
                    'Express genuine appreciation',
                    'Provide additional helpful information',
                    'Encourage future contact',
                    'Ask if there\'s anything else needed'
                ]
            },
            'polite': {
                'tone': 'equally_polite_professional',
                'priority': 'match_courtesy_level',
                'key_phrases': ['Thank you for your courtesy', 'I\'d be happy to help', 'Please let me assist you'],
                'avoid_phrases': ['No problem', 'Sure thing'],
                'suggestions': [
                    'Mirror the polite tone',
                    'Be thorough and detailed in response',
                    'Express appreciation for patience',
                    'Use formal but warm language'
                ]
            },
            'concerned': {
                'tone': 'understanding_informative',
                'priority': 'address_concerns',
                'key_phrases': ['I understand your concern', 'Let me clarify this for you', 'I\'ll make sure this is handled'],
                'avoid_phrases': ['Don\'t worry about it'],
                'suggestions': [
                    'Address specific concerns directly',
                    'Provide detailed explanations',
                    'Offer preventive measures',
                    'Give reassurance with facts'
                ]
            },
            'neutral': {
                'tone': 'professional_informative',
                'priority': 'provide_information',
                'key_phrases': ['I\'ll be happy to help', 'Let me assist you with this', 'Here\'s what I can do'],
                'avoid_phrases': [],
                'suggestions': [
                    'Be clear and informative',
                    'Maintain professional tone',
                    'Ask clarifying questions if needed',
                    'Provide complete information'
                ]
            }
        }
        
        return approaches.get(customer_mood, approaches['neutral'])
    
    def _extract_sentiment_keywords(self, text: str, sentiment_category: str) -> List[str]:
        """Extract keywords that contributed to sentiment analysis"""
        text_lower = text.lower()
        keywords = []
        
        # Extract words that match sentiment patterns
        if sentiment_category == 'positive':
            for pattern in self.positive_patterns:
                matches = re.findall(pattern, text_lower, re.IGNORECASE)
                keywords.extend(matches)
        elif sentiment_category == 'negative':
            for pattern in self.negative_patterns:
                matches = re.findall(pattern, text_lower, re.IGNORECASE)
                keywords.extend(matches)
        
        # Add urgency keywords if detected
        for pattern in self.urgency_patterns:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            keywords.extend(matches)
        
        # Remove duplicates and return top 10
        unique_keywords = list(set(keywords))
        return unique_keywords[:10]
    
    def _get_text_statistics(self, text: str) -> Dict[str, Any]:
        """Get basic text statistics"""
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        
        return {
            'word_count': len(words),
            'char_count': len(text),
            'sentence_count': len([s for s in sentences if s.strip()]),
            'avg_word_length': sum(len(word) for word in words) / len(words) if words else 0,
            'exclamation_count': text.count('!'),
            'question_count': text.count('?'),
            'caps_word_count': len([word for word in words if word.isupper() and len(word) > 1])
        }
    
    def _get_escalation_indicators(self, text: str, frustration_score: float, urgency_score: float) -> Dict[str, Any]:
        """Identify specific escalation indicators"""
        text_lower = text.lower()
        
        escalation_keywords = [
            'manager', 'supervisor', 'escalate', 'higher up', 'someone else',
            'complaint', 'legal action', 'lawyer', 'attorney', 'sue', 'court',
            'better business bureau', 'bbb', 'review', 'social media', 'twitter',
            'cancel', 'refund', 'money back', 'never again', 'last time'
        ]
        
        threat_indicators = [
            'lawyer', 'legal', 'sue', 'court', 'attorney', 'legal action',
            'better business bureau', 'bbb', 'consumer protection', 'report you'
        ]
        
        social_media_threats = [
            'twitter', 'facebook', 'instagram', 'yelp', 'google reviews',
            'social media', 'tell everyone', 'spread the word', 'warn others'
        ]
        
        found_escalation = [kw for kw in escalation_keywords if kw in text_lower]
        found_threats = [kw for kw in threat_indicators if kw in text_lower]
        found_social = [kw for kw in social_media_threats if kw in text_lower]
        
        return {
            'requires_escalation': len(found_escalation) > 0 or frustration_score > 0.7 or urgency_score > 0.8,
            'escalation_keywords': found_escalation,
            'legal_threats': found_threats,
            'social_media_threats': found_social,
            'escalation_risk_score': min(1.0, (len(found_escalation) * 0.2) + (frustration_score * 0.4) + (urgency_score * 0.3))
        }
    
    def _predict_customer_satisfaction(self, sentiment_score: float, customer_mood: str) -> Dict[str, Any]:
        """Predict customer satisfaction and likelihood of resolution"""
        base_satisfaction = 0.5  # Neutral starting point
        
        # Adjust based on sentiment
        base_satisfaction += sentiment_score * 0.3
        
        # Adjust based on mood
        mood_adjustments = {
            'satisfied': 0.3,
            'pleased': 0.2,
            'polite': 0.1,
            'neutral': 0.0,
            'concerned': -0.1,
            'anxious': -0.2,
            'dissatisfied': -0.3,
            'frustrated': -0.4
        }
        
        base_satisfaction += mood_adjustments.get(customer_mood, 0.0)
        
        # Clamp between 0 and 1
        current_satisfaction = max(0.0, min(1.0, base_satisfaction))
        
        # Predict satisfaction after response
        response_boost = {
            'frustrated': 0.4,  # Good response can significantly improve
            'anxious': 0.3,
            'dissatisfied': 0.3,
            'concerned': 0.2,
            'neutral': 0.1,
            'polite': 0.1,
            'satisfied': 0.05,
            'pleased': 0.05
        }
        
        predicted_satisfaction = min(1.0, current_satisfaction + response_boost.get(customer_mood, 0.1))
        
        return {
            'current_satisfaction': round(current_satisfaction, 3),
            'predicted_post_response': round(predicted_satisfaction, 3),
            'improvement_potential': round(predicted_satisfaction - current_satisfaction, 3),
            'satisfaction_category': self._categorize_satisfaction(predicted_satisfaction)
        }
    
    def _categorize_satisfaction(self, satisfaction_score: float) -> str:
        """Categorize satisfaction level"""
        if satisfaction_score >= 0.8:
            return "highly_satisfied"
        elif satisfaction_score >= 0.6:
            return "satisfied"
        elif satisfaction_score >= 0.4:
            return "neutral"
        elif satisfaction_score >= 0.2:
            return "dissatisfied"
        else:
            return "highly_dissatisfied"
    
    def _get_default_sentiment_result(self) -> Dict[str, Any]:
        """Return default sentiment analysis result in case of error"""
        return {
            'sentiment_score': 0.0,
            'sentiment_category': 'neutral',
            'urgency_score': 0.0,
            'urgency_level': 'low',
            'customer_mood': 'neutral',
            'polarity': 0.0,
            'subjectivity': 0.0,
            'intensity_multiplier': 1.0,
            'component_scores': {
                'positive': 0.0,
                'negative': 0.0,
                'frustration': 0.0,
                'politeness': 0.0,
                'raw_positive': 0.0,
                'raw_negative': 0.0,
                'raw_urgency': 0.0,
                'raw_frustration': 0.0
            },
            'analysis_confidence': 0.1,
            'detected_emotions': [],
            'response_priority': 'medium',
            'recommended_approach': self._recommend_response_approach('neutral', 'neutral'),
            'sentiment_keywords': [],
            'text_statistics': {'word_count': 0, 'char_count': 0, 'sentence_count': 0},
            'escalation_indicators': {'requires_escalation': False, 'escalation_risk_score': 0.0},
            'customer_satisfaction_prediction': {
                'current_satisfaction': 0.5,
                'predicted_post_response': 0.6,
                'improvement_potential': 0.1,
                'satisfaction_category': 'neutral'
            }
        }
    
    def analyze_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Analyze sentiment for a batch of texts"""
        results = []
        
        for i, text in enumerate(texts):
            try:
                result = self.analyze_sentiment(text)
                result['batch_index'] = i
                results.append(result)
            except Exception as e:
                self.logger.error(f"Error analyzing text {i}: {str(e)}")
                error_result = self._get_default_sentiment_result()
                error_result['batch_index'] = i
                error_result['error'] = str(e)
                results.append(error_result)
        
        return results
    
    def get_sentiment_summary(self, analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summary statistics for a batch of sentiment analyses"""
        if not analyses:
            return {}
        
        # Count categories
        sentiment_counts = {'positive': 0, 'negative': 0, 'neutral': 0}
        urgency_counts = {'critical': 0, 'high': 0, 'medium': 0, 'low': 0}
        mood_counts = {}
        priority_counts = {'immediate': 0, 'high': 0, 'medium': 0, 'low': 0}
        
        # Collect scores
        sentiment_scores = []
        urgency_scores = []
        confidence_scores = []
        satisfaction_scores = []
        
        # Process each analysis
        for analysis in analyses:
            # Count categories
            sentiment_counts[analysis.get('sentiment_category', 'neutral')] += 1
            urgency_counts[analysis.get('urgency_level', 'low')] += 1
            
            mood = analysis.get('customer_mood', 'neutral')
            mood_counts[mood] = mood_counts.get(mood, 0) + 1
            
            priority_counts[analysis.get('response_priority', 'medium')] += 1
            
            # Collect scores
            sentiment_scores.append(analysis.get('sentiment_score', 0.0))
            urgency_scores.append(analysis.get('urgency_score', 0.0))
            confidence_scores.append(analysis.get('analysis_confidence', 0.0))
            
            csat = analysis.get('customer_satisfaction_prediction', {})
            satisfaction_scores.append(csat.get('current_satisfaction', 0.5))
        
        # Calculate averages
        total = len(analyses)
        avg_sentiment = sum(sentiment_scores) / total if total > 0 else 0.0
        avg_urgency = sum(urgency_scores) / total if total > 0 else 0.0
        avg_confidence = sum(confidence_scores) / total if total > 0 else 0.0
        avg_satisfaction = sum(satisfaction_scores) / total if total > 0 else 0.5
        
        return {
            'total_analyzed': total,
            'sentiment_distribution': sentiment_counts,
            'urgency_distribution': urgency_counts,
            'mood_distribution': mood_counts,
            'priority_distribution': priority_counts,
            'average_scores': {
                'sentiment': round(avg_sentiment, 3),
                'urgency': round(avg_urgency, 3),
                'confidence': round(avg_confidence, 3),
                'satisfaction': round(avg_satisfaction, 3)
            },
            'score_ranges': {
                'sentiment_min': round(min(sentiment_scores), 3) if sentiment_scores else 0.0,
                'sentiment_max': round(max(sentiment_scores), 3) if sentiment_scores else 0.0,
                'urgency_min': round(min(urgency_scores), 3) if urgency_scores else 0.0,
                'urgency_max': round(max(urgency_scores), 3) if urgency_scores else 0.0,
            }
        }
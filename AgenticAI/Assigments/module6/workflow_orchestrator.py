#!/usr/bin/env python3
"""
SupportTriageWorkflow using Agentic Task Orchestration
Implements the complete workflow: Log Ingestion → Preprocessing → Chunking → Embedding → Intent/Sentiment Extraction → Suggested Response → Query Handling
"""

import logging
import asyncio
import uuid
from typing import Dict, Any, List, Optional, Callable, Union
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import json
import traceback

# Workflow task status
class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"

class TaskPriority(Enum):
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class TaskResult:
    """Result of a workflow task execution"""
    task_id: str
    status: TaskStatus
    data: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    execution_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class WorkflowContext:
    """Context passed between workflow tasks"""
    workflow_id: str
    session_id: str
    data: Dict[str, Any] = field(default_factory=dict)
    results: Dict[str, TaskResult] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

class WorkflowTask(ABC):
    """Abstract base class for workflow tasks"""
    
    def __init__(self, task_id: str, name: str, description: str = "", 
                 priority: TaskPriority = TaskPriority.NORMAL,
                 timeout: Optional[float] = None,
                 retry_count: int = 0,
                 dependencies: List[str] = None):
        self.task_id = task_id
        self.name = name
        self.description = description
        self.priority = priority
        self.timeout = timeout
        self.retry_count = retry_count
        self.dependencies = dependencies or []
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    @abstractmethod
    async def execute(self, context: WorkflowContext) -> TaskResult:
        """Execute the task and return results"""
        pass
    
    def validate_dependencies(self, context: WorkflowContext) -> bool:
        """Check if all dependencies are satisfied"""
        for dep_id in self.dependencies:
            if dep_id not in context.results:
                return False
            if context.results[dep_id].status != TaskStatus.COMPLETED:
                return False
        return True
    
    def should_skip(self, context: WorkflowContext) -> bool:
        """Determine if task should be skipped based on context"""
        return False

class LogIngestionTask(WorkflowTask):
    """Task 1: Log Ingestion - Load and validate input data"""
    
    def __init__(self):
        super().__init__(
            task_id="log_ingestion",
            name="Log Ingestion",
            description="Load and validate customer support logs and ticket data",
            priority=TaskPriority.HIGH,
            timeout=30.0
        )
    
    async def execute(self, context: WorkflowContext) -> TaskResult:
        start_time = datetime.now()
        self.logger.info(f"Starting log ingestion for workflow {context.workflow_id}")
        
        try:
            # Extract input data
            input_data = context.data.get('input', {})
            
            # Validate required fields
            if 'ticket_text' not in input_data and 'file_data' not in input_data:
                raise ValueError("Either 'ticket_text' or 'file_data' must be provided")
            
            ingested_data = {
                'source_type': input_data.get('source_type', 'ticket'),
                'timestamp': datetime.now().isoformat(),
                'data_size': 0,
                'records_count': 0,
                'validation_status': 'valid'
            }
            
            # Process ticket text
            if 'ticket_text' in input_data:
                ticket_text = input_data['ticket_text'].strip()
                if len(ticket_text) < 10:
                    raise ValueError("Ticket text too short (minimum 10 characters)")
                
                ingested_data.update({
                    'ticket_text': ticket_text,
                    'ticket_id': input_data.get('ticket_id', f"TICKET_{uuid.uuid4().hex[:8]}"),
                    'customer_id': input_data.get('customer_id'),
                    'priority': input_data.get('priority', 'normal'),
                    'channel': input_data.get('channel', 'email'),
                    'data_size': len(ticket_text),
                    'records_count': 1
                })
            
            # Process file data
            if 'file_data' in input_data:
                file_info = input_data['file_data']
                file_content = file_info.get('content', '')
                
                if not file_content:
                    raise ValueError("File content is empty")
                
                ingested_data.update({
                    'file_name': file_info.get('name', 'unknown.txt'),
                    'file_type': file_info.get('type', 'text/plain'),
                    'file_content': file_content,
                    'data_size': len(file_content),
                    'records_count': file_content.count('\n') + 1 if file_content else 0
                })
            
            # Add to context
            context.data['ingested_data'] = ingested_data
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            self.logger.info(f"Log ingestion completed: {ingested_data['records_count']} records, {ingested_data['data_size']} bytes")
            
            return TaskResult(
                task_id=self.task_id,
                status=TaskStatus.COMPLETED,
                data=ingested_data,
                execution_time=execution_time,
                metadata={
                    'records_processed': ingested_data['records_count'],
                    'bytes_processed': ingested_data['data_size']
                }
            )
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            error_msg = f"Log ingestion failed: {str(e)}"
            self.logger.error(error_msg)
            
            return TaskResult(
                task_id=self.task_id,
                status=TaskStatus.FAILED,
                error=error_msg,
                execution_time=execution_time
            )

class PreprocessingTask(WorkflowTask):
    """Task 2: Preprocessing - Clean and normalize text data"""
    
    def __init__(self):
        super().__init__(
            task_id="preprocessing",
            name="Data Preprocessing",
            description="Clean, normalize, and prepare text data for analysis",
            dependencies=["log_ingestion"],
            timeout=20.0
        )
    
    async def execute(self, context: WorkflowContext) -> TaskResult:
        start_time = datetime.now()
        self.logger.info(f"Starting preprocessing for workflow {context.workflow_id}")
        
        try:
            ingested_data = context.data.get('ingested_data', {})
            
            processed_data = {
                'preprocessing_steps': [],
                'original_length': 0,
                'processed_length': 0,
                'languages_detected': [],
                'quality_score': 0.0
            }
            
            # Process ticket text
            if 'ticket_text' in ingested_data:
                text = ingested_data['ticket_text']
                processed_data['original_length'] = len(text)
                
                # Text cleaning steps
                processed_text = self._clean_text(text)
                processed_data['preprocessing_steps'].append('text_cleaning')
                
                # Language detection (simplified)
                processed_data['languages_detected'] = ['en']  # Assume English for demo
                processed_data['preprocessing_steps'].append('language_detection')
                
                # Quality assessment
                quality_score = self._assess_text_quality(processed_text)
                processed_data['quality_score'] = quality_score
                processed_data['preprocessing_steps'].append('quality_assessment')
                
                processed_data.update({
                    'processed_text': processed_text,
                    'processed_length': len(processed_text),
                    'word_count': len(processed_text.split()),
                    'sentence_count': processed_text.count('.') + processed_text.count('!') + processed_text.count('?')
                })
            
            # Process file content
            if 'file_content' in ingested_data:
                content = ingested_data['file_content']
                processed_content = self._clean_text(content)
                
                processed_data.update({
                    'processed_file_content': processed_content,
                    'file_processing_steps': ['text_cleaning', 'encoding_normalization'],
                    'processed_length': len(processed_content),
                    'word_count': len(processed_content.split()),
                    'sentence_count': processed_content.count('.') + processed_content.count('!') + processed_content.count('?')
                })
            
            context.data['processed_data'] = processed_data
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            self.logger.info(f"Preprocessing completed: {processed_data['word_count']} words, quality score: {processed_data['quality_score']:.2f}")
            
            return TaskResult(
                task_id=self.task_id,
                status=TaskStatus.COMPLETED,
                data=processed_data,
                execution_time=execution_time,
                metadata={
                    'quality_score': processed_data['quality_score'],
                    'word_count': processed_data.get('word_count', 0)
                }
            )
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            error_msg = f"Preprocessing failed: {str(e)}"
            self.logger.error(error_msg)
            
            return TaskResult(
                task_id=self.task_id,
                status=TaskStatus.FAILED,
                error=error_msg,
                execution_time=execution_time
            )
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        import re
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s.,!?;:()\'-]', ' ', text)
        
        # Normalize case (keep original for sentiment analysis)
        # text = text.lower()  # Comment out to preserve original case
        
        return text.strip()
    
    def _assess_text_quality(self, text: str) -> float:
        """Assess text quality on a scale of 0-1"""
        if not text:
            return 0.0
        
        score = 0.0
        
        # Length score (optimal around 50-500 characters)
        length = len(text)
        if 50 <= length <= 500:
            score += 0.3
        elif 20 <= length < 50 or 500 < length <= 1000:
            score += 0.2
        elif length > 10:
            score += 0.1
        
        # Word count score
        words = text.split()
        if 10 <= len(words) <= 100:
            score += 0.2
        elif 5 <= len(words) < 10 or 100 < len(words) <= 200:
            score += 0.1
        
        # Sentence structure score
        sentences = text.count('.') + text.count('!') + text.count('?')
        if sentences > 0:
            score += 0.2
        
        # Grammar indicators (simple check)
        if any(word in text.lower() for word in ['please', 'thank', 'help', 'issue', 'problem']):
            score += 0.2
        
        # Completeness score (has proper punctuation)
        if text.strip().endswith(('.', '!', '?')):
            score += 0.1
        
        return min(score, 1.0)

class ChunkingTask(WorkflowTask):
    """Task 3: Chunking - Split text into manageable chunks"""
    
    def __init__(self):
        super().__init__(
            task_id="chunking",
            name="Text Chunking",
            description="Split text into optimal chunks for processing and embedding",
            dependencies=["preprocessing"],
            timeout=15.0
        )
    
    async def execute(self, context: WorkflowContext) -> TaskResult:
        start_time = datetime.now()
        self.logger.info(f"Starting chunking for workflow {context.workflow_id}")
        
        try:
            processed_data = context.data.get('processed_data', {})
            
            chunking_result = {
                'chunks': [],
                'chunking_method': 'semantic',
                'chunk_size': 500,
                'overlap_size': 50,
                'total_chunks': 0
            }
            
            # Process main text
            if 'processed_text' in processed_data:
                text = processed_data['processed_text']
                chunks = self._create_chunks(text, max_size=500, overlap=50)
                
                for i, chunk in enumerate(chunks):
                    chunk_data = {
                        'chunk_id': f"chunk_{i+1}",
                        'text': chunk,
                        'length': len(chunk),
                        'word_count': len(chunk.split()),
                        'position': i,
                        'source': 'ticket_text'
                    }
                    chunking_result['chunks'].append(chunk_data)
            
            # Process file content
            if 'processed_file_content' in processed_data:
                content = processed_data['processed_file_content']
                file_chunks = self._create_chunks(content, max_size=1000, overlap=100)
                
                for i, chunk in enumerate(file_chunks):
                    chunk_data = {
                        'chunk_id': f"file_chunk_{i+1}",
                        'text': chunk,
                        'length': len(chunk),
                        'word_count': len(chunk.split()),
                        'position': i,
                        'source': 'file_content'
                    }
                    chunking_result['chunks'].append(chunk_data)
            
            chunking_result['total_chunks'] = len(chunking_result['chunks'])
            context.data['chunking_result'] = chunking_result
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            self.logger.info(f"Chunking completed: {chunking_result['total_chunks']} chunks created")
            
            return TaskResult(
                task_id=self.task_id,
                status=TaskStatus.COMPLETED,
                data=chunking_result,
                execution_time=execution_time,
                metadata={
                    'chunks_created': chunking_result['total_chunks'],
                    'avg_chunk_size': sum(c['length'] for c in chunking_result['chunks']) / max(len(chunking_result['chunks']), 1)
                }
            )
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            error_msg = f"Chunking failed: {str(e)}"
            self.logger.error(error_msg)
            
            return TaskResult(
                task_id=self.task_id,
                status=TaskStatus.FAILED,
                error=error_msg,
                execution_time=execution_time
            )
    
    def _create_chunks(self, text: str, max_size: int = 500, overlap: int = 50) -> List[str]:
        """Create overlapping text chunks"""
        if len(text) <= max_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + max_size
            
            # Try to break at sentence boundaries
            if end < len(text):
                # Look for sentence endings near the boundary
                for punct in ['. ', '! ', '? ']:
                    punct_pos = text.rfind(punct, start, end)
                    if punct_pos > start + max_size // 2:
                        end = punct_pos + 1
                        break
                else:
                    # Fall back to word boundaries
                    space_pos = text.rfind(' ', start, end)
                    if space_pos > start + max_size // 2:
                        end = space_pos
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move start position with overlap
            start = max(start + max_size - overlap, end - overlap)
            if start >= len(text):
                break
        
        return chunks

class EmbeddingTask(WorkflowTask):
    """Task 4: Embedding - Generate vector embeddings for text chunks"""
    
    def __init__(self):
        super().__init__(
            task_id="embedding",
            name="Vector Embedding",
            description="Generate vector embeddings for text chunks",
            dependencies=["chunking"],
            timeout=30.0
        )
    
    async def execute(self, context: WorkflowContext) -> TaskResult:
        start_time = datetime.now()
        self.logger.info(f"Starting embedding generation for workflow {context.workflow_id}")
        
        try:
            chunking_result = context.data.get('chunking_result', {})
            chunks = chunking_result.get('chunks', [])
            
            embedding_result = {
                'embeddings': [],
                'embedding_model': 'all-MiniLM-L6-v2',
                'embedding_dimension': 384,
                'total_embeddings': 0,
                'processing_method': 'sentence_transformers'
            }
            
            # Generate embeddings for each chunk
            for chunk_data in chunks:
                try:
                    # Simulate embedding generation (in real implementation, use sentence-transformers)
                    embedding_vector = self._generate_mock_embedding(chunk_data['text'])
                    
                    embedding_info = {
                        'chunk_id': chunk_data['chunk_id'],
                        'embedding': embedding_vector,
                        'dimension': len(embedding_vector),
                        'text_preview': chunk_data['text'][:100] + "..." if len(chunk_data['text']) > 100 else chunk_data['text'],
                        'source': chunk_data['source']
                    }
                    
                    embedding_result['embeddings'].append(embedding_info)
                    
                except Exception as e:
                    self.logger.warning(f"Failed to generate embedding for chunk {chunk_data['chunk_id']}: {str(e)}")
                    continue
            
            embedding_result['total_embeddings'] = len(embedding_result['embeddings'])
            context.data['embedding_result'] = embedding_result
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            self.logger.info(f"Embedding generation completed: {embedding_result['total_embeddings']} vectors created")
            
            return TaskResult(
                task_id=self.task_id,
                status=TaskStatus.COMPLETED,
                data=embedding_result,
                execution_time=execution_time,
                metadata={
                    'embeddings_created': embedding_result['total_embeddings'],
                    'embedding_dimension': embedding_result['embedding_dimension'],
                    'model_used': embedding_result['embedding_model']
                }
            )
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            error_msg = f"Embedding generation failed: {str(e)}"
            self.logger.error(error_msg)
            
            return TaskResult(
                task_id=self.task_id,
                status=TaskStatus.FAILED,
                error=error_msg,
                execution_time=execution_time
            )
    
    def _generate_mock_embedding(self, text: str) -> List[float]:
        """Generate mock embedding vector (replace with real sentence-transformers in production)"""
        import hashlib
        import random
        
        # Create deterministic but varied embedding based on text content
        hash_obj = hashlib.md5(text.encode())
        seed = int(hash_obj.hexdigest()[:8], 16)
        random.seed(seed)
        
        # Generate 384-dimensional vector (MiniLM size)
        embedding = [random.uniform(-1, 1) for _ in range(384)]
        
        # Normalize vector (L2 normalization)
        norm = sum(x**2 for x in embedding) ** 0.5
        if norm > 0:
            embedding = [x/norm for x in embedding]
        
        return embedding

class IntentSentimentExtractionTask(WorkflowTask):
    """Task 5: Intent and Sentiment Extraction - Analyze customer intent and emotions"""
    
    def __init__(self):
        super().__init__(
            task_id="intent_sentiment_extraction",
            name="Intent & Sentiment Analysis",
            description="Extract customer intent and analyze sentiment from processed text",
            dependencies=["preprocessing"],
            timeout=25.0
        )
    
    async def execute(self, context: WorkflowContext) -> TaskResult:
        start_time = datetime.now()
        self.logger.info(f"Starting intent and sentiment extraction for workflow {context.workflow_id}")
        
        try:
            processed_data = context.data.get('processed_data', {})
            
            # Import our existing tools
            from sentiment_analyzer import SentimentAnalyzer
            from intent_classifier import create_sample_classifier
            
            sentiment_analyzer = SentimentAnalyzer()
            intent_classifier = create_sample_classifier()
            
            analysis_result = {
                'sentiment_analysis': {},
                'intent_classification': {},
                'extracted_entities': [],
                'confidence_scores': {},
                'processing_steps': []
            }
            
            # Analyze main text
            if 'processed_text' in processed_data:
                text = processed_data['processed_text']
                
                # Sentiment analysis
                sentiment_result = sentiment_analyzer.analyze_sentiment(text)
                analysis_result['sentiment_analysis'] = sentiment_result
                analysis_result['processing_steps'].append('sentiment_analysis')
                
                # Intent classification
                intent_pred, intent_conf, intent_details = intent_classifier.classify(text)
                intent_result = {
                    'predicted_intent': intent_pred,
                    'confidence': intent_conf,
                    'details': intent_details
                }
                analysis_result['intent_classification'] = intent_result
                analysis_result['processing_steps'].append('intent_classification')
                
                # Extract entities (simplified)
                entities = self._extract_entities(text)
                analysis_result['extracted_entities'] = entities
                analysis_result['processing_steps'].append('entity_extraction')
                
                # Calculate confidence scores
                analysis_result['confidence_scores'] = {
                    'sentiment_confidence': sentiment_result.get('analysis_confidence', 0.0),
                    'intent_confidence': intent_result.get('confidence', 0.0),
                    'overall_confidence': (
                        sentiment_result.get('analysis_confidence', 0.0) + 
                        intent_result.get('confidence', 0.0)
                    ) / 2
                }
            
            context.data['analysis_result'] = analysis_result
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            self.logger.info(f"Intent/sentiment extraction completed: {len(analysis_result['processing_steps'])} analysis steps")
            
            return TaskResult(
                task_id=self.task_id,
                status=TaskStatus.COMPLETED,
                data=analysis_result,
                execution_time=execution_time,
                metadata={
                    'sentiment': analysis_result['sentiment_analysis'].get('sentiment_category', 'unknown'),
                    'intent': analysis_result['intent_classification'].get('predicted_intent', 'unknown'),
                    'confidence': analysis_result['confidence_scores'].get('overall_confidence', 0.0)
                }
            )
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            error_msg = f"Intent/sentiment extraction failed: {str(e)}"
            self.logger.error(error_msg)
            
            return TaskResult(
                task_id=self.task_id,
                status=TaskStatus.FAILED,
                error=error_msg,
                execution_time=execution_time
            )
    
    def _extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract named entities from text (simplified implementation)"""
        import re
        
        entities = []
        
        # Email addresses
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, text)
        for email in emails:
            entities.append({'type': 'email', 'value': email, 'confidence': 0.9})
        
        # Order IDs (pattern: letters + numbers)
        order_pattern = r'\b[A-Z]{2,}\d+\b|\b\d+[A-Z]{2,}\b'
        orders = re.findall(order_pattern, text)
        for order in orders:
            entities.append({'type': 'order_id', 'value': order, 'confidence': 0.8})
        
        # Money amounts
        money_pattern = r'\$\d+(?:\.\d{2})?|\d+(?:\.\d{2})?\s*(?:dollars?|USD)'
        amounts = re.findall(money_pattern, text, re.IGNORECASE)
        for amount in amounts:
            entities.append({'type': 'money', 'value': amount, 'confidence': 0.7})
        
        # Phone numbers
        phone_pattern = r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'
        phones = re.findall(phone_pattern, text)
        for phone in phones:
            entities.append({'type': 'phone', 'value': phone, 'confidence': 0.8})
        
        return entities

class SuggestedResponseTask(WorkflowTask):
    """Task 6: Suggested Response Generation - Generate appropriate responses"""
    
    def __init__(self):
        super().__init__(
            task_id="suggested_response",
            name="Response Generation",
            description="Generate suggested responses based on intent and sentiment analysis",
            dependencies=["intent_sentiment_extraction"],
            timeout=20.0
        )
    
    async def execute(self, context: WorkflowContext) -> TaskResult:
        start_time = datetime.now()
        self.logger.info(f"Starting response generation for workflow {context.workflow_id}")
        
        try:
            analysis_result = context.data.get('analysis_result', {})
            
            # Import response generator
            from response_generator import create_sample_response_generator
            response_generator = create_sample_response_generator()
            
            response_result = {
                'primary_response': {},
                'alternative_responses': [],
                'response_metadata': {},
                'generation_strategy': 'intent_based'
            }
            
            # Generate primary response
            if 'intent_classification' in analysis_result:
                intent = analysis_result['intent_classification'].get('predicted_intent', 'general_inquiry')
                confidence = analysis_result['intent_classification'].get('confidence', 0.5)
                sentiment = analysis_result['sentiment_analysis'].get('sentiment_category', 'neutral')
                
                # Get original text for response generation
                processed_data = context.data.get('processed_data', {})
                user_text = processed_data.get('processed_text', 'No text available')
                
                # Generate response based on intent
                primary_response = response_generator.generate_response(user_text, intent, confidence)
                
                # Enhance response with sentiment-aware adjustments
                enhanced_response = self._enhance_response_with_sentiment(
                    primary_response, sentiment, analysis_result['sentiment_analysis']
                )
                
                response_result['primary_response'] = enhanced_response
                
                # Generate alternative responses with different tones
                alternatives = self._generate_alternative_responses(intent, sentiment)
                response_result['alternative_responses'] = alternatives
                
                # Add metadata
                response_result['response_metadata'] = {
                    'intent_used': intent,
                    'sentiment_considered': sentiment,
                    'urgency_level': analysis_result['sentiment_analysis'].get('urgency_level', 'medium'),
                    'customer_mood': analysis_result['sentiment_analysis'].get('customer_mood', 'neutral'),
                    'response_tone': enhanced_response.get('tone', 'professional'),
                    'estimated_resolution_time': enhanced_response.get('estimated_resolution_time', '24 hours')
                }
            
            context.data['response_result'] = response_result
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            self.logger.info(f"Response generation completed: 1 primary + {len(response_result['alternative_responses'])} alternative responses")
            
            return TaskResult(
                task_id=self.task_id,
                status=TaskStatus.COMPLETED,
                data=response_result,
                execution_time=execution_time,
                metadata={
                    'responses_generated': 1 + len(response_result['alternative_responses']),
                    'primary_tone': response_result['response_metadata'].get('response_tone', 'professional'),
                    'intent_addressed': response_result['response_metadata'].get('intent_used', 'unknown')
                }
            )
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            error_msg = f"Response generation failed: {str(e)}"
            self.logger.error(error_msg)
            
            return TaskResult(
                task_id=self.task_id,
                status=TaskStatus.FAILED,
                error=error_msg,
                execution_time=execution_time
            )
    
    def _enhance_response_with_sentiment(self, response: Dict[str, Any], sentiment: str, sentiment_data: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance response based on sentiment analysis"""
        enhanced = response.copy()
        
        urgency = sentiment_data.get('urgency_level', 'medium')
        customer_mood = sentiment_data.get('customer_mood', 'neutral')
        
        # Adjust tone based on sentiment
        if sentiment == 'negative' or customer_mood in ['frustrated', 'angry']:
            enhanced['tone'] = 'empathetic'
            enhanced['priority'] = 'high'
            # Add empathetic opening
            if 'response' in enhanced:
                enhanced['response'] = f"I sincerely apologize for the frustration this has caused. {enhanced['response']}"
        
        elif sentiment == 'positive':
            enhanced['tone'] = 'friendly'
            enhanced['priority'] = 'normal'
        
        else:
            enhanced['tone'] = 'professional'
            enhanced['priority'] = 'normal'
        
        # Adjust urgency handling
        if urgency in ['high', 'critical']:
            enhanced['estimated_resolution_time'] = '4 hours'
            enhanced['escalation_recommended'] = True
        elif urgency == 'medium':
            enhanced['estimated_resolution_time'] = '24 hours'
        else:
            enhanced['estimated_resolution_time'] = '48 hours'
        
        return enhanced
    
    def _generate_alternative_responses(self, intent: str, sentiment: str) -> List[Dict[str, Any]]:
        """Generate alternative response options"""
        alternatives = []
        
        base_responses = {
            'refund_request': [
                "I'll be happy to process your refund request right away.",
                "I understand you'd like a refund. Let me help you with that immediately.",
                "Thank you for reaching out about your refund. I'll take care of this for you."
            ],
            'delivery_issue': [
                "I apologize for the delivery delay. Let me track your package and provide an update.",
                "I'm sorry your order hasn't arrived as expected. I'll investigate this right away.",
                "Thank you for contacting us about your delivery. Let me help resolve this issue."
            ],
            'account_issue': [
                "I'll help you resolve this account issue immediately.",
                "Let me assist you with your account access problem right away.",
                "I understand your account concern. I'm here to help fix this."
            ]
        }
        
        responses = base_responses.get(intent, ["I'm here to help you with your inquiry."])
        
        tones = ['professional', 'friendly', 'empathetic']
        
        for i, response_text in enumerate(responses[:3]):
            tone = tones[i % len(tones)]
            alternatives.append({
                'response': response_text,
                'tone': tone,
                'variant': f'alternative_{i+1}',
                'confidence': 0.8 - (i * 0.1)
            })
        
        return alternatives

class QueryHandlingTask(WorkflowTask):
    """Task 7: Query Handling - Handle follow-up queries and provide comprehensive results"""
    
    def __init__(self):
        super().__init__(
            task_id="query_handling",
            name="Query Processing",
            description="Process queries against processed data and provide comprehensive results",
            dependencies=["embedding", "suggested_response"],
            timeout=30.0
        )
    
    async def execute(self, context: WorkflowContext) -> TaskResult:
        start_time = datetime.now()
        self.logger.info(f"Starting query handling for workflow {context.workflow_id}")
        
        try:
            # Gather all processed data
            embedding_result = context.data.get('embedding_result', {})
            analysis_result = context.data.get('analysis_result', {})
            response_result = context.data.get('response_result', {})
            
            query_result = {
                'query_capabilities': [],
                'searchable_content': [],
                'response_ready': False,
                'knowledge_base_integration': {},
                'final_output': {}
            }
            
            # Prepare searchable content
            embeddings = embedding_result.get('embeddings', [])
            for embedding_info in embeddings:
                searchable_item = {
                    'content_id': embedding_info['chunk_id'],
                    'text_preview': embedding_info['text_preview'],
                    'source': embedding_info['source'],
                    'embedding_available': True,
                    'searchable': True
                }
                query_result['searchable_content'].append(searchable_item)
            
            # Define query capabilities
            query_result['query_capabilities'] = [
                'semantic_search',
                'intent_based_filtering', 
                'sentiment_based_routing',
                'entity_extraction',
                'response_generation',
                'similarity_matching'
            ]
            
            # Prepare knowledge base integration
            query_result['knowledge_base_integration'] = {
                'total_embeddings': len(embeddings),
                'embedding_dimension': embedding_result.get('embedding_dimension', 384),
                'search_ready': len(embeddings) > 0,
                'content_types': list(set(item['source'] for item in query_result['searchable_content']))
            }
            
            # Create final comprehensive output
            query_result['final_output'] = self._create_final_output(context)
            query_result['response_ready'] = True
            
            context.data['query_result'] = query_result
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            self.logger.info(f"Query handling completed: {len(query_result['query_capabilities'])} capabilities enabled")
            
            return TaskResult(
                task_id=self.task_id,
                status=TaskStatus.COMPLETED,
                data=query_result,
                execution_time=execution_time,
                metadata={
                    'searchable_items': len(query_result['searchable_content']),
                    'capabilities_enabled': len(query_result['query_capabilities']),
                    'response_ready': query_result['response_ready']
                }
            )
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            error_msg = f"Query handling failed: {str(e)}"
            self.logger.error(error_msg)
            
            return TaskResult(
                task_id=self.task_id,
                status=TaskStatus.FAILED,
                error=error_msg,
                execution_time=execution_time
            )
    
    def _create_final_output(self, context: WorkflowContext) -> Dict[str, Any]:
        """Create comprehensive final output combining all workflow results"""
        
        # Extract data from all workflow steps
        ingested_data = context.data.get('ingested_data', {})
        processed_data = context.data.get('processed_data', {})
        chunking_result = context.data.get('chunking_result', {})
        embedding_result = context.data.get('embedding_result', {})
        analysis_result = context.data.get('analysis_result', {})
        response_result = context.data.get('response_result', {})
        
        final_output = {
            'workflow_id': context.workflow_id,
            'session_id': context.session_id,
            'processing_summary': {
                'total_tasks_completed': len([r for r in context.results.values() if r.status == TaskStatus.COMPLETED]),
                'total_execution_time': sum(r.execution_time for r in context.results.values()),
                'data_processed': {
                    'records_count': ingested_data.get('records_count', 0),
                    'chunks_created': chunking_result.get('total_chunks', 0),
                    'embeddings_generated': embedding_result.get('total_embeddings', 0)
                }
            },
            'analysis_results': {
                'sentiment': analysis_result.get('sentiment_analysis', {}),
                'intent': analysis_result.get('intent_classification', {}),
                'entities': analysis_result.get('extracted_entities', []),
                'confidence_scores': analysis_result.get('confidence_scores', {})
            },
            'response_recommendations': {
                'primary_response': response_result.get('primary_response', {}),
                'alternatives': response_result.get('alternative_responses', []),
                'metadata': response_result.get('response_metadata', {})
            },
            'workflow_metadata': {
                'created_at': context.created_at.isoformat(),
                'completed_at': datetime.now().isoformat(),
                'quality_scores': {
                    'text_quality': processed_data.get('quality_score', 0.0),
                    'analysis_confidence': analysis_result.get('confidence_scores', {}).get('overall_confidence', 0.0)
                }
            }
        }
        
        return final_output


class SupportTriageWorkflow:
    """
    Main workflow orchestrator for the Support Triage system
    Manages the execution of all workflow tasks in the correct order
    """
    
    def __init__(self, workflow_id: str = None, session_id: str = None):
        self.workflow_id = workflow_id or f"workflow_{uuid.uuid4().hex[:8]}"
        self.session_id = session_id or f"session_{uuid.uuid4().hex[:8]}"
        self.logger = logging.getLogger(__name__)
        
        # Initialize all workflow tasks
        self.tasks = {
            "log_ingestion": LogIngestionTask(),
            "preprocessing": PreprocessingTask(),
            "chunking": ChunkingTask(),
            "embedding": EmbeddingTask(),
            "intent_sentiment_extraction": IntentSentimentExtractionTask(),
            "suggested_response": SuggestedResponseTask(),
            "query_handling": QueryHandlingTask()
        }
        
        # Define task execution order based on dependencies
        self.execution_order = [
            "log_ingestion",
            "preprocessing", 
            "chunking",
            "embedding",
            "intent_sentiment_extraction",
            "suggested_response",
            "query_handling"
        ]
        
        self.context = None
        self.status = "initialized"
        
    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the complete workflow
        
        Args:
            input_data: Input data containing ticket text, file data, etc.
            
        Returns:
            Complete workflow results including all task outputs
        """
        self.logger.info(f"Starting workflow execution: {self.workflow_id}")
        
        # Initialize workflow context
        self.context = WorkflowContext(
            workflow_id=self.workflow_id,
            session_id=self.session_id,
            data={'input': input_data}
        )
        
        self.status = "running"
        workflow_start_time = datetime.now()
        
        try:
            # Execute tasks in order
            for task_id in self.execution_order:
                if task_id not in self.tasks:
                    self.logger.error(f"Task {task_id} not found in workflow")
                    continue
                
                task = self.tasks[task_id]
                
                # Check dependencies
                if not task.validate_dependencies(self.context):
                    self.logger.error(f"Dependencies not satisfied for task {task_id}")
                    result = TaskResult(
                        task_id=task_id,
                        status=TaskStatus.FAILED,
                        error="Dependencies not satisfied"
                    )
                    self.context.results[task_id] = result
                    continue
                
                # Check if task should be skipped
                if task.should_skip(self.context):
                    self.logger.info(f"Skipping task {task_id}")
                    result = TaskResult(
                        task_id=task_id,
                        status=TaskStatus.SKIPPED
                    )
                    self.context.results[task_id] = result
                    continue
                
                # Execute task
                self.logger.info(f"Executing task: {task_id}")
                
                try:
                    # Apply timeout if specified
                    if task.timeout:
                        result = await asyncio.wait_for(
                            task.execute(self.context),
                            timeout=task.timeout
                        )
                    else:
                        result = await task.execute(self.context)
                    
                    self.context.results[task_id] = result
                    
                    if result.status == TaskStatus.FAILED:
                        self.logger.error(f"Task {task_id} failed: {result.error}")
                        # Decide whether to continue or stop workflow
                        if task.priority == TaskPriority.CRITICAL:
                            self.logger.error("Critical task failed, stopping workflow")
                            break
                    else:
                        self.logger.info(f"Task {task_id} completed successfully")
                
                except asyncio.TimeoutError:
                    self.logger.error(f"Task {task_id} timed out after {task.timeout} seconds")
                    result = TaskResult(
                        task_id=task_id,
                        status=TaskStatus.FAILED,
                        error=f"Task timed out after {task.timeout} seconds"
                    )
                    self.context.results[task_id] = result
                
                except Exception as e:
                    self.logger.error(f"Unexpected error in task {task_id}: {str(e)}")
                    result = TaskResult(
                        task_id=task_id,
                        status=TaskStatus.FAILED,
                        error=f"Unexpected error: {str(e)}"
                    )
                    self.context.results[task_id] = result
            
            # Calculate workflow statistics
            workflow_end_time = datetime.now()
            total_execution_time = (workflow_end_time - workflow_start_time).total_seconds()
            
            completed_tasks = sum(1 for r in self.context.results.values() if r.status == TaskStatus.COMPLETED)
            failed_tasks = sum(1 for r in self.context.results.values() if r.status == TaskStatus.FAILED)
            
            # Determine overall workflow status
            if failed_tasks == 0:
                self.status = "completed"
            elif completed_tasks > failed_tasks:
                self.status = "completed_with_errors"
            else:
                self.status = "failed"
            
            # Create comprehensive workflow result
            workflow_result = {
                'workflow_id': self.workflow_id,
                'session_id': self.session_id,
                'status': self.status,
                'execution_summary': {
                    'total_tasks': len(self.execution_order),
                    'completed_tasks': completed_tasks,
                    'failed_tasks': failed_tasks,
                    'skipped_tasks': sum(1 for r in self.context.results.values() if r.status == TaskStatus.SKIPPED),
                    'total_execution_time': total_execution_time,
                    'started_at': workflow_start_time.isoformat(),
                    'completed_at': workflow_end_time.isoformat()
                },
                'task_results': {task_id: {
                    'status': result.status.value,
                    'execution_time': result.execution_time,
                    'error': result.error,
                    'metadata': result.metadata
                } for task_id, result in self.context.results.items()},
                'final_output': self.context.data.get('query_result', {}).get('final_output', {}),
                'workflow_data': {
                    'processing_steps_completed': list(self.context.results.keys()),
                    'data_artifacts': {
                        'ingested_data': self.context.data.get('ingested_data', {}),
                        'embeddings_count': self.context.data.get('embedding_result', {}).get('total_embeddings', 0),
                        'chunks_created': self.context.data.get('chunking_result', {}).get('total_chunks', 0),
                        'analysis_confidence': self.context.data.get('analysis_result', {}).get('confidence_scores', {}).get('overall_confidence', 0.0)
                    }
                }
            }
            
            self.logger.info(f"Workflow {self.workflow_id} completed with status: {self.status}")
            return workflow_result
            
        except Exception as e:
            self.status = "failed"
            self.logger.error(f"Workflow execution failed: {str(e)}")
            
            return {
                'workflow_id': self.workflow_id,
                'session_id': self.session_id,
                'status': 'failed',
                'error': str(e),
                'execution_summary': {
                    'total_tasks': len(self.execution_order),
                    'completed_tasks': sum(1 for r in (self.context.results.values() if self.context else []) if r.status == TaskStatus.COMPLETED),
                    'failed_tasks': sum(1 for r in (self.context.results.values() if self.context else []) if r.status == TaskStatus.FAILED),
                    'total_execution_time': (datetime.now() - workflow_start_time).total_seconds()
                },
                'final_output': {}
            }
    
    def get_status(self) -> Dict[str, Any]:
        """Get current workflow status"""
        if not self.context:
            return {
                'workflow_id': self.workflow_id,
                'status': self.status,
                'tasks_status': 'not_started'
            }
        
        return {
            'workflow_id': self.workflow_id,
            'session_id': self.session_id,
            'status': self.status,
            'tasks_status': {
                task_id: result.status.value for task_id, result in self.context.results.items()
            },
            'completed_tasks': len([r for r in self.context.results.values() if r.status == TaskStatus.COMPLETED]),
            'total_tasks': len(self.execution_order)
        }
    
    def get_task_details(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific task"""
        if not self.context or task_id not in self.context.results:
            return None
        
        result = self.context.results[task_id]
        task = self.tasks.get(task_id)
        
        return {
            'task_id': task_id,
            'name': task.name if task else 'Unknown',
            'description': task.description if task else '',
            'status': result.status.value,
            'execution_time': result.execution_time,
            'error': result.error,
            'metadata': result.metadata,
            'timestamp': result.timestamp.isoformat()
        }


# Convenience function for easy workflow execution
async def execute_support_triage_workflow(input_data: Dict[str, Any], 
                                        workflow_id: str = None, 
                                        session_id: str = None) -> Dict[str, Any]:
    """
    Convenience function to execute the complete support triage workflow
    
    Args:
        input_data: Input data for the workflow
        workflow_id: Optional workflow ID
        session_id: Optional session ID
        
    Returns:
        Complete workflow results
    """
    workflow = SupportTriageWorkflow(workflow_id, session_id)
    return await workflow.execute(input_data)


# Synchronous wrapper for non-async environments
def execute_support_triage_workflow_sync(input_data: Dict[str, Any], 
                                       workflow_id: str = None, 
                                       session_id: str = None) -> Dict[str, Any]:
    """
    Synchronous wrapper for the workflow execution
    
    Args:
        input_data: Input data for the workflow
        workflow_id: Optional workflow ID
        session_id: Optional session ID
        
    Returns:
        Complete workflow results
    """
    return asyncio.run(execute_support_triage_workflow(input_data, workflow_id, session_id))
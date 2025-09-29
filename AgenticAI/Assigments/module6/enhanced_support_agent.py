#!/usr/bin/env python3
"""
Enhanced Support Agent with Workflow Orchestration
Integrates the SupportTriageWorkflow with the existing SupportTriageAgent
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime

from support_agent import SupportTriageAgent
from workflow_orchestrator import (
    SupportTriageWorkflow, 
    execute_support_triage_workflow_sync,
    execute_support_triage_workflow
)
from persistent_store_integration import PersistentStoreIntegration
from response_suggestions import ResponseSuggestionEngine
from escalation_detector import EscalationDetector
from supervisor_insights import SupervisorInsights
from state_manager import StateManager, EventType
from colored_logging import setup_logging
from pdf_processor import PDFProcessor
from csv_processor import CSVProcessor
from database_manager import DatabaseManager

class EnhancedSupportTriageAgent(SupportTriageAgent):
    """
    Enhanced Support Triage Agent with Workflow Orchestration
    Extends the existing agent with comprehensive workflow capabilities
    """
    
    def __init__(self, name: str = "EnhancedSupportTriageAgent", project_name: str = None, 
                 colored_logs: bool = True, log_level: str = "INFO"):
        # Set up colored logging first
        if colored_logs:
            setup_logging(level=log_level, compact=False)
        
        super().__init__(name)
        self.workflow_enabled = True
        self.workflow_history = []
        
        # Initialize database manager for file storage
        self.database = DatabaseManager()
        
        # Initialize persistent store integration
        self.persistent_store = PersistentStoreIntegration(project_name or "support_system")
        self.persistent_enabled = True
        
        # Initialize state management
        self.state_manager = StateManager(
            storage_dir=f"session_data/{project_name or 'support_system'}",
            auto_save=True
        )
        self.state_enabled = True
        
        # Initialize supervisor tools
        self.response_engine = ResponseSuggestionEngine(self.persistent_store)
        self.escalation_detector = EscalationDetector(self.persistent_store)
        self.supervisor_insights = SupervisorInsights(self.persistent_store, self.escalation_detector)
        
        # Enable supervisor features
        self.supervisor_tools_enabled = True
        
        # Initialize file processors
        self.pdf_processor = PDFProcessor()
        self.csv_processor = CSVProcessor()
        
        self.logger.info(f"Initialized {name} with workflow orchestration, persistent storage, supervisor tools, file processors, and state management")
    
    def add_knowledge_from_file(self, file_path: str, file_type: str = None) -> Dict[str, Any]:
        """
        Add knowledge from various file types with comprehensive processing
        
        Args:
            file_path: Path to the file to process
            file_type: Type of file (pdf, csv, txt, json, etc.)
            
        Returns:
            Processing results including success status and statistics
        """
        import os
        import tempfile
        import uuid
        from pathlib import Path
        
        try:
            if not os.path.exists(file_path):
                return {'error': f'File not found: {file_path}'}
            
            # Get file information
            file_info = Path(file_path)
            filename = file_info.name
            file_size = file_info.stat().st_size
            
            # Detect file type if not provided
            if not file_type:
                file_type = file_info.suffix.lower().lstrip('.')
            
            self.logger.info(f"Processing file: {filename} (type: {file_type}, size: {file_size} bytes)")
            
            # Generate unique file ID
            file_id = f"file_{datetime.now().strftime('%Y%m%d%H%M%S')}_{filename}"
            
            # Process file based on type
            processing_result = {'file_id': file_id, 'filename': filename, 'file_type': file_type, 'file_size': file_size}
            
            if file_type == 'pdf':
                result = self._process_pdf_file(file_path, file_id, filename)
                processing_result.update(result)
                
            elif file_type == 'csv':
                result = self._process_csv_file(file_path, file_id, filename)
                processing_result.update(result)
                
            elif file_type in ['txt', 'md']:
                result = self._process_text_file(file_path, file_id, filename)
                processing_result.update(result)
                
            elif file_type == 'json':
                result = self._process_json_file(file_path, file_id, filename)
                processing_result.update(result)
                
            elif file_type in ['docx', 'doc']:
                result = self._process_document_file(file_path, file_id, filename)
                processing_result.update(result)
                
            else:
                return {'error': f'Unsupported file type: {file_type}'}
            
            self.logger.info(f"Successfully processed {filename}: {processing_result.get('sections_created', 0)} sections created")
            return processing_result
            
        except Exception as e:
            error_msg = f"Error processing file {filename}: {str(e)}"
            self.logger.error(error_msg)
            return {'error': error_msg}
    
    def _process_pdf_file(self, file_path: str, file_id: str, filename: str) -> Dict[str, Any]:
        """Process PDF file with chunking and embedding"""
        try:
            # Process PDF using the PDF processor
            document_analysis = self.pdf_processor.process_pdf(file_path)
            
            sections_created = 0
            chunks_created = 0
            
            if document_analysis.sections:
                # Prepare sections for database storage
                content_sections = []
                for i, section in enumerate(document_analysis.sections):
                    content_sections.append({
                        'content': section.content,
                        'title': section.title,
                        'section_number': i + 1,
                        'category': 'Document',
                        'tags': 'pdf,document,section',
                        'size': len(section.content)
                    })
                    sections_created += 1
                
                # Store all sections in database at once
                if self.database:
                    self.database.store_file_content(
                        file_id=file_id,
                        file_name=filename,
                        file_type='pdf',
                        content_sections=content_sections,
                        file_path=file_path
                    )
                
                # Process each section for vector storage
                for i, section in enumerate(document_analysis.sections):
                    # Chunk the section content for vector storage
                    chunks = self._chunk_text(section.content, max_chunk_size=1000)
                    for chunk_idx, chunk in enumerate(chunks):
                        # Add to vector store with metadata
                        chunk_id = f"chunk_{file_id}_section_{i}_chunk_{chunk_idx}"
                        metadata = {
                            'file_id': file_id,
                            'file_name': filename,
                            'file_type': 'pdf',
                            'section_title': section.title,
                            'section_number': i + 1,
                            'chunk_index': chunk_idx,
                            'source_type': 'uploaded_file',
                            'created_at': datetime.now().isoformat()
                        }
                        
                        # Add to vector store using the simple_vector_response module
                        self._add_to_vector_store(chunk_id, chunk, metadata)
                        chunks_created += 1
            
            return {
                'success': True,
                'message': f'PDF processed successfully',
                'sections_created': sections_created,
                'chunks_created': chunks_created,
                'pages_processed': len(document_analysis.pages),
                'total_text_length': len(document_analysis.full_text)
            }
            
        except Exception as e:
            return {'error': f'PDF processing failed: {str(e)}'}
    
    def _process_text_file(self, file_path: str, file_id: str, filename: str) -> Dict[str, Any]:
        """Process text file with chunking and embedding"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            sections_created = 0
            chunks_created = 0
            
            # Store main content in database
            if self.database:
                content_sections = [{
                    'content': content,
                    'title': 'Text Document Section 1',
                    'section_number': 1,
                    'category': 'Document',
                    'tags': 'text,document,section',
                    'size': len(content)
                }]
                
                self.database.store_file_content(
                    file_id=file_id,
                    file_name=filename,
                    file_type='txt',
                    content_sections=content_sections,
                    file_path=file_path
                )
                sections_created += 1
            
            # Chunk content for vector storage
            chunks = self._chunk_text(content, max_chunk_size=1000)
            for chunk_idx, chunk in enumerate(chunks):
                chunk_id = f"chunk_{file_id}_chunk_{chunk_idx}"
                metadata = {
                    'file_id': file_id,
                    'file_name': filename,
                    'file_type': 'txt',
                    'chunk_index': chunk_idx,
                    'source_type': 'uploaded_file',
                    'created_at': datetime.now().isoformat()
                }
                
                self._add_to_vector_store(chunk_id, chunk, metadata)
                chunks_created += 1
            
            return {
                'success': True,
                'message': 'Text file processed successfully',
                'sections_created': sections_created,
                'chunks_created': chunks_created,
                'total_text_length': len(content)
            }
            
        except Exception as e:
            return {'error': f'Text file processing failed: {str(e)}'}
    
    def _process_csv_file(self, file_path: str, file_id: str, filename: str) -> Dict[str, Any]:
        """Process CSV file and extract content"""
        try:
            # Use CSV processor
            self.csv_processor.load_csv(file_path)
            
            # Generate summary text
            summary_text = f"CSV Summary: {len(self.csv_processor.df)} rows, {len(self.csv_processor.df.columns)} columns. Columns: {', '.join(self.csv_processor.df.columns.tolist())}"
            
            sections_created = 0
            chunks_created = 0
            
            # Store in database
            if self.database:
                content_sections = [{
                    'content': summary_text,
                    'title': 'CSV Data Summary',
                    'section_number': 1,
                    'category': 'Data',
                    'tags': 'csv,data,structured',
                    'size': len(summary_text)
                }]
                
                self.database.store_file_content(
                    file_id=file_id,
                    file_name=filename,
                    file_type='csv',
                    content_sections=content_sections,
                    file_path=file_path
                )
                sections_created += 1
            
            # Add summary to vector store
            chunk_id = f"chunk_{file_id}_summary"
            metadata = {
                'file_id': file_id,
                'file_name': filename,
                'file_type': 'csv',
                'chunk_index': 0,
                'source_type': 'uploaded_file',
                'created_at': datetime.now().isoformat()
            }
            
            self._add_to_vector_store(chunk_id, summary_text, metadata)
            chunks_created += 1
            
            return {
                'success': True,
                'message': 'CSV file processed successfully',
                'sections_created': sections_created,
                'chunks_created': chunks_created,
                'rows_processed': len(self.csv_processor.df),
                'columns_processed': len(self.csv_processor.df.columns)
            }
            
        except Exception as e:
            return {'error': f'CSV processing failed: {str(e)}'}
    
    def _process_json_file(self, file_path: str, file_id: str, filename: str) -> Dict[str, Any]:
        """Process JSON file"""
        try:
            import json
            
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Convert JSON to text representation
            content = json.dumps(data, indent=2)
            
            sections_created = 0
            chunks_created = 0
            
            # Store in database
            if self.database:
                content_sections = [{
                    'content': content,
                    'title': 'JSON Data',
                    'section_number': 1,
                    'category': 'Data',
                    'tags': 'json,data,structured',
                    'size': len(content)
                }]
                
                self.database.store_file_content(
                    file_id=file_id,
                    file_name=filename,
                    file_type='json',
                    content_sections=content_sections,
                    file_path=file_path
                )
                sections_created += 1
            
            # Add to vector store
            chunk_id = f"chunk_{file_id}_json"
            metadata = {
                'file_id': file_id,
                'file_name': filename,
                'file_type': 'json',
                'chunk_index': 0,
                'source_type': 'uploaded_file',
                'created_at': datetime.now().isoformat()
            }
            
            self._add_to_vector_store(chunk_id, content, metadata)
            chunks_created += 1
            
            return {
                'success': True,
                'message': 'JSON file processed successfully',
                'sections_created': sections_created,
                'chunks_created': chunks_created,
                'total_text_length': len(content)
            }
            
        except Exception as e:
            return {'error': f'JSON processing failed: {str(e)}'}
    
    def _process_document_file(self, file_path: str, file_id: str, filename: str) -> Dict[str, Any]:
        """Process DOCX files (placeholder - requires python-docx)"""
        try:
            # For now, return a placeholder - would need python-docx library
            return {'error': 'DOCX processing not yet implemented - please convert to PDF or TXT'}
        except Exception as e:
            return {'error': f'Document processing failed: {str(e)}'}
    
    def _chunk_text(self, text: str, max_chunk_size: int = 1000, overlap: int = 100) -> List[str]:
        """Split text into chunks for vector storage"""
        if len(text) <= max_chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + max_chunk_size
            
            # Try to break at sentence boundaries
            if end < len(text):
                # Look for sentence endings near the break point
                for i in range(end, max(start + max_chunk_size // 2, end - 200), -1):
                    if text[i] in '.!?':
                        end = i + 1
                        break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = max(start + max_chunk_size - overlap, end)
            
        return chunks
    
    def _add_to_vector_store(self, doc_id: str, content: str, metadata: Dict[str, Any]):
        """Add content to vector store with proper embedding generation"""
        try:
            # Import and use the vector store
            from simple_vector_response import VectorStore
            
            # Use the inherited vector store from parent class
            if not hasattr(self, 'vector_store'):
                self.vector_store = VectorStore()
            
            # Generate embedding for the content
            embedding = self.vector_store.generate_embeddings([content])[0]
            
            # Add to in-memory store with proper structure (same as working TXT vectors)
            self.vector_store.in_memory_store[doc_id] = {
                'embedding': embedding,
                'text': content,
                'metadata': metadata
            }
            
            # Save to persistent storage
            self.vector_store._save_persistent_data()
            
            self.logger.debug(f"Added to vector store: {doc_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to add to vector store: {str(e)}")
    
    def create_session(self, user_id: str, context: Dict[str, Any] = None) -> str:
        """
        Create a new session for state tracking
        
        Args:
            user_id: User identifier
            context: Initial session context
            
        Returns:
            Session ID
        """
        if not self.state_enabled:
            return "state_disabled"
        
        session_id = self.state_manager.create_session(user_id, context)
        self.logger.info(f"Created session {session_id} for user {user_id}")
        return session_id
    
    def set_session(self, session_id: str) -> bool:
        """
        Set the current active session
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if session was set successfully
        """
        if not self.state_enabled:
            return False
        
        return self.state_manager.set_current_session(session_id)
    
    def get_session_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get current session history
        
        Args:
            limit: Maximum number of events to return
            
        Returns:
            List of session events
        """
        if not self.state_enabled:
            return []
        
        events = self.state_manager.get_session_history(limit=limit)
        return [event.to_dict() for event in events]
    
    def get_session_statistics(self) -> Dict[str, Any]:
        """
        Get current session statistics
        
        Returns:
            Session statistics
        """
        if not self.state_enabled:
            return {}
        
        return self.state_manager.get_session_statistics()
    
    def export_session_data(self, include_events: bool = True) -> Dict[str, Any]:
        """
        Export current session data
        
        Args:
            include_events: Whether to include all events
            
        Returns:
            Complete session data
        """
        if not self.state_enabled:
            return {}
        
        return self.state_manager.export_session(include_events=include_events)
    
    def process_support_ticket_with_workflow(self, 
                                           ticket_text: str, 
                                           ticket_id: str = None,
                                           use_async: bool = False) -> Dict[str, Any]:
        """
        Process support ticket using the complete workflow orchestration
        
        Args:
            ticket_text: Customer message text
            ticket_id: Optional ticket ID
            use_async: Whether to use async execution
            
        Returns:
            Complete workflow results with enhanced analysis
        """
        self.logger.info(f"Processing ticket with workflow: {ticket_id or 'AUTO-GENERATED'}")
        
        # Log ticket processing start
        start_time = datetime.now()
        if self.state_enabled:
            self.state_manager.log_event(
                EventType.TICKET_PROCESSED,
                {
                    'ticket_id': ticket_id or 'AUTO-GENERATED',
                    'ticket_text_length': len(ticket_text),
                    'use_async': use_async,
                    'processing_start': start_time.isoformat()
                }
            )
        
        # Prepare input data for workflow
        input_data = {
            'ticket_text': ticket_text,
            'ticket_id': ticket_id,
            'source_type': 'ticket',
            'channel': 'support_agent',
            'priority': 'normal',
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            if use_async:
                # Use async execution
                workflow_result = asyncio.run(execute_support_triage_workflow(input_data))
            else:
                # Use sync execution
                workflow_result = execute_support_triage_workflow_sync(input_data)
            
            # Store workflow in history
            self.workflow_history.append({
                'workflow_id': workflow_result['workflow_id'],
                'ticket_id': ticket_id,
                'timestamp': datetime.now(),
                'status': workflow_result['status']
            })
            
            # Convert workflow result to agent format for compatibility
            agent_result = self._convert_workflow_to_agent_format(workflow_result, ticket_text, ticket_id)
            
            # Store ticket in persistent store if enabled
            if self.persistent_enabled:
                try:
                    complaint_id = self.persistent_store.store_processed_ticket(agent_result)
                    agent_result['persistent_complaint_id'] = complaint_id
                    self.logger.info(f"Stored ticket in persistent store: {complaint_id}")
                except Exception as e:
                    self.logger.error(f"Failed to store ticket in persistent store: {str(e)}")
                    agent_result['persistent_storage_error'] = str(e)
            
            # Log successful completion
            if self.state_enabled:
                end_time = datetime.now()
                processing_time_ms = int((end_time - start_time).total_seconds() * 1000)
                self.state_manager.log_event(
                    EventType.SYSTEM_EVENT,
                    {
                        'action': 'ticket_processing_completed',
                        'ticket_id': ticket_id or 'AUTO-GENERATED',
                        'status': workflow_result['status'],
                        'workflow_id': workflow_result['workflow_id'],
                        'persistent_stored': 'persistent_complaint_id' in agent_result,
                        'processing_end': end_time.isoformat()
                    },
                    duration_ms=processing_time_ms
                )
            
            # Add explicit routing decision
            routing_decision = self.determine_routing_decision(agent_result)
            agent_result['routing_decision'] = routing_decision
            
            self.logger.info(f"Workflow processing completed: {workflow_result['status']}")
            return agent_result
            
        except Exception as e:
            self.logger.error(f"Workflow processing failed: {str(e)}")
            
            # Fallback to original agent processing
            self.logger.info("Falling back to original agent processing")
            return super().process_support_ticket(ticket_text, ticket_id)
    
    def process_file_with_workflow(self, 
                                 file_content: str, 
                                 file_name: str = "uploaded_file.txt",
                                 file_type: str = "text/plain") -> Dict[str, Any]:
        """
        Process file content using workflow orchestration
        
        Args:
            file_content: File content as string
            file_name: Name of the file
            file_type: MIME type of the file
            
        Returns:
            Workflow processing results
        """
        self.logger.info(f"Processing file with workflow: {file_name}")
        
        input_data = {
            'file_data': {
                'content': file_content,
                'name': file_name,
                'type': file_type
            },
            'source_type': 'file_upload',
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            workflow_result = execute_support_triage_workflow_sync(input_data)
            
            # If workflow completed successfully, also store in knowledge base
            if workflow_result.get('status') == 'completed':
                try:
                    # Save file to temporary location for base class processing
                    import tempfile
                    import os
                    
                    with tempfile.NamedTemporaryFile(mode='w', suffix=f'_{file_name}', delete=False, encoding='utf-8') as temp_file:
                        temp_file.write(file_content)
                        temp_file_path = temp_file.name
                    
                    # Convert MIME type to simple file type for parent class
                    simple_file_type = self._convert_mime_to_file_type(file_type, file_name)
                    
                    # Use parent class method to store in knowledge base
                    kb_result = super().upload_and_process_file(temp_file_path, simple_file_type)
                    
                    # Clean up temporary file
                    os.unlink(temp_file_path)
                    
                    # Merge workflow and knowledge base results
                    if kb_result.get('status') == 'success':
                        workflow_result['knowledge_base_stored'] = True
                        workflow_result['chunks_created'] = kb_result.get('chunks_created', 0)
                        workflow_result['records_processed'] = kb_result.get('records_processed', 0)
                        self.logger.info(f"File {file_name} successfully stored in knowledge base")
                    else:
                        workflow_result['knowledge_base_stored'] = False
                        self.logger.warning(f"Workflow completed but failed to store {file_name} in knowledge base")
                        
                except Exception as kb_error:
                    self.logger.error(f"Error storing file in knowledge base: {str(kb_error)}")
                    workflow_result['knowledge_base_stored'] = False
                    workflow_result['kb_error'] = str(kb_error)
            
            # Store in workflow history
            self.workflow_history.append({
                'workflow_id': workflow_result['workflow_id'],
                'file_name': file_name,
                'timestamp': datetime.now(),
                'status': workflow_result['status'],
                'kb_stored': workflow_result.get('knowledge_base_stored', False)
            })
            
            self.logger.info(f"File workflow processing completed: {workflow_result['status']}")
            return workflow_result
            
        except Exception as e:
            self.logger.error(f"File workflow processing failed: {str(e)}")
            return {
                'status': 'failed',
                'error': str(e),
                'file_name': file_name
            }
    
    def batch_process_with_workflow(self, tickets: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """
        Process multiple tickets using workflow orchestration
        
        Args:
            tickets: List of ticket dictionaries with 'text' and optional 'id'
            
        Returns:
            List of workflow results
        """
        self.logger.info(f"Batch processing {len(tickets)} tickets with workflow")
        
        results = []
        
        for i, ticket in enumerate(tickets):
            ticket_text = ticket.get('text', '')
            ticket_id = ticket.get('id', f'BATCH-{i+1:03d}')
            
            try:
                result = self.process_support_ticket_with_workflow(ticket_text, ticket_id)
                results.append(result)
                
            except Exception as e:
                self.logger.error(f"Error processing batch ticket {ticket_id}: {str(e)}")
                results.append({
                    'ticket_id': ticket_id,
                    'status': 'failed',
                    'error': str(e)
                })
        
        self.logger.info(f"Batch processing completed: {len(results)} results")
        return results
    
    def get_workflow_analytics(self) -> Dict[str, Any]:
        """
        Get analytics about workflow usage and performance
        
        Returns:
            Workflow analytics data
        """
        if not self.workflow_history:
            return {
                'total_workflows': 0,
                'success_rate': 0.0,
                'average_processing_time': 0.0,
                'workflow_types': {}
            }
        
        total_workflows = len(self.workflow_history)
        successful_workflows = sum(1 for w in self.workflow_history if w['status'] == 'completed')
        success_rate = (successful_workflows / total_workflows) * 100
        
        # Count workflow types
        workflow_types = {}
        for workflow in self.workflow_history:
            if 'ticket_id' in workflow:
                workflow_types['ticket_processing'] = workflow_types.get('ticket_processing', 0) + 1
            elif 'file_name' in workflow:
                workflow_types['file_processing'] = workflow_types.get('file_processing', 0) + 1
        
        return {
            'total_workflows': total_workflows,
            'successful_workflows': successful_workflows,
            'failed_workflows': total_workflows - successful_workflows,
            'success_rate': success_rate,
            'workflow_types': workflow_types,
            'last_workflow': self.workflow_history[-1]['timestamp'].isoformat() if self.workflow_history else None
        }
    
    def get_comprehensive_tools_status(self) -> Dict[str, Any]:
        """
        Get status of all tools including workflow capabilities
        
        Returns:
            Comprehensive tools status
        """
        # Get base tools status from parent class
        base_status = super().get_comprehensive_tools_status()
        
        # Add workflow-specific status
        workflow_analytics = self.get_workflow_analytics()
        
        # Get persistent store status
        persistent_status = 'active' if self.persistent_enabled else 'disabled'
        
        base_status.update({
            'workflow_orchestrator': 'active' if self.workflow_enabled else 'disabled',
            'persistent_store': persistent_status,
            'workflow_analytics': workflow_analytics,
            'enhanced_capabilities': [
                'agentic_task_orchestration',
                'async_processing',
                'batch_workflow_processing',
                'file_workflow_processing',
                'workflow_analytics',
                'persistent_complaint_storage',
                'historical_case_analysis',
                'resolution_template_management',
                'cross_session_learning',
                'intelligent_response_suggestions',
                'escalation_detection_and_highlighting',
                'supervisor_insights_dashboard',
                'sla_monitoring_and_alerts',
                'issue_spike_detection',
                'response_quality_evaluation'
            ]
        })
        
        # Add persistent store statistics if available
        if self.persistent_enabled:
            try:
                persistent_insights = self.get_persistent_store_insights()
                if 'project_stats' in persistent_insights:
                    base_status['persistent_store_stats'] = persistent_insights['project_stats']
            except Exception as e:
                self.logger.warning(f"Could not get persistent store stats: {str(e)}")
        
        return base_status
    
    def process_natural_language_query(self, query: str) -> str:
        """
        Enhanced natural language query processing with workflow context
        
        Args:
            query: Natural language query
            
        Returns:
            Response string
        """
        self.logger.info(f"Processing natural language query: {query[:50]}...")
        
        # Check if query is about workflow analytics
        if any(term in query.lower() for term in ['workflow', 'processing', 'analytics', 'performance']):
            analytics = self.get_workflow_analytics()
            
            if analytics['total_workflows'] > 0:
                response = f"Based on workflow analytics: I've processed {analytics['total_workflows']} workflows with a {analytics['success_rate']:.1f}% success rate. "
                
                if 'ticket_processing' in analytics['workflow_types']:
                    response += f"Processed {analytics['workflow_types']['ticket_processing']} support tickets. "
                
                if 'file_processing' in analytics['workflow_types']:
                    response += f"Processed {analytics['workflow_types']['file_processing']} files. "
                
                return response
            else:
                return "No workflow processing history available yet. Upload some data or process tickets to see analytics."
        
        # Use enhanced hybrid search functionality with historical data
        try:
            # Ensure we have some data for testing (adds demo data if empty)
            self.ensure_demo_data_available()
            
            # Extract keywords for hybrid search
            keywords = self._extract_keywords_from_query(query)
            
            # Perform hybrid search combining keyword and semantic search
            search_results = self.enhanced_knowledge_search(query, keywords)
            
            # If we have persistent store enabled, also search historical cases
            historical_context = ""
            if self.persistent_enabled and keywords:
                try:
                    # Extract category from query if possible
                    category = None
                    if any(kw in query.lower() for kw in ['refund', 'return']):
                        category = 'refund'
                    elif any(kw in query.lower() for kw in ['delivery', 'shipping']):
                        category = 'delivery'
                    elif any(kw in query.lower() for kw in ['billing', 'payment', 'charge']):
                        category = 'billing'
                    
                    similar_cases = self.get_similar_historical_cases(query, category, 3)
                    if similar_cases:
                        historical_context = f"\n\n📊 **Historical Context**: Found {len(similar_cases)} similar past cases"
                        if category:
                            historical_context += f" in {category} category"
                        historical_context += ". This suggests this is a recurring issue type."
                        
                        # Add resolution recommendations if available
                        if category:
                            recommended_resolutions = self.get_recommended_resolutions(category, query, 2)
                            if recommended_resolutions:
                                historical_context += f"\n\n💡 **Recommended Resolutions** (based on past effectiveness):"
                                for i, res in enumerate(recommended_resolutions[:2], 1):
                                    effectiveness = res.get('effectiveness_score', 0) * 100
                                    historical_context += f"\n{i}. {res.get('title', 'Resolution')} (Effectiveness: {effectiveness:.0f}%)"
                
                except Exception as e:
                    self.logger.warning(f"Could not get historical context: {str(e)}")
            
            if search_results.get('total_results', 0) > 0:
                response = self._format_search_response(query, search_results)
                return response + historical_context
            else:
                # Try a broader search with just keywords if no results
                if keywords:
                    broader_search = self.enhanced_knowledge_search(' '.join(keywords), keywords, top_k=10)
                    if broader_search.get('total_results', 0) > 0:
                        response = self._format_search_response(query, broader_search, is_broader=True)
                        return response + historical_context
                
                base_response = f"I couldn't find specific data matching your query '{query}'. This might be because:\n• No relevant data has been uploaded yet\n• Try using different keywords (e.g., 'refund', 'delivery', 'billing')\n• Upload support logs or documents first"
                return base_response + historical_context
                
        except Exception as e:
            self.logger.error(f"Error processing natural language query: {str(e)}")
            return f"I encountered an error processing your query: {str(e)}"
    
    def run_analytics_query(self, query_type: str) -> Dict[str, Any]:
        """
        Run predefined analytics queries with workflow data
        
        Args:
            query_type: Type of analytics query
            
        Returns:
            Analytics results
        """
        self.logger.info(f"Running analytics query: {query_type}")
        
        try:
            workflow_analytics = self.get_workflow_analytics()
            
            if query_type == "workflow_performance":
                return {
                    'status': 'success',
                    'summary': f"Workflow Performance: {workflow_analytics['success_rate']:.1f}% success rate across {workflow_analytics['total_workflows']} workflows",
                    'data': workflow_analytics
                }
            
            elif query_type == "processing_volume":
                return {
                    'status': 'success',
                    'summary': f"Processing Volume: {workflow_analytics['total_workflows']} total workflows processed",
                    'data': {
                        'total_workflows': workflow_analytics['total_workflows'],
                        'workflow_types': workflow_analytics['workflow_types']
                    }
                }
            
            else:
                # Fallback to demo analytics for other query types
                from streamlit_app import generate_demo_analytics
                return generate_demo_analytics(query_type, f"Analytics for {query_type}")
                
        except Exception as e:
            self.logger.error(f"Error running analytics query: {str(e)}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def _convert_workflow_to_agent_format(self, 
                                        workflow_result: Dict[str, Any], 
                                        ticket_text: str, 
                                        ticket_id: str) -> Dict[str, Any]:
        """
        Convert workflow result to original agent format for compatibility
        
        Args:
            workflow_result: Result from workflow execution
            ticket_text: Original ticket text
            ticket_id: Ticket ID
            
        Returns:
            Agent-formatted result
        """
        final_output = workflow_result.get('final_output', {})
        analysis_results = final_output.get('analysis_results', {})
        response_recommendations = final_output.get('response_recommendations', {})
        
        # Extract sentiment analysis
        sentiment_analysis = analysis_results.get('sentiment', {})
        
        # Extract intent classification
        intent_classification = analysis_results.get('intent', {})
        
        # Extract response generation
        primary_response = response_recommendations.get('primary_response', {})
        alternatives = response_recommendations.get('alternatives', [])
        
        # Create agent-compatible result
        agent_result = {
            'ticket_id': ticket_id or workflow_result.get('workflow_id', 'UNKNOWN'),
            'original_text': ticket_text,
            'processing_timestamp': datetime.now().isoformat(),
            'processing_time_ms': workflow_result.get('execution_summary', {}).get('total_execution_time', 0) * 1000,
            'workflow_id': workflow_result.get('workflow_id'),
            'workflow_status': workflow_result.get('status'),
            
            # Core analysis results
            'sentiment_analysis': sentiment_analysis,
            'intent_classification': intent_classification,
            'response_generation': {
                'suggested_response': primary_response.get('response', 'No response generated'),
                'response_tone': primary_response.get('tone', 'professional'),
                'estimated_resolution_time': primary_response.get('estimated_resolution_time', '24 hours'),
                'confidence': primary_response.get('confidence', 0.0),
                'alternative_responses': alternatives
            },
            
            # Enhanced workflow data
            'comprehensive_analysis': {
                'tools_used': ['workflow_orchestrator'] + workflow_result.get('workflow_data', {}).get('processing_steps_completed', []),
                'chunks_created': workflow_result.get('workflow_data', {}).get('data_artifacts', {}).get('chunks_created', 0),
                'embeddings_generated': workflow_result.get('workflow_data', {}).get('data_artifacts', {}).get('embeddings_count', 0),
                'workflow_execution_time': workflow_result.get('execution_summary', {}).get('total_execution_time', 0)
            },
            
            # Entities if available
            'extracted_entities': analysis_results.get('entities', []),
            
            # Workflow-specific metadata
            'workflow_metadata': {
                'tasks_completed': workflow_result.get('execution_summary', {}).get('completed_tasks', 0),
                'tasks_failed': workflow_result.get('execution_summary', {}).get('failed_tasks', 0),
                'processing_quality': final_output.get('workflow_metadata', {}).get('quality_scores', {})
            }
        }
        
        return agent_result
    
    def _convert_mime_to_file_type(self, mime_type: str, file_name: str) -> str:
        """
        Convert MIME type to simple file type for parent class compatibility
        
        Args:
            mime_type: MIME type (e.g., 'text/csv', 'application/pdf')
            file_name: File name for extension fallback
            
        Returns:
            Simple file type string ('csv', 'pdf', 'txt')
        """
        # MIME type mappings
        mime_mappings = {
            'text/csv': 'csv',
            'application/csv': 'csv',
            'text/comma-separated-values': 'csv',
            'application/pdf': 'pdf',
            'text/plain': 'txt',
            'text/html': 'txt',
            'application/json': 'txt',
            'text/markdown': 'txt'
        }
        
        # Check MIME type first
        if mime_type in mime_mappings:
            return mime_mappings[mime_type]
        
        # Fallback to file extension
        if '.' in file_name:
            extension = file_name.split('.')[-1].lower()
            if extension in ['csv']:
                return 'csv'
            elif extension in ['pdf']:
                return 'pdf'
            elif extension in ['txt', 'text', 'md', 'json']:
                return 'txt'
        
        # Default to text
        return 'txt'
    
    def _extract_keywords_from_query(self, query: str) -> List[str]:
        """
        Extract important keywords from query for hybrid search
        
        Args:
            query: Natural language query
            
        Returns:
            List of extracted keywords
        """
        import re
        
        # Common support keywords
        support_keywords = [
            'refund', 'delivery', 'shipping', 'billing', 'payment', 'charge',
            'order', 'account', 'login', 'password', 'cancel', 'return',
            'exchange', 'defective', 'broken', 'quality', 'complaint',
            'technical', 'support', 'help', 'issue', 'problem', 'error',
            'urgent', 'escalate', 'manager', 'supervisor'
        ]
        
        # Extract keywords from query
        words = re.findall(r'\b\w+\b', query.lower())
        
        # Find matching support keywords
        found_keywords = [word for word in words if word in support_keywords]
        
        # Add important words (longer than 3 characters)
        other_keywords = [word for word in words if len(word) > 3 and word not in found_keywords]
        
        # Combine and limit
        all_keywords = found_keywords + other_keywords[:3]
        
        return all_keywords[:5]  # Limit to 5 keywords
    
    def enhanced_knowledge_search(self, query: str, keywords: List[str] = None, top_k: int = 5) -> Dict[str, Any]:
        """
        Enhanced hybrid search combining keyword and semantic search with source references
        
        Args:
            query: Natural language query
            keywords: List of keywords for keyword search
            top_k: Number of results to return
            
        Returns:
            Enhanced search results with source references
        """
        try:
            self.logger.info(f"Enhanced search for: {query[:50]}... Keywords: {keywords}")
            
            # 1. Semantic search using parent class method
            semantic_results = super().search_knowledge_base(query, search_type='semantic', top_k=top_k)
            
            # 2. Keyword search if keywords provided
            keyword_results = {'similar_tickets': []}
            if keywords:
                for keyword in keywords:
                    kw_results = super().search_knowledge_base(keyword, search_type='hybrid', top_k=3)
                    keyword_results['similar_tickets'].extend(kw_results.get('similar_tickets', []))
            
            # 3. Combine and deduplicate results
            all_results = []
            seen_content = set()
            
            # Process semantic results first (higher priority)
            # Parent class returns results in 'similar_tickets' field, not 'results'
            for result in semantic_results.get('similar_tickets', []):
                content_key = str(result.get('content', ''))[:100]
                if content_key not in seen_content:
                    result['search_type'] = 'semantic'
                    result['relevance_boost'] = 1.2  # Boost semantic results
                    all_results.append(self._add_source_references(result))
                    seen_content.add(content_key)
            
            # Process keyword results
            # Also check 'similar_tickets' for keyword results
            for result in keyword_results.get('similar_tickets', []):
                content_key = str(result.get('content', ''))[:100]
                if content_key not in seen_content:
                    result['search_type'] = 'keyword'
                    result['relevance_boost'] = 1.0
                    all_results.append(self._add_source_references(result))
                    seen_content.add(content_key)
            
            # 4. Score and rank results
            scored_results = self._score_search_results(all_results, query, keywords)
            
            # 5. Sort by final score and limit
            scored_results.sort(key=lambda x: x.get('final_score', 0), reverse=True)
            final_results = scored_results[:top_k]
            
            return {
                'query': query,
                'keywords': keywords,
                'search_type': 'enhanced_hybrid',
                'total_results': len(final_results),
                'semantic_count': len([r for r in final_results if r.get('search_type') == 'semantic']),
                'keyword_count': len([r for r in final_results if r.get('search_type') == 'keyword']),
                'results': final_results,
                'search_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Enhanced search failed: {str(e)}")
            return {
                'query': query,
                'total_results': 0,
                'results': [],
                'error': str(e)
            }
    
    def _add_source_references(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add source references to search results
        
        Args:
            result: Search result dictionary
            
        Returns:
            Result with source references added
        """
        # Extract source information from metadata
        metadata = result.get('metadata', {})
        
        source_ref = {
            'file_name': metadata.get('file_name', 'Unknown Source'),
            'file_type': metadata.get('file_type', 'Unknown'),
            'chunk_id': metadata.get('chunk_id', 'N/A'),
            'upload_date': metadata.get('created_at', 'Unknown'),
            'source_type': metadata.get('source_type', 'Document')
        }
        
        # Create readable source reference
        if source_ref['file_name'] != 'Unknown Source':
            if source_ref['chunk_id'] != 'N/A':
                source_ref['reference'] = f"{source_ref['file_name']} (Section {source_ref['chunk_id']})"
            else:
                source_ref['reference'] = source_ref['file_name']
        else:
            source_ref['reference'] = f"Internal Database Record"
        
        result['source'] = source_ref
        return result
    
    def _score_search_results(self, results: List[Dict[str, Any]], query: str, keywords: List[str]) -> List[Dict[str, Any]]:
        """
        Score search results based on relevance, keyword matches, and other factors
        
        Args:
            results: List of search results
            query: Original query
            keywords: Extracted keywords
            
        Returns:
            Results with scoring added
        """
        for result in results:
            base_score = result.get('relevance_score', 0.5)
            boost = result.get('relevance_boost', 1.0)
            
            # Keyword matching bonus
            keyword_bonus = 0
            content = str(result.get('content', '')).lower()
            query_lower = query.lower()
            
            if keywords:
                matching_keywords = sum(1 for kw in keywords if kw in content)
                keyword_bonus = (matching_keywords / len(keywords)) * 0.3
            
            # Query phrase matching bonus
            if query_lower in content:
                keyword_bonus += 0.2
            
            # Content quality bonus (longer, structured content)
            content_length = len(content)
            if content_length > 200:
                content_bonus = min(0.1, content_length / 5000)
            else:
                content_bonus = 0
            
            # Calculate final score
            final_score = (base_score * boost) + keyword_bonus + content_bonus
            result['final_score'] = min(1.0, final_score)  # Cap at 1.0
            result['scoring_details'] = {
                'base_score': base_score,
                'boost': boost,
                'keyword_bonus': keyword_bonus,
                'content_bonus': content_bonus
            }
        
        return results
    
    def _format_search_response(self, query: str, search_results: Dict[str, Any], is_broader: bool = False) -> str:
        """
        Format search results into a human-readable response with source references
        
        Args:
            query: Original query
            search_results: Search results dictionary
            is_broader: Whether this is a broader search result
            
        Returns:
            Formatted response string
        """
        total_results = search_results.get('total_results', 0)
        results = search_results.get('results', [])
        
        if is_broader:
            response = f"I found {total_results} broader results related to your query '{query}':\n\n"
        else:
            response = f"I found {total_results} relevant results for your query '{query}':\n\n"
        
        # Add search type info
        semantic_count = search_results.get('semantic_count', 0)
        keyword_count = search_results.get('keyword_count', 0)
        if semantic_count > 0 and keyword_count > 0:
            response += f"📊 Results: {semantic_count} semantic matches, {keyword_count} keyword matches\n\n"
        
        # Format top results with source references
        for i, result in enumerate(results[:3], 1):  # Show top 3 results
            content = result.get('content', 'No content available')[:300]
            source = result.get('source', {})
            score = result.get('final_score', 0)
            
            response += f"**Result {i}** (Relevance: {score:.1%})\n"
            response += f"📄 Source: {source.get('reference', 'Unknown')}\n"
            response += f"📝 Content: {content}...\n"
            
            if source.get('file_type'):
                response += f"📋 Type: {source.get('file_type', 'Unknown')}"
                if source.get('upload_date', 'Unknown') != 'Unknown':
                    try:
                        upload_date = source.get('upload_date', '')[:10]  # Just date part
                        response += f" | 📅 Added: {upload_date}"
                    except:
                        pass
                response += "\n"
            
            response += "\n"
        
        # Add summary if more results available
        if len(results) > 3:
            response += f"💡 Found {len(results) - 3} additional results. Try a more specific query for better results.\n\n"
        
        # Add helpful suggestions
        keywords = search_results.get('keywords', [])
        if keywords:
            response += f"🔍 Search included keywords: {', '.join(keywords[:3])}\n"
        
        return response
    
    def get_similar_historical_cases(self, query: str, category: str = None, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Find similar historical cases from persistent store
        
        Args:
            query: Description to match against
            category: Optional category filter
            limit: Maximum results
            
        Returns:
            List of similar historical cases
        """
        if not self.persistent_enabled:
            return []
        
        try:
            similar_cases = self.persistent_store.find_similar_complaints(query, category, limit)
            self.logger.info(f"Found {len(similar_cases)} similar historical cases")
            return similar_cases
        except Exception as e:
            self.logger.error(f"Error finding similar cases: {str(e)}")
            return []
    
    def get_recommended_resolutions(self, category: str, description: str = None, limit: int = 3) -> List[Dict[str, Any]]:
        """
        Get recommended resolutions from persistent store
        
        Args:
            category: Issue category
            description: Optional description for better matching
            limit: Maximum results
            
        Returns:
            List of recommended resolutions
        """
        if not self.persistent_enabled:
            return []
        
        try:
            resolutions = self.persistent_store.get_best_resolutions(category, description, limit)
            self.logger.info(f"Found {len(resolutions)} recommended resolutions for category '{category}'")
            return resolutions
        except Exception as e:
            self.logger.error(f"Error getting recommended resolutions: {str(e)}")
            return []
    
    def get_persistent_store_insights(self) -> Dict[str, Any]:
        """
        Get comprehensive insights from persistent store
        
        Returns:
            Project insights and analytics
        """
        if not self.persistent_enabled:
            return {'error': 'Persistent store not enabled'}
        
        try:
            insights = self.persistent_store.get_project_insights()
            self.logger.info("Retrieved persistent store insights")
            return insights
        except Exception as e:
            self.logger.error(f"Error getting persistent store insights: {str(e)}")
            return {'error': str(e)}
    
    def export_learning_data(self, include_resolutions: bool = True, min_effectiveness: float = 0.7) -> Dict[str, Any]:
        """
        Export high-quality learning data from persistent store
        
        Args:
            include_resolutions: Whether to include resolution data
            min_effectiveness: Minimum effectiveness score for resolutions
            
        Returns:
            Structured learning data for training or analysis
        """
        if not self.persistent_enabled:
            return {'error': 'Persistent store not enabled'}
        
        try:
            learning_data = self.persistent_store.export_learning_data(include_resolutions, min_effectiveness)
            self.logger.info(f"Exported learning data with {len(learning_data.get('complaints', []))} complaints")
            return learning_data
        except Exception as e:
            self.logger.error(f"Error exporting learning data: {str(e)}")
            return {'error': str(e)}
    
    def create_resolution_template(self, category: str, title: str, description: str, 
                                 solution_steps: List[str]) -> str:
        """
        Create a reusable resolution template in persistent store
        
        Args:
            category: Resolution category
            title: Template title
            description: Template description
            solution_steps: List of solution steps
            
        Returns:
            Resolution template ID
        """
        if not self.persistent_enabled:
            return None
        
        try:
            template_id = self.persistent_store.create_resolution_template(
                category, title, description, solution_steps, "enhanced_agent"
            )
            self.logger.info(f"Created resolution template: {template_id}")
            return template_id
        except Exception as e:
            self.logger.error(f"Error creating resolution template: {str(e)}")
            return None
    
    def track_resolution_usage(self, resolution_id: str, effectiveness_feedback: float = None) -> bool:
        """
        Track when a resolution is used and update its effectiveness
        
        Args:
            resolution_id: ID of the resolution used
            effectiveness_feedback: Feedback score (0.0 to 1.0)
            
        Returns:
            Success status
        """
        if not self.persistent_enabled:
            return False
        
        try:
            success = self.persistent_store.track_resolution_usage(resolution_id, effectiveness_feedback)
            if success:
                self.logger.info(f"Tracked usage for resolution {resolution_id}")
            return success
        except Exception as e:
            self.logger.error(f"Error tracking resolution usage: {str(e)}")
            return False
    
    def generate_response_suggestions(self, ticket_analysis: Dict[str, Any], 
                                    customer_context: Dict[str, Any] = None,
                                    num_suggestions: int = 3) -> List[Dict[str, Any]]:
        """
        Generate intelligent response suggestions for agents
        
        Args:
            ticket_analysis: Ticket analysis results
            customer_context: Customer context information
            num_suggestions: Number of suggestions to generate
            
        Returns:
            List of response suggestions with confidence scores
        """
        if not self.supervisor_tools_enabled:
            return []
        
        try:
            suggestions = self.response_engine.generate_response_suggestions(
                ticket_analysis, customer_context, num_suggestions
            )
            
            # Convert to serializable format
            suggestions_data = []
            for suggestion in suggestions:
                suggestions_data.append({
                    'id': suggestion.id,
                    'response_text': suggestion.response_text,
                    'confidence_score': suggestion.confidence_score,
                    'response_type': suggestion.response_type,
                    'tone': suggestion.tone,
                    'estimated_resolution_time': suggestion.estimated_resolution_time,
                    'requires_escalation': suggestion.requires_escalation,
                    'suggested_actions': suggestion.suggested_actions,
                    'source_reference': suggestion.source_reference,
                    'metadata': suggestion.metadata
                })
            
            # Log response generation
            if self.state_enabled:
                avg_confidence = sum(s['confidence_score'] for s in suggestions_data) / len(suggestions_data) if suggestions_data else 0
                escalation_count = sum(1 for s in suggestions_data if s['requires_escalation'])
                
                self.state_manager.log_response_generated(
                    response_type="auto_suggestions",
                    confidence=avg_confidence,
                    suggestions_count=len(suggestions_data),
                    escalation_triggered=escalation_count > 0
                )
            
            self.logger.info(f"Generated {len(suggestions_data)} response suggestions")
            return suggestions_data
            
        except Exception as e:
            self.logger.error(f"Error generating response suggestions: {str(e)}")
            return []
    
    def evaluate_response_quality(self, response_text: str, ticket_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate the quality of an agent's response
        
        Args:
            response_text: The response to evaluate
            ticket_analysis: Original ticket analysis
            
        Returns:
            Quality evaluation with scores and feedback
        """
        if not self.supervisor_tools_enabled:
            return {'error': 'Supervisor tools not enabled'}
        
        try:
            quality_score = self.response_engine.get_response_quality_score(response_text, ticket_analysis)
            self.logger.info(f"Response quality evaluated: {quality_score.get('grade', 'N/A')}")
            return quality_score
            
        except Exception as e:
            self.logger.error(f"Error evaluating response quality: {str(e)}")
            return {'error': str(e)}
    
    def analyze_escalation_need(self, ticket_analysis: Dict[str, Any], 
                              customer_context: Dict[str, Any] = None,
                              interaction_history: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Analyze if a ticket needs escalation and provide recommendations
        
        Args:
            ticket_analysis: Ticket analysis results
            customer_context: Customer context information
            interaction_history: Previous interactions
            
        Returns:
            Escalation analysis with recommendations
        """
        if not self.supervisor_tools_enabled:
            return {'escalation_needed': False, 'error': 'Supervisor tools not enabled'}
        
        try:
            escalation_alert = self.escalation_detector.analyze_escalation_need(
                ticket_analysis, customer_context, interaction_history
            )
            
            # Convert to serializable format
            escalation_data = {
                'escalation_needed': escalation_alert.level.value in ['high', 'critical', 'immediate'],
                'level': escalation_alert.level.value,
                'confidence_score': escalation_alert.confidence_score,
                'priority_score': escalation_alert.priority_score,
                'reasons': [reason.value for reason in escalation_alert.reasons],
                'recommended_action': escalation_alert.recommended_action,
                'suggested_assignee': escalation_alert.suggested_assignee,
                'timeframe': escalation_alert.timeframe,
                'risk_factors': escalation_alert.risk_factors,
                'extracted_indicators': escalation_alert.extracted_indicators,
                'metadata': escalation_alert.metadata
            }
            
            # Log escalation analysis
            if self.state_enabled and escalation_data['escalation_needed']:
                self.state_manager.log_escalation(
                    escalation_level=escalation_data['level'],
                    reasons=escalation_data['reasons'],
                    priority_score=escalation_data['priority_score'],
                    recommended_action=escalation_data['recommended_action']
                )
            
            self.logger.info(f"Escalation analysis complete: Level {escalation_alert.level.value}")
            return escalation_data
            
        except Exception as e:
            self.logger.error(f"Error analyzing escalation need: {str(e)}")
            return {'escalation_needed': False, 'error': str(e)}
    
    def get_supervisor_dashboard(self, timeframe_hours: int = 24) -> Dict[str, Any]:
        """
        Get comprehensive supervisor dashboard data
        
        Args:
            timeframe_hours: Timeframe for analysis in hours
            
        Returns:
            Complete supervisor dashboard data
        """
        if not self.supervisor_tools_enabled:
            return {'error': 'Supervisor tools not enabled'}
        
        try:
            dashboard = self.supervisor_insights.get_comprehensive_dashboard(timeframe_hours)
            self.logger.info(f"Generated supervisor dashboard for {timeframe_hours}h timeframe")
            return dashboard
            
        except Exception as e:
            self.logger.error(f"Error generating supervisor dashboard: {str(e)}")
            return {'error': str(e)}
    
    def get_sla_monitoring(self, timeframe_hours: int = 24) -> Dict[str, Any]:
        """
        Get SLA monitoring data and alerts
        
        Args:
            timeframe_hours: Timeframe for analysis
            
        Returns:
            SLA monitoring data
        """
        if not self.supervisor_tools_enabled:
            return {'error': 'Supervisor tools not enabled'}
        
        try:
            sla_metrics = self.supervisor_insights.monitor_sla_compliance(timeframe_hours)
            
            # Convert to serializable format
            sla_data = []
            for metric in sla_metrics:
                sla_data.append({
                    'metric_name': metric.metric_name,
                    'target_value': metric.target_value,
                    'current_value': metric.current_value,
                    'status': metric.status.value,
                    'compliance_percentage': metric.compliance_percentage,
                    'trend': metric.trend,
                    'violations_count': metric.violations_count,
                    'last_violation': metric.last_violation.isoformat() if metric.last_violation else None
                })
            
            self.logger.info(f"Retrieved SLA monitoring data for {len(sla_data)} metrics")
            return {'sla_metrics': sla_data, 'timestamp': datetime.now().isoformat()}
            
        except Exception as e:
            self.logger.error(f"Error getting SLA monitoring data: {str(e)}")
            return {'error': str(e)}
    
    def detect_issue_spikes(self, timeframe_hours: int = 24) -> Dict[str, Any]:
        """
        Detect and analyze issue volume spikes
        
        Args:
            timeframe_hours: Timeframe for spike detection
            
        Returns:
            Issue spike detection results
        """
        if not self.supervisor_tools_enabled:
            return {'spikes': [], 'error': 'Supervisor tools not enabled'}
        
        try:
            spikes = self.supervisor_insights.detect_and_analyze_spikes(timeframe_hours)
            
            # Convert to serializable format
            spikes_data = []
            for spike in spikes:
                spikes_data.append({
                    'category': spike.category,
                    'severity': spike.severity.value,
                    'volume_increase': spike.volume_increase,
                    'current_volume': spike.current_volume,
                    'baseline_volume': spike.baseline_volume,
                    'start_time': spike.spike_start.isoformat(),
                    'duration_hours': spike.spike_duration.total_seconds() / 3600,
                    'probable_causes': spike.probable_causes,
                    'recommended_actions': spike.recommended_actions,
                    'affected_metrics': spike.affected_metrics
                })
            
            self.logger.info(f"Detected {len(spikes_data)} issue spikes")
            return {'spikes': spikes_data, 'timestamp': datetime.now().isoformat()}
            
        except Exception as e:
            self.logger.error(f"Error detecting issue spikes: {str(e)}")
            return {'spikes': [], 'error': str(e)}
    
    def get_supervisor_alerts(self, severity_threshold: str = 'medium') -> Dict[str, Any]:
        """
        Get supervisor alerts based on current conditions
        
        Args:
            severity_threshold: Minimum alert severity ('low', 'medium', 'high', 'critical')
            
        Returns:
            List of supervisor alerts
        """
        if not self.supervisor_tools_enabled:
            return {'alerts': [], 'error': 'Supervisor tools not enabled'}
        
        try:
            from supervisor_insights import AlertSeverity
            
            # Convert string to AlertSeverity enum
            severity_map = {
                'low': AlertSeverity.LOW,
                'medium': AlertSeverity.MEDIUM,
                'high': AlertSeverity.HIGH,
                'critical': AlertSeverity.CRITICAL
            }
            
            threshold = severity_map.get(severity_threshold.lower(), AlertSeverity.MEDIUM)
            alerts = self.supervisor_insights.generate_supervisor_alerts(threshold)
            
            # Convert to serializable format
            alerts_data = []
            for alert in alerts:
                alerts_data.append({
                    'id': alert.alert_id,
                    'title': alert.title,
                    'description': alert.description,
                    'severity': alert.severity.value,
                    'category': alert.category,
                    'created_at': alert.created_at.isoformat(),
                    'requires_action': alert.requires_action,
                    'recommended_actions': alert.recommended_actions,
                    'affected_agents': alert.affected_agents,
                    'metrics_impacted': alert.metrics_impacted,
                    'estimated_impact': alert.estimated_impact
                })
            
            self.logger.info(f"Generated {len(alerts_data)} supervisor alerts")
            return {'alerts': alerts_data, 'timestamp': datetime.now().isoformat()}
            
        except Exception as e:
            self.logger.error(f"Error generating supervisor alerts: {str(e)}")
            return {'alerts': [], 'error': str(e)}
    
    def process_ticket_with_supervisor_analysis(self, ticket_text: str, ticket_id: str = None,
                                              customer_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process ticket with comprehensive supervisor analysis
        
        Args:
            ticket_text: Customer message text
            ticket_id: Optional ticket ID
            customer_context: Customer context information
            
        Returns:
            Complete ticket processing with supervisor insights
        """
        try:
            # Process ticket with workflow
            ticket_result = self.process_support_ticket_with_workflow(ticket_text, ticket_id)
            
            if not self.supervisor_tools_enabled:
                return ticket_result
            
            # Add supervisor analysis
            supervisor_analysis = {}
            
            # Generate response suggestions
            suggestions = self.generate_response_suggestions(ticket_result, customer_context, 3)
            supervisor_analysis['response_suggestions'] = suggestions
            
            # Analyze escalation need
            escalation_analysis = self.analyze_escalation_need(ticket_result, customer_context)
            supervisor_analysis['escalation_analysis'] = escalation_analysis
            
            # If escalation is needed, highlight it
            if escalation_analysis.get('escalation_needed', False):
                supervisor_analysis['escalation_alert'] = {
                    'requires_immediate_attention': True,
                    'escalation_level': escalation_analysis.get('level', 'medium'),
                    'priority_score': escalation_analysis.get('priority_score', 5),
                    'timeframe': escalation_analysis.get('timeframe', 'Standard'),
                    'recommended_action': escalation_analysis.get('recommended_action', 'Review manually')
                }
            
            # Add supervisor analysis to result
            ticket_result['supervisor_analysis'] = supervisor_analysis
            
            self.logger.info(f"Processed ticket with supervisor analysis: {ticket_id or 'AUTO'}")
            return ticket_result
            
        except Exception as e:
            self.logger.error(f"Error processing ticket with supervisor analysis: {str(e)}")
            # Return basic processing result if supervisor analysis fails
            return self.process_support_ticket_with_workflow(ticket_text, ticket_id)
    
    def process_query(self, query: str, query_type: str = "general") -> Dict[str, Any]:
        """
        Process a user query with state tracking
        
        Args:
            query: User query text
            query_type: Type of query
            
        Returns:
            Query result with response
        """
        start_time = datetime.now()
        
        # Use existing natural language processing
        response = self.process_natural_language_query(query)
        
        # Log the query and response
        if self.state_enabled:
            end_time = datetime.now()
            response_time_ms = int((end_time - start_time).total_seconds() * 1000)
            
            self.state_manager.log_query(
                query=query,
                query_type=query_type,
                response=response,
                response_time_ms=response_time_ms
            )
        
        return {
            'query': query,
            'response': response,
            'query_type': query_type,
            'timestamp': start_time.isoformat(),
            'response_time_ms': response_time_ms if self.state_enabled else None
        }
    
    def process_file_upload(self, file_path: str, file_type: str = None) -> Dict[str, Any]:
        """
        Process a file upload with state tracking
        
        Args:
            file_path: Path to uploaded file
            file_type: Type of file (optional, will be detected)
            
        Returns:
            File processing result
        """
        import os
        
        if not os.path.exists(file_path):
            return {'error': 'File not found', 'file_path': file_path}
        
        # Get file information
        filename = os.path.basename(file_path)
        file_size = os.path.getsize(file_path)
        
        if not file_type:
            # Detect file type from extension
            _, ext = os.path.splitext(filename)
            file_type = ext.lower().lstrip('.')
        
        # Process file using existing file processing capabilities
        try:
            processing_result = {}
            
            # Use existing file processing methods based on type
            if file_type in ['txt', 'csv', 'pdf', 'json']:
                # Process using existing add_knowledge_from_file method
                processing_result = self.add_knowledge_from_file(file_path, file_type)
            else:
                processing_result = {'message': f'File type {file_type} processed but not indexed'}
            
            # Log the file upload
            if self.state_enabled:
                self.state_manager.log_file_upload(
                    filename=filename,
                    file_size=file_size,
                    file_type=file_type,
                    processing_result=processing_result
                )
            
            return {
                'filename': filename,
                'file_size': file_size,
                'file_type': file_type,
                'processing_result': processing_result,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            error_result = {'error': str(e)}
            
            # Log failed upload
            if self.state_enabled:
                self.state_manager.log_file_upload(
                    filename=filename,
                    file_size=file_size,
                    file_type=file_type,
                    processing_result=error_result
                )
            
            return {
                'filename': filename,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def determine_routing_decision(self, ticket_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Determine routing decision based on ticket analysis
        
        Args:
            ticket_analysis: Complete ticket analysis results
            
        Returns:
            Routing decision with destination, priority, and reasoning
        """
        try:
            # Extract key analysis components
            sentiment = ticket_analysis.get('sentiment_analysis', {})
            intent = ticket_analysis.get('intent_classification', {})
            original_text = ticket_analysis.get('original_text', '').lower()
            
            urgency_level = sentiment.get('urgency_level', 'low')
            sentiment_score = sentiment.get('sentiment_score', 0)
            customer_mood = sentiment.get('customer_mood', 'neutral')
            intent_category = intent.get('intent_category', 'unknown')
            
            # Initialize routing decision
            routing = {
                'decision': 'STANDARD_QUEUE',
                'priority': 'normal',
                'reason': 'Standard support inquiry',
                'escalation_needed': False,
                'specialist_required': False,
                'estimated_complexity': 'low',
                'suggested_sla': '24h'
            }
            
            # Check for legal threats (highest priority)
            legal_keywords = ['legal', 'sue', 'lawyer', 'attorney', 'court', 'lawsuit', 'litigation']
            if any(keyword in original_text for keyword in legal_keywords):
                routing.update({
                    'decision': 'ESCALATE_LEGAL',
                    'priority': 'critical',
                    'reason': 'Legal threat detected',
                    'escalation_needed': True,
                    'specialist_required': True,
                    'estimated_complexity': 'high',
                    'suggested_sla': '1h'
                })
                return routing
            
            # Check for high urgency or very negative sentiment
            if urgency_level in ['critical', 'high'] or sentiment_score < -0.7:
                routing.update({
                    'decision': 'ESCALATE_SUPERVISOR',
                    'priority': 'high',
                    'reason': f'High urgency ({urgency_level}) or very negative sentiment ({sentiment_score:.2f})',
                    'escalation_needed': True,
                    'estimated_complexity': 'medium',
                    'suggested_sla': '4h'
                })
                return routing
            
            # Check for specific domain expertise needed
            billing_keywords = ['billing', 'charge', 'payment', 'refund', 'subscription', 'invoice']
            technical_keywords = ['crash', 'bug', 'error', 'technical', 'website', 'app', 'login', 'password']
            account_keywords = ['account', 'profile', 'settings', 'data', 'privacy', 'security']
            
            if any(keyword in original_text for keyword in billing_keywords):
                routing.update({
                    'decision': 'BILLING_SPECIALIST',
                    'priority': 'normal' if sentiment_score > -0.3 else 'high',
                    'reason': 'Billing or payment related inquiry',
                    'specialist_required': True,
                    'estimated_complexity': 'medium',
                    'suggested_sla': '12h'
                })
            elif any(keyword in original_text for keyword in technical_keywords):
                routing.update({
                    'decision': 'TECHNICAL_TEAM',
                    'priority': 'high' if urgency_level in ['high', 'critical'] else 'normal',
                    'reason': 'Technical issue requiring specialized support',
                    'specialist_required': True,
                    'estimated_complexity': 'medium',
                    'suggested_sla': '8h'
                })
            elif any(keyword in original_text for keyword in account_keywords):
                routing.update({
                    'decision': 'ACCOUNT_SPECIALIST',
                    'priority': 'normal',
                    'reason': 'Account or privacy related inquiry',
                    'specialist_required': True,
                    'estimated_complexity': 'low',
                    'suggested_sla': '12h'
                })
            
            # Adjust priority based on customer mood and sentiment
            if customer_mood in ['angry', 'frustrated'] and sentiment_score < -0.5:
                if routing['priority'] == 'normal':
                    routing['priority'] = 'high'
                routing['reason'] += f' with frustrated customer (mood: {customer_mood})'
                routing['suggested_sla'] = '6h'
            
            # Check if escalation is needed based on escalation detector
            try:
                escalation_analysis = self.escalation_detector.analyze_ticket(
                    original_text, 
                    sentiment_score, 
                    urgency_level
                )
                if escalation_analysis.get('escalation_needed', False):
                    routing['escalation_needed'] = True
                    routing['escalation_triggers'] = escalation_analysis.get('triggers', [])
                    if routing['decision'] == 'STANDARD_QUEUE':
                        routing['decision'] = 'ESCALATE_SUPERVISOR'
                        routing['priority'] = 'high'
                        routing['reason'] += ' (escalation triggers detected)'
            except Exception as e:
                self.logger.warning(f"Escalation analysis failed: {str(e)}")
            
            return routing
            
        except Exception as e:
            self.logger.error(f"Error determining routing decision: {str(e)}")
            return {
                'decision': 'SUPERVISOR_REVIEW',
                'priority': 'normal',
                'reason': f'Unable to determine routing: {str(e)}',
                'escalation_needed': True,
                'specialist_required': False,
                'estimated_complexity': 'unknown',
                'suggested_sla': '24h'
            }

    def get_similar_historical_cases(self, query: str, category: str = None, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Get similar historical cases from persistent store
        
        Args:
            query: Search query text
            category: Optional category filter
            limit: Maximum number of results
            
        Returns:
            List of similar historical cases
        """
        try:
            self.logger.info(f"Searching for similar historical cases: {query}")
            
            # Get complaints from persistent store (access the actual store)
            store = self.persistent_store.store if hasattr(self.persistent_store, 'store') else self.persistent_store
            complaints = store.get_complaints_by_category(category) if category else store.get_all_complaints()
            
            if not complaints:
                self.logger.info("No complaints found in persistent store")
                return []
            
            # Use semantic search to find similar cases
            search_results = self.search_knowledge_base(query, search_type='semantic', top_k=limit * 2)
            semantic_matches = search_results.get('results', [])
            
            # Combine with complaint data for better results
            historical_cases = []
            
            for complaint in complaints[:limit]:
                # Calculate basic similarity based on content
                complaint_text = complaint.get('description', '') + ' ' + complaint.get('resolution', '')
                similarity_score = self._calculate_text_similarity(query.lower(), complaint_text.lower())
                
                # Apply category filter if specified
                if category and complaint.get('category', '').lower() != category.lower():
                    continue
                
                case_data = {
                    'id': complaint.get('id', 'unknown'),
                    'title': complaint.get('title', complaint.get('description', 'No title')[:50]),
                    'description': complaint.get('description', 'No description'),
                    'category': complaint.get('category', 'general'),
                    'status': complaint.get('status', 'completed'),
                    'created_at': complaint.get('created_at', 'Unknown'),
                    'resolution': complaint.get('resolution', 'No resolution'),
                    'sentiment_score': complaint.get('sentiment_score', 0),
                    'urgency_level': complaint.get('urgency_level', 'normal'),
                    'resolution_time_hours': complaint.get('resolution_time_hours', 0),
                    'satisfaction_rating': complaint.get('satisfaction_rating', 0),
                    'similarity_score': similarity_score
                }
                
                historical_cases.append(case_data)
            
            # Sort by similarity score (descending)
            historical_cases.sort(key=lambda x: x['similarity_score'], reverse=True)
            
            # Take top results
            result = historical_cases[:limit]
            
            self.logger.info(f"Found {len(result)} similar historical cases")
            return result
            
        except Exception as e:
            self.logger.error(f"Error getting similar historical cases: {str(e)}")
            return []
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate basic text similarity score
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score between 0 and 1
        """
        try:
            # Simple keyword-based similarity
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            
            if not words1 or not words2:
                return 0.0
            
            intersection = words1.intersection(words2)
            union = words1.union(words2)
            
            similarity = len(intersection) / len(union) if union else 0.0
            
            # Boost score if key terms match
            key_terms = ['billing', 'refund', 'payment', 'delivery', 'technical', 'account', 'order', 'cancel']
            for term in key_terms:
                if term in text1 and term in text2:
                    similarity += 0.2
            
            return min(similarity, 1.0)
            
        except Exception:
            return 0.0

    def get_state_summary(self) -> Dict[str, Any]:
        """
        Get a comprehensive summary of current session state
        
        Returns:
            Complete state summary
        """
        if not self.state_enabled:
            return {'state_management': 'disabled'}
        
        return {
            'session_statistics': self.get_session_statistics(),
            'recent_events': self.get_session_history(limit=10),
            'state_management': 'enabled',
            'session_id': self.state_manager.current_session.session_id if self.state_manager.current_session else None,
            'supervisor_tools': {
                'response_suggestions': self.supervisor_tools_enabled,
                'escalation_detection': self.supervisor_tools_enabled,
                'supervisor_insights': self.supervisor_tools_enabled
            },
            'persistent_storage': self.persistent_enabled
        }
    
    def ensure_demo_data_available(self) -> bool:
        """
        Ensure there's some demo data available for search testing
        
        Returns:
            True if demo data was added or already exists
        """
        try:
            # Check if we have any data
            test_search = super().search_knowledge_base("test", top_k=1)
            if test_search.get('results_count', 0) > 0:
                return True  # We have data
            
            # Add some demo data for testing
            demo_content = [
                {
                    'chunk_id': 'demo_001',
                    'text': 'Customer requesting refund for damaged product. Item arrived broken and customer wants full refund processed within 3-5 business days.',
                    'chunk_type': 'support_case',
                    'word_count': 20,
                    'char_count': 135,
                    'metadata': {
                        'file_name': 'sample_refund_cases.txt',
                        'file_type': 'txt',
                        'chunk_id': '001',
                        'source_type': 'Support Log',
                        'created_at': datetime.now().isoformat()
                    }
                },
                {
                    'chunk_id': 'demo_002',
                    'text': 'Delivery delay complaint - customer order was promised in 2 days but took 7 days to arrive. Customer wants compensation.',
                    'chunk_type': 'support_case',
                    'word_count': 20,
                    'char_count': 125,
                    'metadata': {
                        'file_name': 'sample_delivery_issues.txt',
                        'file_type': 'txt',
                        'chunk_id': '002',
                        'source_type': 'Support Log',
                        'created_at': datetime.now().isoformat()
                    }
                },
                {
                    'chunk_id': 'demo_003',
                    'text': 'Billing dispute - customer was charged twice for the same order. Needs immediate credit card refund.',
                    'chunk_type': 'support_case',
                    'word_count': 16,
                    'char_count': 105,
                    'metadata': {
                        'file_name': 'sample_billing_cases.txt',
                        'file_type': 'txt',
                        'chunk_id': '003',
                        'source_type': 'Support Log',
                        'created_at': datetime.now().isoformat()
                    }
                },
                {
                    'chunk_id': 'demo_004',
                    'text': 'Technical support request - customer cannot login to account. Password reset not working. Needs account recovery assistance.',
                    'chunk_type': 'support_case',
                    'word_count': 18,
                    'char_count': 125,
                    'metadata': {
                        'file_name': 'sample_technical_issues.txt',
                        'file_type': 'txt',
                        'chunk_id': '004',
                        'source_type': 'Support Log',
                        'created_at': datetime.now().isoformat()
                    }
                },
                {
                    'chunk_id': 'demo_005',
                    'text': 'Return policy: Items can be returned within 30 days of purchase for full refund. Item must be in original condition.',
                    'chunk_type': 'policy',
                    'word_count': 20,
                    'char_count': 120,
                    'metadata': {
                        'file_name': 'return_policy.txt',
                        'file_type': 'txt',
                        'chunk_id': '005',
                        'source_type': 'Policy Document',
                        'created_at': datetime.now().isoformat()
                    }
                }
            ]
            
            # Add demo data to vector store for semantic search
            try:
                vector_result = self.vector_store.upsert_chunks(demo_content)
                if vector_result.get('successful_upserts', 0) > 0:
                    self.logger.info(f"Added {vector_result['successful_upserts']} demo chunks to vector store")
            except Exception as e:
                self.logger.warning(f"Could not store demo data in vector store: {str(e)}")
            
            self.logger.info("Added demo data for search testing")
            return True
            
        except Exception as e:
            self.logger.error(f"Could not ensure demo data: {str(e)}")
            return False
    
    def get_similar_historical_cases(self, query: str, category: str = None, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Get similar historical cases from persistent store using semantic search
        
        Args:
            query: Search query describing the issue
            category: Optional category to filter by
            limit: Maximum number of cases to return
            
        Returns:
            List of similar historical cases with details
        """
        try:
            self.logger.info(f"Searching historical cases for: {query[:50]}... Category: {category}")
            
            # Get historical cases from persistent store
            if category:
                historical_cases = self.persistent_store.store.get_complaints_by_category(category)
            else:
                historical_cases = self.persistent_store.store.get_all_complaints()
            
            if not historical_cases:
                self.logger.info("No historical cases found in database")
                return []
            
            # Use semantic search to find similar cases
            query_lower = query.lower()
            scored_cases = []
            
            # Define related keywords to expand search
            delivery_keywords = ['delivery', 'shipping', 'order', 'product', 'arrived', 'late', 'delayed']
            billing_keywords = ['billing', 'charge', 'payment', 'refund', 'account', 'money']
            problem_keywords = ['problem', 'issue', 'frustrated', 'help', 'trouble', 'error']
            
            for case in historical_cases:
                # Calculate similarity based on description and title
                case_text = f"{case.get('title', '')} {case.get('description', '')}".lower()
                
                # Extract words from both query and case
                query_words = set(query_lower.split())
                case_words = set(case_text.split())
                common_words = query_words.intersection(case_words)
                
                # Calculate base similarity
                similarity_score = 0.0
                
                if common_words:
                    # Direct word match
                    similarity_score = len(common_words) / max(len(query_words), len(case_words))
                
                # Boost for related keywords
                if 'delivery' in query_lower or 'shipping' in query_lower:
                    delivery_matches = sum(1 for word in delivery_keywords if word in case_text)
                    if delivery_matches > 0:
                        similarity_score += 0.3 * (delivery_matches / len(delivery_keywords))
                
                if 'billing' in query_lower or 'payment' in query_lower:
                    billing_matches = sum(1 for word in billing_keywords if word in case_text)
                    if billing_matches > 0:
                        similarity_score += 0.3 * (billing_matches / len(billing_keywords))
                
                if 'problem' in query_lower or 'issue' in query_lower:
                    problem_matches = sum(1 for word in problem_keywords if word in case_text)
                    if problem_matches > 0:
                        similarity_score += 0.2 * (problem_matches / len(problem_keywords))
                
                # Boost score for exact category match
                if category and case.get('category', '').lower() == category.lower():
                    similarity_score *= 1.2
                
                # Boost score for resolved cases
                if case.get('status', '').lower() in ['resolved', 'closed']:
                    similarity_score *= 1.1
                
                # Add cases with any similarity > 0
                if similarity_score > 0:
                    scored_cases.append({
                        'case': case,
                        'similarity_score': similarity_score
                    })
            
            # Sort by similarity score and return top results
            scored_cases.sort(key=lambda x: x['similarity_score'], reverse=True)
            
            # Format results
            similar_cases = []
            for scored_case in scored_cases[:limit]:
                case = scored_case['case']
                formatted_case = {
                    'id': case.get('id', ''),
                    'title': case.get('title', 'No title'),
                    'description': case.get('description', 'No description')[:200] + '...' if len(case.get('description', '')) > 200 else case.get('description', ''),
                    'category': case.get('category', 'Unknown'),
                    'priority': case.get('priority', 'Normal'),
                    'status': case.get('status', 'Unknown'),
                    'resolution': case.get('resolution', 'No resolution available'),
                    'similarity_score': round(scored_case['similarity_score'], 3),
                    'created_at': case.get('created_at', ''),
                    'resolution_time_hours': case.get('resolution_time_hours', 0),
                    'satisfaction_rating': case.get('satisfaction_rating', 0)
                }
                similar_cases.append(formatted_case)
            
            self.logger.info(f"Found {len(similar_cases)} similar historical cases")
            return similar_cases
            
        except Exception as e:
            self.logger.error(f"Error searching historical cases: {str(e)}")
            return []
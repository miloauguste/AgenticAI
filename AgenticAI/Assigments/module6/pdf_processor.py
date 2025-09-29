#!/usr/bin/env python3
"""
PDF Processor for Document Analysis and LLM Applications

This module provides comprehensive PDF processing capabilities including
text extraction, document analysis, structure detection, and preparation
for LLM training/inference. Essential for handling document-based AI applications.
"""

import json
import re
import os
from typing import List, Dict, Any, Optional, Union, Tuple, NamedTuple
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
import hashlib
import base64

# Core dependencies for PDF processing
try:
    import PyPDF2
    import pdfplumber
    from PIL import Image
    import io
    HAS_PDF_LIBS = True
except ImportError:
    HAS_PDF_LIBS = False
    print("Warning: PDF processing libraries not installed.")
    print("Install with: pip install PyPDF2 pdfplumber Pillow")

# Optional OCR support
try:
    import pytesseract
    HAS_OCR = True
except ImportError:
    HAS_OCR = False


class PageInfo(NamedTuple):
    """Information about a single PDF page."""
    page_num: int
    text: str
    word_count: int
    char_count: int
    has_images: bool
    has_tables: bool


@dataclass
class DocumentMetadata:
    """Comprehensive PDF document metadata."""
    filename: str
    file_size: int
    creation_date: Optional[datetime]
    modification_date: Optional[datetime]
    author: Optional[str]
    title: Optional[str]
    subject: Optional[str]
    creator: Optional[str]
    producer: Optional[str]
    page_count: int
    is_encrypted: bool
    has_images: bool
    has_tables: bool
    language: Optional[str] = None
    document_hash: Optional[str] = None


@dataclass
class TextSection:
    """Represents a logical section of text with metadata."""
    title: str
    content: str
    page_range: Tuple[int, int]
    section_type: str  # 'header', 'paragraph', 'list', 'table', 'footer'
    confidence: float
    word_count: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DocumentAnalysis:
    """Comprehensive document analysis results."""
    metadata: DocumentMetadata
    pages: List[PageInfo]
    sections: List[TextSection]
    full_text: str
    word_count: int
    char_count: int
    reading_time_minutes: float
    language_detected: Optional[str]
    topics_detected: List[str]
    quality_score: float
    processing_errors: List[str] = field(default_factory=list)


class PDFProcessor:
    """
    Comprehensive PDF processor for document analysis and LLM applications.
    
    This processor demonstrates key concepts:
    - Multi-library text extraction (PyPDF2, pdfplumber)
    - Document structure analysis
    - Metadata extraction and validation
    - Text preprocessing for NLP tasks
    - Section detection and classification
    - Quality assessment and error handling
    """
    
    def __init__(self, use_ocr: bool = False, ocr_language: str = 'eng'):
        """
        Initialize PDF processor.
        
        Args:
            use_ocr: Whether to use OCR for image-based PDFs
            ocr_language: Language code for OCR (e.g., 'eng', 'spa', 'fra')
        """
        if not HAS_PDF_LIBS:
            raise ImportError("PDF processing libraries not installed. Run: pip install PyPDF2 pdfplumber Pillow")
        
        self.use_ocr = use_ocr and HAS_OCR
        self.ocr_language = ocr_language
        self.processing_log: List[Dict] = []
        
        if use_ocr and not HAS_OCR:
            print("Warning: OCR requested but pytesseract not installed. OCR disabled.")
            self.use_ocr = False
    
    def _log_operation(self, operation: str, details: str, metadata: Dict = None) -> None:
        """Log processing operations for audit trail."""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'operation': operation,
            'details': details,
            'metadata': metadata or {}
        }
        self.processing_log.append(log_entry)
    
    def extract_metadata(self, pdf_path: Union[str, Path]) -> DocumentMetadata:
        """
        Extract comprehensive metadata from PDF file.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            DocumentMetadata object with all available metadata
        """
        pdf_path = Path(pdf_path)
        
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        metadata = DocumentMetadata(
            filename=pdf_path.name,
            file_size=pdf_path.stat().st_size,
            creation_date=None,
            modification_date=datetime.fromtimestamp(pdf_path.stat().st_mtime),
            author=None,
            title=None,
            subject=None,
            creator=None,
            producer=None,
            page_count=0,
            is_encrypted=False,
            has_images=False,
            has_tables=False
        )
        
        # Generate file hash for deduplication
        with open(pdf_path, 'rb') as f:
            metadata.document_hash = hashlib.md5(f.read()).hexdigest()
        
        # Extract PDF metadata using PyPDF2
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                metadata.page_count = len(reader.pages)
                metadata.is_encrypted = reader.is_encrypted
                
                if reader.metadata:
                    metadata.author = reader.metadata.get('/Author')
                    metadata.title = reader.metadata.get('/Title')
                    metadata.subject = reader.metadata.get('/Subject')
                    metadata.creator = reader.metadata.get('/Creator')
                    metadata.producer = reader.metadata.get('/Producer')
                    
                    # Parse creation date
                    creation_date = reader.metadata.get('/CreationDate')
                    if creation_date:
                        try:
                            # PDF date format: D:YYYYMMDDHHmmSSOHH'mm
                            date_str = creation_date.replace('D:', '').split('+')[0].split('-')[0]
                            if len(date_str) >= 8:
                                metadata.creation_date = datetime.strptime(date_str[:14], '%Y%m%d%H%M%S')
                        except ValueError:
                            pass
                
                # Check for images and tables using pdfplumber
                try:
                    import pdfplumber
                    with pdfplumber.open(pdf_path) as pdf:
                        for page in pdf.pages:
                            if page.images:
                                metadata.has_images = True
                            if page.extract_tables():
                                metadata.has_tables = True
                            if metadata.has_images and metadata.has_tables:
                                break
                except Exception:
                    pass
        
        except Exception as e:
            self._log_operation('extract_metadata', f"Error extracting metadata: {str(e)}")
        
        self._log_operation('extract_metadata', f"Extracted metadata for {metadata.filename}")
        return metadata
    
    def extract_text_pypdf2(self, pdf_path: Union[str, Path]) -> List[PageInfo]:
        """Extract text using PyPDF2 (fast but basic)."""
        pages = []
        
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                
                for page_num, page in enumerate(reader.pages, 1):
                    try:
                        text = page.extract_text()
                        word_count = len(text.split())
                        char_count = len(text)
                        
                        page_info = PageInfo(
                            page_num=page_num,
                            text=text,
                            word_count=word_count,
                            char_count=char_count,
                            has_images=False,  # PyPDF2 doesn't detect images easily
                            has_tables=False
                        )
                        pages.append(page_info)
                        
                    except Exception as e:
                        self._log_operation('extract_text_pypdf2', 
                                          f"Error on page {page_num}: {str(e)}")
                        
        except Exception as e:
            self._log_operation('extract_text_pypdf2', f"Fatal error: {str(e)}")
        
        return pages
    
    def extract_text_pdfplumber(self, pdf_path: Union[str, Path]) -> List[PageInfo]:
        """Extract text using pdfplumber (slower but more accurate)."""
        pages = []
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    try:
                        text = page.extract_text() or ""
                        word_count = len(text.split())
                        char_count = len(text)
                        
                        # Detect images and tables
                        has_images = bool(page.images)
                        has_tables = bool(page.extract_tables())
                        
                        page_info = PageInfo(
                            page_num=page_num,
                            text=text,
                            word_count=word_count,
                            char_count=char_count,
                            has_images=has_images,
                            has_tables=has_tables
                        )
                        pages.append(page_info)
                        
                    except Exception as e:
                        self._log_operation('extract_text_pdfplumber', 
                                          f"Error on page {page_num}: {str(e)}")
                        
        except Exception as e:
            self._log_operation('extract_text_pdfplumber', f"Fatal error: {str(e)}")
        
        return pages
    
    def extract_text_with_ocr(self, pdf_path: Union[str, Path]) -> List[PageInfo]:
        """Extract text using OCR for image-based PDFs."""
        if not self.use_ocr:
            return []
        
        pages = []
        
        try:
            import fitz  # PyMuPDF for image extraction
            doc = fitz.open(pdf_path)
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                
                # Convert page to image
                mat = fitz.Matrix(2, 2)  # 2x zoom for better OCR
                pix = page.get_pixmap(matrix=mat)
                img_data = pix.tobytes("png")
                
                # Run OCR
                image = Image.open(io.BytesIO(img_data))
                text = pytesseract.image_to_string(image, lang=self.ocr_language)
                
                word_count = len(text.split())
                char_count = len(text)
                
                page_info = PageInfo(
                    page_num=page_num + 1,
                    text=text,
                    word_count=word_count,
                    char_count=char_count,
                    has_images=True,  # Assume images if using OCR
                    has_tables=False  # OCR doesn't detect table structure
                )
                pages.append(page_info)
            
            doc.close()
            
        except ImportError:
            self._log_operation('extract_text_with_ocr', 
                              "PyMuPDF not installed. Install with: pip install PyMuPDF")
        except Exception as e:
            self._log_operation('extract_text_with_ocr', f"OCR error: {str(e)}")
        
        return pages
    
    def detect_document_structure(self, pages: List[PageInfo]) -> List[TextSection]:
        """
        Detect and classify document sections.
        
        Args:
            pages: List of PageInfo objects
            
        Returns:
            List of detected text sections with classification
        """
        sections = []
        current_section = None
        
        for page in pages:
            if not page.text.strip():
                continue
            
            # Split page text into paragraphs
            paragraphs = [p.strip() for p in page.text.split('\n\n') if p.strip()]
            
            for paragraph in paragraphs:
                section_type = self._classify_text_section(paragraph)
                
                # If this is a header, start a new section
                if section_type == 'header':
                    if current_section:
                        sections.append(current_section)
                    
                    current_section = TextSection(
                        title=paragraph[:100] + ('...' if len(paragraph) > 100 else ''),
                        content=paragraph,
                        page_range=(page.page_num, page.page_num),
                        section_type=section_type,
                        confidence=0.8,
                        word_count=len(paragraph.split())
                    )
                else:
                    # Add to current section or create new generic section
                    if current_section:
                        current_section.content += '\n\n' + paragraph
                        current_section.page_range = (
                            current_section.page_range[0], 
                            page.page_num
                        )
                        current_section.word_count += len(paragraph.split())
                    else:
                        current_section = TextSection(
                            title=f"Section starting on page {page.page_num}",
                            content=paragraph,
                            page_range=(page.page_num, page.page_num),
                            section_type='paragraph',
                            confidence=0.6,
                            word_count=len(paragraph.split())
                        )
        
        # Add the last section
        if current_section:
            sections.append(current_section)
        
        self._log_operation('detect_document_structure', f"Detected {len(sections)} sections")
        return sections
    
    def _classify_text_section(self, text: str) -> str:
        """Classify a text block into section types."""
        text_lower = text.lower().strip()
        
        # Header patterns
        header_patterns = [
            r'^\d+\.?\s+[A-Z]',  # "1. Introduction" or "1 INTRODUCTION"
            r'^[A-Z][A-Z\s]{3,}$',  # "INTRODUCTION", "METHODOLOGY"
            r'^(chapter|section|part)\s+\d+',  # "Chapter 1", "Section 2"
            r'^(introduction|conclusion|abstract|summary|references|bibliography)',
            r'^[ivx]+\.\s+[A-Z]',  # Roman numerals
        ]
        
        for pattern in header_patterns:
            if re.match(pattern, text, re.IGNORECASE):
                return 'header'
        
        # List patterns
        if re.match(r'^\s*[-•*]\s+', text) or re.match(r'^\s*\d+\.\s+', text):
            return 'list'
        
        # Table patterns (simple heuristic)
        if '\t' in text or re.search(r'\s{3,}', text):
            tab_count = text.count('\t')
            space_runs = len(re.findall(r'\s{3,}', text))
            if tab_count > 2 or space_runs > 3:
                return 'table'
        
        # Footer patterns
        if len(text) < 100 and re.search(r'(page \d+|\d+ of \d+|©|\(c\))', text, re.IGNORECASE):
            return 'footer'
        
        # Default to paragraph
        return 'paragraph'
    
    def detect_language(self, text: str) -> Optional[str]:
        """Simple language detection based on common words."""
        text_lower = text.lower()
        
        # Language detection patterns (basic implementation)
        language_patterns = {
            'english': ['the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had'],
            'spanish': ['la', 'de', 'que', 'el', 'en', 'un', 'es', 'se', 'no', 'te'],
            'french': ['le', 'de', 'et', 'la', 'les', 'des', 'en', 'du', 'un', 'sur'],
            'german': ['der', 'die', 'und', 'in', 'den', 'von', 'zu', 'das', 'mit', 'sich'],
        }
        
        scores = {}
        words = text_lower.split()
        
        for lang, common_words in language_patterns.items():
            score = sum(1 for word in words if word in common_words)
            scores[lang] = score / len(words) if words else 0
        
        if scores:
            best_lang = max(scores, key=scores.get)
            if scores[best_lang] > 0.02:  # At least 2% common words
                return best_lang
        
        return None
    
    def detect_topics(self, text: str) -> List[str]:
        """Simple topic detection based on keyword frequency."""
        text_lower = text.lower()
        
        # Topic keywords (expandable)
        topic_keywords = {
            'technology': ['computer', 'software', 'digital', 'internet', 'data', 'algorithm', 'ai', 'machine learning'],
            'business': ['company', 'market', 'revenue', 'profit', 'customer', 'sales', 'strategy', 'management'],
            'science': ['research', 'study', 'experiment', 'hypothesis', 'analysis', 'method', 'results', 'conclusion'],
            'legal': ['law', 'court', 'legal', 'contract', 'agreement', 'regulation', 'compliance', 'liability'],
            'medical': ['patient', 'treatment', 'diagnosis', 'medical', 'health', 'disease', 'therapy', 'clinical'],
            'education': ['student', 'learning', 'education', 'school', 'university', 'course', 'teaching', 'academic']
        }
        
        detected_topics = []
        words = text_lower.split()
        
        for topic, keywords in topic_keywords.items():
            keyword_count = sum(1 for word in words if any(kw in word for kw in keywords))
            if keyword_count > len(words) * 0.005:  # At least 0.5% of words
                detected_topics.append(topic)
        
        return detected_topics
    
    def calculate_quality_score(self, pages: List[PageInfo], sections: List[TextSection]) -> float:
        """Calculate document quality score (0-100)."""
        score = 100.0
        
        # Penalize empty or very short pages
        empty_pages = sum(1 for page in pages if page.word_count < 10)
        if pages:
            empty_page_ratio = empty_pages / len(pages)
            score -= empty_page_ratio * 30
        
        # Reward good structure
        if sections:
            header_sections = sum(1 for section in sections if section.section_type == 'header')
            if header_sections > 0:
                score += min(header_sections * 5, 20)  # Up to 20 points for structure
        
        # Penalize extraction errors
        total_chars = sum(page.char_count for page in pages)
        if total_chars == 0:
            score = 0
        elif total_chars < 1000:
            score -= 20  # Very short document
        
        # Average confidence from sections
        if sections:
            avg_confidence = sum(section.confidence for section in sections) / len(sections)
            score *= avg_confidence
        
        return max(0.0, min(100.0, score))
    
    def process_pdf(self, pdf_path: Union[str, Path], 
                   extraction_method: str = 'auto') -> DocumentAnalysis:
        """
        Process PDF file and return comprehensive analysis.
        
        Args:
            pdf_path: Path to PDF file
            extraction_method: 'auto', 'pypdf2', 'pdfplumber', 'ocr'
            
        Returns:
            Complete document analysis
        """
        pdf_path = Path(pdf_path)
        
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        self._log_operation('process_pdf', f"Starting processing of {pdf_path.name}")
        
        # Extract metadata
        metadata = self.extract_metadata(pdf_path)
        
        # Extract text based on method
        pages = []
        processing_errors = []
        
        if extraction_method == 'auto':
            # Try pdfplumber first, fall back to PyPDF2
            try:
                pages = self.extract_text_pdfplumber(pdf_path)
                if not any(page.text.strip() for page in pages) and self.use_ocr:
                    pages = self.extract_text_with_ocr(pdf_path)
            except Exception as e:
                processing_errors.append(f"pdfplumber failed: {str(e)}")
                try:
                    pages = self.extract_text_pypdf2(pdf_path)
                except Exception as e2:
                    processing_errors.append(f"PyPDF2 failed: {str(e2)}")
        
        elif extraction_method == 'pypdf2':
            pages = self.extract_text_pypdf2(pdf_path)
        elif extraction_method == 'pdfplumber':
            pages = self.extract_text_pdfplumber(pdf_path)
        elif extraction_method == 'ocr':
            pages = self.extract_text_with_ocr(pdf_path)
        else:
            raise ValueError(f"Unknown extraction method: {extraction_method}")
        
        # Combine all text
        full_text = '\n\n'.join(page.text for page in pages if page.text.strip())
        word_count = sum(page.word_count for page in pages)
        char_count = sum(page.char_count for page in pages)
        
        # Calculate reading time (average 200 words per minute)
        reading_time_minutes = word_count / 200.0
        
        # Detect document structure
        sections = self.detect_document_structure(pages)
        
        # Language and topic detection
        language_detected = self.detect_language(full_text)
        topics_detected = self.detect_topics(full_text)
        
        # Calculate quality score
        quality_score = self.calculate_quality_score(pages, sections)
        
        analysis = DocumentAnalysis(
            metadata=metadata,
            pages=pages,
            sections=sections,
            full_text=full_text,
            word_count=word_count,
            char_count=char_count,
            reading_time_minutes=reading_time_minutes,
            language_detected=language_detected,
            topics_detected=topics_detected,
            quality_score=quality_score,
            processing_errors=processing_errors
        )
        
        self._log_operation('process_pdf', 
                          f"Completed processing: {word_count} words, {len(sections)} sections, quality: {quality_score:.1f}")
        
        return analysis
    
    def extract_for_llm(self, analysis: DocumentAnalysis, 
                       chunk_size: int = 1000, overlap: int = 200) -> List[Dict[str, Any]]:
        """
        Extract and chunk text for LLM training or inference.
        
        Args:
            analysis: Document analysis results
            chunk_size: Target size of text chunks (in characters)
            overlap: Overlap between chunks (in characters)
            
        Returns:
            List of text chunks with metadata
        """
        chunks = []
        
        # Method 1: Chunk by sections (preserves structure)
        for i, section in enumerate(analysis.sections):
            if len(section.content) <= chunk_size:
                # Section fits in one chunk
                chunk = {
                    'chunk_id': f"{analysis.metadata.filename}_section_{i}",
                    'text': section.content,
                    'metadata': {
                        'document': analysis.metadata.filename,
                        'section_title': section.title,
                        'section_type': section.section_type,
                        'page_range': section.page_range,
                        'word_count': section.word_count,
                        'chunk_type': 'section',
                        'document_topics': analysis.topics_detected,
                        'language': analysis.language_detected
                    }
                }
                chunks.append(chunk)
            else:
                # Split large section into smaller chunks
                section_chunks = self._split_text_into_chunks(
                    section.content, chunk_size, overlap
                )
                
                for j, chunk_text in enumerate(section_chunks):
                    chunk = {
                        'chunk_id': f"{analysis.metadata.filename}_section_{i}_chunk_{j}",
                        'text': chunk_text,
                        'metadata': {
                            'document': analysis.metadata.filename,
                            'section_title': section.title,
                            'section_type': section.section_type,
                            'page_range': section.page_range,
                            'chunk_index': j,
                            'chunk_type': 'section_fragment',
                            'document_topics': analysis.topics_detected,
                            'language': analysis.language_detected
                        }
                    }
                    chunks.append(chunk)
        
        # If no sections were detected, chunk the full text
        if not chunks and analysis.full_text:
            text_chunks = self._split_text_into_chunks(
                analysis.full_text, chunk_size, overlap
            )
            
            for i, chunk_text in enumerate(text_chunks):
                chunk = {
                    'chunk_id': f"{analysis.metadata.filename}_chunk_{i}",
                    'text': chunk_text,
                    'metadata': {
                        'document': analysis.metadata.filename,
                        'chunk_index': i,
                        'chunk_type': 'text_fragment',
                        'document_topics': analysis.topics_detected,
                        'language': analysis.language_detected,
                        'total_chunks': len(text_chunks)
                    }
                }
                chunks.append(chunk)
        
        self._log_operation('extract_for_llm', f"Created {len(chunks)} text chunks")
        return chunks
    
    def _split_text_into_chunks(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """Split text into overlapping chunks."""
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence ending within the last 200 characters
                last_period = text.rfind('.', end - 200, end)
                last_exclamation = text.rfind('!', end - 200, end)
                last_question = text.rfind('?', end - 200, end)
                
                sentence_end = max(last_period, last_exclamation, last_question)
                
                if sentence_end > start:
                    end = sentence_end + 1
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - overlap
            
            # Avoid infinite loop
            if start <= 0:
                break
        
        return chunks
    
    def export_analysis(self, analysis: DocumentAnalysis, output_path: str, 
                       format: str = 'json') -> None:
        """
        Export document analysis results.
        
        Args:
            analysis: Document analysis to export
            output_path: Output file path
            format: Export format ('json', 'txt', 'csv')
        """
        output_path = Path(output_path)
        
        if format.lower() == 'json':
            # Convert to serializable format
            data = {
                'metadata': {
                    'filename': analysis.metadata.filename,
                    'file_size': analysis.metadata.file_size,
                    'page_count': analysis.metadata.page_count,
                    'author': analysis.metadata.author,
                    'title': analysis.metadata.title,
                    'creation_date': analysis.metadata.creation_date.isoformat() if analysis.metadata.creation_date else None,
                    'document_hash': analysis.metadata.document_hash,
                    'has_images': analysis.metadata.has_images,
                    'has_tables': analysis.metadata.has_tables
                },
                'analysis': {
                    'word_count': analysis.word_count,
                    'char_count': analysis.char_count,
                    'reading_time_minutes': analysis.reading_time_minutes,
                    'language_detected': analysis.language_detected,
                    'topics_detected': analysis.topics_detected,
                    'quality_score': analysis.quality_score
                },
                'sections': [
                    {
                        'title': section.title,
                        'section_type': section.section_type,
                        'page_range': section.page_range,
                        'word_count': section.word_count,
                        'confidence': section.confidence,
                        'content_preview': section.content[:200] + ('...' if len(section.content) > 200 else '')
                    }
                    for section in analysis.sections
                ],
                'processing_errors': analysis.processing_errors,
                'processing_log': self.processing_log
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        
        elif format.lower() == 'txt':
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(f"Document Analysis: {analysis.metadata.filename}\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Word Count: {analysis.word_count}\n")
                f.write(f"Reading Time: {analysis.reading_time_minutes:.1f} minutes\n")
                f.write(f"Quality Score: {analysis.quality_score:.1f}/100\n")
                f.write(f"Language: {analysis.language_detected or 'Unknown'}\n")
                f.write(f"Topics: {', '.join(analysis.topics_detected) or 'None detected'}\n\n")
                
                f.write("Full Text:\n")
                f.write("-" * 20 + "\n")
                f.write(analysis.full_text)
        
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        self._log_operation('export_analysis', f"Exported analysis to {output_path} ({format} format)")
    
    def process_file(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Process a PDF file and return summary information.
        Compatible method for support agent integration.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Dictionary with processing results and statistics
        """
        try:
            # Extract text and analyze document
            analysis = self.analyze_document(file_path)
            
            # Calculate summary statistics
            total_pages = len(analysis.pages)
            total_words = sum(page.word_count for page in analysis.pages)
            total_chars = sum(page.char_count for page in analysis.pages)
            
            # Return compatible format for support agent
            return {
                'status': 'success',
                'pages_processed': total_pages,
                'words_extracted': total_words,
                'characters_extracted': total_chars,
                'sections_found': len(analysis.sections),
                'file_type': 'pdf',
                'document_title': analysis.metadata.title or 'Unknown',
                'document_author': analysis.metadata.author or 'Unknown',
                'processing_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'file_type': 'pdf',
                'processing_timestamp': datetime.now().isoformat()
            }


def create_sample_pdf_analysis():
    """Create a sample PDF analysis for demonstration purposes."""
    from datetime import datetime
    
    # Create sample metadata
    metadata = DocumentMetadata(
        filename="sample_document.pdf",
        file_size=1024000,  # 1MB
        creation_date=datetime(2023, 1, 15, 10, 30),
        modification_date=datetime(2023, 2, 10, 14, 45),
        author="Dr. Jane Smith",
        title="Introduction to Machine Learning",
        subject="Artificial Intelligence",
        creator="LaTeX",
        producer="pdfTeX-1.40.21",
        page_count=25,
        is_encrypted=False,
        has_images=True,
        has_tables=True,
        language="english",
        document_hash="a1b2c3d4e5f6"
    )
    
    # Create sample pages
    pages = [
        PageInfo(1, "Abstract\n\nMachine learning is a subset of artificial intelligence...", 45, 250, False, False),
        PageInfo(2, "Table of Contents\n\n1. Introduction\n2. Methodology\n3. Results...", 25, 150, False, True),
        PageInfo(3, "1. Introduction\n\nMachine learning has revolutionized...", 180, 980, True, False),
        PageInfo(4, "2. Methodology\n\nOur approach uses supervised learning...", 165, 890, False, True),
        PageInfo(5, "3. Results\n\nThe experimental results show significant...", 200, 1100, True, True)
    ]
    
    # Create sample sections
    sections = [
        TextSection("Abstract", "Machine learning is a subset of artificial intelligence that focuses on algorithms...", (1, 1), "header", 0.95, 45),
        TextSection("Introduction", "Machine learning has revolutionized how we approach data analysis...", (3, 3), "header", 0.90, 180),
        TextSection("Methodology", "Our approach uses supervised learning techniques with cross-validation...", (4, 4), "header", 0.88, 165),
        TextSection("Results", "The experimental results show significant improvements over baseline methods...", (5, 5), "header", 0.92, 200)
    ]
    
    # Create full analysis
    full_text = "\n\n".join(section.content for section in sections)
    
    analysis = DocumentAnalysis(
        metadata=metadata,
        pages=pages,
        sections=sections,
        full_text=full_text,
        word_count=590,
        char_count=3370,
        reading_time_minutes=2.95,
        language_detected="english",
        topics_detected=["technology", "science"],
        quality_score=87.5,
        processing_errors=[]
    )
    
    return analysis


def demo_pdf_processing():
    """Demonstrate PDF processing capabilities."""
    print("PDF Processor Demo")
    print("=" * 50)
    
    if not HAS_PDF_LIBS:
        print("PDF libraries not installed. Using sample data for demonstration.")
        print("To use with real PDFs, install: pip install PyPDF2 pdfplumber Pillow")
        print()
        
        # Create sample analysis
        analysis = create_sample_pdf_analysis()
        
        print("Sample PDF Analysis Results:")
        print("-" * 30)
        print(f"Document: {analysis.metadata.filename}")
        print(f"Pages: {analysis.metadata.page_count}")
        print(f"Author: {analysis.metadata.author}")
        print(f"Title: {analysis.metadata.title}")
        print(f"File Size: {analysis.metadata.file_size / 1024:.1f} KB")
        print()
        
        print("Content Analysis:")
        print(f"Word Count: {analysis.word_count}")
        print(f"Reading Time: {analysis.reading_time_minutes:.1f} minutes")
        print(f"Language: {analysis.language_detected}")
        print(f"Topics: {', '.join(analysis.topics_detected)}")
        print(f"Quality Score: {analysis.quality_score:.1f}/100")
        print()
        
        print("Document Structure:")
        for i, section in enumerate(analysis.sections, 1):
            print(f"  {i}. {section.title} (Page {section.page_range[0]})")
            print(f"     Type: {section.section_type}, Words: {section.word_count}")
        print()
        
        # Demonstrate LLM chunk extraction
        processor = PDFProcessor()
        chunks = processor.extract_for_llm(analysis, chunk_size=500, overlap=100)
        
        print("LLM-Ready Text Chunks:")
        print("-" * 25)
        for i, chunk in enumerate(chunks[:3], 1):  # Show first 3 chunks
            print(f"Chunk {i} (ID: {chunk['chunk_id']}):")
            print(f"  Text: {chunk['text'][:100]}...")
            print(f"  Metadata: {chunk['metadata']['section_type']}, {chunk['metadata']['word_count']} words")
            print()
        
        print(f"Total chunks created: {len(chunks)}")
        
        # Export sample analysis
        processor.export_analysis(analysis, 'sample_pdf_analysis.json', 'json')
        processor.export_analysis(analysis, 'sample_pdf_analysis.txt', 'txt')
        print("Sample analysis exported to JSON and TXT formats")
        
        return
    
    # If libraries are available, show how to process real PDFs
    processor = PDFProcessor(use_ocr=HAS_OCR)
    
    print("PDF Processing Capabilities:")
    print("- Text extraction with PyPDF2 and pdfplumber")
    print("- Metadata extraction and validation")
    print("- Document structure analysis")
    print("- Language and topic detection")
    print("- Quality scoring and assessment")
    print("- LLM-ready text chunking")
    if HAS_OCR:
        print("- OCR support for image-based PDFs")
    print()
    
    print("To process a PDF file:")
    print("  processor = PDFProcessor()")
    print("  analysis = processor.process_pdf('your_document.pdf')")
    print("  chunks = processor.extract_for_llm(analysis)")
    print("  processor.export_analysis(analysis, 'analysis.json')")
    print()
    
    print("Processing Methods Available:")
    print("  - 'auto': Intelligent method selection")
    print("  - 'pypdf2': Fast but basic extraction")
    print("  - 'pdfplumber': Slower but more accurate")
    if HAS_OCR:
        print("  - 'ocr': OCR for image-based PDFs")
    
    # Show sample with mock data
    analysis = create_sample_pdf_analysis()
    chunks = processor.extract_for_llm(analysis)
    print(f"\nSample processing created {len(chunks)} chunks from {analysis.metadata.page_count} pages")


# Utility functions for integration with other processors

def batch_process_pdfs(pdf_directory: str, processor: PDFProcessor) -> List[DocumentAnalysis]:
    """
    Process multiple PDFs in a directory.
    
    Args:
        pdf_directory: Directory containing PDF files
        processor: Configured PDFProcessor instance
        
    Returns:
        List of document analyses
    """
    pdf_dir = Path(pdf_directory)
    analyses = []
    
    if not pdf_dir.exists():
        raise FileNotFoundError(f"Directory not found: {pdf_directory}")
    
    pdf_files = list(pdf_dir.glob("*.pdf"))
    
    for pdf_file in pdf_files:
        try:
            print(f"Processing {pdf_file.name}...")
            analysis = processor.process_pdf(pdf_file)
            analyses.append(analysis)
        except Exception as e:
            print(f"Error processing {pdf_file.name}: {str(e)}")
    
    return analyses


def create_training_dataset(analyses: List[DocumentAnalysis], 
                           output_path: str, chunk_size: int = 1000) -> None:
    """
    Create LLM training dataset from multiple PDF analyses.
    
    Args:
        analyses: List of document analyses
        output_path: Path for output JSONL file
        chunk_size: Size of text chunks
    """
    processor = PDFProcessor()
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for analysis in analyses:
            chunks = processor.extract_for_llm(analysis, chunk_size=chunk_size)
            
            for chunk in chunks:
                # Format for training (can be customized)
                training_example = {
                    'text': chunk['text'],
                    'source': chunk['metadata']['document'],
                    'section_type': chunk['metadata'].get('section_type', 'unknown'),
                    'topics': chunk['metadata'].get('document_topics', []),
                    'language': chunk['metadata'].get('language', 'unknown')
                }
                
                f.write(json.dumps(training_example, ensure_ascii=False) + '\n')


def search_documents(analyses: List[DocumentAnalysis], query: str, 
                    top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Simple text search across processed documents.
    
    Args:
        analyses: List of document analyses
        query: Search query
        top_k: Number of top results to return
        
    Returns:
        List of search results with relevance scores
    """
    results = []
    query_lower = query.lower()
    
    for analysis in analyses:
        # Simple keyword matching (can be enhanced with TF-IDF, embeddings, etc.)
        score = 0
        matches = []
        
        # Search in full text
        text_lower = analysis.full_text.lower()
        query_count = text_lower.count(query_lower)
        score += query_count * 10
        
        # Search in sections
        for section in analysis.sections:
            section_text_lower = section.content.lower()
            section_matches = section_text_lower.count(query_lower)
            if section_matches > 0:
                score += section_matches * 5
                matches.append({
                    'section': section.title,
                    'page_range': section.page_range,
                    'matches': section_matches
                })
        
        # Bonus for title/topic matches
        if analysis.metadata.title and query_lower in analysis.metadata.title.lower():
            score += 20
        
        if query_lower in [topic.lower() for topic in analysis.topics_detected]:
            score += 15
        
        if score > 0:
            results.append({
                'document': analysis.metadata.filename,
                'score': score,
                'matches': matches,
                'word_count': analysis.word_count,
                'quality_score': analysis.quality_score,
                'topics': analysis.topics_detected
            })
    
    # Sort by relevance score
    results.sort(key=lambda x: x['score'], reverse=True)
    return results[:top_k]


if __name__ == "__main__":
    demo_pdf_processing()
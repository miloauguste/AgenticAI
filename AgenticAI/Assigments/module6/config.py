"""
Configuration settings for the Customer Support Triage Agent
Centralized configuration management with environment variable support,
validation, and intelligent defaults for different deployment scenarios.
"""
import os
import sys
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import logging
from datetime import datetime

# Try to load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
    HAS_DOTENV = True
except ImportError:
    HAS_DOTENV = False
    print("Warning: python-dotenv not installed. Environment variables must be set manually.")

class ConfigurationError(Exception):
    """Custom exception for configuration-related errors"""
    pass

class Settings:
    """
    Comprehensive application configuration settings with intelligent defaults,
    validation, and support for multiple deployment environments.
    """
    
    # ============================================================================
    # API CONFIGURATION
    # ============================================================================
    
    # OpenAI Configuration
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_ORG_ID: str = os.getenv("OPENAI_ORG_ID", "")
    OPENAI_API_BASE: str = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
    
    # Google AI Configuration  
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
    GOOGLE_PROJECT_ID: str = os.getenv("GOOGLE_PROJECT_ID", "")
    
    # Pinecone Configuration
    PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY", "")
    PINECONE_ENVIRONMENT: str = os.getenv("PINECONE_ENVIRONMENT", "us-west1-gcp-free")
    PINECONE_INDEX_NAME: str = os.getenv("PINECONE_INDEX_NAME", "support-triage")
    
    # Hugging Face Configuration (for local models)
    HUGGINGFACE_API_KEY: str = os.getenv("HUGGINGFACE_API_KEY", "")
    
    # ============================================================================
    # MODEL CONFIGURATION
    # ============================================================================
    
    # Primary LLM Settings
    DEFAULT_LLM_MODEL: str = os.getenv("DEFAULT_LLM_MODEL", "gpt-3.5-turbo")
    FALLBACK_LLM_MODEL: str = os.getenv("FALLBACK_LLM_MODEL", "gpt-3.5-turbo")
    MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", "2000"))
    TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.3"))
    TOP_P: float = float(os.getenv("TOP_P", "1.0"))
    FREQUENCY_PENALTY: float = float(os.getenv("FREQUENCY_PENALTY", "0.0"))
    PRESENCE_PENALTY: float = float(os.getenv("PRESENCE_PENALTY", "0.0"))
    
    # Model Selection Strategy
    MODEL_SELECTION_STRATEGY: str = os.getenv("MODEL_SELECTION_STRATEGY", "cost_optimized")  # options: cost_optimized, performance, balanced
    
    # Available Models Configuration
    AVAILABLE_MODELS: Dict[str, Dict[str, Any]] = {
        "gpt-4": {
            "provider": "openai",
            "cost_per_token": 0.00003,
            "context_window": 8192,
            "capabilities": ["text", "reasoning", "complex_tasks"],
            "recommended_for": ["complex_analysis", "high_quality_responses"]
        },
        "gpt-4-turbo": {
            "provider": "openai", 
            "cost_per_token": 0.00001,
            "context_window": 128000,
            "capabilities": ["text", "reasoning", "long_context"],
            "recommended_for": ["document_analysis", "batch_processing"]
        },
        "gpt-3.5-turbo": {
            "provider": "openai",
            "cost_per_token": 0.0000015,
            "context_window": 4096,
            "capabilities": ["text", "fast_responses"],
            "recommended_for": ["quick_analysis", "high_volume"]
        },
        "gemini-pro": {
            "provider": "google",
            "cost_per_token": 0.000001,
            "context_window": 32768,
            "capabilities": ["text", "multimodal"],
            "recommended_for": ["cost_effective", "mixed_content"]
        }
    }
    
    # Embedding Models
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    EMBEDDING_DIMENSION: int = int(os.getenv("EMBEDDING_DIMENSION", "384"))
    
    AVAILABLE_EMBEDDING_MODELS: Dict[str, Dict[str, Any]] = {
        "all-MiniLM-L6-v2": {"dimension": 384, "max_seq_length": 256, "size_mb": 80},
        "all-mpnet-base-v2": {"dimension": 768, "max_seq_length": 384, "size_mb": 420},
        "text-embedding-ada-002": {"dimension": 1536, "provider": "openai", "cost_per_token": 0.0000001}
    }
    
    # ============================================================================
    # VECTOR DATABASE CONFIGURATION  
    # ============================================================================
    
    VECTOR_DIMENSION: int = int(os.getenv("VECTOR_DIMENSION", str(EMBEDDING_DIMENSION)))
    TOP_K_RESULTS: int = int(os.getenv("TOP_K_RESULTS", "5"))
    SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.7"))
    
    # Vector Database Selection
    VECTOR_DB_PROVIDER: str = os.getenv("VECTOR_DB_PROVIDER", "pinecone")  # options: pinecone, chroma, faiss, memory
    
    # Pinecone Specific Settings
    PINECONE_METRIC: str = os.getenv("PINECONE_METRIC", "cosine")
    PINECONE_PODS: int = int(os.getenv("PINECONE_PODS", "1"))
    PINECONE_POD_TYPE: str = os.getenv("PINECONE_POD_TYPE", "p1.x1")
    
    # ============================================================================
    # FILE PROCESSING CONFIGURATION
    # ============================================================================
    
    # File Size Limits
    MAX_FILE_SIZE_MB: int = int(os.getenv("MAX_FILE_SIZE_MB", "50"))
    MAX_BATCH_SIZE: int = int(os.getenv("MAX_BATCH_SIZE", "100"))
    
    # Supported File Types
    SUPPORTED_FILE_TYPES: List[str] = ['.csv', '.txt', '.pdf', '.docx', '.xlsx', '.json']
    SUPPORTED_ENCODINGS: List[str] = ['utf-8', 'utf-16', 'latin-1', 'iso-8859-1', 'cp1252']
    
    # Text Processing
    MIN_TEXT_LENGTH: int = int(os.getenv("MIN_TEXT_LENGTH", "10"))
    MAX_TEXT_LENGTH: int = int(os.getenv("MAX_TEXT_LENGTH", "50000"))
    
    # Chunking Configuration
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))
    MIN_CHUNK_SIZE: int = int(os.getenv("MIN_CHUNK_SIZE", "100"))
    MAX_CHUNKS_PER_FILE: int = int(os.getenv("MAX_CHUNKS_PER_FILE", "1000"))
    
    # Chunking Strategies
    CHUNKING_STRATEGIES: Dict[str, Dict[str, Any]] = {
        "ticket": {
            "strategy": "individual",
            "max_size": 2000,
            "overlap": 0,
            "preserve_structure": True
        },
        "policy": {
            "strategy": "section_based", 
            "max_size": 1500,
            "overlap": 200,
            "preserve_hierarchy": True
        },
        "conversation": {
            "strategy": "turn_based",
            "max_size": 2000,
            "overlap": 100,
            "preserve_context": True
        }
    }
    
    # ============================================================================
    # DIRECTORY CONFIGURATION
    # ============================================================================
    
    # Base Paths
    BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT: str = os.getenv("PROJECT_ROOT", BASE_DIR)
    
    # Data Directories
    DATA_DIR: str = os.getenv("DATA_DIR", os.path.join(PROJECT_ROOT, "data"))
    UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", os.path.join(DATA_DIR, "uploads"))
    PROCESSED_DIR: str = os.getenv("PROCESSED_DIR", os.path.join(DATA_DIR, "processed"))
    TEMP_DIR: str = os.getenv("TEMP_DIR", os.path.join(DATA_DIR, "temp"))
    BACKUP_DIR: str = os.getenv("BACKUP_DIR", os.path.join(DATA_DIR, "backups"))
    
    # Log Configuration
    LOG_DIR: str = os.getenv("LOG_DIR", os.path.join(PROJECT_ROOT, "logs"))
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_MAX_SIZE: int = int(os.getenv("LOG_MAX_SIZE", "10485760"))  # 10MB
    LOG_BACKUP_COUNT: int = int(os.getenv("LOG_BACKUP_COUNT", "5"))
    
    # ============================================================================
    # UI CONFIGURATION
    # ============================================================================
    
    # Streamlit Settings
    STREAMLIT_PORT: int = int(os.getenv("STREAMLIT_PORT", "8501"))
    STREAMLIT_HOST: str = os.getenv("STREAMLIT_HOST", "localhost")
    PAGE_TITLE: str = os.getenv("PAGE_TITLE", "Customer Support Triage Agent")
    PAGE_ICON: str = os.getenv("PAGE_ICON", "🎯")
    
    # UI Feature Flags
    ENABLE_FILE_UPLOAD: bool = os.getenv("ENABLE_FILE_UPLOAD", "true").lower() == "true"
    ENABLE_BATCH_PROCESSING: bool = os.getenv("ENABLE_BATCH_PROCESSING", "true").lower() == "true"
    ENABLE_ANALYTICS: bool = os.getenv("ENABLE_ANALYTICS", "true").lower() == "true"
    ENABLE_EXPORT: bool = os.getenv("ENABLE_EXPORT", "true").lower() == "true"
    
    # Pagination and Display
    ITEMS_PER_PAGE: int = int(os.getenv("ITEMS_PER_PAGE", "10"))
    MAX_DISPLAY_RESULTS: int = int(os.getenv("MAX_DISPLAY_RESULTS", "100"))
    
    # ============================================================================
    # SENTIMENT ANALYSIS CONFIGURATION
    # ============================================================================
    
    # Thresholds
    SENTIMENT_THRESHOLD: float = float(os.getenv("SENTIMENT_THRESHOLD", "0.6"))
    URGENCY_THRESHOLD: float = float(os.getenv("URGENCY_THRESHOLD", "0.7"))
    ESCALATION_THRESHOLD: float = float(os.getenv("ESCALATION_THRESHOLD", "0.8"))
    
    # Keywords and Patterns
    URGENCY_KEYWORDS: List[str] = [
        "urgent", "asap", "immediately", "emergency", "critical", "priority",
        "frustrated", "angry", "disappointed", "unacceptable", "terrible",
        "deadline", "time sensitive", "running out of time", "need help now"
    ]
    
    ESCALATION_KEYWORDS: List[str] = [
        "manager", "supervisor", "escalate", "complaint", "legal action",
        "lawyer", "attorney", "sue", "court", "better business bureau",
        "social media", "review", "cancel", "refund", "never again"
    ]
    
    POLITENESS_KEYWORDS: List[str] = [
        "please", "thank you", "thanks", "appreciate", "kindly",
        "would you mind", "if possible", "when convenient"
    ]
    
    # ============================================================================
    # INTENT CLASSIFICATION CONFIGURATION
    # ============================================================================
    
    # Intent Categories
    INTENT_CATEGORIES: Dict[str, List[str]] = {
        "refund": ["refund", "money back", "return", "cancel order", "reimbursement", "get my money back"],
        "delivery": ["shipping", "delivery", "tracking", "delayed", "lost package", "where is my order"],
        "product_issue": ["defective", "broken", "quality", "not working", "damaged", "wrong item"],
        "account": ["login", "password", "account", "profile", "subscription", "access"],
        "billing": ["charge", "payment", "invoice", "billing", "transaction", "credit card"],
        "technical_support": ["technical", "bug", "error", "not responding", "crash", "fix"],
        "complaint": ["complaint", "unsatisfied", "disappointed", "poor service", "manager"],
        "cancellation": ["cancel", "stop", "discontinue", "unsubscribe", "terminate"],
        "exchange": ["exchange", "swap", "different", "size", "color", "model"],
        "information": ["question", "how to", "information", "help", "support", "explain"]
    }
    
    # Department Routing
    DEPARTMENT_ROUTING: Dict[str, str] = {
        "refund": "billing",
        "delivery": "logistics", 
        "product_issue": "quality_assurance",
        "account": "technical_support",
        "billing": "billing",
        "technical_support": "technical_support", 
        "complaint": "customer_service",
        "cancellation": "customer_retention",
        "exchange": "customer_service",
        "information": "customer_service"
    }
    
    # Confidence Thresholds
    INTENT_CONFIDENCE_THRESHOLD: float = float(os.getenv("INTENT_CONFIDENCE_THRESHOLD", "0.6"))
    LOW_CONFIDENCE_THRESHOLD: float = float(os.getenv("LOW_CONFIDENCE_THRESHOLD", "0.3"))
    
    # ============================================================================
    # RESPONSE GENERATION CONFIGURATION
    # ============================================================================
    
    # Response Templates
    RESPONSE_TEMPLATES: Dict[str, Dict[str, str]] = {
        "refund": {
            "opening": "Thank you for contacting us regarding your refund request.",
            "empathy": "I understand your concern about this matter and want to help resolve it quickly.",
            "action": "I'll review your case and process your refund request according to our policy.", 
            "closing": "Is there anything else I can help you with regarding this refund?",
            "timeline": "Refunds typically take 3-5 business days to process once approved."
        },
        "delivery": {
            "opening": "Thank you for reaching out about your delivery.",
            "empathy": "I apologize for any inconvenience with your shipment.",
            "action": "Let me track your order and provide you with an immediate update.",
            "closing": "I'll keep you informed of any developments with your delivery.",
            "timeline": "Most delivery issues can be resolved within 24-48 hours."
        },
        "product_issue": {
            "opening": "Thank you for bringing this product issue to our attention.",
            "empathy": "I'm sorry to hear that your item isn't working as expected.",
            "action": "I'll help you resolve this issue and ensure you're completely satisfied.",
            "closing": "We're committed to making sure you have a great experience with our products.",
            "timeline": "Product issues are typically resolved within 1-3 business days."
        },
        "account": {
            "opening": "Thank you for contacting us about your account.",
            "empathy": "I understand how frustrating account access issues can be.",
            "action": "I'll help you regain access to your account securely and quickly.",
            "closing": "Please let me know if you need any additional assistance with your account.",
            "timeline": "Account issues are usually resolved within a few hours."
        },
        "billing": {
            "opening": "Thank you for your billing inquiry.",
            "empathy": "I understand your concern about the charges on your account.",
            "action": "Let me review your billing details and provide a clear explanation.",
            "closing": "I'm here to help resolve any billing questions or concerns you may have.",
            "timeline": "Billing inquiries are typically resolved within 1-2 business days."
        },
        "complaint": {
            "opening": "Thank you for bringing this matter to our attention.",
            "empathy": "I sincerely apologize for the experience you've had, and I take your concerns seriously.",
            "action": "I'll personally ensure this issue is addressed and work to make things right.",
            "closing": "Your feedback is valuable and helps us improve our service for everyone.",
            "timeline": "We aim to address all complaints within 24 hours and provide resolution within 3-5 days."
        },
        "general": {
            "opening": "Thank you for contacting our support team.",
            "empathy": "I'm here to help you with your inquiry and ensure you have a positive experience.",
            "action": "Let me assist you with this matter and find the best solution.",
            "closing": "Please don't hesitate to reach out if you have any other questions or concerns.",
            "timeline": "Most inquiries are handled within 24 hours."
        }
    }
    
    # Response Quality Settings
    MIN_RESPONSE_LENGTH: int = int(os.getenv("MIN_RESPONSE_LENGTH", "50"))
    MAX_RESPONSE_LENGTH: int = int(os.getenv("MAX_RESPONSE_LENGTH", "500"))
    TARGET_RESPONSE_LENGTH: int = int(os.getenv("TARGET_RESPONSE_LENGTH", "150"))
    
    # ============================================================================
    # PERFORMANCE AND CACHING
    # ============================================================================
    
    # Rate Limiting
    API_RATE_LIMIT: int = int(os.getenv("API_RATE_LIMIT", "60"))  # requests per minute
    BATCH_PROCESSING_DELAY: float = float(os.getenv("BATCH_PROCESSING_DELAY", "1.0"))  # seconds
    
    # Caching
    ENABLE_CACHING: bool = os.getenv("ENABLE_CACHING", "true").lower() == "true"
    CACHE_TTL: int = int(os.getenv("CACHE_TTL", "3600"))  # seconds
    MAX_CACHE_SIZE: int = int(os.getenv("MAX_CACHE_SIZE", "1000"))  # items
    
    # Performance Monitoring
    ENABLE_PERFORMANCE_MONITORING: bool = os.getenv("ENABLE_PERFORMANCE_MONITORING", "true").lower() == "true"
    SLOW_QUERY_THRESHOLD: float = float(os.getenv("SLOW_QUERY_THRESHOLD", "5.0"))  # seconds
    
    # ============================================================================
    # SECURITY AND PRIVACY
    # ============================================================================
    
    # Data Privacy
    ENABLE_DATA_ANONYMIZATION: bool = os.getenv("ENABLE_DATA_ANONYMIZATION", "false").lower() == "true"
    RETAIN_CUSTOMER_DATA: bool = os.getenv("RETAIN_CUSTOMER_DATA", "false").lower() == "true"
    DATA_RETENTION_DAYS: int = int(os.getenv("DATA_RETENTION_DAYS", "30"))
    
    # Security
    ENABLE_REQUEST_VALIDATION: bool = os.getenv("ENABLE_REQUEST_VALIDATION", "true").lower() == "true"
    MAX_REQUEST_SIZE: int = int(os.getenv("MAX_REQUEST_SIZE", "10485760"))  # 10MB
    
    # ============================================================================
    # ENVIRONMENT AND DEPLOYMENT
    # ============================================================================
    
    # Environment Detection
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")  # development, staging, production
    DEBUG_MODE: bool = os.getenv("DEBUG", "false").lower() == "true"
    
    # Feature Flags based on Environment
    @property
    def is_production(self) -> bool:
        return self.ENVIRONMENT.lower() == "production"
    
    @property 
    def is_development(self) -> bool:
        return self.ENVIRONMENT.lower() == "development"
        
    @property
    def is_staging(self) -> bool:
        return self.ENVIRONMENT.lower() == "staging"
    
    # ============================================================================
    # METHODS
    # ============================================================================
    
    @classmethod
    def create_directories(cls) -> None:
        """Create necessary directories if they don't exist"""
        directories = [
            cls.DATA_DIR, cls.UPLOAD_DIR, cls.PROCESSED_DIR, 
            cls.TEMP_DIR, cls.BACKUP_DIR, cls.LOG_DIR
        ]
        
        for directory in directories:
            try:
                Path(directory).mkdir(parents=True, exist_ok=True)
            except Exception as e:
                raise ConfigurationError(f"Failed to create directory {directory}: {str(e)}")
    
    @classmethod
    def validate_config(cls) -> Dict[str, Any]:
        """
        Validate configuration and return detailed status
        
        Returns:
            Dictionary with validation results and recommendations
        """
        validation_results = {
            "overall_status": "unknown",
            "api_keys": {},
            "directories": {},
            "model_config": {},
            "warnings": [],
            "errors": [],
            "recommendations": []
        }
        
        try:
            # Validate API Keys
            validation_results["api_keys"] = {
                "openai": {
                    "configured": bool(cls.OPENAI_API_KEY),
                    "format_valid": cls.OPENAI_API_KEY.startswith("sk-") if cls.OPENAI_API_KEY else False
                },
                "pinecone": {
                    "configured": bool(cls.PINECONE_API_KEY),
                    "environment_set": bool(cls.PINECONE_ENVIRONMENT)
                },
                "google": {
                    "configured": bool(cls.GOOGLE_API_KEY)
                }
            }
            
            # Validate Directories
            try:
                cls.create_directories()
                validation_results["directories"]["created"] = True
            except Exception as e:
                validation_results["directories"]["created"] = False
                validation_results["errors"].append(f"Directory creation failed: {str(e)}")
            
            # Validate Model Configuration
            validation_results["model_config"] = {
                "default_model": cls.DEFAULT_LLM_MODEL in cls.AVAILABLE_MODELS,
                "embedding_model": cls.EMBEDDING_MODEL in cls.AVAILABLE_EMBEDDING_MODELS,
                "temperature_valid": 0.0 <= cls.TEMPERATURE <= 2.0,
                "max_tokens_valid": 1 <= cls.MAX_TOKENS <= 32768
            }
            
            # Generate Warnings
            if not validation_results["api_keys"]["openai"]["configured"]:
                validation_results["warnings"].append("OpenAI API key not configured - will use template responses only")
            
            if not validation_results["api_keys"]["pinecone"]["configured"]:
                validation_results["warnings"].append("Pinecone API key not configured - will use in-memory vector storage")
                
            if cls.CHUNK_SIZE > 2000:
                validation_results["warnings"].append("Large chunk size may impact processing performance")
            
            if cls.TEMPERATURE > 1.0:
                validation_results["warnings"].append("High temperature setting may produce inconsistent responses")
            
            # Generate Recommendations
            if cls.is_development:
                validation_results["recommendations"].append("Consider using lighter models for faster development")
            
            if not cls.ENABLE_CACHING and cls.is_production:
                validation_results["recommendations"].append("Enable caching for better production performance")
                
            if cls.MAX_FILE_SIZE_MB > 100:
                validation_results["recommendations"].append("Consider reducing max file size for better memory usage")
            
            # Determine Overall Status
            error_count = len(validation_results["errors"])
            warning_count = len(validation_results["warnings"])
            
            if error_count == 0:
                if warning_count == 0:
                    validation_results["overall_status"] = "excellent"
                elif warning_count <= 2:
                    validation_results["overall_status"] = "good"
                else:
                    validation_results["overall_status"] = "acceptable"
            else:
                validation_results["overall_status"] = "issues_found"
            
            return validation_results
            
        except Exception as e:
            validation_results["overall_status"] = "validation_failed"
            validation_results["errors"].append(f"Configuration validation failed: {str(e)}")
            return validation_results
    
    @classmethod
    def get_model_config(cls, model_name: Optional[str] = None) -> Dict[str, Any]:
        """Get configuration for a specific model"""
        model_name = model_name or cls.DEFAULT_LLM_MODEL
        
        if model_name not in cls.AVAILABLE_MODELS:
            raise ConfigurationError(f"Model '{model_name}' not found in available models")
        
        config = cls.AVAILABLE_MODELS[model_name].copy()
        config.update({
            "temperature": cls.TEMPERATURE,
            "max_tokens": cls.MAX_TOKENS,
            "top_p": cls.TOP_P,
            "frequency_penalty": cls.FREQUENCY_PENALTY,
            "presence_penalty": cls.PRESENCE_PENALTY
        })
        
        return config
    
    @classmethod 
    def get_embedding_config(cls, model_name: Optional[str] = None) -> Dict[str, Any]:
        """Get configuration for embedding model"""
        model_name = model_name or cls.EMBEDDING_MODEL
        
        if model_name not in cls.AVAILABLE_EMBEDDING_MODELS:
            raise ConfigurationError(f"Embedding model '{model_name}' not found")
        
        return cls.AVAILABLE_EMBEDDING_MODELS[model_name].copy()
    
    @classmethod
    def update_from_dict(cls, config_dict: Dict[str, Any]) -> None:
        """Update configuration from dictionary"""
        for key, value in config_dict.items():
            if hasattr(cls, key):
                setattr(cls, key, value)
            else:
                logging.warning(f"Unknown configuration key: {key}")
    
    @classmethod
    def export_config(cls, include_secrets: bool = False) -> Dict[str, Any]:
        """
        Export current configuration
        
        Args:
            include_secrets: Whether to include API keys and sensitive data
        """
        config = {}
        
        # Get all class attributes that are configuration settings
        for attr_name in dir(cls):
            if not attr_name.startswith('_') and not callable(getattr(cls, attr_name)):
                attr_value = getattr(cls, attr_name)
                
                # Skip sensitive data unless explicitly requested
                if not include_secrets and any(sensitive in attr_name.lower() for sensitive in ['key', 'token', 'password', 'secret']):
                    config[attr_name] = "***HIDDEN***"
                else:
                    # Convert Path objects to strings for JSON serialization
                    if isinstance(attr_value, Path):
                        attr_value = str(attr_value)
                    config[attr_name] = attr_value
        
        return config
    
    @classmethod
    def setup_logging(cls) -> None:
        """Setup logging configuration"""
        import logging.handlers
        
        # Create log directory
        Path(cls.LOG_DIR).mkdir(parents=True, exist_ok=True)
        
        # Setup logger
        logger = logging.getLogger()
        logger.setLevel(getattr(logging, cls.LOG_LEVEL.upper()))
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_format)
        
        # File handler with rotation
        log_file = os.path.join(cls.LOG_DIR, 'support_agent.log')
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, 
            maxBytes=cls.LOG_MAX_SIZE, 
            backupCount=cls.LOG_BACKUP_COUNT
        )
        file_handler.setLevel(getattr(logging, cls.LOG_LEVEL.upper()))
        file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s')
        file_handler.setFormatter(file_format)
        
        # Add handlers
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
        
        # Log startup message
        logger.info(f"Logging initialized - Level: {cls.LOG_LEVEL}, Environment: {cls.ENVIRONMENT}")

    @classmethod
    def get_operational_mode(cls) -> str:
        """
        Determine operational mode based on available API keys and configuration
        
        Returns:
            String indicating operational mode: full, basic, demo
        """
        has_openai = bool(cls.OPENAI_API_KEY)
        has_pinecone = bool(cls.PINECONE_API_KEY)
        
        if has_openai and has_pinecone:
            return "full"
        elif has_openai or has_pinecone:
            return "basic"
        else:
            return "demo"
    
    @classmethod
    def print_startup_summary(cls) -> None:
        """Print a startup summary showing configuration status"""
        print("=" * 60)
        print("🎯 CUSTOMER SUPPORT TRIAGE AGENT")
        print("=" * 60)
        print(f"Environment: {cls.ENVIRONMENT.upper()}")
        print(f"Operational Mode: {cls.get_operational_mode().upper()}")
        print(f"Default Model: {cls.DEFAULT_LLM_MODEL}")
        print(f"Embedding Model: {cls.EMBEDDING_MODEL}")
        print(f"Vector DB: {cls.VECTOR_DB_PROVIDER.title()}")
        print(f"Debug Mode: {'ON' if cls.DEBUG_MODE else 'OFF'}")
        
        validation = cls.validate_config()
        print(f"Configuration Status: {validation['overall_status'].replace('_', ' ').title()}")
        
        if validation['warnings']:
            print("\n⚠️  Warnings:")
            for warning in validation['warnings'][:3]:  # Show only first 3
                print(f"  • {warning}")
        
        if validation['errors']:
            print("\n❌ Errors:")
            for error in validation['errors'][:3]:  # Show only first 3
                print(f"  • {error}")
        
        print("=" * 60)

# ============================================================================
# GLOBAL SETTINGS INSTANCE
# ============================================================================

# Create global settings instance
settings = Settings()

# Setup logging on import
try:
    settings.setup_logging()
except Exception as e:
    print(f"Warning: Failed to setup logging: {e}")

# Create directories on import
try:
    settings.create_directories()
except Exception as e:
    print(f"Warning: Failed to create directories: {e}")

# ============================================================================
# ENVIRONMENT-SPECIFIC CONFIGURATIONS
# ============================================================================

class DevelopmentSettings(Settings):
    """Development environment specific settings"""
    DEBUG_MODE = True
    LOG_LEVEL = "DEBUG"
    ENABLE_PERFORMANCE_MONITORING = True
    CHUNK_SIZE = 500  # Smaller chunks for faster processing
    MAX_TOKENS = 1000  # Lower token limit for cost control
    ENABLE_CACHING = False  # Disable caching for development

class ProductionSettings(Settings):
    """Production environment specific settings"""
    DEBUG_MODE = False
    LOG_LEVEL = "INFO"
    ENABLE_PERFORMANCE_MONITORING = True
    ENABLE_CACHING = True
    API_RATE_LIMIT = 100  # Higher rate limit for production
    MAX_FILE_SIZE_MB = 100  # Larger files allowed in production

class TestingSettings(Settings):
    """Testing environment specific settings"""
    DEBUG_MODE = True
    LOG_LEVEL = "DEBUG"
    ENABLE_CACHING = False
    DATA_DIR = "/tmp/support_agent_test"
    CHUNK_SIZE = 200  # Very small chunks for fast testing
    MAX_TOKENS = 500

# ============================================================================
# CONFIGURATION FACTORY
# ============================================================================

def get_settings(environment: Optional[str] = None) -> Settings:
    """
    Factory function to get appropriate settings based on environment
    
    Args:
        environment: Environment name (development, production, testing)
        If None, uses ENVIRONMENT variable
    
    Returns:
        Settings instance for the specified environment
    """
    env = environment or os.getenv("ENVIRONMENT", "development")
    
    if env.lower() == "production":
        return ProductionSettings()
    elif env.lower() == "testing":
        return TestingSettings()
    else:
        return DevelopmentSettings()

# ============================================================================
# CONFIGURATION UTILITIES
# ============================================================================

def load_config_from_file(file_path: str) -> Dict[str, Any]:
    """
    Load configuration from JSON or YAML file
    
    Args:
        file_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    import json
    
    if not os.path.exists(file_path):
        raise ConfigurationError(f"Configuration file not found: {file_path}")
    
    file_extension = Path(file_path).suffix.lower()
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            if file_extension == '.json':
                return json.load(f)
            elif file_extension in ['.yml', '.yaml']:
                try:
                    import yaml
                    return yaml.safe_load(f)
                except ImportError:
                    raise ConfigurationError("PyYAML not installed. Cannot load YAML configuration.")
            else:
                raise ConfigurationError(f"Unsupported configuration file format: {file_extension}")
                
    except Exception as e:
        raise ConfigurationError(f"Failed to load configuration file: {str(e)}")

def save_config_to_file(config: Dict[str, Any], file_path: str) -> None:
    """
    Save configuration to JSON file
    
    Args:
        config: Configuration dictionary
        file_path: Path to save configuration
    """
    import json
    
    try:
        # Create directory if it doesn't exist
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, default=str)
            
    except Exception as e:
        raise ConfigurationError(f"Failed to save configuration file: {str(e)}")

def validate_api_key_format(api_key: str, provider: str) -> bool:
    """
    Validate API key format for different providers
    
    Args:
        api_key: API key to validate
        provider: Provider name (openai, pinecone, google)
        
    Returns:
        True if format is valid
    """
    if not api_key:
        return False
    
    format_patterns = {
        "openai": r"^sk-[a-zA-Z0-9]{48}$",
        "pinecone": r"^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$",
        "google": r"^AIza[0-9A-Za-z_-]{35}$"
    }
    
    if provider.lower() not in format_patterns:
        return True  # Unknown provider, assume valid
    
    import re
    pattern = format_patterns[provider.lower()]
    return bool(re.match(pattern, api_key))

def get_recommended_model(task_type: str, priority: str = "balanced") -> str:
    """
    Get recommended model based on task type and priority
    
    Args:
        task_type: Type of task (analysis, generation, classification)
        priority: Priority (cost, performance, balanced)
        
    Returns:
        Recommended model name
    """
    recommendations = {
        "cost": {
            "analysis": "gpt-3.5-turbo",
            "generation": "gpt-3.5-turbo", 
            "classification": "gpt-3.5-turbo",
            "embedding": "all-MiniLM-L6-v2"
        },
        "performance": {
            "analysis": "gpt-4",
            "generation": "gpt-4",
            "classification": "gpt-4",
            "embedding": "all-mpnet-base-v2"
        },
        "balanced": {
            "analysis": "gpt-3.5-turbo",
            "generation": "gpt-4-turbo",
            "classification": "gpt-3.5-turbo", 
            "embedding": "all-MiniLM-L6-v2"
        }
    }
    
    return recommendations.get(priority, recommendations["balanced"]).get(task_type, "gpt-3.5-turbo")

def estimate_costs(num_tokens: int, model: str) -> float:
    """
    Estimate costs for API usage
    
    Args:
        num_tokens: Number of tokens to process
        model: Model name
        
    Returns:
        Estimated cost in USD
    """
    model_costs = settings.AVAILABLE_MODELS.get(model, {})
    cost_per_token = model_costs.get("cost_per_token", 0.00002)  # Default fallback
    
    return num_tokens * cost_per_token

def get_optimal_chunk_size(content_type: str, model: str) -> int:
    """
    Get optimal chunk size based on content type and model
    
    Args:
        content_type: Type of content (ticket, policy, conversation)
        model: Model name
        
    Returns:
        Optimal chunk size
    """
    model_config = settings.AVAILABLE_MODELS.get(model, {})
    context_window = model_config.get("context_window", 4096)
    
    # Reserve space for prompt and response
    available_tokens = context_window - 1000
    
    # Approximate 4 characters per token
    base_chunk_size = available_tokens * 4
    
    # Adjust based on content type
    content_multipliers = {
        "ticket": 0.5,      # Smaller chunks for individual tickets
        "policy": 0.8,      # Larger chunks for policy documents
        "conversation": 0.6  # Medium chunks for conversations
    }
    
    multiplier = content_multipliers.get(content_type, 0.6)
    optimal_size = int(base_chunk_size * multiplier)
    
    # Clamp to reasonable bounds
    return max(500, min(optimal_size, 3000))

# ============================================================================
# CONFIGURATION VALIDATION HELPERS
# ============================================================================

def check_disk_space(path: str, min_gb: float = 1.0) -> bool:
    """Check if there's enough disk space"""
    import shutil
    
    try:
        total, used, free = shutil.disk_usage(path)
        free_gb = free / (1024**3)
        return free_gb >= min_gb
    except Exception:
        return True  # Assume OK if we can't check

def check_memory_usage() -> Dict[str, float]:
    """Check current memory usage"""
    try:
        import psutil
        memory = psutil.virtual_memory()
        return {
            "total_gb": memory.total / (1024**3),
            "available_gb": memory.available / (1024**3),
            "used_percentage": memory.percent
        }
    except ImportError:
        return {"total_gb": 0, "available_gb": 0, "used_percentage": 0}

def validate_environment() -> Dict[str, Any]:
    """Comprehensive environment validation"""
    validation = {
        "python_version": sys.version_info[:2],
        "disk_space_ok": check_disk_space(settings.DATA_DIR),
        "memory_info": check_memory_usage(),
        "required_packages": {},
        "optional_packages": {}
    }
    
    # Check required packages
    required_packages = ["pandas", "numpy", "streamlit"]
    for package in required_packages:
        try:
            __import__(package)
            validation["required_packages"][package] = True
        except ImportError:
            validation["required_packages"][package] = False
    
    # Check optional packages
    optional_packages = ["openai", "pinecone", "sentence_transformers", "textblob"]
    for package in optional_packages:
        try:
            __import__(package)
            validation["optional_packages"][package] = True
        except ImportError:
            validation["optional_packages"][package] = False
    
    return validation

# ============================================================================
# STARTUP CONFIGURATION
# ============================================================================

def initialize_application() -> bool:
    """
    Initialize application with full configuration validation
    
    Returns:
        True if initialization successful
    """
    try:
        print("🚀 Initializing Customer Support Triage Agent...")
        
        # Print startup summary
        settings.print_startup_summary()
        
        # Validate configuration
        validation = settings.validate_config()
        
        if validation["overall_status"] == "validation_failed":
            print("❌ Configuration validation failed!")
            for error in validation["errors"]:
                print(f"  • {error}")
            return False
        
        # Validate environment
        env_validation = validate_environment()
        
        # Check Python version
        if env_validation["python_version"] < (3, 8):
            print(f"⚠️  Warning: Python {env_validation['python_version']} detected. Python 3.8+ recommended.")
        
        # Check required packages
        missing_required = [pkg for pkg, installed in env_validation["required_packages"].items() if not installed]
        if missing_required:
            print(f"❌ Missing required packages: {', '.join(missing_required)}")
            print("   Run: pip install -r requirements.txt")
            return False
        
        # Check disk space
        if not env_validation["disk_space_ok"]:
            print("⚠️  Warning: Low disk space detected")
        
        print("✅ Application initialized successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Initialization failed: {str(e)}")
        return False

# Print configuration summary on import (only in non-production)
if not settings.is_production and os.getenv("SUPPRESS_CONFIG_SUMMARY", "false").lower() != "true":
    try:
        settings.print_startup_summary()
    except Exception:
        pass  # Suppress errors during import
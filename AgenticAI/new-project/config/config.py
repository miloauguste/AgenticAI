"""
Configuration settings for the Customer Support Agent
"""

import os
from typing import Optional


class Config:
    """Application configuration class"""
    
    # Environment
    DEBUG: bool = os.getenv('DEBUG', 'False').lower() in ('true', '1', 'yes')
    
    # OpenAI Configuration
    OPENAI_API_KEY: Optional[str] = os.getenv('OPENAI_API_KEY')
    OPENAI_MODEL: str = os.getenv('OPENAI_MODEL', 'gpt-4')
    OPENAI_TEMPERATURE: float = float(os.getenv('OPENAI_TEMPERATURE', '0.1'))
    
    # Database Configuration
    DATABASE_PATH: str = os.getenv('DATABASE_PATH', 'customer_support.db')
    
    # Streamlit Configuration
    STREAMLIT_PAGE_TITLE: str = "TechTrend Innovations - Customer Support"
    STREAMLIT_PAGE_ICON: str = "🤖"
    STREAMLIT_LAYOUT: str = "wide"
    
    # Support Configuration
    SUPPORT_EMAIL: str = os.getenv('SUPPORT_EMAIL', 'support@techtrend.com')
    SUPPORT_PHONE: str = os.getenv('SUPPORT_PHONE', '+1-800-TECHTREND')
    ESCALATION_TIMEOUT_HOURS: int = int(os.getenv('ESCALATION_TIMEOUT_HOURS', '2'))
    
    # Memory and Context Management
    MAX_CONVERSATION_HISTORY: int = int(os.getenv('MAX_CONVERSATION_HISTORY', '10'))
    USER_HISTORY_LIMIT: int = int(os.getenv('USER_HISTORY_LIMIT', '5'))
    
    # Logging
    LOG_LEVEL: str = os.getenv('LOG_LEVEL', 'INFO')
    
    @classmethod
    def validate(cls) -> bool:
        """Validate configuration settings"""
        if not cls.OPENAI_API_KEY:
            print("Warning: OPENAI_API_KEY not set. Please configure it in the Streamlit interface.")
        
        if cls.OPENAI_TEMPERATURE < 0 or cls.OPENAI_TEMPERATURE > 2:
            print("Warning: OPENAI_TEMPERATURE should be between 0 and 2")
            return False
            
        return True
    
    @classmethod
    def get_escalation_keywords(cls) -> list:
        """Get keywords that trigger escalation"""
        return ["refund", "legal", "urgent", "emergency", "complaint", "cancel subscription", "lawsuit", "attorney"]
    
    @classmethod
    def get_common_issues(cls) -> dict:
        """Get common issues and their resolutions"""
        return {
            "password": "To reset your password: 1) Go to login page 2) Click 'Forgot Password' 3) Enter your email 4) Check your inbox for reset link",
            "billing": f"For billing inquiries: 1) Check your account dashboard 2) Review recent transactions 3) Contact {cls.SUPPORT_EMAIL} for disputes",
            "login": "Login issues: 1) Clear browser cache 2) Try incognito mode 3) Check if Caps Lock is on 4) Reset password if needed",
            "feature": "For feature requests: 1) Check our roadmap at techtrend.com/roadmap 2) Submit requests via feedback form 3) Join our beta program for early access",
            "bug": "To report bugs: 1) Note exact steps to reproduce 2) Include screenshots if applicable 3) Submit via support portal 4) Include browser/OS details"
        }
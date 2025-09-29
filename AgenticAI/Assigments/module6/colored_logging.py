#!/usr/bin/env python3
"""
Colored Logging Configuration
Adds color support to logging output for better visibility and user experience
"""

import logging
import sys
from datetime import datetime

class ColoredFormatter(logging.Formatter):
    """
    Custom formatter to add colors to log levels
    """
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m'   # Magenta
    }
    
    RESET = '\033[0m'           # Reset color
    BOLD = '\033[1m'            # Bold text
    
    def __init__(self, use_colors: bool = True, show_timestamp: bool = True, 
                 show_module: bool = True, compact: bool = False):
        """
        Initialize colored formatter
        
        Args:
            use_colors: Whether to use colors (disable for file logging)
            show_timestamp: Whether to show timestamp
            show_module: Whether to show module name
            compact: Whether to use compact format
        """
        self.use_colors = use_colors and self._supports_color()
        self.show_timestamp = show_timestamp
        self.show_module = show_module
        self.compact = compact
        
        # Build format string
        format_parts = []
        
        if show_timestamp:
            if compact:
                format_parts.append('%(asctime)s')
            else:
                format_parts.append('%(asctime)s')
        
        if show_module and not compact:
            format_parts.append('%(name)s')
        
        format_parts.append('%(levelname)s')
        format_parts.append('%(message)s')
        
        if compact:
            format_string = ' '.join(format_parts)
            date_format = '%H:%M:%S'
        else:
            format_string = ' - '.join(format_parts)
            date_format = '%Y-%m-%d %H:%M:%S'
        
        super().__init__(format_string, datefmt=date_format)
    
    def _supports_color(self) -> bool:
        """Check if the current terminal supports colors"""
        # Check if we're in a terminal that supports colors
        if not hasattr(sys.stdout, 'isatty') or not sys.stdout.isatty():
            return False
        
        # Check for common terminals that support colors
        import os
        term = os.environ.get('TERM', '').lower()
        colorterm = os.environ.get('COLORTERM', '').lower()
        
        # Windows terminal detection
        if sys.platform == 'win32':
            try:
                import colorama
                colorama.init()
                return True
            except ImportError:
                # Try to enable Windows 10 ANSI support
                try:
                    import os
                    os.system('')  # Enables ANSI sequences on Windows 10
                    return True
                except:
                    return False
        
        # Unix-like systems
        return ('color' in term or 'ansi' in term or 'xterm' in term or 
                '256' in term or colorterm in ['truecolor', '24bit'])
    
    def format(self, record):
        """Format the log record with colors"""
        # Get the original formatted message
        formatted = super().format(record)
        
        if not self.use_colors:
            return formatted
        
        # Get color for log level
        level_color = self.COLORS.get(record.levelname, '')
        
        if level_color:
            # Color the entire message
            if self.compact:
                # Just color the level name in compact mode
                formatted = formatted.replace(
                    record.levelname,
                    f"{level_color}{self.BOLD}{record.levelname}{self.RESET}"
                )
            else:
                # Color the entire message for full format
                formatted = f"{level_color}{formatted}{self.RESET}"
        
        return formatted

class SupervisorToolsLogger:
    """
    Specialized logger configuration for supervisor tools
    """
    
    @staticmethod
    def setup_colored_logging(level: str = 'INFO', compact: bool = False, 
                            show_module: bool = True) -> None:
        """
        Set up colored logging for the supervisor tools system
        
        Args:
            level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            compact: Use compact format for less verbose output
            show_module: Show module names in logs
        """
        # Convert string level to logging constant
        numeric_level = getattr(logging, level.upper(), logging.INFO)
        
        # Create colored formatter
        colored_formatter = ColoredFormatter(
            use_colors=True,
            show_timestamp=True,
            show_module=show_module,
            compact=compact
        )
        
        # Create console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(colored_formatter)
        console_handler.setLevel(numeric_level)
        
        # Configure root logger
        root_logger = logging.getLogger()
        
        # Remove existing handlers to avoid duplicates
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        root_logger.addHandler(console_handler)
        root_logger.setLevel(numeric_level)
        
        # Add specific formatting for supervisor tools loggers
        supervisor_loggers = [
            'response_suggestions',
            'escalation_detector', 
            'supervisor_insights',
            'state_manager',
            'enhanced_support_agent',
            'support_agent',
            'persistent_store_integration'
        ]
        
        for logger_name in supervisor_loggers:
            logger = logging.getLogger(logger_name)
            logger.setLevel(numeric_level)
    
    @staticmethod
    def setup_file_logging(filename: str, level: str = 'DEBUG') -> None:
        """
        Set up file logging (without colors) for detailed logs
        
        Args:
            filename: Log file path
            level: Logging level for file
        """
        numeric_level = getattr(logging, level.upper(), logging.DEBUG)
        
        # Create file formatter (no colors)
        file_formatter = ColoredFormatter(
            use_colors=False,
            show_timestamp=True,
            show_module=True,
            compact=False
        )
        
        # Create file handler
        file_handler = logging.FileHandler(filename, encoding='utf-8')
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(numeric_level)
        
        # Add to root logger
        root_logger = logging.getLogger()
        root_logger.addHandler(file_handler)
    
    @staticmethod
    def log_system_status():
        """Log system status with colors for testing"""
        logger = logging.getLogger('supervisor_tools')
        
        logger.info("[SYSTEM] Supervisor Tools System Status")
        logger.info("[ACTIVE] Response Suggestions: Active")
        logger.info("[ACTIVE] Escalation Detection: Active") 
        logger.info("[ACTIVE] Supervisor Insights: Active")
        logger.info("[ACTIVE] State Management: Active")
        logger.warning("[DEMO] Demo mode - using sample data")
        logger.error("[TEST] Example error message for color testing")
        logger.critical("[TEST] Example critical message for color testing")

def setup_logging(level: str = 'INFO', compact: bool = False, 
                 file_log: str = None) -> None:
    """
    Convenience function to set up colored logging
    
    Args:
        level: Logging level
        compact: Use compact format
        file_log: Optional file for logging (in addition to console)
    """
    SupervisorToolsLogger.setup_colored_logging(level, compact)
    
    if file_log:
        SupervisorToolsLogger.setup_file_logging(file_log, 'DEBUG')
    
    # Log setup confirmation
    logger = logging.getLogger('colored_logging')
    logger.info(f"Colored logging initialized - Level: {level.upper()}, Compact: {compact}")

if __name__ == "__main__":
    # Test colored logging
    print("Testing Colored Logging System")
    print("=" * 40)
    
    # Set up colored logging
    setup_logging('DEBUG', compact=False)
    
    # Test different log levels
    SupervisorToolsLogger.log_system_status()
    
    print("\nTesting compact format:")
    setup_logging('INFO', compact=True)
    SupervisorToolsLogger.log_system_status()
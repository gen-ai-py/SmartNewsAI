"""
Logging Configuration Module for Smart News AI
Implements comprehensive logging with rotation, levels, and multiple handlers.

Features:
- Rotating file handlers (daily and size-based)
- Separate log files for different components
- Configurable log levels per module
- Console and file output
- Structured logging with timestamps
- Log file compression for archived logs
"""

import logging
import logging.handlers
import os
import sys
from pathlib import Path
from datetime import datetime
import yaml


class LoggingConfig:
    """
    Centralized logging configuration manager.
    
    Provides:
    - Multiple handlers (console, file, rotating file)
    - Component-specific loggers (classifier, recommender, API)
    - Configurable via YAML or environment variables
    - Log rotation (daily/size-based)
    - Structured log formatting
    """
    
    DEFAULT_LOG_DIR = "logs"
    DEFAULT_LOG_LEVEL = logging.INFO
    DEFAULT_MAX_BYTES = 10 * 1024 * 1024  # 10 MB
    DEFAULT_BACKUP_COUNT = 5
    DEFAULT_ROTATION_WHEN = 'midnight'
    
    # Standard log format
    DEFAULT_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
    DETAILED_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(funcName)s() - %(message)s'
    SIMPLE_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
    
    def __init__(self, config_file=None, log_dir=None, log_level=None):
        """
        Initialize logging configuration.
        
        Args:
            config_file (str): Path to YAML configuration file
            log_dir (str): Directory for log files
            log_level (int): Default logging level
        """
        self.config_file = config_file
        self.log_dir = log_dir or self.DEFAULT_LOG_DIR
        self.log_level = log_level or self.DEFAULT_LOG_LEVEL
        self.config = {}
        self.loggers = {}
        
        # Create log directory
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)
        
        # Load configuration
        if config_file and os.path.exists(config_file):
            self._load_config_file(config_file)
        else:
            self._load_default_config()
    
    def _load_config_file(self, config_file):
        """Load logging configuration from YAML file."""
        try:
            with open(config_file, 'r') as f:
                self.config = yaml.safe_load(f)
            logging.info(f"Loaded logging configuration from {config_file}")
        except Exception as e:
            logging.warning(f"Failed to load config file {config_file}: {e}. Using defaults.")
            self._load_default_config()
    
    def _load_default_config(self):
        """Load default logging configuration."""
        self.config = {
            'version': 1,
            'log_dir': self.log_dir,
            'log_level': logging.getLevelName(self.log_level),
            'handlers': {
                'console': {
                    'level': 'INFO',
                    'format': 'simple'
                },
                'file': {
                    'level': 'DEBUG',
                    'format': 'detailed',
                    'rotation': 'size',
                    'max_bytes': self.DEFAULT_MAX_BYTES,
                    'backup_count': self.DEFAULT_BACKUP_COUNT
                }
            },
            'loggers': {
                'classifier': {'level': 'INFO', 'file': 'classifier.log'},
                'recommender': {'level': 'INFO', 'file': 'recommender.log'},
                'api': {'level': 'INFO', 'file': 'api.log'},
                'data': {'level': 'INFO', 'file': 'data.log'}
            }
        }
    
    def _get_formatter(self, format_type='default'):
        """Get log formatter based on type."""
        formats = {
            'default': self.DEFAULT_FORMAT,
            'detailed': self.DETAILED_FORMAT,
            'simple': self.SIMPLE_FORMAT
        }
        return logging.Formatter(formats.get(format_type, self.DEFAULT_FORMAT))
    
    def _create_console_handler(self, level='INFO', format_type='simple'):
        """Create console handler."""
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(getattr(logging, level.upper()))
        handler.setFormatter(self._get_formatter(format_type))
        return handler
    
    def _create_rotating_file_handler(self, filename, level='DEBUG', 
                                     format_type='detailed', rotation='size',
                                     max_bytes=None, backup_count=None,
                                     when='midnight'):
        """
        Create rotating file handler.
        
        Args:
            filename (str): Log file name
            level (str): Log level
            format_type (str): Format type
            rotation (str): 'size' or 'time' based rotation
            max_bytes (int): Max file size for size-based rotation
            backup_count (int): Number of backup files to keep
            when (str): When to rotate for time-based rotation
        """
        filepath = os.path.join(self.log_dir, filename)
        
        if rotation == 'time':
            # Time-based rotation (daily, hourly, etc.)
            handler = logging.handlers.TimedRotatingFileHandler(
                filepath,
                when=when,
                interval=1,
                backupCount=backup_count or self.DEFAULT_BACKUP_COUNT,
                encoding='utf-8'
            )
        else:
            # Size-based rotation
            handler = logging.handlers.RotatingFileHandler(
                filepath,
                maxBytes=max_bytes or self.DEFAULT_MAX_BYTES,
                backupCount=backup_count or self.DEFAULT_BACKUP_COUNT,
                encoding='utf-8'
            )
        
        handler.setLevel(getattr(logging, level.upper()))
        handler.setFormatter(self._get_formatter(format_type))
        return handler
    
    def setup_root_logger(self):
        """Setup root logger with console and main file handler."""
        root_logger = logging.getLogger()
        root_logger.setLevel(self.log_level)
        
        # Clear existing handlers
        root_logger.handlers.clear()
        
        # Console handler
        console_config = self.config.get('handlers', {}).get('console', {})
        console_handler = self._create_console_handler(
            level=console_config.get('level', 'INFO'),
            format_type=console_config.get('format', 'simple')
        )
        root_logger.addHandler(console_handler)
        
        # Main file handler
        file_config = self.config.get('handlers', {}).get('file', {})
        main_file_handler = self._create_rotating_file_handler(
            filename='smart_news_ai.log',
            level=file_config.get('level', 'DEBUG'),
            format_type=file_config.get('format', 'detailed'),
            rotation=file_config.get('rotation', 'size'),
            max_bytes=file_config.get('max_bytes', self.DEFAULT_MAX_BYTES),
            backup_count=file_config.get('backup_count', self.DEFAULT_BACKUP_COUNT)
        )
        root_logger.addHandler(main_file_handler)
        
        logging.info("Root logger configured successfully")
        logging.info(f"Log directory: {os.path.abspath(self.log_dir)}")
        logging.info(f"Log level: {logging.getLevelName(self.log_level)}")
    
    def get_logger(self, name, component=None):
        """
        Get or create a logger for a specific component.
        
        Args:
            name (str): Logger name (typically __name__)
            component (str): Component type ('classifier', 'recommender', 'api', 'data')
        
        Returns:
            logging.Logger: Configured logger instance
        """
        # Use existing logger if available
        if name in self.loggers:
            return self.loggers[name]
        
        logger = logging.getLogger(name)
        
        # If component specified, add component-specific file handler
        if component and component in self.config.get('loggers', {}):
            component_config = self.config['loggers'][component]
            
            # Set component-specific log level
            log_level = component_config.get('level', 'INFO')
            logger.setLevel(getattr(logging, log_level.upper()))
            
            # Add component-specific file handler
            file_config = self.config.get('handlers', {}).get('file', {})
            component_handler = self._create_rotating_file_handler(
                filename=component_config.get('file', f'{component}.log'),
                level=log_level,
                format_type=file_config.get('format', 'detailed'),
                rotation=file_config.get('rotation', 'size'),
                max_bytes=file_config.get('max_bytes', self.DEFAULT_MAX_BYTES),
                backup_count=file_config.get('backup_count', self.DEFAULT_BACKUP_COUNT)
            )
            logger.addHandler(component_handler)
            
            logging.info(f"Component logger configured: {name} ({component})")
        
        self.loggers[name] = logger
        return logger
    
    def set_log_level(self, level, logger_name=None):
        """
        Set log level for a specific logger or root logger.
        
        Args:
            level (str or int): Log level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
            logger_name (str): Logger name (None for root logger)
        """
        if isinstance(level, str):
            level = getattr(logging, level.upper())
        
        if logger_name:
            logger = logging.getLogger(logger_name)
        else:
            logger = logging.getLogger()
        
        logger.setLevel(level)
        logging.info(f"Log level set to {logging.getLevelName(level)} for {logger_name or 'root'}")
    
    def log_to_console_only(self, logger_name=None):
        """Configure logger to output only to console (useful for production)."""
        logger = logging.getLogger(logger_name) if logger_name else logging.getLogger()
        
        # Remove all handlers
        logger.handlers.clear()
        
        # Add only console handler
        console_handler = self._create_console_handler()
        logger.addHandler(console_handler)
        
        logging.info(f"Logger {logger_name or 'root'} configured for console-only output")
    
    def log_to_file_only(self, logger_name=None, filename='app.log'):
        """Configure logger to output only to file (useful for background jobs)."""
        logger = logging.getLogger(logger_name) if logger_name else logging.getLogger()
        
        # Remove all handlers
        logger.handlers.clear()
        
        # Add only file handler
        file_handler = self._create_rotating_file_handler(filename)
        logger.addHandler(file_handler)
        
        logging.info(f"Logger {logger_name or 'root'} configured for file-only output: {filename}")
    
    def get_log_files(self):
        """Get list of all log files in the log directory."""
        log_dir_path = Path(self.log_dir)
        return sorted([f.name for f in log_dir_path.glob('*.log*')])
    
    def cleanup_old_logs(self, days=30):
        """
        Clean up log files older than specified days.
        
        Args:
            days (int): Delete logs older than this many days
        """
        from datetime import timedelta
        
        cutoff_time = datetime.now() - timedelta(days=days)
        log_dir_path = Path(self.log_dir)
        removed_count = 0
        
        for log_file in log_dir_path.glob('*.log*'):
            if log_file.stat().st_mtime < cutoff_time.timestamp():
                try:
                    log_file.unlink()
                    removed_count += 1
                except Exception as e:
                    logging.warning(f"Failed to remove old log file {log_file}: {e}")
        
        logging.info(f"Cleaned up {removed_count} old log files (older than {days} days)")
        return removed_count


# Global logging configuration instance
_logging_config = None


def setup_logging(config_file=None, log_dir=None, log_level=None):
    """
    Setup application-wide logging configuration.
    
    Args:
        config_file (str): Path to YAML configuration file
        log_dir (str): Directory for log files
        log_level (str or int): Default logging level
    
    Returns:
        LoggingConfig: Configured logging instance
    """
    global _logging_config
    
    if isinstance(log_level, str):
        log_level = getattr(logging, log_level.upper())
    
    _logging_config = LoggingConfig(
        config_file=config_file,
        log_dir=log_dir,
        log_level=log_level
    )
    _logging_config.setup_root_logger()
    
    return _logging_config


def get_logger(name, component=None):
    """
    Get a logger instance.
    
    Args:
        name (str): Logger name (typically __name__)
        component (str): Component type ('classifier', 'recommender', 'api', 'data')
    
    Returns:
        logging.Logger: Configured logger
    """
    global _logging_config
    
    if _logging_config is None:
        # Setup with defaults if not already configured
        setup_logging()
    
    return _logging_config.get_logger(name, component)


def set_log_level(level, logger_name=None):
    """Set log level for specific logger or root logger."""
    global _logging_config
    
    if _logging_config is None:
        setup_logging()
    
    _logging_config.set_log_level(level, logger_name)


# Convenience functions
def get_classifier_logger(name):
    """Get logger for classifier component."""
    return get_logger(name, component='classifier')


def get_recommender_logger(name):
    """Get logger for recommender component."""
    return get_logger(name, component='recommender')


def get_api_logger(name):
    """Get logger for API component."""
    return get_logger(name, component='api')


def get_data_logger(name):
    """Get logger for data processing component."""
    return get_logger(name, component='data')

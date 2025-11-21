# src/logger.py
import logging
import sys

def setup_logger():
    """
    Sets up a logger that outputs pretty, detailed logs to the terminal.
    """
    logger = logging.getLogger("RAG_System")
    logger.setLevel(logging.DEBUG)
    
    # Create console handler with a higher log level
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)
    
    # Create formatter and add it to the handlers
    formatter = logging.Formatter('🔍 %(asctime)s | %(levelname)s | %(message)s')
    ch.setFormatter(formatter)
    
    # Avoid adding duplicates
    if not logger.handlers:
        logger.addHandler(ch)
        
    return logger

logger = setup_logger()
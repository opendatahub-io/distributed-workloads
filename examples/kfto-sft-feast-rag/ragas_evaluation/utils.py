#!/usr/bin/env python3
"""
Utilities module for RAGAS evaluation
"""

import sys
import os
import logging
from datetime import datetime

# Set dummy OpenAI API key to avoid RAGAS errors
if not os.getenv("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = "dummy-key-for-evaluation-only"

# Suppress external API warnings and errors
import warnings
warnings.filterwarnings("ignore", message=".*API key.*")
warnings.filterwarnings("ignore", message=".*OpenAI.*")
warnings.filterwarnings("ignore", message=".*external.*")
warnings.filterwarnings("ignore", message=".*deprecated.*")
warnings.filterwarnings("ignore", message=".*DeprecationWarning.*")

# Suppress logging from external libraries
logging.getLogger("openai").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("requests").setLevel(logging.ERROR)

def setup_logging(output_dir="."):
    """Setup logging to both file and console"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(output_dir, f"ragas_evaluation_{timestamp}.log")
    
    # Suppress verbose logging from external libraries
    logging.getLogger("feast").setLevel(logging.WARNING)
    logging.getLogger("pymilvus").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return log_filename 
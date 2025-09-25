#!/usr/bin/env python3
"""
Utilities module for RAGAS evaluation
"""

import sys
import os
import logging
from datetime import datetime
from logging.handlers import RotatingFileHandler
import threading
import queue

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

class ComprehensiveTeeHandler:
    """Handler that captures ALL terminal output including library output"""
    def __init__(self, filename, max_bytes=2*1024*1024, backup_count=5):
        self.filename = filename
        self.max_bytes = max_bytes
        self.backup_count = backup_count
        self.current_file = None
        self.current_size = 0
        self.file_number = 0
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        self.output_queue = queue.Queue()
        self.running = True
        
        # Start output thread
        self.output_thread = threading.Thread(target=self._output_worker, daemon=True)
        self.output_thread.start()
        
        self._open_file()
    
    def _open_file(self):
        """Open the current log file"""
        if self.current_file:
            self.current_file.close()
        
        if self.file_number == 0:
            current_filename = self.filename
        else:
            current_filename = f"{self.filename}.{self.file_number}"
        
        self.current_file = open(current_filename, 'w', encoding='utf-8', buffering=1)
        self.current_size = 0
    
    def _rotate_file(self):
        """Rotate to next file if current one is too large"""
        if self.current_size >= self.max_bytes:
            self.current_file.close()
            
            # Remove oldest backup if we have too many
            if self.backup_count > 0:
                oldest_file = f"{self.filename}.{self.backup_count}"
                if os.path.exists(oldest_file):
                    os.remove(oldest_file)
                
                # Rename existing backups
                for i in range(self.backup_count - 1, 0, -1):
                    old_name = f"{self.filename}.{i}"
                    new_name = f"{self.filename}.{i + 1}"
                    if os.path.exists(old_name):
                        os.rename(old_name, new_name)
                
                # Rename current file to .1
                if os.path.exists(self.filename):
                    os.rename(self.filename, f"{self.filename}.1")
            
            self.file_number = 0
            self._open_file()
    
    def _output_worker(self):
        """Worker thread to handle output"""
        while self.running:
            try:
                text = self.output_queue.get(timeout=0.1)
                if text is None:  # Shutdown signal
                    break
                
                # Write to original console
                self.original_stdout.write(text)
                self.original_stdout.flush()
                
                # Write to file
                if self.current_file:
                    self.current_file.write(text)
                    self.current_file.flush()
                    self.current_size += len(text.encode('utf-8'))
                    self._rotate_file()
                    
            except queue.Empty:
                continue
            except Exception as e:
                # Fallback to direct output if queue fails
                try:
                    self.original_stdout.write(f"TeeHandler error: {e}\n")
                    self.original_stdout.flush()
                except:
                    pass
    
    def write(self, text):
        """Write text to queue for processing"""
        try:
            self.output_queue.put(text, block=False)
        except queue.Full:
            # If queue is full, write directly
            self.original_stdout.write(text)
            self.original_stdout.flush()
    
    def flush(self):
        """Flush output"""
        self.original_stdout.flush()
        if self.current_file:
            self.current_file.flush()
    
    def close(self):
        """Close the handler"""
        self.running = False
        self.output_queue.put(None)  # Shutdown signal
        if self.output_thread.is_alive():
            self.output_thread.join(timeout=1.0)
        if self.current_file:
            self.current_file.close()

def setup_logging(output_dir="."):
    """Setup logging with comprehensive tee-like behavior to capture ALL terminal output"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_log_filename = os.path.join(output_dir, f"ragas_evaluation_{timestamp}")
    
    # Suppress verbose logging from external libraries
    logging.getLogger("feast").setLevel(logging.WARNING)
    logging.getLogger("pymilvus").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    
    # Create comprehensive tee handler
    tee_handler = ComprehensiveTeeHandler(f"{base_log_filename}.log")
    
    # Replace both stdout and stderr
    sys.stdout = tee_handler
    sys.stderr = tee_handler
    
    # Configure basic logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(tee_handler)
        ]
    )
    
    return f"{base_log_filename}.log" 
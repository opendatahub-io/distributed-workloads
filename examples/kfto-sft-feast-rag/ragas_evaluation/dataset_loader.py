#!/usr/bin/env python3
"""
Dataset loading utilities for RAGAS evaluation
"""

import signal
import logging
from typing import List, Tuple
from datasets import load_dataset

def load_natural_questions_dataset(sample_percentage: float = 0.1, max_questions: int = 100, timeout: int = 300) -> Tuple[List[str], List[str]]:
    """Load Natural Questions dataset with timeout handling"""
    def timeout_handler(signum, frame):
        raise TimeoutError("Dataset loading timed out")
    
    # Set timeout
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)
    
    try:
        logging.info("Loading Natural Questions dataset...")
        dataset = load_dataset("natural_questions", split="validation")
        
        # Sample the dataset
        total_samples = len(dataset)
        sample_size = min(int(total_samples * sample_percentage), max_questions)
        
        # Take a random sample
        sampled_dataset = dataset.shuffle(seed=42).select(range(sample_size))
        
        # Extract questions and answers
        questions = []
        ground_truth = []
        
        for item in sampled_dataset:
            # Get the question text
            question = item['question']['text']
            
            # Get the first answer (most common approach)
            if item['annotations']['short_answers']:
                answer = item['annotations']['short_answers'][0]['text']
            elif item['annotations']['yes_no_answers']:
                answer = str(item['annotations']['yes_no_answers'][0])
            else:
                # Skip items without answers
                continue
            
            questions.append(question)
            ground_truth.append(answer)
        
        signal.alarm(0)  # Cancel timeout
        logging.info(f"Loaded {len(questions)} questions from Natural Questions dataset")
        return questions, ground_truth
        
    except TimeoutError:
        logging.error("Dataset loading timed out, using curated questions instead")
        return load_curated_questions()
    except Exception as e:
        logging.error(f"Error loading Natural Questions dataset: {e}")
        logging.info("Falling back to curated questions")
        return load_curated_questions()

def load_curated_questions() -> Tuple[List[str], List[str]]:
    """Load curated questions for evaluation"""
    questions = [
        "What is the capital of France?",
        "Who wrote Romeo and Juliet?",
        "What is the largest planet in our solar system?",
        "When was the Declaration of Independence signed?",
        "What is the chemical symbol for gold?",
        "Who painted the Mona Lisa?",
        "What is the speed of light?",
        "What is the largest ocean on Earth?",
        "Who discovered penicillin?",
        "What is the main component of the sun?"
    ]
    
    ground_truth = [
        "Paris",
        "William Shakespeare",
        "Jupiter",
        "1776",
        "Au",
        "Leonardo da Vinci",
        "299,792,458 meters per second",
        "Pacific Ocean",
        "Alexander Fleming",
        "Hydrogen"
    ]
    
    logging.info(f"Loaded {len(questions)} curated questions")
    return questions, ground_truth 
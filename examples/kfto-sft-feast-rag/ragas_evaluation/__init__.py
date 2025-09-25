#!/usr/bin/env python3
"""
RAGAS Evaluation Package

A comprehensive evaluation framework for RAG models using RAGAS metrics.
"""

from .config import ModelConfig
from .dataset_loader import load_natural_questions_dataset, load_curated_questions
from .evaluator import RAGEvaluator

from .utils import setup_logging


__all__ = [
    "ModelConfig",
    "load_natural_questions_dataset",
    "load_curated_questions", 
    "RAGEvaluator",
    "setup_logging"
] 
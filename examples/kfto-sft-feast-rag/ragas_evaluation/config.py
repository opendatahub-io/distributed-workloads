#!/usr/bin/env python3
"""
Configuration for RAGAS evaluation
"""

import torch
from dataclasses import dataclass

@dataclass
class ModelConfig:
    """Configuration for model paths and settings"""
    # Paths
    finetuned_checkpoint_dir: str = "/opt/app-root/src/shared/fine_tuned_rag_model/inference_sft-rag-master-0"
    feast_repo_path: str = "/opt/app-root/src/distributed-workloads/examples/kfto-sft-feast-rag/feature_repo"
    
    # Original model names
    original_qe_model: str = "facebook/dpr-question_encoder-single-nq-base"
    original_gen_model: str = "facebook/bart-large"
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Evaluation settings
    max_new_tokens: int = 200
    num_beams: int = 1
    do_sample: bool = False
    
    # Dataset settings
    use_natural_questions: bool = True
    nq_sample_percentage: float = 0.1
    max_evaluation_questions: int = 100 
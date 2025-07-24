#!/usr/bin/env python3
"""
Main script for RAGAS evaluation
"""

import logging
from datetime import datetime

from .config import ModelConfig
from .dataset_loader import load_natural_questions_dataset, load_curated_questions
from .evaluator import RAGEvaluator

from .utils import setup_logging

def main():
    """Main function to run comprehensive RAG evaluation"""
    log_filename = setup_logging()
    
    logging.info("🔍 COMPREHENSIVE RAG EVALUATION SCRIPT")
    logging.info("=" * 60)
    logging.info("📊 Evaluates 4 combinations to test RAG functionality:")
    logging.info("   1. Original QE + Original Generator (baseline with retrieval)")
    logging.info("   2. Fine-tuned Generator only (no retrieval - test memorization)")
    logging.info("   3. Fine-tuned QE + Original Generator (test retrieval quality)")
    logging.info("   4. Fine-tuned QE + Fine-tuned Generator (full pipeline)")
    logging.info("=" * 60)
    
    # Configuration - modify these paths as needed
    config = ModelConfig(
        finetuned_checkpoint_dir="/opt/app-root/src/shared/fine_tuned_rag_model/inference_sft-rag-master-0",
        feast_repo_path="/opt/app-root/src/distributed-workloads/examples/kfto-sft-feast-rag/feature_repo",
        max_new_tokens=200,
        num_beams=1,
        do_sample=False,
        use_natural_questions=True,
        nq_sample_percentage=0.1,
        max_evaluation_questions=100
    )
    
    logging.info(f"📁 Fine-tuned checkpoint: {config.finetuned_checkpoint_dir}")
    logging.info(f"📁 Feast repo path: {config.feast_repo_path}")
    logging.info(f"🔧 Device: {config.device}")
    
    # Load evaluation dataset based on configuration
    if config.use_natural_questions:
        logging.info("📥 Loading Natural Questions dataset...")
        test_questions, ground_truth = load_natural_questions_dataset(
            sample_percentage=config.nq_sample_percentage,
            max_questions=config.max_evaluation_questions
        )
    else:
        logging.info("📝 Using curated questions...")
        test_questions, ground_truth = load_curated_questions()
    
    logging.info(f"📊 Using {len(test_questions)} questions for evaluation")
    
    # Initialize evaluator
    logging.info("🚀 Initializing RAG evaluator...")
    evaluator = RAGEvaluator(config)
    
    # Test Feast connection first to diagnose context retrieval issues
    logging.info("🔍 Testing Feast connection before evaluation...")
    feast_test = evaluator.test_feast_connection()
    if feast_test['status'] == 'success':
        logging.info(f"✅ Feast connection successful. Found {feast_test['sample_count']} sample passages")
        if feast_test['sample_passages']:
            logging.info("📄 Sample passages available for retrieval")
        else:
            logging.warning("⚠️ No sample passages found - this may cause context retrieval issues")
        
        if feast_test.get('generation_works', False):
            logging.info("✅ Feast generation test successful")
        else:
            logging.warning("⚠️ Feast generation test failed - will use fallback generation")
            
    else:
        logging.error(f"❌ Feast connection failed: {feast_test.get('error', 'Unknown error')}")
    
    # Run comprehensive evaluation
    logging.info("🎯 Starting comprehensive evaluation...")
    results = evaluator.run_comprehensive_evaluation(test_questions, ground_truth)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"comprehensive_rag_evaluation_{timestamp}.json"
    evaluator.save_results(results, results_file)
    
    # Save answer comparison text file
    answer_comparison_file = f"answer_comparison_{timestamp}.txt"
    evaluator.save_answer_comparison_text(results, answer_comparison_file)
    
    # Generate comparison charts
    charts = evaluator.generate_comparison_charts(results)
    
    # Print comparison table
    evaluator.print_comparison_table(results)
    
    logging.info("✅ Evaluation complete! Results saved to:")
    logging.info(f"   📊 Metrics: {results_file}")
    logging.info(f"   📝 Answers: {answer_comparison_file}")
    if charts:
        logging.info(f"   📈 Charts: {len(charts)} charts generated")
    
    logging.info(f"📝 Log file saved to: {log_filename}")

if __name__ == "__main__":
    main() 
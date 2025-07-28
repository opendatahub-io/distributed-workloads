#!/usr/bin/env python3
"""
Main script for RAGAS evaluation
"""

import logging
import argparse
from datetime import datetime

from config import ModelConfig
from dataset_loader import load_natural_questions_dataset, load_curated_questions
from evaluator import RAGEvaluator

from utils import setup_logging

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="RAGAS Evaluation Script")
    parser.add_argument(
        "--use-natural-questions", 
        action="store_true",
        help="Use Natural Questions dataset (~50GB download, takes 10-30 minutes) (default: use curated questions)"
    )
    parser.add_argument(
        "--nq-sample-percentage",
        type=float,
        default=0.1,
        help="Percentage of Natural Questions to sample (default: 0.1)"
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=100,
        help="Maximum number of questions to evaluate (default: 100)"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Directory to cache Natural Questions dataset (default: use default cache location)"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="/opt/app-root/src/shared/fine_tuned_rag_model/inference_sft-rag-master-0",
        help="Path to fine-tuned checkpoint directory"
    )
    parser.add_argument(
        "--feast-repo-path",
        type=str,
        default="/opt/app-root/src/distributed-workloads/examples/kfto-sft-feast-rag/feature_repo",
        help="Path to Feast repository"
    )
    parser.add_argument(
        "--combinations",
        type=str,
        nargs="+",
        choices=["1", "2", "3", "all"],
        default=["all"],
        help="Select combinations to test: 1=original_qe_original_gen, 2=original_qe_finetuned_gen, 3=finetuned_qe_finetuned_gen, all=all combinations (default: all)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="ragas_evaluation_results",
        help="Output directory for all results (default: ragas_evaluation_results)"
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="Directory to store log files (default: same as output-dir)"
    )
    return parser.parse_args()

def main():
    """Main function to run comprehensive RAG evaluation"""
    args = parse_args()
    
    # Create output directory
    import os
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Determine log directory (use log-dir if specified, otherwise use output-dir)
    log_dir = args.log_dir if args.log_dir is not None else args.output_dir
    os.makedirs(log_dir, exist_ok=True)
    
    # Setup logging to specified log directory
    log_filename = setup_logging(output_dir=log_dir)
    
    print("🔍 COMPREHENSIVE RAG EVALUATION SCRIPT")
    print("=" * 60)
    print("📊 Evaluates combinations to test RAG functionality:")
    print("   1. Original QE + Original Generator (baseline with retrieval)")
    print("   2. Original QE + Fine-tuned Generator (notebook configuration)")
    print("   3. Fine-tuned QE + Fine-tuned Generator (full pipeline)")
    print("=" * 60)
    
    # Configuration - modify these paths as needed
    config = ModelConfig(
        finetuned_checkpoint_dir=args.checkpoint_dir,
        feast_repo_path=args.feast_repo_path,
        max_new_tokens=200,
        num_beams=1,
        do_sample=False,
        use_natural_questions=args.use_natural_questions,
        nq_sample_percentage=args.nq_sample_percentage,
        max_evaluation_questions=args.max_questions
    )
    
    print(f"📁 Fine-tuned checkpoint: {config.finetuned_checkpoint_dir}")
    print(f"📁 Feast repo path: {config.feast_repo_path}")
    print(f"🔧 Device: {config.device}")
    
    # Load dataset based on configuration
    if config.use_natural_questions:
        print("🔄 Loading Natural Questions dataset...")
        try:
            test_questions, ground_truth_metadata = load_natural_questions_dataset(
                sample_percentage=config.nq_sample_percentage,
                max_questions=config.max_evaluation_questions,
                cache_dir=args.cache_dir
            )
            print(f"✅ Loaded {len(test_questions)} questions with enhanced metadata")
        except Exception as e:
            print(f"❌ Failed to load Natural Questions dataset: {e}")
            print("🔄 Falling back to curated questions...")
            test_questions, ground_truth_metadata = load_curated_questions()
    else:
        print("🔄 Loading curated questions...")
        test_questions, ground_truth_metadata = load_curated_questions()
    
    print(f"📊 Dataset loaded: {len(test_questions)} questions")
    print(f"📊 Ground truth metadata available: {len(ground_truth_metadata)} entries")
    
    # Show sample of ground truth metadata
    if ground_truth_metadata:
        print("\n📝 Sample ground truth metadata:")
        for i in range(min(3, len(ground_truth_metadata))):
            gt_info = ground_truth_metadata[i]
            print(f"   {i+1}. Type: {gt_info['answer_type']}, Confidence: {gt_info['confidence']:.2f}")
            print(f"      Answer: {gt_info['answer']}")
            print(f"      Total answers available: {gt_info['answer_count']}")
    
    # Initialize evaluator
    print("\n🔧 Initializing RAG evaluator...")
    evaluator = RAGEvaluator(config)
    
    # Test Feast connection
    print("\n🔍 Testing Feast connection...")
    feast_test = evaluator.test_feast_connection()
    if feast_test["status"] == "success":
        print(f"✅ Feast connection successful!")
        print(f"   Retrieved {feast_test['sample_count']} passages")
        print(f"   Generation works: {feast_test['generation_works']}")
    else:
        print(f"⚠️  Feast connection test failed: {feast_test['error']}")
        print("   Continuing with evaluation...")
    
    # Run comprehensive evaluation
    print(f"\n🎯 Starting comprehensive evaluation...")
    print(f"📊 Questions: {len(test_questions)}")
    print(f"📊 Combinations: {args.combinations}")
    print(f"📊 Output directory: {args.output_dir}")
    
    results = evaluator.run_comprehensive_evaluation(test_questions, ground_truth_metadata, args.combinations)
    
    # Save results to output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(args.output_dir, f"comprehensive_rag_evaluation_{timestamp}.json")
    evaluator.save_results(results, results_file)
    
    # Save answer comparison text file
    answer_comparison_file = os.path.join(args.output_dir, f"answer_comparison_{timestamp}.txt")
    evaluator.save_answer_comparison_text(results, answer_comparison_file)
    
    # Generate comparison charts
    charts = evaluator.generate_comparison_charts(results, output_dir=args.output_dir)
    
    # Print comparison table
    evaluator.print_comparison_table(results)
    
    print("\n✅ Evaluation complete! All results saved to:")
    print(f"   📁 Output directory: {args.output_dir}")
    if args.log_dir:
        print(f"   📝 Log directory: {args.log_dir}")
    print(f"   📊 Metrics: {os.path.basename(results_file)}")
    print(f"   📝 Answers: {os.path.basename(answer_comparison_file)}")
    if charts:
        print(f"   📈 Charts: {len(charts)} charts generated")
    
    print(f"📝 Log file saved to: {log_filename}")

if __name__ == "__main__":
    main() 
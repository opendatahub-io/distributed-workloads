#!/usr/bin/env python3
"""
Dataset loading utilities for RAGAS evaluation
"""

import signal
import logging
from typing import List, Tuple
from datasets import load_dataset

def load_natural_questions_dataset(sample_percentage: float = 0.1, max_questions: int = 100, 
                                 cache_dir: str = None) -> Tuple[List[str], List[str]]:
    """Load Natural Questions dataset with optional caching to avoid re-downloading"""
    
    # Set cache directory
    if cache_dir:
        cache_path = os.path.expanduser(cache_dir)
        os.makedirs(cache_path, exist_ok=True)
        print(f"   📁 Using cache directory: {cache_path}")
    else:
        cache_path = None
        print("   🚫 No cache directory specified - dataset will be downloaded to default location")
    
    print("   🚫 No timeout - allowing full dataset download")
    
    try:
        print("🔄 Loading Natural Questions dataset...")
        logging.info("Loading Natural Questions dataset...")
        
        # Load the dataset with progress indication
        print("   📥 Downloading/loading dataset (this may take a few minutes)...")
        logging.info("Starting dataset download/load...")
        
        # Load the dataset with cache directory and progress tracking
        print("   📥 Loading dataset files...")
        
        # Enable progress bars for HuggingFace
        os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '0'
        
        if cache_path:
            dataset = load_dataset("natural_questions", split="validation", cache_dir=cache_path)
        else:
            dataset = load_dataset("natural_questions", split="validation")
        
        print("   ✅ Dataset loaded successfully!")
        logging.info("Dataset loaded successfully")
        
        # Sample the dataset
        total_samples = len(dataset)
        sample_size = min(int(total_samples * sample_percentage), max_questions)
        
        print(f"📊 Natural Questions dataset: {total_samples:,} total questions")
        print(f"📊 Sampling: {sample_percentage*100:.1f}% = {sample_size} questions (max: {max_questions})")
        
        # Take a random sample
        sampled_dataset = dataset.shuffle(seed=42).select(range(sample_size))
        
        # Extract questions and answers
        questions = []
        ground_truth = []
        
        print("🔄 Processing questions and answers...")
        
        # Add progress bar for processing
        try:
            from tqdm import tqdm
            progress_bar = tqdm(sampled_dataset, desc="Processing questions", unit="q")
        except ImportError:
            progress_bar = sampled_dataset
            print("   (Install 'tqdm' for progress bar: pip install tqdm)")
        
        for item in progress_bar:
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
        
        print(f"✅ Successfully loaded {len(questions)} questions from Natural Questions dataset")
        logging.info(f"Loaded {len(questions)} questions from Natural Questions dataset")
        
        # Show sample questions
        print("\n📝 Sample questions loaded:")
        for i in range(min(3, len(questions))):
            print(f"   {i+1}. {questions[i][:80]}{'...' if len(questions[i]) > 80 else ''}")
        
        return questions, ground_truth
        
    except Exception as e:
        print(f"❌ Error loading Natural Questions dataset: {e}")
        logging.error(f"Error loading Natural Questions dataset: {e}")
        
        # Try to provide more helpful error information
        if "Connection" in str(e) or "timeout" in str(e).lower():
            print("   💡 This might be a network issue. Try:")
            print("      - Check your internet connection")
            print("      - Try again later")
            print("      - Use curated questions instead")
        elif "disk space" in str(e).lower():
            print("   💡 This might be a disk space issue. Try:")
            print("      - Free up disk space")
            print("      - Use curated questions instead")
        else:
            print("   💡 Try using curated questions instead")
        
        print("🔄 Falling back to curated questions...")
        logging.info("Falling back to curated questions")
        return load_curated_questions()

def download_natural_questions_dataset(cache_dir: str, split: str = "validation") -> bool:
    """Download Natural Questions dataset to specified directory"""
    
    # Import os at the beginning to avoid scope issues
    import os
    
    cache_path = os.path.expanduser(cache_dir)
    os.makedirs(cache_path, exist_ok=True)
    
    print(f"📥 Downloading Natural Questions dataset to: {cache_path}")
    print(f"📊 Split: {split}")
    print("⚠️  This will download ~50GB of data and may take 10-30 minutes")
    print("💡 For better progress bars, install tqdm: pip install tqdm")
    print("=" * 60)
    
    try:
        # Download the dataset with progress tracking
        print("🔄 Starting download...")
        
        # Set up progress tracking
        import os
        os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '0'  # Enable HuggingFace progress bars
        
        # Download with progress
        print("   📊 Download progress will be shown below:")
        dataset = load_dataset(
            "natural_questions", 
            split=split, 
            cache_dir=cache_path,
            download_mode="force_redownload"  # Force download to show progress
        )
        
        print(f"\n✅ Successfully downloaded dataset!")
        print(f"📊 Total questions: {len(dataset):,}")
        
        # Show sample questions
        print("\n📝 Sample questions:")
        for i in range(min(3, len(dataset))):
            question = dataset[i]['question']['text']
            print(f"   {i+1}. {question[:80]}{'...' if len(question) > 80 else ''}")
        
        print(f"\n💾 Dataset cached at: {cache_path}")
        print("🎯 You can now use --cache-dir option to load from this location")
        
        return True
        
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        return False

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

def main():
    """Main function for dataset operations"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Natural Questions Dataset Operations")
    parser.add_argument(
        "operation",
        choices=["download", "test"],
        help="Operation to perform: download or test"
    )
    parser.add_argument(
        "--cache-dir",
        required=True,
        help="Directory to cache/download the dataset"
    )
    parser.add_argument(
        "--split",
        default="validation",
        choices=["train", "validation"],
        help="Dataset split to use (default: validation)"
    )
    parser.add_argument(
        "--sample-percentage",
        type=float,
        default=0.0001,
        help="Sample percentage for testing (default: 0.0001)"
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=5,
        help="Maximum questions for testing (default: 5)"
    )
    
    args = parser.parse_args()
    
    if args.operation == "download":
        success = download_natural_questions_dataset(args.cache_dir, args.split)
        if success:
            print("\n🎉 Download completed successfully!")
            print(f"💡 To use this cached dataset, run:")
            print(f"   python main.py --use-natural-questions --cache-dir {args.cache_dir}")
        else:
            print("\n❌ Download failed!")
            sys.exit(1)
    
    elif args.operation == "test":
        print("🧪 Testing Natural Questions dataset loading...")
        print("=" * 50)
        
        try:
            questions, answers = load_natural_questions_dataset(
                sample_percentage=args.sample_percentage,
                max_questions=args.max_questions,
                cache_dir=args.cache_dir
            )
            
            print(f"✅ Success! Loaded {len(questions)} questions")
            print("\n📝 Sample questions:")
            for i, (q, a) in enumerate(zip(questions[:3], answers[:3])):
                print(f"   {i+1}. Q: {q[:60]}...")
                print(f"      A: {a}")
                print()
                
        except Exception as e:
            print(f"❌ Failed to load Natural Questions: {e}")
            print("\n🔄 Testing curated questions fallback...")
            
            try:
                questions, answers = load_curated_questions()
                print(f"✅ Curated questions work! Loaded {len(questions)} questions")
            except Exception as e2:
                print(f"❌ Even curated questions failed: {e2}")

if __name__ == "__main__":
    main() 
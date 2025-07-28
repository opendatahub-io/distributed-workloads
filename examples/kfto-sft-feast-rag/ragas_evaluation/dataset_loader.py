#!/usr/bin/env python3
"""
Dataset loading utilities for RAGAS evaluation
"""

import signal
import logging
from typing import List, Tuple, Dict
from datasets import load_dataset

def load_natural_questions_dataset(sample_percentage: float = 0.1, max_questions: int = 100, 
                                 cache_dir: str = None) -> Tuple[List[str], List[Dict]]:
    """Load Natural Questions dataset with enhanced answer selection and metadata"""
    
    # Import os at the beginning
    import os
    
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
        
        # Add progress indication for dataset loading
        try:
            from tqdm import tqdm
            print("   📊 Downloading Natural Questions dataset...")
            print("   ⏳ This may take several minutes for the first download...")
        except ImportError:
            print("   📊 Downloading Natural Questions dataset...")
            print("   ⏳ This may take several minutes for the first download...")
            print("   💡 Install 'tqdm' for better progress indication: pip install tqdm")
        
        if cache_path:
            # Check if dataset is already cached
            cache_dataset_path = os.path.join(cache_path, "natural_questions")
            if os.path.exists(cache_dataset_path):
                print(f"   📁 Found cached dataset at: {cache_dataset_path}")
                print("   🔄 Loading from cache...")
                try:
                    # Load only the validation split to avoid train split generation
                    dataset = load_dataset("natural_questions", split="validation", cache_dir=cache_path)
                    print("   ✅ Successfully loaded from cache!")
                except Exception as e:
                    print(f"   ⚠️ Cache loading failed: {e}")
                    print("   🔄 Falling back to streaming...")
                    # If cache loading fails, try streaming first
                    try:
                        dataset = load_dataset("natural_questions", split="validation", streaming=True)
                        # Convert to limited list
                        limited_dataset = []
                        print("   📥 Converting streaming dataset to limited samples...")
                        
                        try:
                            from tqdm import tqdm
                            for i, item in enumerate(tqdm(dataset, desc="Loading samples", unit="sample")):
                                if i >= max_questions:  # Use max_questions parameter
                                    break
                                limited_dataset.append(item)
                        except ImportError:
                            for i, item in enumerate(dataset):
                                if i >= max_questions:  # Use max_questions parameter
                                    break
                                limited_dataset.append(item)
                        
                        dataset = limited_dataset
                        print("   ✅ Successfully loaded via streaming!")
                    except Exception as stream_error:
                        print(f"   ⚠️ Streaming failed: {stream_error}")
                        print("   🔄 Falling back to download...")
                        # If streaming fails, try download
                        dataset = load_dataset("natural_questions", split="validation", cache_dir=cache_path)
                        print("   ✅ Successfully downloaded and cached!")
            else:
                print(f"   📁 Cache directory exists but dataset not found at: {cache_dataset_path}")
                print("   📥 Downloading dataset to cache...")
                try:
                    # Load only the validation split to avoid train split generation
                    dataset = load_dataset("natural_questions", split="validation", cache_dir=cache_path)
                    print("   ✅ Successfully downloaded and cached!")
                except Exception as e:
                    print(f"   ⚠️ Download failed: {e}")
                    print("   🔄 Falling back to streaming...")
                    # If download fails, use streaming for limited samples
                    dataset = load_dataset("natural_questions", split="validation", streaming=True)
                    # Convert to limited list
                    limited_dataset = []
                    print("   📥 Converting streaming dataset to limited samples...")
                    
                    try:
                        from tqdm import tqdm
                        for i, item in enumerate(tqdm(dataset, desc="Loading samples", unit="sample")):
                            if i >= max_questions:  # Use max_questions parameter
                                break
                            limited_dataset.append(item)
                    except ImportError:
                        for i, item in enumerate(dataset):
                            if i >= max_questions:  # Use max_questions parameter
                                break
                            limited_dataset.append(item)
                    
                    dataset = limited_dataset
        else:
            print("   📥 Downloading dataset (no cache directory specified)...")
            try:
                dataset = load_dataset("natural_questions", split="validation")
                print("   ✅ Successfully downloaded!")
            except Exception as e:
                print(f"   ⚠️ Download failed: {e}")
                print("   🔄 Falling back to streaming...")
                # If download fails, use streaming for limited samples
                dataset = load_dataset("natural_questions", split="validation", streaming=True)
                # Convert to limited list
                limited_dataset = []
                print("   📥 Converting streaming dataset to limited samples...")
                
                try:
                    from tqdm import tqdm
                    for i, item in enumerate(tqdm(dataset, desc="Loading samples", unit="sample")):
                        if i >= max_questions:  # Use max_questions parameter
                            break
                        limited_dataset.append(item)
                except ImportError:
                    for i, item in enumerate(dataset):
                        if i >= max_questions:  # Use max_questions parameter
                            break
                        limited_dataset.append(item)
                
                dataset = limited_dataset
        
        print("   ✅ Dataset loaded successfully!")
        logging.info("Dataset loaded successfully")
        
        # Sample the dataset
        total_samples = len(dataset)
        
        # If we have a limited dataset (from streaming), prioritize max_questions over percentage
        if total_samples <= max_questions and max_questions > total_samples * sample_percentage:
            sample_size = min(total_samples, max_questions)
            print(f"📊 Natural Questions dataset: {total_samples:,} total questions (limited subset)")
            print(f"📊 Using: {sample_size} questions (max: {max_questions})")
        else:
            sample_size = min(int(total_samples * sample_percentage), max_questions)
            print(f"📊 Natural Questions dataset: {total_samples:,} total questions")
            print(f"📊 Sampling: {sample_percentage*100:.1f}% = {sample_size} questions (max: {max_questions})")
        
        # Take a random sample - handle both Dataset and list objects
        print("🔄 Sampling dataset...")
        if hasattr(dataset, 'shuffle'):
            # HuggingFace Dataset object
            sampled_dataset = dataset.shuffle(seed=42).select(range(sample_size))
        else:
            # Python list - use random sampling
            import random
            random.seed(42)
            sampled_dataset = random.sample(dataset, min(sample_size, len(dataset)))
        
        print(f"✅ Sampled {len(sampled_dataset)} questions from dataset")
        
        # Extract questions and enhanced answer metadata
        questions = []
        ground_truth_metadata = []
        
        print("🔄 Processing questions and answers with enhanced metadata...")
        
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
            
            # Enhanced answer selection with metadata
            answer_info = select_best_answer_with_metadata(item['annotations'])
            
            if answer_info is None:
                # Skip items without valid answers
                print(f"   ⚠️ Skipping question without valid answers: {question[:50]}...")
                continue
            
            # Additional validation: ensure we have a non-empty answer
            if not answer_info.get('answer') or answer_info['answer'].strip() == '':
                print(f"   ⚠️ Skipping question with empty answer: {question[:50]}...")
                continue
            
            questions.append(question)
            ground_truth_metadata.append(answer_info)
        
        print(f"✅ Successfully loaded {len(questions)} questions from Natural Questions dataset")
        logging.info(f"Loaded {len(questions)} questions from Natural Questions dataset")
        
        # Show sample questions with answer types
        print("\n📝 Sample questions loaded:")
        for i in range(min(3, len(questions))):
            answer_info = ground_truth_metadata[i]
            print(f"   {i+1}. Q: {questions[i][:60]}...")
            print(f"      A: {answer_info['answer']} (Type: {answer_info['answer_type']})")
            if answer_info['confidence']:
                print(f"      Confidence: {answer_info['confidence']}")
        
        return questions, ground_truth_metadata
        
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

def select_best_answer_with_metadata(annotations):
    """Enhanced answer selection with metadata preservation"""
    
    answer_info = {
        'answer': None,
        'answer_type': None,
        'confidence': None,
        'all_answers': [],
        'answer_count': 0
    }
    
    # Helper function to safely extract text from various formats
    def safe_extract_text(text_field):
        """Safely extract text from field that might be string, list, or other format"""
        if isinstance(text_field, str):
            return text_field.strip()
        elif isinstance(text_field, list):
            # If it's a list, join the elements
            if text_field:
                return " ".join(str(item) for item in text_field if item).strip()
            else:
                return ""
        elif text_field is None:
            return ""
        else:
            return str(text_field).strip()
    
    # Collect all available answers
    all_answers = []
    
    # Process short answers
    if annotations.get('short_answers'):
        for short_ans in annotations['short_answers']:
            text = safe_extract_text(short_ans.get('text'))
            if text:  # Only add if we have actual content
                all_answers.append({
                    'text': text,
                    'type': 'short_text',
                    'confidence': 1.0,  # Short answers are typically high confidence
                    'metadata': short_ans
                })
    
    # Process yes/no answers
    if annotations.get('yes_no_answers'):
        yes_no_answers = annotations['yes_no_answers']
        if yes_no_answers:  # Ensure we have actual answers
            # Count occurrences for majority vote
            yes_count = sum(1 for ans in yes_no_answers if ans == True)
            no_count = sum(1 for ans in yes_no_answers if ans == False)
            
            if yes_count > no_count:
                majority_answer = "Yes"
                confidence = yes_count / len(yes_no_answers)
            elif no_count > yes_count:
                majority_answer = "No"
                confidence = no_count / len(yes_no_answers)
            else:
                majority_answer = "Uncertain"
                confidence = 0.5
            
            all_answers.append({
                'text': majority_answer,
                'type': 'yes_no',
                'confidence': confidence,
                'metadata': {
                    'yes_count': yes_count,
                    'no_count': no_count,
                    'total_votes': len(yes_no_answers)
                }
            })
    
    # Process long answers (extract key facts)
    if annotations.get('long_answers'):
        long_answers = annotations['long_answers']
        # For long answers, we'll extract the first one and summarize
        if long_answers and long_answers[0].get('text'):
            long_text = safe_extract_text(long_answers[0].get('text'))
            if long_text:  # Only process if we have actual content
                # Simple extraction of key facts (first sentence or key phrases)
                key_facts = extract_key_facts_from_long_answer(long_text)
                if key_facts.strip():  # Only add if we have actual content
                    all_answers.append({
                        'text': key_facts,
                        'type': 'long_text',
                        'confidence': 0.8,  # Long answers might be less precise
                        'metadata': {'original_length': len(long_text)}
                    })
    
    if not all_answers:
        return None
    
    # Select the best answer based on type priority and confidence
    answer_info['all_answers'] = all_answers
    answer_info['answer_count'] = len(all_answers)
    
    # Priority order: short_text > yes_no > long_text
    type_priority = {'short_text': 3, 'yes_no': 2, 'long_text': 1}
    
    best_answer = max(all_answers, key=lambda x: (type_priority.get(x['type'], 0), x['confidence']))
    
    answer_info['answer'] = best_answer['text']
    answer_info['answer_type'] = best_answer['type']
    answer_info['confidence'] = best_answer['confidence']
    
    return answer_info

def extract_key_facts_from_long_answer(long_text, max_length=100):
    """Extract key facts from long answer text"""
    if not long_text:
        return ""
    
    # Simple approach: take first sentence or first N characters
    sentences = long_text.split('.')
    if sentences:
        first_sentence = sentences[0].strip()
        if len(first_sentence) <= max_length:
            return first_sentence
        else:
            # Truncate to max_length
            return first_sentence[:max_length].rsplit(' ', 1)[0] + "..."
    
    # Fallback: truncate the entire text
    return long_text[:max_length].rsplit(' ', 1)[0] + "..." if len(long_text) > max_length else long_text

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
        
        # Download the dataset with streaming to avoid memory issues
        try:
            print("   🔄 Using streaming to download only what we need...")
            dataset = load_dataset(
                "natural_questions", 
                split=split, 
                cache_dir=cache_path,
                streaming=True
            )
            
            # Convert streaming dataset to a limited list
            print("   📊 Processing samples...")
            limited_dataset = []
            target_samples = max(1000, max_questions)  # Process at least 1000 or max_questions
            
            for i, item in enumerate(dataset):
                if i >= target_samples:
                    break
                limited_dataset.append(item)
                
                if (i + 1) % 100 == 0:
                    print(f"      Processed {i + 1}/{target_samples} samples...")
            
            print(f"   ✅ Successfully processed {len(limited_dataset)} samples")
            
        except Exception as e:
            print(f"   ❌ Download failed: {e}")
            print("   💡 This might be due to:")
            print("      - Network connectivity issues")
            print("      - Insufficient disk space")
            print("      - Memory limitations")
            print("      - HuggingFace server issues")
            raise e
        
        print(f"\n✅ Successfully downloaded dataset!")
        print(f"📊 Total questions: {len(limited_dataset):,}")
        
        # Show sample questions
        print("\n📝 Sample questions:")
        for i in range(min(3, len(limited_dataset))):
            question = limited_dataset[i]['question']['text']
            print(f"   {i+1}. {question[:80]}{'...' if len(question) > 80 else ''}")
        
        print(f"\n💾 Dataset cached at: {cache_path}")
        print("🎯 You can now use --cache-dir option to load from this location")
        print("⚠️  Note: This is a limited subset (1000 samples) for faster processing")
        
        return True
        
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        return False

def load_curated_questions() -> Tuple[List[str], List[Dict]]:
    """Load curated questions for evaluation with enhanced metadata format"""
    questions = [
        "Who painted the Mona Lisa?",
        "Who was the first woman to win a Nobel Prize?",
        "What is the capital of France?",
        "When was the Declaration of Independence signed?",
        "What is the capital city of Australia?",
        "Who wrote Romeo and Juliet?",
        "Who discovered penicillin?",
        "What is the main component of the sun?",
        "Who invented the telephone?",
        "Who wrote 'Alice's Adventures in Wonderland'?",
    ]
    
    # Enhanced metadata format matching Natural Questions structure
    ground_truth_metadata = [
        {
            'answer': 'Leonardo da Vinci',
            'answer_type': 'short_text',
            'confidence': 1.0,
            'all_answers': [{'text': 'Leonardo da Vinci', 'type': 'short_text', 'confidence': 1.0}],
            'answer_count': 1
        },
        {
            'answer': 'Marie Curie',
            'answer_type': 'short_text',
            'confidence': 1.0,
            'all_answers': [{'text': 'Marie Curie', 'type': 'short_text', 'confidence': 1.0}],
            'answer_count': 1
        },
        {
            'answer': 'Paris',
            'answer_type': 'short_text',
            'confidence': 1.0,
            'all_answers': [{'text': 'Paris', 'type': 'short_text', 'confidence': 1.0}],
            'answer_count': 1
        },
        {
            'answer': '1776',
            'answer_type': 'short_text',
            'confidence': 1.0,
            'all_answers': [{'text': '1776', 'type': 'short_text', 'confidence': 1.0}],
            'answer_count': 1
        },
        {
            'answer': 'Canberra',
            'answer_type': 'short_text',
            'confidence': 1.0,
            'all_answers': [{'text': 'Canberra', 'type': 'short_text', 'confidence': 1.0}],
            'answer_count': 1
        },
        {
            'answer': 'William Shakespeare',
            'answer_type': 'short_text',
            'confidence': 1.0,
            'all_answers': [{'text': 'William Shakespeare', 'type': 'short_text', 'confidence': 1.0}],
            'answer_count': 1
        },
        {
            'answer': 'Alexander Fleming',
            'answer_type': 'short_text',
            'confidence': 1.0,
            'all_answers': [{'text': 'Alexander Fleming', 'type': 'short_text', 'confidence': 1.0}],
            'answer_count': 1
        },
        {
            'answer': 'Hydrogen',
            'answer_type': 'short_text',
            'confidence': 1.0,
            'all_answers': [{'text': 'Hydrogen', 'type': 'short_text', 'confidence': 1.0}],
            'answer_count': 1
        },
        {
            'answer': 'Alexander Graham Bell',
            'answer_type': 'short_text',
            'confidence': 1.0,
            'all_answers': [{'text': 'Alexander Graham Bell', 'type': 'short_text', 'confidence': 1.0}],
            'answer_count': 1
        },
        {
            'answer': 'Lewis Carroll',
            'answer_type': 'short_text',
            'confidence': 1.0,
            'all_answers': [{'text': 'Lewis Carroll', 'type': 'short_text', 'confidence': 1.0}],
            'answer_count': 1
        }
    ]
    
    logging.info(f"Loaded {len(questions)} curated questions with enhanced metadata")
    return questions, ground_truth_metadata

def cache_natural_questions_dataset(cache_dir: str = None, split: str = "validation"):
    """Pre-download and cache the Natural Questions dataset"""
    import os
    
    if cache_dir:
        cache_path = os.path.expanduser(cache_dir)
        os.makedirs(cache_path, exist_ok=True)
        print(f"📁 Using cache directory: {cache_path}")
    else:
        cache_path = None
        print("🚫 No cache directory specified - using default location")
    
    print(f"📥 Downloading Natural Questions dataset ({split} split only)...")
    print("⏳ This may take 10-30 minutes depending on your internet connection...")
    print("💡 Only downloading the validation split to avoid unnecessary train split generation")
    
    try:
        # Download only the specified split to avoid train split generation
        dataset = load_dataset("natural_questions", split=split, cache_dir=cache_path)
        
        total_samples = len(dataset)
        print(f"✅ Successfully downloaded and cached {total_samples:,} samples!")
        
        if cache_path:
            cache_dataset_path = os.path.join(cache_path, "natural_questions")
            print(f"📁 Dataset cached at: {cache_dataset_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to download dataset: {e}")
        return False

def main():
    """Main function for standalone dataset operations"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Dataset operations")
    parser.add_argument("--cache-dataset", action="store_true", 
                       help="Download and cache Natural Questions dataset")
    parser.add_argument("--cache-dir", type=str, default=None,
                       help="Directory to cache dataset")
    parser.add_argument("--split", type=str, default="validation",
                       choices=["train", "validation", "test"],
                       help="Dataset split to download")
    
    args = parser.parse_args()
    
    if args.cache_dataset:
        success = cache_natural_questions_dataset(args.cache_dir, args.split)
        if success:
            print("🎉 Dataset caching completed successfully!")
        else:
            print("💥 Dataset caching failed!")
            return 1
    
    return 0

if __name__ == "__main__":
    main() 
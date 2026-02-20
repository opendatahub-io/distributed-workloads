#!/usr/bin/env python3
"""
Ray based Synthetic Data Generation Script
"""

import os
import json
import ray
import ray.data
import torch
import warnings
import time
import argparse
import numpy as np
import threading
import signal
from typing import List, Dict, Optional, Tuple
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import random
import re
from dataclasses import dataclass
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")

# Configuration constants
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
# Alternative models that can be used: "microsoft/DialoGPT-medium", "microsoft/DialoGPT-small"

@dataclass
class QualityMetrics:
    """Quality assessment metrics for generated content."""
    mathematical_correctness: float = 0.0
    reasoning_quality: float = 0.0
    problem_complexity: float = 0.0
    answer_completeness: float = 0.0
    overall_quality: float = 0.0

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy arrays."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        return super().default(obj)

class CheckpointManager:
    """Manages checkpointing and resume functionality."""
    
    def __init__(self, output_dir: str, save_every: int = 5):
        self.output_dir = Path(output_dir)
        self.save_every = save_every
        self.checkpoint_file = self.output_dir / "checkpoint.json"
        self.dataset_file = self.output_dir / "synthetic_dataset.json"
        self.metadata_file = self.output_dir / "dataset_metadata.json"
        self.processed_seeds = set()
        self.current_data = []
        self.checkpoint_count = 0
        self.lock = threading.Lock()
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def load_checkpoint(self) -> Dict:
        """Load existing checkpoint if available."""
        if not self.checkpoint_file.exists():
            return {}
            
        try:
            with open(self.checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
            
            self.processed_seeds = set(checkpoint.get('processed_seeds', []))
            self.checkpoint_count = checkpoint.get('checkpoint_count', 0)
            
            if self.dataset_file.exists():
                with open(self.dataset_file, 'r') as f:
                    self.current_data = json.load(f)
            
            print(f"Loaded checkpoint: {len(self.processed_seeds)} seeds processed, {len(self.current_data)} samples saved")
            return checkpoint
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")
            return {}
    
    def save_checkpoint(self, processed_seeds: set, total_expected: int, force: bool = False):
        """Save current progress to checkpoint."""
        with self.lock:
            self.processed_seeds.update(processed_seeds)
            
            if not (force or len(self.processed_seeds) % self.save_every == 0):
                return
                
            checkpoint = {
                'processed_seeds': list(self.processed_seeds),
                'checkpoint_count': self.checkpoint_count + 1,
                'total_expected': total_expected,
                'timestamp': time.time(),
                'progress_percentage': (len(self.processed_seeds) / total_expected) * 100 if total_expected > 0 else 0
            }
            
            try:
                with open(self.checkpoint_file, 'w') as f:
                    json.dump(checkpoint, f, indent=2)
                
                self.checkpoint_count += 1
                print(f"Checkpoint saved: {len(self.processed_seeds)}/{total_expected} seeds processed ({checkpoint['progress_percentage']:.1f}%)")
            except Exception as e:
                print(f"Failed to save checkpoint: {e}")
    
    def save_batch_data(self, batch_data: List[Dict]):
        """Save batch data incrementally."""
        with self.lock:
            self.current_data.extend(batch_data)
            
            try:
                with open(self.dataset_file, 'w') as f:
                    json.dump(self.current_data, f, indent=2, cls=NumpyEncoder)
                
                metadata = self._create_metadata()
                with open(self.metadata_file, 'w') as f:
                    json.dump(metadata, f, indent=2, cls=NumpyEncoder)
                
            except Exception as e:
                print(f"Failed to save batch data: {e}")
    
    def _create_metadata(self) -> Dict:
        """Create metadata for current dataset."""
        total_samples = len(self.current_data)
        high_quality_count = sum(1 for item in self.current_data if item.get('overall_quality', 0) >= 0.3)
        
        metadata = {
            'total_samples': total_samples,
            'high_quality_count': high_quality_count,
            'last_update': time.time(),
            'checkpoint_count': self.checkpoint_count
        }
        
        if total_samples > 0:
            metadata['quality_pass_rate'] = (high_quality_count / total_samples) * 100
            metadata['avg_quality_score'] = np.mean([item.get('overall_quality', 0) for item in self.current_data])
        
        return metadata
    
    def get_remaining_seeds(self, all_seed_ids: List[int]) -> List[int]:
        """Get list of seeds that haven't been processed yet."""
        return [seed_id for seed_id in all_seed_ids if seed_id not in self.processed_seeds]
    
    def is_seed_processed(self, seed_id: int) -> bool:
        """Check if a seed has already been processed."""
        return seed_id in self.processed_seeds

class ModelInferenceCallable:
    """Ray Data callable class for distributed model inference with checkpointing."""
    
    def __init__(self, model_name: str = MODEL_NAME, variations_per_seed: int = 1, 
                 output_dir: str = "/tmp/synthetic_data", processed_seeds: set = None):
        self.model_name = model_name
        self.variations_per_seed = variations_per_seed
        self.output_dir = output_dir
        self.processed_seeds = processed_seeds or set()
        self.model = None
        self.tokenizer = None
        self.device = None
        
    def __call__(self, batch: Dict[str, List]) -> Dict[str, List]:
        """Process a batch of seed samples to generate synthetic data."""
        if self.model is None:
            self._initialize_model()
        
        results = {
            "questions": [], "answers": [], "sources": [], "difficulties": [],
            "concepts": [], "quality_scores": [], "model_confidences": [],
            "seed_ids": [], "variation_ids": []
        }
        
        processed_seed_ids = set()
        batch_data = []
        
        for i, seed_sample in enumerate(batch["seed_samples"]):
            seed_id = batch.get("seed_ids", [i])[i]
            
            if seed_id in self.processed_seeds:
                print(f"[Batch] Skipping already processed seed {seed_id}")
                continue
            
            print(f"[Batch] Processing seed {seed_id}: {seed_sample.get('question', 'No question')[:50]}...")
            
            for var_id in range(self.variations_per_seed):
                try:
                    generated = self._generate_variation(seed_sample, var_id)
                    if not generated or not self._validate_quality(generated):
                        continue
                    
                    quality_metrics = self._assess_quality(generated)
                    
                    # Add to results
                    self._add_to_results(results, generated, quality_metrics, seed_id, var_id)
                    self._add_to_batch_data(batch_data, generated, quality_metrics, seed_id, var_id)
                        
                except Exception as e:
                    print(f"[Batch] Error processing seed {seed_id}, variation {var_id}: {e}")
                    continue
            
            processed_seed_ids.add(seed_id)
        
        if batch_data:
            self._save_batch_data(batch_data)
        
        self._cleanup_gpu_memory()
        return results
    
    def _initialize_model(self):
        """Initialize model and tokenizer on worker."""
        print(f"[Worker] Loading model: {self.model_name}")
        
        cache_dir = self._get_cache_directory()
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                # Load tokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name, 
                    trust_remote_code=True, 
                    cache_dir=cache_dir,
                    resume_download=True,
                    force_download=False,
                    local_files_only=False
                )
                
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                
                # Setup device
                self.device, device_map = self._setup_device()
                
                # Load model
                model_dtype = torch.bfloat16 if self.device.startswith("cuda") else torch.float32
                
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    trust_remote_code=True,
                    torch_dtype=model_dtype,
                    cache_dir=cache_dir,
                    device_map=device_map,
                    low_cpu_mem_usage=True,
                    use_safetensors=True,
                    resume_download=True,
                    force_download=False,
                    local_files_only=False
                )
                
                if device_map is None and self.device != "cpu":
                    self.model = self.model.to(self.device)
                
                self.model.eval()
                
                if self.device.startswith("cuda"):
                    torch.cuda.empty_cache()
                
                print(f"[Worker] Model loaded successfully on {self.device}")
                return
                
            except Exception as e:
                print(f"[Worker] Attempt {attempt + 1} failed: {e}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 10
                    print(f"[Worker] Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                else:
                    raise Exception(f"Failed to load model after {max_retries} attempts. Last error: {e}")
    
    def _setup_device(self) -> Tuple[str, Optional[str]]:
        """Setup device configuration."""
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        
        if not torch.cuda.is_available():
            return "cpu", None
        
        available_gpus = torch.cuda.device_count()
        ray_gpu_ids = ray.get_gpu_ids()
        
        print(f"[Worker] CUDA available: {torch.cuda.is_available()}, GPUs: {available_gpus}, Ray GPUs: {ray_gpu_ids}")
        
        if not ray_gpu_ids:
            return ("cuda:0" if available_gpus > 0 else "cpu"), None
        
        valid_gpus = [gpu_id for gpu_id in ray_gpu_ids if gpu_id < available_gpus]
        if not valid_gpus:
            return "cuda:0", None
        
        primary_gpu = valid_gpus[0]
        device_map = "auto" if len(valid_gpus) > 1 else None
        
        return f"cuda:{primary_gpu}", device_map
    
    def _get_cache_directory(self) -> str:
        """Get cache directory for model storage."""
        for path in ["/shared/cache", os.path.expanduser("~/.cache"), "/tmp/.cache"]:
            if os.path.exists(os.path.dirname(path)):
                os.makedirs(path, exist_ok=True)
                return path
        return "/tmp/.cache"
    
    def _generate_variation(self, seed_sample: Dict, var_id: int) -> Optional[Dict]:
        """Generate a single variation from a seed sample."""
        difficulties = ["easy", "medium", "hard"]
        difficulty = difficulties[var_id % len(difficulties)]
        
        prompt = self._create_variation_prompt(seed_sample, difficulty)
        
        try:
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=512
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                    top_k=50,
                    repetition_penalty=1.1,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            
            response = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:], 
                skip_special_tokens=True
            ).strip()
            
            parsed = self._parse_response(response)
            if parsed:
                parsed["difficulty"] = difficulty
                return parsed
            
        except Exception as e:
            print(f"[Worker] Generation error: {e}")
        
        return None
    
    def _create_variation_prompt(self, seed_sample: Dict, difficulty: str = "medium") -> str:
        """Create prompt for generating variations."""
        seed_question = seed_sample["question"]
        seed_answer = seed_sample["answer"]
        
        difficulty_constraints = {
            "easy": "single-step calculation with whole numbers, clear and simple scenario",
            "medium": "2-3 step calculation, may include decimals, requires logical reasoning",
            "hard": "multi-step problem requiring multiple operations and complex logical reasoning"
        }
        
        quality_examples = {
            "easy": {
                "question": "A bakery sold 45 cupcakes in the morning and 38 cupcakes in the afternoon. How many cupcakes did they sell in total?",
                "answer": "To find the total cupcakes sold, I need to add the morning and afternoon sales.\nMorning sales: 45 cupcakes\nAfternoon sales: 38 cupcakes\nTotal = 45 + 38 = 83 cupcakes\nTherefore, the bakery sold 83 cupcakes in total."
            },
            "medium": {
                "question": "A school bought 15 notebooks for $45. If each notebook costs the same amount, how much would it cost to buy 27 notebooks?",
                "answer": "First, I need to find the cost per notebook.\nCost per notebook = Total cost ÷ Number of notebooks = $45 ÷ 15 = $3 per notebook\nNext, I'll calculate the cost for 27 notebooks.\nCost for 27 notebooks = 27 × $3 = $81\nTherefore, it would cost $81 to buy 27 notebooks."
            },
            "hard": {
                "question": "A company produces widgets at a rate of 12 per hour. If they work 8 hours per day and need to fulfill an order of 2,400 widgets, how many full days will it take to complete the order?",
                "answer": "First, I'll calculate how many widgets are produced per day.\nWidgets per hour = 12\nHours per day = 8\nWidgets per day = 12 × 8 = 96 widgets per day\nNext, I'll find how many days are needed for 2,400 widgets.\nDays needed = Total widgets ÷ Widgets per day = 2,400 ÷ 96 = 25 days\nTherefore, it will take 25 full days to complete the order."
            }
        }
        
        example = quality_examples.get(difficulty, quality_examples["medium"])
        constraint = difficulty_constraints.get(difficulty, difficulty_constraints["medium"])
        
        return f"""<|im_start|>system
You are an expert mathematics educator creating high-quality word problems for student practice.

DIFFICULTY LEVEL: {difficulty.upper()}
REQUIREMENTS: {constraint}

QUALITY STANDARDS:
1. Create a realistic, engaging scenario (shopping, cooking, travel, business, etc.)
2. Question must be clear, specific, and unambiguous
3. Provide a complete step-by-step solution showing ALL calculations
4. Each calculation must be mathematically correct (double-check your arithmetic!)
5. Include the final numerical answer clearly stated with appropriate units
6. Use proper mathematical language and logical flow

EXAMPLE OF HIGH QUALITY {difficulty.upper()} PROBLEM:
Question: {example["question"]}
Answer: {example["answer"]}

SEED INSPIRATION (create something similar but different):
Question: {seed_question}
Answer: {seed_answer}

Return ONLY a JSON object with "question" and "answer" fields. Ensure your answer shows step-by-step work and arrives at a correct final numerical answer.
<|im_end|>
<|im_start|>user
Generate a high-quality {difficulty} math word problem with complete step-by-step solution:
<|im_end|>
<|im_start|>assistant
"""
    
    def _parse_response(self, response: str) -> Optional[Dict]:
        """Parse model response into structured format."""
        try:
            response = response.strip()
            
            # Try JSON parsing first
            if response.startswith('{') and response.endswith('}'):
                return json.loads(response)
            
            # Handle Qwen format
            if response.startswith('"') and '"answer":' in response:
                if not response.startswith('{'):
                    response = "{" + response
                if not response.endswith('}'):
                    response = response + "}"
                return json.loads(response)
            
            # Try to find JSON-like content
            json_match = re.search(r'\{[^{}]*"question"[^{}]*"answer"[^{}]*\}', response, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group(0))
                except:
                    pass
            
            return self._fallback_parse(response)
            
        except Exception:
            return self._fallback_parse(response)
    
    def _fallback_parse(self, response: str) -> Optional[Dict]:
        """Fallback parsing for non-JSON responses."""
        # Extract question and answer with regex
        question_match = re.search(r'"question":\s*"([^"]*(?:\\.[^"]*)*?)"', response, re.DOTALL)
        answer_match = re.search(r'"answer":\s*"([^"]*(?:\\.[^"]*)*?)"', response, re.DOTALL)
        
        if question_match and answer_match:
            question = question_match.group(1).replace('\\"', '"').replace('\\n', '\n')
            answer = answer_match.group(1).replace('\\"', '"').replace('\\n', '\n')
            return {"question": question, "answer": answer, "confidence": 0.7}
        
        # Try alternative patterns
        alt_question_match = re.search(r'Question:\s*(.+?)(?=Answer:|$)', response, re.DOTALL | re.IGNORECASE)
        alt_answer_match = re.search(r'Answer:\s*(.+?)(?=Question:|$)', response, re.DOTALL | re.IGNORECASE)
        
        if alt_question_match and alt_answer_match:
            return {
                "question": alt_question_match.group(1).strip(),
                "answer": alt_answer_match.group(1).strip(),
                "confidence": 0.6
            }
        
        # If response looks like a math problem
        if any(keyword in response.lower() for keyword in ["how many", "what is", "calculate", "find"]):
            return {
                "question": response.strip(),
                "answer": "This is a mathematical word problem that requires calculation.",
                "confidence": 0.5
            }
        
        return None
    
    def _validate_quality(self, generated: Dict) -> bool:
        """Enhanced quality validation."""
        if not generated or not generated.get("question") or not generated.get("answer"):
            return False
        
        question = str(generated["question"]).lower()
        answer = str(generated["answer"]).lower()
        
        # Check for mathematical content
        math_indicators = ["calculate", "solve", "find", "how many", "total", "cost", "price", "sum", "difference"]
        has_math = any(indicator in question for indicator in math_indicators)
        
        # Check minimum length and completeness
        min_length = len(question) >= 25 and len(answer) >= 50
        
        # Check if answer seems complete
        has_calculation = any(word in answer for word in ["=", "total", "result", "answer is", "solution", "therefore"])
        has_numbers = len(re.findall(r'\d+', answer)) >= 2
        
        # Avoid incomplete answers
        incomplete_indicators = ["let's", "step by step:", "to determine", "we need to", "first,", "i need to", "let me"]
        is_incomplete = any(indicator in answer and len(answer) < 150 for indicator in incomplete_indicators)
        
        # Check for realistic scenarios
        scenario_words = ["bakery", "school", "store", "farmer", "company", "trip", "recipe", "budget", "restaurant", "library"]
        has_scenario = any(word in question for word in scenario_words)
        
        # Additional quality checks
        has_proper_question = any(phrase in question for phrase in ["how many", "how much", "what is", "calculate"])
        has_final_answer = any(phrase in answer for phrase in ["therefore", "the answer is", "total", "result"])
        
        return (has_math and min_length and (has_calculation or has_numbers) and 
                not is_incomplete and (has_scenario or has_proper_question) and has_final_answer)
    
    def _assess_quality(self, generated: Dict) -> QualityMetrics:
        """Assess quality of generated content."""
        question = str(generated.get("question", ""))
        answer = str(generated.get("answer", ""))
        
        math_score = self._validate_mathematical_correctness(question, answer)
        reasoning_score = self._assess_reasoning_quality(answer)
        complexity_score = self._assess_problem_complexity(question)
        completeness_score = self._assess_answer_completeness(answer)
        
        overall_quality = (math_score * 0.4 + reasoning_score * 0.3 + 
                          complexity_score * 0.2 + completeness_score * 0.1)
        
        return QualityMetrics(
            mathematical_correctness=math_score,
            reasoning_quality=reasoning_score,
            problem_complexity=complexity_score,
            answer_completeness=completeness_score,
            overall_quality=overall_quality
        )
    
    def _validate_mathematical_correctness(self, question: str, answer: str) -> float:
        """Validate mathematical correctness by checking calculations."""
        calculations = re.findall(r'(\d+(?:\.\d+)?)\s*([+\-*/÷×])\s*(\d+(?:\.\d+)?)\s*=\s*(\d+(?:\.\d+)?)', answer)
        
        if not calculations:
            math_indicators = ["total", "sum", "difference", "product", "quotient", "result"]
            has_math = any(indicator in answer.lower() for indicator in math_indicators)
            has_numbers = len(re.findall(r'\d+', answer)) >= 2
            return 0.6 if has_math and has_numbers else 0.3
        
        correct_count = 0
        for calc in calculations:
            try:
                num1, op, num2, result = float(calc[0]), calc[1], float(calc[2]), float(calc[3])
                
                if op == '+':
                    expected = num1 + num2
                elif op == '-':
                    expected = num1 - num2
                elif op in ['*', '×']:
                    expected = num1 * num2
                elif op in ['/', '÷'] and num2 != 0:
                    expected = num1 / num2
                else:
                    continue
                
                if abs(expected - result) < 0.01:
                    correct_count += 1
            except:
                continue
        
        return correct_count / len(calculations) if calculations else 0.3
    
    def _assess_reasoning_quality(self, answer: str) -> float:
        """Assess quality of step-by-step reasoning."""
        reasoning_score = 0.0
        
        # Check for logical flow indicators
        flow_indicators = ["first", "then", "next", "therefore", "so", "thus", "finally"]
        flow_count = sum(1 for indicator in flow_indicators if indicator in answer.lower())
        reasoning_score += min(flow_count * 0.15, 0.4)
        
        # Check for explanation words
        explanation_words = ["because", "since", "as", "given that", "we know", "to find"]
        explanation_count = sum(1 for word in explanation_words if word in answer.lower())
        reasoning_score += min(explanation_count * 0.1, 0.3)
        
        # Check for step-by-step structure
        step_patterns = ["step 1", "step 2", "1.", "2.", "3."]
        if any(pattern in answer.lower() for pattern in step_patterns):
            reasoning_score += 0.2
        
        # Check for calculation explanation
        calc_explanations = ["calculate", "multiply", "add", "subtract", "divide"]
        if any(word in answer.lower() for word in calc_explanations):
            reasoning_score += 0.1
        
        return min(reasoning_score, 1.0)
    
    def _assess_problem_complexity(self, question: str) -> float:
        """Assess problem complexity and clarity."""
        complexity_score = 0.0
        
        # Check question length
        word_count = len(question.split())
        if 15 <= word_count <= 50:
            complexity_score += 0.3
        elif 10 <= word_count <= 60:
            complexity_score += 0.2
        else:
            complexity_score += 0.1
        
        # Check for realistic scenario words
        scenario_words = ["bakery", "school", "store", "farmer", "company", "trip", "recipe", "budget"]
        if any(word in question.lower() for word in scenario_words):
            complexity_score += 0.2
        
        # Check for clear question structure
        question_words = ["how many", "how much", "what is", "calculate", "find"]
        if any(phrase in question.lower() for phrase in question_words):
            complexity_score += 0.3
        
        # Check for multiple numbers
        numbers = re.findall(r'\d+', question)
        if len(numbers) >= 2:
            complexity_score += 0.2
        
        return min(complexity_score, 1.0)
    
    def _assess_answer_completeness(self, answer: str) -> float:
        """Check if answer is complete with final numerical result."""
        completeness_score = 0.0
        
        # Check for final answer indicators
        final_indicators = ["therefore", "the answer is", "total", "result", "final answer"]
        if any(indicator in answer.lower() for indicator in final_indicators):
            completeness_score += 0.4
        
        # Check for numerical answer
        if re.findall(r'\d+(?:\.\d+)?', answer):
            completeness_score += 0.3
        
        # Check minimum answer length
        if len(answer.split()) >= 20:
            completeness_score += 0.2
        
        # Check for units or context
        unit_words = ["dollars", "cents", "items", "people", "days", "hours", "pounds", "kilograms"]
        if any(unit in answer.lower() for unit in unit_words):
            completeness_score += 0.1
        
        return min(completeness_score, 1.0)
    
    def _add_to_results(self, results: Dict, generated: Dict, quality_metrics: QualityMetrics, 
                       seed_id: int, var_id: int):
        """Add generated content to results."""
        results["questions"].append(str(generated["question"]))
        results["answers"].append(str(generated["answer"]))
        results["sources"].append("ray_data_sdg_qwen")
        results["difficulties"].append(str(generated.get("difficulty", "medium")))
        results["concepts"].append("arithmetic,word_problems")
        results["quality_scores"].append(quality_metrics.overall_quality)
        results["model_confidences"].append(float(generated.get("confidence", 0.5)))
        results["seed_ids"].append(seed_id)
        results["variation_ids"].append(var_id)
    
    def _add_to_batch_data(self, batch_data: List[Dict], generated: Dict, 
                          quality_metrics: QualityMetrics, seed_id: int, var_id: int):
        """Add generated content to batch data."""
        batch_item = {
            "question": str(generated["question"]),
            "answer": str(generated["answer"]),
            "source": "ray_data_sdg_qwen",
            "difficulty": str(generated.get("difficulty", "medium")),
            "concepts": ["arithmetic", "word_problems"],
            "overall_quality": quality_metrics.overall_quality,
            "model_confidence": float(generated.get("confidence", 0.5)),
            "seed_id": seed_id,
            "variation_id": var_id
        }
        batch_data.append(batch_item)
    
    def _save_batch_data(self, batch_data: List[Dict]):
        """Save batch data incrementally."""
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            
            dataset_path = os.path.join(self.output_dir, "synthetic_dataset.json")
            existing_data = []
            
            if os.path.exists(dataset_path):
                try:
                    with open(dataset_path, "r") as f:
                        existing_data = json.load(f)
                except:
                    pass
            
            existing_data.extend(batch_data)
            
            with open(dataset_path, "w") as f:
                json.dump(existing_data, f, indent=2, cls=NumpyEncoder)
            
            # Update metadata
            metadata = {
                "total_samples": len(existing_data),
                "high_quality_count": sum(1 for item in existing_data if item.get("overall_quality", 0) >= 0.3),
                "last_update": time.time(),
                "batch_count": len(batch_data)
            }
            
            if metadata["total_samples"] > 0:
                metadata["quality_pass_rate"] = (metadata["high_quality_count"] / metadata["total_samples"]) * 100
                metadata["avg_quality_score"] = np.mean([item.get("overall_quality", 0) for item in existing_data])
            
            metadata_path = os.path.join(self.output_dir, "dataset_metadata.json")
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2, cls=NumpyEncoder)
            
            print(f"Saved batch: {len(batch_data)} new samples (Total: {len(existing_data)})")
            
        except Exception as e:
            print(f"Warning: Failed to save batch data: {e}")
    
    def _cleanup_gpu_memory(self):
        """Clean up GPU memory after batch processing."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

def get_cluster_info() -> Dict:
    """Get detailed cluster information for multi-node setup."""
    cluster_resources = ray.cluster_resources()
    node_resources = ray.nodes()
    
    gpu_nodes = []
    total_gpus = total_cpus = 0
    
    for node in node_resources:
        if node['Alive']:
            node_gpus = node['Resources'].get('GPU', 0)
            node_cpus = node['Resources'].get('CPU', 0)
            
            if node_gpus > 0:
                gpu_nodes.append({
                    'node_id': node['NodeID'],
                    'gpus': int(node_gpus),
                    'cpus': int(node_cpus),
                    'node_ip': node.get('NodeManagerAddress', 'unknown')
                })
            
            total_gpus += node_gpus
            total_cpus += node_cpus
    
    return {
        'total_nodes': len([n for n in node_resources if n['Alive']]),
        'gpu_nodes': len(gpu_nodes),
        'total_gpus': int(total_gpus),
        'total_cpus': int(total_cpus),
        'gpu_nodes_info': gpu_nodes,
        'cluster_resources': cluster_resources
    }

def calculate_optimal_resources(cluster_info: Dict, args) -> Dict:
    """Calculate optimal resource allocation for multi-node multi-GPU setup."""
    total_gpus = cluster_info['total_gpus']
    gpu_nodes = cluster_info['gpu_nodes']
    
    if args.cpu_only or total_gpus == 0:
        return {
            'compute_resources': {'num_cpus': min(args.num_cpus or 4, cluster_info['total_cpus'])},
            'concurrency': min(4, cluster_info['total_cpus'] // 2),
            'gpus_per_worker': 0,
            'total_workers': min(4, cluster_info['total_cpus'] // 2)
        }
    
    gpus_per_worker = min(args.gpus_per_worker, total_gpus)
    max_workers = total_gpus // gpus_per_worker
    
    if args.max_concurrent_workers:
        concurrency = min(args.max_concurrent_workers, max_workers)
    else:
        if args.enable_multi_node and gpu_nodes > 1:
            concurrency = min(max_workers, gpu_nodes * 2)
        else:
            concurrency = max_workers
    
    return {
        'compute_resources': {'num_gpus': gpus_per_worker},
        'concurrency': concurrency,
        'gpus_per_worker': gpus_per_worker,
        'total_workers': concurrency
    }

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Ray Data Synthetic Data Generation for Mathematical Word Problems",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test mode - generate few samples quickly
  python ray_data_sdg_job.py --test-mode
  
  # Production mode - generate full dataset
  python ray_data_sdg_job.py
  
  # Custom configuration
  python ray_data_sdg_job.py --seeds 100 --variations 2 --batch-size 16
        """
    )
    
    parser.add_argument("--test-mode", action="store_true", help="Enable test mode with minimal samples")
    parser.add_argument("--seeds", type=int, help="Number of seed samples to use")
    parser.add_argument("--variations", type=int, help="Number of variations per seed")
    parser.add_argument("--batch-size", type=int, help="Batch size for Ray Data processing")
    parser.add_argument("--quality-threshold", type=float, help="Quality threshold for filtering (0.0-1.0)")
    parser.add_argument("--output-path", type=str, default="/tmp/synthetic_data", help="Output path for generated dataset")
    parser.add_argument("--num-cpus", type=int, help="Number of CPUs to use")
    parser.add_argument("--save-every", type=int, default=5, help="Save checkpoint every N processed seeds")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint if available")
    parser.add_argument("--checkpoint-dir", type=str, help="Directory for checkpoint files")
    parser.add_argument("--cpu-only", action="store_true", help="Force CPU-only execution")
    parser.add_argument("--gpus-per-worker", type=int, default=1, help="Number of GPUs per worker")
    parser.add_argument("--max-concurrent-workers", type=int, help="Maximum number of concurrent workers")
    parser.add_argument("--enable-multi-node", action="store_true", help="Enable multi-node distributed processing")
    
    return parser.parse_args()

def create_seed_dataset(num_seeds: int, cache_dir: str) -> ray.data.Dataset:
    """Create Ray Dataset from GSM8K seed samples."""
    print("Loading GSM8K dataset...")
    gsm8k_dataset = load_dataset("gsm8k", "main", cache_dir=f"{cache_dir}/datasets")
    print(f"Dataset loaded: {len(gsm8k_dataset['train'])} train samples")
    
    train_data = gsm8k_dataset["train"]
    seed_samples = []
    
    for i in range(min(num_seeds, len(train_data))):
        sample = train_data[i]
        seed_samples.append({
            "seed_samples": {"question": sample["question"], "answer": sample["answer"]},
            "seed_ids": i
        })
    
    print(f"Created {len(seed_samples)} seed samples")
    return ray.data.from_items(seed_samples)

def create_seed_dataset_filtered(seed_ids: List[int], cache_dir: str) -> ray.data.Dataset:
    """Create Ray Dataset from GSM8K seed samples for specific seed IDs."""
    print(f"Loading GSM8K dataset for {len(seed_ids)} specific seeds...")
    gsm8k_dataset = load_dataset("gsm8k", "main", cache_dir=f"{cache_dir}/datasets")
    
    train_data = gsm8k_dataset["train"]
    seed_samples = []
    
    for seed_id in seed_ids:
        if seed_id < len(train_data):
            sample = train_data[seed_id]
            seed_samples.append({
                "seed_samples": {"question": sample["question"], "answer": sample["answer"]},
                "seed_ids": seed_id
            })
    
    print(f"Created {len(seed_samples)} filtered seed samples")
    return ray.data.from_items(seed_samples)

def quality_filter(batch: Dict[str, List]) -> Dict[str, List]:
    """Enhanced quality filter with detailed logging."""
    print(f"[Filter] Input batch has {len(batch.get('quality_scores', []))} items")
    
    filtered_indices = []
    quality_stats = {"total": 0, "high_quality": 0, "medium_quality": 0, "low_quality": 0}
    
    for i, quality_score in enumerate(batch["quality_scores"]):
        quality_stats["total"] += 1
        
        if quality_score >= 0.7:
            filtered_indices.append(i)
            quality_stats["high_quality"] += 1
            print(f"[Filter] Item {i}: ACCEPTED (quality_score = {quality_score:.3f}) - High Quality")
        elif quality_score >= 0.5:
            filtered_indices.append(i)
            quality_stats["medium_quality"] += 1
            print(f"[Filter] Item {i}: ACCEPTED (quality_score = {quality_score:.3f}) - Medium Quality")
        else:
            quality_stats["low_quality"] += 1
            print(f"[Filter] Item {i}: REJECTED (quality_score = {quality_score:.3f}) - Low Quality")
    
    print(f"[Filter] Quality Statistics:")
    print(f"  - Total items: {quality_stats['total']}")
    print(f"  - High quality (≥0.7): {quality_stats['high_quality']}")
    print(f"  - Medium quality (≥0.5): {quality_stats['medium_quality']}")
    print(f"  - Low quality (<0.5): {quality_stats['low_quality']}")
    print(f"  - Acceptance rate: {len(filtered_indices)/quality_stats['total']*100:.1f}%")
    
    # Filter all fields based on quality
    return {key: [values[i] for i in filtered_indices] for key, values in batch.items()}

def format_for_output(batch: Dict[str, List]) -> Dict[str, List]:
    """Format batch for final output."""
    formatted_items = []
    
    for i in range(len(batch["questions"])):
        item = {
            "question": str(batch["questions"][i]),
            "answer": str(batch["answers"][i]),
            "source": str(batch["sources"][i]),
            "difficulty": str(batch["difficulties"][i]),
            "concepts": batch["concepts"][i].split(",") if isinstance(batch["concepts"][i], str) else ["arithmetic", "word_problems"],
            "overall_quality": float(batch["quality_scores"][i]),
            "model_confidence": float(batch["model_confidences"][i]),
            "seed_id": int(batch["seed_ids"][i]),
            "variation_id": int(batch["variation_ids"][i])
        }
        formatted_items.append(item)
    
    return {"items": formatted_items}

def setup_signal_handlers(checkpoint_manager: CheckpointManager, total_expected: int):
    """Setup signal handlers for graceful shutdown."""
    def signal_handler(signum, frame):
        print(f"\nReceived signal {signum}. Saving checkpoint and shutting down gracefully...")
        checkpoint_manager.save_checkpoint(checkpoint_manager.processed_seeds, total_expected, force=True)
        print("Checkpoint saved. Exiting...")
        if ray.is_initialized():
            ray.shutdown()
        exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

def save_final_dataset(all_items: List[Dict], output_path: str, metadata: Dict):
    """Save final formatted dataset."""
    os.makedirs(output_path, exist_ok=True)
    
    # Split into train/test (80/20)
    random.shuffle(all_items)
    split_idx = int(len(all_items) * 0.8)
    
    final_dataset = {
        "train": all_items[:split_idx],
        "test": all_items[split_idx:],
        "metadata": metadata
    }
    
    final_dataset_path = os.path.join(output_path, "final_synthetic_dataset.json")
    with open(final_dataset_path, "w") as f:
        json.dump(final_dataset, f, indent=2, cls=NumpyEncoder)
    
    print(f"Final dataset saved: {len(final_dataset['train'])} train / {len(final_dataset['test'])} test")
    print(f"Saved to: {final_dataset_path}")

def main():
    """Main function using Ray Data pipeline with checkpointing."""
    args = parse_args()
    
    print("Starting Ray Data distributed synthetic data generation with checkpointing...")
    print(f"Mode: {'TEST' if args.test_mode else 'PRODUCTION'}")
    
    # Setup checkpoint directory
    checkpoint_dir = args.checkpoint_dir if args.checkpoint_dir else args.output_path
    checkpoint_manager = CheckpointManager(checkpoint_dir, args.save_every)
    
    # Load existing checkpoint if resuming
    checkpoint_data = {}
    if args.resume:
        print("Attempting to resume from checkpoint...")
        checkpoint_data = checkpoint_manager.load_checkpoint()
        if checkpoint_data:
            print(f"Resuming from checkpoint with {len(checkpoint_manager.processed_seeds)} seeds already processed")
        else:
            print("No checkpoint found, starting fresh")
    else:
        print("Starting fresh (not resuming from checkpoint)")
    
    # Initialize Ray
    if not ray.is_initialized():
        print("Initializing Ray...")
        is_ray_job = os.environ.get('RAY_JOB_ID') is not None or os.environ.get('RAY_ADDRESS') is not None
        
        if is_ray_job:
            ray.init()
            print("Ray initialized in cluster mode (running as Ray job)")
        else:
            ray.init(num_cpus=min(8, os.cpu_count()))
            print("Ray initialized in standalone mode")
    else:
        print("Ray already initialized")
    
    # Get cluster information
    cluster_info = get_cluster_info()
    print(f"Connected to Ray cluster:")
    print(f"  - Total nodes: {cluster_info['total_nodes']}")
    print(f"  - GPU nodes: {cluster_info['gpu_nodes']}")
    print(f"  - Total GPUs: {cluster_info['total_gpus']}")
    print(f"  - Total CPUs: {cluster_info['total_cpus']}")
    
    if args.enable_multi_node and cluster_info['gpu_nodes'] > 1:
        print(f"  - Multi-node mode enabled with {cluster_info['gpu_nodes']} GPU nodes")
        for i, node_info in enumerate(cluster_info['gpu_nodes_info']):
            print(f"    Node {i+1}: {node_info['gpus']} GPUs, {node_info['cpus']} CPUs @ {node_info['node_ip']}")
    
    # Configure based on mode
    if args.test_mode:
        default_seeds, default_variations, default_batch_size = 2, 1, 1
        default_quality_threshold, default_num_cpus = 0.5, 2
        print("TEST MODE: Using minimal samples for quick testing")
    else:
        default_seeds, default_variations = 50, 3
        default_batch_size = max(4, cluster_info['total_gpus'])
        default_quality_threshold, default_num_cpus = 0.7, 4
        print("PRODUCTION MODE: Generating full-scale dataset")
    
    # Apply user overrides
    num_seeds = args.seeds if args.seeds is not None else default_seeds
    variations_per_seed = args.variations if args.variations is not None else default_variations
    batch_size = args.batch_size if args.batch_size is not None else default_batch_size
    quality_threshold = args.quality_threshold if args.quality_threshold is not None else default_quality_threshold
    num_cpus = args.num_cpus if args.num_cpus is not None else default_num_cpus
    
    # Calculate optimal resource allocation
    resource_config = calculate_optimal_resources(cluster_info, args)
    total_expected_seeds = num_seeds
    
    print(f"Configuration:")
    print(f"  - Seeds: {num_seeds}")
    print(f"  - Variations per seed: {variations_per_seed}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Quality threshold: {quality_threshold}")
    print(f"  - Save every: {args.save_every} seeds")
    print(f"  - Checkpoint dir: {checkpoint_dir}")
    print(f"  - Expected total: {num_seeds} x {variations_per_seed} = {num_seeds * variations_per_seed} problems")
    print(f"Resource Allocation:")
    print(f"  - GPUs per worker: {resource_config['gpus_per_worker']}")
    print(f"  - Concurrent workers: {resource_config['concurrency']}")
    print(f"  - Total workers: {resource_config['total_workers']}")
    print(f"  - Compute resources: {resource_config['compute_resources']}")
    
    # Setup signal handlers
    setup_signal_handlers(checkpoint_manager, total_expected_seeds)
    
    # Get cache directory
    cache_dir = "/shared/cache" if os.path.exists("/shared/cache") else os.path.expanduser("~/.cache")
    
    try:
        # Create seed dataset
        all_seed_ids = list(range(num_seeds))
        
        if args.resume and checkpoint_manager.processed_seeds:
            remaining_seed_ids = checkpoint_manager.get_remaining_seeds(all_seed_ids)
            print(f"Resuming: {len(remaining_seed_ids)} seeds remaining out of {num_seeds} total")
            if not remaining_seed_ids:
                print("All seeds already processed! Nothing to do.")
                return
        else:
            remaining_seed_ids = all_seed_ids
        
        seed_ds = create_seed_dataset_filtered(remaining_seed_ids, cache_dir)
        
        print(f"\nStarting Ray Data pipeline with checkpointing...")
        
        compute_resources = resource_config['compute_resources']
        concurrency = resource_config['concurrency']
        
        print(f"Pipeline Configuration:")
        print(f"  - Compute resources per worker: {compute_resources}")
        print(f"  - Concurrency (parallel workers): {concurrency}")
        print(f"  - Batch size: {batch_size}")
        
        # Enhanced Ray Data Pipeline
        results_ds = (seed_ds
            .map_batches(
                ModelInferenceCallable(MODEL_NAME, variations_per_seed, checkpoint_dir, checkpoint_manager.processed_seeds),
                batch_size=batch_size,
                concurrency=concurrency,
                **compute_resources,
                max_retries=3,
                retry_exceptions=True,
            )
            .filter(lambda batch: len(batch["questions"]) > 0)
            .map_batches(quality_filter, batch_size=batch_size * 2)
            .map_batches(format_for_output, batch_size=batch_size * 2)
        )
        
        # Execute pipeline
        print("Executing Ray Data pipeline with incremental saving...")
        
        total_generated = high_quality_count = processed_batches = 0
        quality_scores = []
        batches_without_progress = last_total_generated = 0
        max_batches_without_progress = 10
        
        pipeline_start_time = start_time = time.time()
        max_pipeline_time = 3600  # 1 hour timeout
        
        try:
            for batch in results_ds.iter_batches(batch_size=None):
                if time.time() - pipeline_start_time > max_pipeline_time:
                    print(f"Pipeline timeout after {max_pipeline_time} seconds, stopping...")
                    break
                
                items = batch["items"]
                total_generated += len(items)
                processed_batches += 1
                
                batch_seed_ids = set()
                for item in items:
                    if item["overall_quality"] >= quality_threshold:
                        high_quality_count += 1
                    quality_scores.append(item["overall_quality"])
                    batch_seed_ids.add(item["seed_id"])
                
                if batch_seed_ids:
                    checkpoint_manager.save_checkpoint(batch_seed_ids, total_expected_seeds)
                
                print(f"Processed batch {processed_batches}: {len(items)} items, {len(batch_seed_ids)} seeds")
                
                # Check for progress stall
                if total_generated == last_total_generated:
                    batches_without_progress += 1
                    if batches_without_progress >= max_batches_without_progress:
                        print(f"No progress for {max_batches_without_progress} batches, stopping pipeline...")
                        break
                else:
                    batches_without_progress = 0
                    last_total_generated = total_generated
                    
        except KeyboardInterrupt:
            print("Pipeline interrupted by user")
        except Exception as e:
            print(f"Pipeline error: {e}")
            print("Saving progress and continuing...")
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Create metadata
        metadata = {
            "total_generated": total_generated,
            "high_quality_count": high_quality_count,
            "quality_pass_rate": (high_quality_count / total_generated * 100) if total_generated > 0 else 0,
            "quality_threshold": quality_threshold,
            "avg_quality_score": sum(quality_scores) / len(quality_scores) if quality_scores else 0,
            "processing_time_seconds": processing_time,
            "model_used": MODEL_NAME,
            "generation_method": "ray_data_distributed",
            "ray_data_features": [
                "map_batches_inference",
                "automatic_scaling",
                "fault_tolerance",
                "streaming_processing",
                "quality_filtering"
            ]
        }
        
        print("\n" + "="*60)
        print("RAY DATA SDG PIPELINE SUMMARY")
        print("="*60)
        print(f"Total problems generated: {total_generated}")
        print(f"High quality problems: {high_quality_count}")
        print(f"Quality pass rate: {metadata['quality_pass_rate']:.1f}%")
        print(f"Average quality score: {metadata['avg_quality_score']:.3f}")
        print(f"Processing time: {processing_time:.1f} seconds")
        print(f"Throughput: {total_generated/processing_time:.2f} problems/second")
        
        if cluster_info['total_gpus'] > 0:
            gpu_efficiency = total_generated / (cluster_info['total_gpus'] * processing_time)
            print(f"GPU efficiency: {gpu_efficiency:.2f} problems/GPU/second")
            print(f"GPU utilization: {resource_config['total_workers']}/{cluster_info['total_gpus']} GPUs used")
        
        print("="*60)
        
        # Final checkpoint save
        checkpoint_manager.save_checkpoint(checkpoint_manager.processed_seeds, total_expected_seeds, force=True)
        
        # Save final results
        if checkpoint_manager.current_data:
            save_final_dataset(checkpoint_manager.current_data, args.output_path, metadata)
        
        print(f"\nPipeline completed successfully!")
        print(f"Results saved to: {args.output_path}")
        print(f"Checkpoints saved to: {checkpoint_dir}")
        
    except KeyboardInterrupt:
        print(f"\nPipeline interrupted by user")
        checkpoint_manager.save_checkpoint(checkpoint_manager.processed_seeds, total_expected_seeds, force=True)
        print("Progress saved to checkpoint")
    except Exception as e:
        print(f"Error in Ray Data pipeline: {e}")
        checkpoint_manager.save_checkpoint(checkpoint_manager.processed_seeds, total_expected_seeds, force=True)
        print("Progress saved to checkpoint before exit")
        raise
    finally:
        if ray.is_initialized():
            ray.shutdown()

if __name__ == "__main__":
    main()
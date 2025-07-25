#!/usr/bin/env python3
"""
RAG Evaluator for RAGAS evaluation
"""

import sys
import os
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Tuple
import json
from datetime import datetime
import logging
import matplotlib.pyplot as plt
import seaborn as sns

# Transformers imports
from transformers import (
    RagSequenceForGeneration,
    RagConfig,
    RagTokenizer,
    DPRQuestionEncoder,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DPRQuestionEncoderTokenizer,
    GenerationConfig,
    AutoConfig,
)

# RAGAS imports
from ragas import evaluate
from ragas.metrics import (
    context_recall,
    context_precision
)

from datasets import Dataset

# Feast imports
from feast import FeastRAGRetriever, FeastIndex
from feature_repo.ragproject_repo import wiki_passage_feature_view

from .config import ModelConfig

class RAGEvaluator:
    """RAG model evaluator using RAGAS metrics"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.device = torch.device(config.device)
        logging.info(f"Using device: {self.device}")
        
        # Initialize tokenizers
        self._init_tokenizers()
        
        # Initialize Feast components
        self._init_feast_components()
        
        # Store models for reuse
        self.models = {}
        
    def _init_tokenizers(self):
        """Initialize tokenizers"""
        logging.info("Initializing tokenizers...")
        
        # Question encoder tokenizer
        self.qe_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(
            self.config.original_qe_model
        )
        
        # Generator tokenizer
        self.gen_tokenizer = AutoTokenizer.from_pretrained(
            self.config.original_gen_model
        )
        
        # Add padding token if not present
        if self.gen_tokenizer.pad_token is None:
            self.gen_tokenizer.pad_token = self.gen_tokenizer.eos_token
            
        logging.info("Tokenizers initialized successfully")
        
    def _init_feast_components(self):
        """Initialize Feast RAG retriever components"""
        logging.info("Initializing Feast components...")
        
        try:
            # Initialize Feast index
            self.feast_index = FeastIndex(
                feature_view=wiki_passage_feature_view,
                repo_path=self.config.feast_repo_path
            )
            
            # Initialize Feast RAG retriever
            self.feast_retriever = FeastRAGRetriever(
                index=self.feast_index,
                question_encoder_tokenizer=self.qe_tokenizer,
                question_encoder=self.load_question_encoder("original")
            )
            
            logging.info("Feast components initialized successfully")
            
        except Exception as e:
            logging.error(f"Error initializing Feast components: {e}")
            self.feast_retriever = None
            self.feast_index = None
    
    def load_question_encoder(self, encoder_type: str) -> DPRQuestionEncoder:
        """Load question encoder based on type"""
        if encoder_type == "original":
            model_path = self.config.original_qe_model
        elif encoder_type == "finetuned":
            model_path = os.path.join(self.config.finetuned_checkpoint_dir, "question_encoder")
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")
            
        logging.info(f"Loading question encoder from: {model_path}")
        return DPRQuestionEncoder.from_pretrained(model_path).to(self.device)
    
    def load_generator(self, generator_type: str) -> AutoModelForSeq2SeqLM:
        """Load generator based on type"""
        if generator_type == "original":
            model_path = self.config.original_gen_model
        elif generator_type == "finetuned":
            model_path = os.path.join(self.config.finetuned_checkpoint_dir, "generator")
        else:
            raise ValueError(f"Unknown generator type: {generator_type}")
            
        logging.info(f"Loading generator from: {model_path}")
        return AutoModelForSeq2SeqLM.from_pretrained(model_path).to(self.device)
    
    def setup_rag_model(self, qe_type: str, gen_type: str) -> RagSequenceForGeneration:
        """Setup RAG model with specified components"""
        model_key = f"{qe_type}_{gen_type}"
        
        if model_key in self.models:
            return self.models[model_key]
        
        logging.info(f"Setting up RAG model: QE={qe_type}, Generator={gen_type}")
        
        # Load components
        question_encoder = self.load_question_encoder(qe_type)
        generator = self.load_generator(gen_type)
        
        # Create RAG config
        rag_config = RagConfig.from_pretrained("facebook/rag-sequence-base")
        rag_config.question_encoder = question_encoder.config
        rag_config.generator = generator.config
        
        # Create RAG model
        rag_model = RagSequenceForGeneration(
            config=rag_config,
            question_encoder=question_encoder,
            generator=generator,
            retriever=self.feast_retriever
        )
        
        # Set generation config
        rag_model.generation_config = GenerationConfig(
            max_new_tokens=self.config.max_new_tokens,
            num_beams=self.config.num_beams,
            do_sample=self.config.do_sample,
            pad_token_id=self.gen_tokenizer.pad_token_id,
            eos_token_id=self.gen_tokenizer.eos_token_id
        )
        
        self.models[model_key] = rag_model
        return rag_model
    
    def test_feast_connection(self) -> Dict[str, Any]:
        """Test Feast connection and retrieve sample passages"""
        if not self.feast_retriever:
            return {"status": "error", "error": "Feast retriever not initialized"}
        
        try:
            test_question = "What is machine learning?"
            
            # Test retrieval
            retrieved_docs = self.feast_retriever.get_relevant_documents(test_question)
            
            # Test generation if possible
            try:
                answer = self.feast_retriever.generate_answer(test_question)
                generation_works = True
            except Exception as gen_error:
                logging.warning(f"Generation test failed: {gen_error}")
                generation_works = False
                answer = "Generation test failed"
            
            return {
                "status": "success",
                "sample_count": len(retrieved_docs) if retrieved_docs else 0,
                "sample_passages": [doc.page_content[:100] + "..." for doc in retrieved_docs[:3]] if retrieved_docs else [],
                "generation_works": generation_works,
                "sample_answer": answer[:200] + "..." if len(answer) > 200 else answer
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def generate_response(self, model: RagSequenceForGeneration, question: str, use_retrieval: bool = True) -> Dict[str, Any]:
        """Generate response using RAG model"""
        try:
            if use_retrieval and self.feast_retriever:
                # Use Feast RAG retriever's generate_answer method
                try:
                    # Use the Feast RAG retriever's generate_answer method
                    answer = self.feast_retriever.generate_answer(question)
                    
                    # Get retrieved context
                    retrieved_docs = self.feast_retriever.get_relevant_documents(question)
                    context = [doc.page_content for doc in retrieved_docs]
                    
                except Exception as feast_error:
                    logging.warning(f"Feast retriever failed, falling back to manual generation: {feast_error}")
                    # Fallback to manual generation
                    inputs = self.gen_tokenizer(question, return_tensors="pt").to(self.device)
                    
                    with torch.no_grad():
                        outputs = model.generate(**inputs)
                    
                    answer = self.gen_tokenizer.decode(outputs[0], skip_special_tokens=True)
                    
                    # Get retrieved context
                    retrieved_docs = self.feast_retriever.get_relevant_documents(question)
                    context = [doc.page_content for doc in retrieved_docs]
                
            else:
                # Use generator directly without retrieval
                inputs = self.gen_tokenizer(question, return_tensors="pt").to(self.device)
                
                with torch.no_grad():
                    outputs = model.generator.generate(**inputs)
                
                answer = self.gen_tokenizer.decode(outputs[0], skip_special_tokens=True)
                context = []
            
            return {
                "answer": answer,
                "context": context,
                "question": question,
                "use_retrieval": use_retrieval
            }
            
        except Exception as e:
            logging.error(f"Error generating response: {e}")
            return {
                "answer": f"Error: {str(e)}",
                "context": [],
                "question": question,
                "use_retrieval": use_retrieval,
                "error": True
            }
    
    def run_ragas_evaluation(self, eval_dataset: Dataset, use_retrieval: bool = True) -> Dict[str, float]:
        """Run RAGAS evaluation on dataset"""
        try:
            logging.info("Running RAGAS evaluation...")
            
            # Run RAGAS evaluation
            results = evaluate(
                eval_dataset,
                metrics=[context_recall, context_precision]
            )
            
            return {
                "context_recall": results["context_recall"],
                "context_precision": results["context_precision"]
            }
            
        except Exception as e:
            logging.error(f"RAGAS evaluation failed: {e}")
            return {"context_recall": 0.0, "context_precision": 0.0}
    
    def create_evaluation_dataset(self, questions: List[str], responses: List[Dict], 
                                ground_truth: List[str] = None) -> Dataset:
        """Create evaluation dataset for RAGAS"""
        dataset_dict = {
            "question": questions,
            "answer": [r["answer"] for r in responses],
            "contexts": [r["context"] for r in responses]
        }
        
        if ground_truth:
            dataset_dict["ground_truth"] = ground_truth
            
        return Dataset.from_dict(dataset_dict)
    
    def evaluate_combination(self, qe_type: str, gen_type: str, 
                           test_questions: List[str], ground_truth: List[str] = None,
                           use_retrieval: bool = True) -> Tuple[Dict[str, float], List[Dict]]:
        """Evaluate a specific model combination"""
        logging.info(f"Evaluating combination: QE={qe_type}, Generator={gen_type}, Retrieval={use_retrieval}")
        
        # Setup model
        model = self.setup_rag_model(qe_type, gen_type)
        
        # Generate responses
        responses = []
        for question in test_questions:
            response = self.generate_response(model, question, use_retrieval)
            responses.append(response)
        
        # Create evaluation dataset
        eval_dataset = self.create_evaluation_dataset(test_questions, responses, ground_truth)
        
        # Run RAGAS evaluation
        ragas_metrics = self.run_ragas_evaluation(eval_dataset, use_retrieval)
        
        # Calculate fallback metrics
        fallback_metrics = self._calculate_fallback_metrics(responses, ground_truth, use_retrieval)
        
        # Combine metrics
        combined_metrics = {**ragas_metrics, **fallback_metrics}
        
        return combined_metrics, responses
    
    def _calculate_fallback_metrics(self, responses: List[Dict], 
                                  ground_truth: List[str] = None,
                                  use_retrieval: bool = True) -> Dict[str, float]:
        """Calculate fallback metrics when RAGAS fails"""
        def is_error_response(answer):
            return answer.startswith("Error:") or "error" in answer.lower()
        
        def calculate_quality_score(answer, use_retrieval):
            if is_error_response(answer):
                return 0.0
            
            # Simple quality heuristics
            if len(answer.strip()) < 10:
                return 0.1
            
            if use_retrieval and "I don't know" in answer.lower():
                return 0.3
            
            return 0.8  # Default quality score
        
        # Calculate metrics
        total_responses = len(responses)
        error_count = sum(1 for r in responses if is_error_response(r["answer"]))
        error_rate = error_count / total_responses if total_responses > 0 else 0.0
        
        quality_scores = [calculate_quality_score(r["answer"], use_retrieval) for r in responses]
        avg_quality = np.mean(quality_scores) if quality_scores else 0.0
        
        context_lengths = [len(r["context"]) for r in responses]
        avg_context_length = np.mean(context_lengths) if context_lengths else 0.0
        
        return {
            "error_rate": error_rate,
            "avg_quality": avg_quality,
            "avg_context_length": avg_context_length,
            "total_responses": total_responses
        }
    
    def run_comprehensive_evaluation(self, test_questions: List[str], 
                                   ground_truth: List[str] = None) -> Dict[str, Dict[str, float]]:
        """Run comprehensive evaluation across different model combinations"""
        print("🎯 Starting comprehensive RAG evaluation...")
        print("=" * 80)
        print("🔄 Will evaluate 4 combinations to test RAG functionality:")
        print("   1. ORIGINAL QE + ORIGINAL Generator (baseline with retrieval)")
        print("   2. FINE-TUNED Generator only (no retrieval - test memorization)")
        print("   3. FINE-TUNED QE + ORIGINAL Generator (test retrieval quality)")
        print("   4. FINE-TUNED QE + FINE-TUNED Generator (full pipeline)")
        print("")
        
        results = {}
        all_responses = {}  # Store all responses for answer comparison
        
        # Combination 1: Original QE + Original Generator (baseline with retrieval)
        print("🔄 [1/4] Starting evaluation...")
        print("=" * 60)
        print("🔍 EVALUATING: ORIGINAL Question Encoder + ORIGINAL Generator")
        print("=" * 60)
        try:
            result, responses = self.evaluate_combination("original", "original", test_questions, ground_truth)
            results["original_qe_original_gen"] = result
            all_responses["original_qe_original_gen"] = responses
            print("✅ [1/4] Completed successfully")
        except Exception as e:
            print(f"❌ [1/4] Failed: {e}")
            results["original_qe_original_gen"] = {"error": str(e), "skipped": True}
            all_responses["original_qe_original_gen"] = []
        
        # Combination 2: Fine-tuned Generator only (no retrieval - test memorization)
        print("🔄 [2/4] Starting evaluation...")
        print("=" * 60)
        print("🔍 EVALUATING: FINE-TUNED Generator only (no retrieval)")
        print("=" * 60)
        try:
            result, responses = self.evaluate_combination("original", "finetuned", test_questions, ground_truth, use_retrieval=False)
            results["finetuned_gen_only"] = result
            all_responses["finetuned_gen_only"] = responses
            print("✅ [2/4] Completed successfully")
        except Exception as e:
            print(f"❌ [2/4] Failed: {e}")
            results["finetuned_gen_only"] = {"error": str(e), "skipped": True}
            all_responses["finetuned_gen_only"] = []
        
        # Combination 3: Fine-tuned QE + Original Generator (test retrieval quality)
        print("🔄 [3/4] Starting evaluation...")
        print("=" * 60)
        print("🔍 EVALUATING: FINE-TUNED QE + ORIGINAL Generator")
        print("=" * 60)
        try:
            result, responses = self.evaluate_combination("finetuned", "original", test_questions, ground_truth)
            results["finetuned_qe_original_gen"] = result
            all_responses["finetuned_qe_original_gen"] = responses
            print("✅ [3/4] Completed successfully")
        except Exception as e:
            print(f"❌ [3/4] Failed: {e}")
            results["finetuned_qe_original_gen"] = {"error": str(e), "skipped": True}
            all_responses["finetuned_qe_original_gen"] = []
        
        # Combination 4: Fine-tuned QE + Fine-tuned Generator (full pipeline)
        print("🔄 [4/4] Starting evaluation...")
        print("=" * 60)
        print("🔍 EVALUATING: FINE-TUNED QE + FINE-TUNED Generator")
        print("=" * 60)
        try:
            result, responses = self.evaluate_combination("finetuned", "finetuned", test_questions, ground_truth)
            results["finetuned_qe_finetuned_gen"] = result
            all_responses["finetuned_qe_finetuned_gen"] = responses
            print("✅ [4/4] Completed successfully")
        except Exception as e:
            print(f"❌ [4/4] Failed: {e}")
            results["finetuned_qe_finetuned_gen"] = {"error": str(e), "skipped": True}
            all_responses["finetuned_qe_finetuned_gen"] = []
        
        print("")
        print("🎉 All evaluations completed! Processed 4 combinations")
        print("")
        
        # Store responses in results for later use
        results["_all_responses"] = all_responses
        results["_test_questions"] = test_questions
        
        return results
    
    def save_results(self, results: Dict[str, Dict[str, float]], 
                    output_path: str = "ragas_evaluation_results.json"):
        """Save evaluation results to file"""
        output_data = {
            "timestamp": datetime.now().isoformat(),
            "config": {
                "finetuned_checkpoint_dir": self.config.finetuned_checkpoint_dir,
                "feast_repo_path": self.config.feast_repo_path,
                "device": str(self.device),
                "max_new_tokens": self.config.max_new_tokens
            },
            "results": results
        }
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"✓ Results saved to {output_path}")
    
    def print_comparison_table(self, results: Dict[str, Dict[str, float]]):
        """Print results in a comparison table format with RAGAS and fallback metrics"""
        print("\n" + "="*120)
        print("COMPREHENSIVE RAG EVALUATION RESULTS")
        print("="*120)
        
        # Scenario descriptions
        scenario_descriptions = {
            "original_qe_original_gen": "Original QE + Original Gen (Baseline with retrieval)",
            "finetuned_gen_only": "Fine-tuned Gen Only (No retrieval - test memorization)",
            "finetuned_qe_original_gen": "Fine-tuned QE + Original Gen (Test retrieval quality)",
            "finetuned_qe_finetuned_gen": "Fine-tuned QE + Fine-tuned Gen (Full pipeline)"
        }
        
        # Get all metric names, separating RAGAS and fallback metrics
        ragas_metrics = set()
        fallback_metrics = set()
        scenario_metrics = set()
        metadata_fields = {'qe_type', 'gen_type', 'total_questions', 'successful_responses', 'use_retrieval', 'scenario'}
        
        for result in results.values():
            if isinstance(result, dict) and "error" not in result:
                for metric in result.keys():
                    if metric.startswith('ragas_'):
                        ragas_metrics.add(metric)
                    elif metric in ['scenario', 'retrieval_success_rate', 'context_utilization_rate', 'memorization_indicator']:
                        scenario_metrics.add(metric)
                    elif metric not in metadata_fields:
                        fallback_metrics.add(metric)
        
        # Print scenario overview
        print("\n🎯 EVALUATION SCENARIOS:")
        print("-" * 80)
        for combo, desc in scenario_descriptions.items():
            if combo in results:
                result = results[combo]
                if not isinstance(result, dict):
                    status = "❌ INVALID"
                elif "error" in result:
                    status = "❌ FAILED"
                elif "skipped" in result:
                    status = "⚠️ SKIPPED"
                else:
                    status = "✅ COMPLETED"
                print(f"{combo:<30} | {status:<12} | {desc}")
        
        # Print RAGAS metrics section (only for retrieval scenarios)
        if ragas_metrics:
            print("\n🔍 RAGAS METRICS (Industry Standard RAG Evaluation)")
            print("-" * 80)
            
            # Print header
            header = f"{'Combination':<30}"
            for metric in sorted(ragas_metrics):
                metric_name = metric.replace('ragas_', '').replace('_', ' ').title()
                header += f"{metric_name:<15}"
            print(header)
            print("-" * (30 + 15 * len(ragas_metrics)))
            
            # Print data rows
            for combo, desc in scenario_descriptions.items():
                if combo in results:
                    result = results[combo]
                    if isinstance(result, dict) and "error" not in result and result.get('use_retrieval', True):
                        row = f"{combo:<30}"
                        for metric in sorted(ragas_metrics):
                            value = result.get(metric, 0.0)
                            # Convert value to string for formatting
                            if isinstance(value, (list, dict)):
                                value_str = str(value)[:14]  # Truncate long lists/dicts
                            else:
                                value_str = f"{value:.3f}" if isinstance(value, (int, float)) else str(value)
                            row += f"{value_str:<15}"
                        print(row)
        
        # Print fallback metrics section
        if fallback_metrics:
            print("\n📊 FALLBACK METRICS (When RAGAS Fails)")
            print("-" * 80)
            
            # Print header
            header = f"{'Combination':<30}"
            for metric in sorted(fallback_metrics):
                metric_name = metric.replace('_', ' ').title()
                header += f"{metric_name:<15}"
            print(header)
            print("-" * (30 + 15 * len(fallback_metrics)))
            
            # Print data rows
            for combo, desc in scenario_descriptions.items():
                if combo in results:
                    result = results[combo]
                    if isinstance(result, dict) and "error" not in result:
                        row = f"{combo:<30}"
                        for metric in sorted(fallback_metrics):
                            value = result.get(metric, 0.0)
                            value_str = f"{value:.3f}" if isinstance(value, (int, float)) else str(value)
                            row += f"{value_str:<15}"
                        print(row)
        
        print("\n" + "="*120)
    
    def generate_comparison_charts(self, results: Dict[str, Dict[str, float]], 
                                 output_dir: str = ".") -> List[str]:
        """Generate comprehensive comparison charts"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        charts_dir = os.path.join(output_dir, f"charts_{timestamp}")
        os.makedirs(charts_dir, exist_ok=True)
        
        charts = []
        
        # Generate different chart types
        charts.append(self._create_metrics_bar_chart(results, charts_dir))
        charts.append(self._create_radar_chart(results, charts_dir))
        charts.append(self._create_heatmap(results, charts_dir))
        
        logging.info(f"Generated {len(charts)} charts in {charts_dir}")
        return charts
    
    def _create_metrics_bar_chart(self, results: Dict[str, Dict[str, float]], 
                                charts_dir: str) -> str:
        """Create bar chart comparing metrics across combinations"""
        # Prepare data
        combinations = list(results.keys())
        metrics = ['context_recall', 'context_precision', 'avg_quality', 'error_rate']
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('RAG Model Performance Comparison', fontsize=16, fontweight='bold')
        
        for i, metric in enumerate(metrics):
            ax = axes[i//2, i%2]
            values = [results[combo].get(metric, 0) for combo in combinations]
            
            bars = ax.bar(combinations, values, alpha=0.7)
            ax.set_title(f'{metric.replace("_", " ").title()}')
            ax.set_ylabel('Score')
            ax.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        chart_path = os.path.join(charts_dir, 'metrics_comparison.png')
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return chart_path
    
    def _create_radar_chart(self, results: Dict[str, Dict[str, float]], 
                          charts_dir: str) -> str:
        """Create radar chart for comprehensive comparison"""
        # Select key metrics for radar chart
        metrics = ['context_recall', 'context_precision', 'avg_quality']
        combinations = list(results.keys())
        
        # Prepare data
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        for combo in combinations:
            values = [results[combo].get(metric, 0) for metric in metrics]
            values += values[:1]  # Complete the circle
            
            ax.plot(angles, values, 'o-', linewidth=2, label=combo.replace('_', ' '))
            ax.fill(angles, values, alpha=0.25)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics])
        ax.set_ylim(0, 1)
        ax.set_title('RAG Model Performance Radar Chart', pad=20, fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.grid(True)
        
        chart_path = os.path.join(charts_dir, 'radar_chart.png')
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return chart_path
    
    def _create_heatmap(self, results: Dict[str, Dict[str, float]], 
                       charts_dir: str) -> str:
        """Create heatmap of all metrics"""
        # Convert to DataFrame
        df = pd.DataFrame(results).T
        
        # Select numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df_numeric = df[numeric_cols]
        
        # Create heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(df_numeric, annot=True, cmap='RdYlBu_r', center=0.5, 
                   fmt='.3f', cbar_kws={'label': 'Score'})
        plt.title('RAG Model Metrics Heatmap', fontsize=14, fontweight='bold')
        plt.xlabel('Metrics')
        plt.ylabel('Model Combinations')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        
        chart_path = os.path.join(charts_dir, 'metrics_heatmap.png')
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return chart_path
    
    def save_answer_comparison_text(self, results: Dict[str, Dict[str, float]], 
                                  output_path: str = "answer_comparison.txt") -> str:
        """Save detailed answer comparison to text file"""
        try:
            with open(output_path, 'w') as f:
                f.write("RAG MODEL ANSWER COMPARISON\n")
                f.write("=" * 50 + "\n\n")
                
                # This would need access to the actual responses
                # For now, just save the metrics
                f.write("EVALUATION METRICS:\n")
                f.write("-" * 30 + "\n")
                
                for combo, metrics in results.items():
                    f.write(f"\n{combo}:\n")
                    for metric, value in metrics.items():
                        f.write(f"  {metric}: {value:.3f}\n")
                
            logging.info(f"Answer comparison saved to {output_path}")
            return output_path
            
        except Exception as e:
            logging.error(f"Error saving answer comparison: {e}")
            return "" 
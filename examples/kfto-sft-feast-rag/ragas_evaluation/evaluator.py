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
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from feast_rag_retriever import FeastRAGRetriever, FeastIndex
from feature_repo.ragproject_repo import wiki_passage_feature_view

from config import ModelConfig

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
        
        # Store models and retrievers for reuse
        self.models = {}
        self.retrievers = {}  # Store retrievers for each combination
        self.fallback_used = False
        
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
        
        # Load fine-tuned tokenizers if available (like in notebook)
        try:
            logging.info("Loading fine-tuned tokenizers...")
            rag_tokenizer = RagTokenizer.from_pretrained(self.config.finetuned_checkpoint_dir)
            self.finetuned_qe_tokenizer = rag_tokenizer.question_encoder
            self.finetuned_gen_tokenizer = rag_tokenizer.generator
            logging.info("Fine-tuned tokenizers loaded successfully")
        except Exception as e:
            logging.warning(f"Could not load fine-tuned tokenizers: {e}")
            self.finetuned_qe_tokenizer = None
            self.finetuned_gen_tokenizer = None
            
        logging.info("Tokenizers initialized successfully")
        
    def _init_feast_components(self):
        """Initialize Feast RAG retriever components"""
        logging.info("Initializing Feast components...")
        
        try:
            # Initialize Feast index (no parameters needed)
            self.feast_index = FeastIndex()
            
            # Defer Feast retriever initialization until needed
            self.feast_retriever = None
            self._feast_initialized = False
            
            logging.info("Feast index initialized successfully")
            
        except Exception as e:
            logging.error(f"Error initializing Feast components: {e}")
            self.feast_retriever = None
            self.feast_index = None
            self._feast_initialized = False
    
    def _create_feast_retriever(self, question_encoder, generator_model, use_finetuned_tokenizers=False):
        """Create a fresh Feast retriever for a specific combination"""
        try:
            logging.info("Creating fresh Feast RAG retriever...")
            
            # Choose tokenizers based on whether we're using fine-tuned models
            if use_finetuned_tokenizers and self.finetuned_qe_tokenizer and self.finetuned_gen_tokenizer:
                qe_tokenizer = self.finetuned_qe_tokenizer
                gen_tokenizer = self.finetuned_gen_tokenizer
                logging.info("Using fine-tuned tokenizers")
            else:
                qe_tokenizer = self.qe_tokenizer
                gen_tokenizer = self.gen_tokenizer
                logging.info("Using original tokenizers")
            
            # Create RAG config for the retriever
            if question_encoder and generator_model:
                rag_config = RagConfig(
                    question_encoder=question_encoder.config.to_dict(),
                    generator=generator_model.config.to_dict(),
                    index_name="custom",
                    index={"index_name": "feast_dummy_index", "custom_type": "FeastIndex"},
                    n_docs=1,
                )
            else:
                # Use config from fine-tuned model
                rag_config = RagConfig.from_pretrained(self.config.finetuned_checkpoint_dir)
            
            feast_retriever = FeastRAGRetriever(
                question_encoder_tokenizer=qe_tokenizer,
                generator_tokenizer=gen_tokenizer,
                question_encoder=question_encoder,
                generator_model=generator_model,
                feast_repo_path=self.config.feast_repo_path,
                feature_view=wiki_passage_feature_view,
                features=[
                    "wiki_passages:passage_text",
                    "wiki_passages:embedding", 
                    "wiki_passages:passage_id",
                ],
                search_type="vector",
                config=rag_config,
                index=self.feast_index
            )
            
            logging.info("Fresh Feast RAG retriever created successfully")
            return feast_retriever
            
        except Exception as e:
            logging.error(f"Error creating Feast retriever: {e}")
            return None
    
    def load_question_encoder(self, encoder_type: str) -> DPRQuestionEncoder:
        """Load question encoder based on type"""
        if encoder_type == "original":
            model_path = self.config.original_qe_model
            logging.info(f"Loading original question encoder from: {model_path}")
            return DPRQuestionEncoder.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype="bfloat16"
            ).to(self.device)
        elif encoder_type == "finetuned":
            model_path = os.path.join(self.config.finetuned_checkpoint_dir, "question_encoder")
            logging.info(f"Loading fine-tuned question encoder from: {model_path}")
            return DPRQuestionEncoder.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype="bfloat16"
            ).to(self.device)
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")
    
    def load_generator(self, generator_type: str) -> AutoModelForSeq2SeqLM:
        """Load generator based on type"""
        if generator_type == "original":
            model_path = self.config.original_gen_model
            logging.info(f"Loading original generator from: {model_path}")
            return AutoModelForSeq2SeqLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype="bfloat16"
            ).to(self.device)
        elif generator_type == "finetuned":
            model_path = os.path.join(self.config.finetuned_checkpoint_dir, "generator")
            logging.info(f"Loading fine-tuned generator from: {model_path}")
            return AutoModelForSeq2SeqLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype="bfloat16"
            ).to(self.device)
        else:
            raise ValueError(f"Unknown generator type: {generator_type}")
    
    def _init_feast_retriever(self, question_encoder, generator_model, use_finetuned_tokenizers=False):
        """Initialize Feast retriever with specific models"""
        if self._feast_initialized:
            return
            
        try:
            logging.info("Initializing Feast RAG retriever...")
            
            # Choose tokenizers based on whether we're using fine-tuned models
            if use_finetuned_tokenizers and self.finetuned_qe_tokenizer and self.finetuned_gen_tokenizer:
                qe_tokenizer = self.finetuned_qe_tokenizer
                gen_tokenizer = self.finetuned_gen_tokenizer
                logging.info("Using fine-tuned tokenizers")
            else:
                qe_tokenizer = self.qe_tokenizer
                gen_tokenizer = self.gen_tokenizer
                logging.info("Using original tokenizers")
            
            # Create RAG config for the retriever
            if question_encoder and generator_model:
                rag_config = RagConfig(
                    question_encoder=question_encoder.config.to_dict(),
                    generator=generator_model.config.to_dict(),
                    index_name="custom",
                    index={"index_name": "feast_dummy_index", "custom_type": "FeastIndex"},
                    n_docs=10,
                )
            else:
                # Use config from fine-tuned model
                rag_config = RagConfig.from_pretrained(self.config.finetuned_checkpoint_dir)
            
            self.feast_retriever = FeastRAGRetriever(
                question_encoder_tokenizer=qe_tokenizer,
                generator_tokenizer=gen_tokenizer,
                question_encoder=question_encoder,
                generator_model=generator_model,
                feast_repo_path=self.config.feast_repo_path,
                feature_view=wiki_passage_feature_view,
                features=[
                    "wiki_passages:passage_text",
                    "wiki_passages:embedding", 
                    "wiki_passages:passage_id",
                ],
                search_type="vector",
                config=rag_config,
                index=self.feast_index
            )
            self._feast_initialized = True
            logging.info("Feast RAG retriever initialized successfully")
            
        except Exception as e:
            logging.error(f"Error initializing Feast retriever: {e}")
            self.feast_retriever = None
            self._feast_initialized = False
    
    def setup_rag_model(self, qe_type: str, gen_type: str) -> RagSequenceForGeneration:
        """Setup RAG model with specified components - NO FALLBACKS"""
        model_key = f"{qe_type}_{gen_type}"
        
        if model_key in self.models:
            return self.models[model_key]
        
        logging.info(f"Setting up RAG model: QE={qe_type}, Generator={gen_type}")
        
        # Create a fresh Feast retriever for each combination to avoid sharing
        feast_retriever = None
        
        if qe_type == "original" and gen_type == "original":
            # Combination 1: Original QE + Original Generator
            logging.info("Loading ORIGINAL QE + ORIGINAL Generator")
            question_encoder = self.load_question_encoder("original")
            generator = self.load_generator("original")
            
            # Initialize fresh Feast retriever for this combination
            feast_retriever = self._create_feast_retriever(question_encoder, generator, use_finetuned_tokenizers=False)
            if not feast_retriever:
                raise RuntimeError("Failed to create Feast retriever for original models")
            
            # Create RAG config
            rag_config = RagConfig.from_pretrained("facebook/rag-sequence-base")
            rag_config.question_encoder = question_encoder.config
            rag_config.generator = generator.config
            
            # Create RAG model
            rag_model = RagSequenceForGeneration(
                config=rag_config,
                question_encoder=question_encoder,
                generator=generator,
                retriever=feast_retriever
            )
            
        elif qe_type == "original" and gen_type == "finetuned":
            # Combination 2: Original QE + Fine-tuned Generator
            logging.info("Loading ORIGINAL QE + FINE-TUNED Generator")
            
            # Load the entire fine-tuned RAG model first (like in notebook)
            rag_config = RagConfig.from_pretrained(self.config.finetuned_checkpoint_dir)
            feast_retriever = self._create_feast_retriever(None, None, use_finetuned_tokenizers=True)
            if not feast_retriever:
                raise RuntimeError("Failed to create Feast retriever for fine-tuned models")
            
            rag_model = RagSequenceForGeneration.from_pretrained(
                self.config.finetuned_checkpoint_dir,
                retriever=feast_retriever,
                config=rag_config
            )
            
            # Replace with original QE (like in notebook)
            original_qe = self.load_question_encoder("original")
            rag_model.rag.question_encoder = original_qe
            feast_retriever.question_encoder = original_qe
            feast_retriever.question_encoder_tokenizer = self.qe_tokenizer
            
            # Set the generator from the loaded RAG model
            feast_retriever.generator_model = rag_model.rag.generator
            
        elif qe_type == "finetuned" and gen_type == "finetuned":
            # Combination 3: Fine-tuned QE + Fine-tuned Generator
            logging.info("Loading FINE-TUNED QE + FINE-TUNED Generator")
            
            # Load the entire fine-tuned RAG model first (like in notebook)
            rag_config = RagConfig.from_pretrained(self.config.finetuned_checkpoint_dir)
            feast_retriever = self._create_feast_retriever(None, None, use_finetuned_tokenizers=True)
            if not feast_retriever:
                raise RuntimeError("Failed to create Feast retriever for fine-tuned models")
            
            rag_model = RagSequenceForGeneration.from_pretrained(
                self.config.finetuned_checkpoint_dir,
                retriever=feast_retriever,
                config=rag_config
            )
            
            # Set the models in the retriever from the loaded RAG model
            feast_retriever.question_encoder = rag_model.rag.question_encoder
            feast_retriever.generator_model = rag_model.rag.generator
            
        else:
            raise ValueError(f"Unsupported combination: QE={qe_type}, Generator={gen_type}. Only 'original' and 'finetuned' are supported.")
        
        # Set generation config
        rag_model.generation_config = GenerationConfig(
            max_new_tokens=self.config.max_new_tokens,
            num_beams=self.config.num_beams,
            do_sample=self.config.do_sample,
            pad_token_id=self.gen_tokenizer.pad_token_id,
            eos_token_id=self.gen_tokenizer.eos_token_id
        )
        
        # Move to device
        rag_model = rag_model.to(self.device)
        
        # Store both model and its retriever
        self.models[model_key] = rag_model
        self.retrievers[model_key] = feast_retriever
        
        return rag_model
    
    def test_feast_connection(self) -> Dict[str, Any]:
        """Test Feast connection and retrieve sample passages"""
        try:
            # Initialize a basic Feast retriever for testing
            test_question = "What is machine learning?"
            
            # Load basic models for testing
            test_qe = self.load_question_encoder("original")
            test_gen = self.load_generator("original")
            
            # Initialize Feast retriever for testing
            self._init_feast_retriever(test_qe, test_gen, use_finetuned_tokenizers=False)
            
            if not self.feast_retriever:
                return {"status": "error", "error": "Failed to initialize Feast retriever"}
            
            # Test retrieval using the retrieve method
            inputs = self.qe_tokenizer(test_question, return_tensors="pt").to(self.device)
            question_embeddings = self.feast_retriever.question_encoder(**inputs).pooler_output
            question_embeddings = question_embeddings.detach().cpu().to(torch.float32).numpy()
            
            _, _, doc_batch = self.feast_retriever.retrieve(question_embeddings, n_docs=3, query=test_question)
            retrieved_texts = doc_batch[0]["text"] if doc_batch else []
            
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
                "sample_count": len(retrieved_texts) if retrieved_texts else 0,
                "sample_passages": [text[:100] + "..." for text in retrieved_texts[:3]] if retrieved_texts else [],
                "generation_works": generation_works,
                "sample_answer": answer[:200] + "..." if len(answer) > 200 else answer
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def generate_response(self, model: RagSequenceForGeneration, question: str, use_retrieval: bool = True) -> Dict[str, Any]:
        """Generate response using RAG model"""
        # Find the correct retriever for this model
        model_key = None
        for key, stored_model in self.models.items():
            if stored_model is model:
                model_key = key
                break
        
        if not model_key or model_key not in self.retrievers:
            raise ValueError(f"Could not find retriever for model")
        
        retriever = self.retrievers[model_key]
        
        # Use Feast RAG retriever's generate_answer method
        answer = retriever.generate_answer(question)
        
        # Get retrieved context using the retrieve method
        inputs = self.qe_tokenizer(question, return_tensors="pt").to(self.device)
        question_embeddings = retriever.question_encoder(**inputs).pooler_output
        question_embeddings = question_embeddings.detach().cpu().to(torch.float32).numpy()
        
        _, _, doc_batch = retriever.retrieve(question_embeddings, n_docs=1, query=question)
        context = doc_batch[0]["text"] if doc_batch else []
        
        return {
            "answer": answer,
            "context": context,
            "question": question,
            "use_retrieval": use_retrieval
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
        # RAGAS expects contexts as a list of lists of strings
        contexts = []
        for r in responses:
            if r["context"]:
                # Keep as list of strings for RAGAS
                contexts.append(r["context"])
            else:
                contexts.append([])
        
        dataset_dict = {
            "question": questions,
            "answer": [r["answer"] for r in responses],
            "contexts": contexts
        }
        
        if ground_truth:
            dataset_dict["ground_truth"] = ground_truth
            
        return Dataset.from_dict(dataset_dict)
    
    def evaluate_combination(self, qe_type: str, gen_type: str, 
                           test_questions: List[str], ground_truth: List[str] = None,
                           use_retrieval: bool = True) -> Tuple[Dict[str, float], List[Dict]]:
        """Evaluate a specific model combination"""
        print(f"\n🔍 EVALUATING: QE={qe_type.upper()}, Generator={gen_type.upper()}, Retrieval={use_retrieval}")
        print("=" * 80)
        
        try:
            # Setup model
            print("📦 Setting up RAG model...")
            model = self.setup_rag_model(qe_type, gen_type)
            print("✅ RAG model ready")
            
            # Generate responses
            print(f"\n🤖 Generating responses for {len(test_questions)} questions...")
            responses = []
            
            for i, question in enumerate(test_questions):
                print(f"\n{'='*80}")
                print(f"Question {i+1}/{len(test_questions)}")
                print(f"{'='*80}")
                print(f"❓ {question}")
                print(f"{'-'*80}")
                
                response = self.generate_response(model, question, use_retrieval)
                responses.append(response)
                
                if response['context']:
                    context_text = response['context'][0]
                    # Truncate context to first 200 characters for cleaner display
                    display_context = context_text[:200] + "..." if len(context_text) > 200 else context_text
                    print(f"📚 Context: {display_context}")
                else:
                    print("📚 Context: None")
                print(f"{'-'*80}")
                print(f"💬 Answer: {response['answer']}")
                
                # Debug: Check if answer seems relevant
                if "location service" in response['answer'].lower() or len(response['answer']) < 10:
                    print(f"⚠️  WARNING: Answer seems irrelevant or too short!")
                    print(f"   Question was about: {question}")
                    print(f"   Answer length: {len(response['answer'])}")
                    if response['context']:
                        print(f"   Context length: {len(response['context'][0])}")
                
                # Show quality indicators
                if response['context']:
                    context_text = " ".join(response['context'])
                    question_lower = question.lower()
                    
                    # Extract key terms
                    key_terms = []
                    for word in question_lower.split():
                        if len(word) > 3 and word not in ['what', 'when', 'where', 'which', 'whose', 'about', 'tell', 'explain', 'describe']:
                            key_terms.append(word)
                    
                    # Calculate relevance
                    term_matches = sum(1 for term in key_terms if term in context_text.lower())
                    relevance = term_matches / len(key_terms) if key_terms else 0.5
                    
                    # Calculate answer quality
                    answer_lower = response['answer'].lower()
                    answer_length = len(response['answer'].split())
                    context_utilization = len(set(answer_lower.split()).intersection(set(context_text.lower().split()))) / len(answer_lower.split()) if answer_lower.split() else 0
                    
                    print(f"📊 Quality metrics:")
                    print(f"   Context relevance: {relevance:.2f} ({term_matches}/{len(key_terms)} key terms)")
                    print(f"   Answer length: {answer_length} words")
                    print(f"   Context utilization: {context_utilization:.2f}")
                else:
                    print(f"📊 Quality metrics: No context retrieved")
                
                print(f"{'='*80}")
            
            print(f"\n✅ Generated {len(responses)} responses")
            
            # Create evaluation dataset
            print("📊 Creating evaluation dataset...")
            eval_dataset = self.create_evaluation_dataset(test_questions, responses, ground_truth)
            
            # Run RAGAS evaluation
            print("🎯 Running RAGAS evaluation...")
            ragas_metrics = self.run_ragas_evaluation(eval_dataset, use_retrieval)
            
            # Calculate fallback metrics
            print("📈 Calculating fallback metrics...")
            fallback_metrics = self._calculate_fallback_metrics(responses, ground_truth, use_retrieval)
            
            # Combine metrics
            combined_metrics = {**ragas_metrics, **fallback_metrics}
            print(f"\n📊 FINAL METRICS: {combined_metrics}")
            
            return combined_metrics, responses
            
        except Exception as e:
            print(f"❌ ERROR in evaluate_combination: {e}")
            import traceback
            logging.error(f"Traceback: {traceback.format_exc()}")
            # Return error metrics
            error_metrics = {
                "error": str(e),
                "context_recall": 0.0,
                "context_precision": 0.0,
                "error_rate": 1.0,
                "avg_quality": 0.0,
                "avg_context_length": 0.0,
                "total_responses": 0
            }
            return error_metrics, []
    
    def _calculate_fallback_metrics(self, responses: List[Dict], 
                                  ground_truth: List[str] = None,
                                  use_retrieval: bool = True) -> Dict[str, float]:
        """Calculate comprehensive metrics for RAG evaluation"""
        def is_error_response(answer):
            return answer.startswith("Error:") or "error" in answer.lower()
        
        def calculate_answer_quality(answer, question, context):
            """Calculate answer quality based on multiple factors"""
            if is_error_response(answer):
                return 0.0
            
            # Length-based scoring
            answer_length = len(answer.strip())
            if answer_length < 5:
                return 0.1
            elif answer_length < 20:
                return 0.3
            elif answer_length > 500:
                return 0.6  # Penalize overly long answers
            
            # Content-based scoring
            score = 0.3  # Lower base score for more differentiation
            
            # Check for question-answer relevance (key terms matching)
            question_lower = question.lower()
            answer_lower = answer.lower()
            
            # Extract key terms from question (nouns, proper nouns)
            key_terms = []
            for word in question_lower.split():
                if len(word) > 3 and word not in ['what', 'when', 'where', 'which', 'whose', 'about', 'tell', 'explain', 'describe']:
                    key_terms.append(word)
            
            # Check if key terms appear in answer
            term_matches = sum(1 for term in key_terms if term in answer_lower)
            term_score = term_matches / len(key_terms) if key_terms else 0.5
            score += term_score * 0.4
            
            # Check for context utilization (if retrieval is used)
            if use_retrieval and context:
                context_text = " ".join(context).lower()
                context_words = set(context_text.split())
                answer_words = set(answer_lower.split())
                context_utilization = len(answer_words.intersection(context_words)) / len(answer_words) if answer_words else 0
                score += context_utilization * 0.3
            
            return min(score, 1.0)
        
        def calculate_context_relevance(question, context):
            """Calculate how relevant the retrieved context is to the question"""
            if not context:
                return 0.0
            
            question_lower = question.lower()
            context_text = " ".join(context).lower()
            
            # Extract key terms from question (nouns, proper nouns)
            key_terms = []
            for word in question_lower.split():
                if len(word) > 3 and word not in ['what', 'when', 'where', 'which', 'whose', 'about', 'tell', 'explain', 'describe']:
                    key_terms.append(word)
            
            if not key_terms:
                return 0.5  # Default score if no key terms found
            
            # Check if key terms appear in context
            term_matches = sum(1 for term in key_terms if term in context_text)
            relevance = term_matches / len(key_terms)
            
            # Bonus for exact phrase matches
            if any(term in context_text for term in key_terms):
                relevance += 0.2
            
            return min(relevance, 1.0)
        
        def calculate_context_coverage(context):
            """Calculate how much context information is available"""
            if not context:
                return 0.0
            
            total_length = sum(len(c) for c in context)
            if total_length < 50:
                return 0.2
            elif total_length < 200:
                return 0.5
            elif total_length < 500:
                return 0.8
            else:
                return 1.0
        
        # Calculate metrics
        total_responses = len(responses)
        error_count = sum(1 for r in responses if is_error_response(r["answer"]))
        error_rate = error_count / total_responses if total_responses > 0 else 0.0
        
        # Calculate quality scores
        quality_scores = []
        context_relevance_scores = []
        context_coverage_scores = []
        
        for r in responses:
            quality = calculate_answer_quality(r["answer"], r["question"], r["context"])
            quality_scores.append(quality)
            
            relevance = calculate_context_relevance(r["question"], r["context"])
            context_relevance_scores.append(relevance)
            
            coverage = calculate_context_coverage(r["context"])
            context_coverage_scores.append(coverage)
        
        avg_quality = np.mean(quality_scores) if quality_scores else 0.0
        avg_context_relevance = np.mean(context_relevance_scores) if context_relevance_scores else 0.0
        avg_context_coverage = np.mean(context_coverage_scores) if context_coverage_scores else 0.0
        
        # Calculate context length
        context_lengths = [len(r["context"]) for r in responses]
        avg_context_length = np.mean(context_lengths) if context_lengths else 0.0
        
        # Calculate answer length
        answer_lengths = [len(r["answer"]) for r in responses]
        avg_answer_length = np.mean(answer_lengths) if answer_lengths else 0.0
        
        # Calculate answer correctness based on question type
        correctness_scores = []
        for r in responses:
            question = r["question"].lower()
            answer = r["answer"].lower()
            
            # Different scoring for different question types
            if "who" in question:
                # For "who" questions, look for person names or titles
                if any(word in answer for word in ['artist', 'painter', 'author', 'writer', 'director', 'scientist', 'president', 'king', 'queen']):
                    correctness_scores.append(0.9)
                elif len(answer.split()) > 2 and not is_error_response(r["answer"]):
                    correctness_scores.append(0.7)
                else:
                    correctness_scores.append(0.3)
                    
            elif "what" in question:
                # For "what" questions, look for descriptive answers
                if len(answer.split()) > 5 and not is_error_response(r["answer"]):
                    correctness_scores.append(0.8)
                elif len(answer.split()) > 2:
                    correctness_scores.append(0.6)
                else:
                    correctness_scores.append(0.3)
                    
            elif "when" in question:
                # For "when" questions, look for dates or time periods
                if any(word in answer for word in ['year', 'century', 'decade', 'period', 'era']):
                    correctness_scores.append(0.9)
                elif len(answer.split()) > 2:
                    correctness_scores.append(0.6)
                else:
                    correctness_scores.append(0.3)
                    
            else:
                # Default scoring
                if len(answer.split()) > 3 and not is_error_response(r["answer"]):
                    correctness_scores.append(0.7)
                else:
                    correctness_scores.append(0.4)
        
        avg_correctness = np.mean(correctness_scores) if correctness_scores else 0.0
        
        # Calculate additional meaningful metrics
        def calculate_answer_specificity(answer):
            """Calculate how specific vs generic an answer is"""
            if is_error_response(answer):
                return 0.0
            
            # Count unique words vs total words
            words = answer.lower().split()
            if not words:
                return 0.0
            
            unique_words = set(words)
            specificity = len(unique_words) / len(words)
            
            # Bonus for technical/specific terms
            technical_terms = ['temperature', 'pressure', 'chemical', 'molecular', 'atomic', 'biological', 'physical', 'mathematical']
            technical_bonus = sum(1 for word in words if word in technical_terms) * 0.1
            
            return min(specificity + technical_bonus, 1.0)
        
        def calculate_context_utilization(answer, context):
            """Calculate how much of the retrieved context is actually used"""
            if not context or is_error_response(answer):
                return 0.0
            
            context_text = " ".join(context).lower()
            answer_lower = answer.lower()
            
            # Count words from context that appear in answer
            context_words = set(context_text.split())
            answer_words = set(answer_lower.split())
            
            if not context_words:
                return 0.0
            
            utilization = len(answer_words.intersection(context_words)) / len(context_words)
            return utilization
        
        def calculate_response_coherence(answer):
            """Calculate how coherent and well-structured the answer is"""
            if is_error_response(answer):
                return 0.0
            
            sentences = answer.split('.')
            if len(sentences) < 2:
                return 0.3  # Single sentence answers are less coherent
            
            # Check for logical connectors
            connectors = ['because', 'therefore', 'however', 'although', 'furthermore', 'additionally', 'moreover']
            connector_count = sum(1 for sentence in sentences for connector in connectors if connector in sentence.lower())
            
            # Check for proper sentence structure (basic)
            proper_sentences = sum(1 for sentence in sentences if len(sentence.strip()) > 10)
            
            coherence = (connector_count * 0.2 + proper_sentences / len(sentences) * 0.8)
            return min(coherence, 1.0)
        
        # Calculate new metrics
        specificity_scores = [calculate_answer_specificity(r["answer"]) for r in responses]
        utilization_scores = [calculate_context_utilization(r["answer"], r["context"]) for r in responses]
        coherence_scores = [calculate_response_coherence(r["answer"]) for r in responses]
        
        avg_specificity = np.mean(specificity_scores) if specificity_scores else 0.0
        avg_utilization = np.mean(utilization_scores) if utilization_scores else 0.0
        avg_coherence = np.mean(coherence_scores) if coherence_scores else 0.0
        
        return {
            "error_rate": error_rate,
            "avg_quality": avg_quality,
            "avg_correctness": avg_correctness,
            "avg_context_relevance": avg_context_relevance,
            "avg_context_coverage": avg_context_coverage,
            "avg_context_length": avg_context_length,
            "avg_answer_length": avg_answer_length,
            "avg_specificity": avg_specificity,
            "avg_context_utilization": avg_utilization,
            "avg_coherence": avg_coherence,
            "total_responses": total_responses,
            "successful_responses": total_responses - error_count
        }
    
    def run_comprehensive_evaluation(self, test_questions: List[str], 
                                   ground_truth: List[str] = None,
                                   combinations: List[str] = None) -> Dict[str, Dict[str, float]]:
        """Run comprehensive evaluation across different model combinations"""
        print("🎯 Starting comprehensive RAG evaluation...")
        print("=" * 80)
        print("🔄 Will evaluate combinations to test RAG functionality:")
        print("   1. ORIGINAL QE + ORIGINAL Generator (baseline with retrieval)")
        print("   2. ORIGINAL QE + FINE-TUNED Generator (notebook configuration)")
        print("   3. FINE-TUNED QE + FINE-TUNED Generator (full pipeline)")
        print("")
        
        # Define combination mappings
        combination_configs = {
            "1": ("original", "original", True, "original_qe_original_gen"),
            "2": ("original", "finetuned", True, "original_qe_finetuned_gen"),
            "3": ("finetuned", "finetuned", True, "finetuned_qe_finetuned_gen")
        }
        
        # Default to all combinations if none specified
        if combinations is None or "all" in combinations:
            combinations = ["1", "2", "3"]
        
        print(f"🔍 Testing combinations: {', '.join(combinations)}")
        print("")
        
        results = {}
        all_responses = {}  # Store all responses for answer comparison
        
        # Evaluate selected combinations
        for i, combo_id in enumerate(combinations, 1):
            if combo_id not in combination_configs:
                print(f"⚠️ Skipping unknown combination: {combo_id}")
                continue
                
            qe_type, gen_type, use_retrieval, result_key = combination_configs[combo_id]
            
            print(f"🔄 [{i}/{len(combinations)}] Starting evaluation...")
            print("=" * 60)
            
            # Get description for this combination
            descriptions = {
                "1": "ORIGINAL Question Encoder + ORIGINAL Generator",
                "2": "ORIGINAL QE + FINE-TUNED Generator (notebook config)",
                "3": "FINE-TUNED QE + FINE-TUNED Generator"
            }
            print(f"🔍 EVALUATING: {descriptions[combo_id]}")
            print("=" * 60)
            
            try:
                result, responses = self.evaluate_combination(qe_type, gen_type, test_questions, ground_truth, use_retrieval=use_retrieval)
                results[result_key] = result
                all_responses[result_key] = responses
                print(f"✅ [{i}/{len(combinations)}] Completed successfully")
            except Exception as e:
                print(f"❌ [{i}/{len(combinations)}] Failed: {e}")
                results[result_key] = {"error": str(e), "skipped": True}
                all_responses[result_key] = []
        
        print("")
        print(f"🎉 All evaluations completed! Processed {len(combinations)} combination(s)")        
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
        """Print results in a clean, readable comparison table format"""
        print("\n" + "="*100)
        print("COMPREHENSIVE RAG EVALUATION RESULTS")
        print("="*100)
        
        # Scenario descriptions
        scenario_descriptions = {
            "original_qe_original_gen": "Original QE + Original Gen (Baseline)",
            "original_qe_finetuned_gen": "Original QE + Fine-tuned Gen (Notebook config)",
            "finetuned_qe_finetuned_gen": "Fine-tuned QE + Fine-tuned Gen (Full pipeline)"
        }
        
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
        
        # Define the most important metrics to display
        key_metrics = [
            'avg_quality',
            'avg_correctness', 
            'avg_context_relevance',
            'avg_answer_length',
            'avg_context_coverage',
            'avg_specificity',
            'avg_context_utilization',
            'avg_coherence',
            'error_rate'
        ]
        
        # Print key metrics table
        print("\n📊 KEY PERFORMANCE METRICS:")
        print("-" * 100)
        
        # Print header
        header = f"{'Combination':<30}"
        for metric in key_metrics:
            metric_name = metric.replace('_', ' ').title()
            header += f"{metric_name:<15}"
        print(header)
        print("-" * 100)
        
        # Print data rows
        for combo, desc in scenario_descriptions.items():
            if combo in results:
                result = results[combo]
                if isinstance(result, dict) and "error" not in result:
                    row = f"{combo:<30}"
                    for metric in key_metrics:
                        value = result.get(metric, 0.0)
                        if isinstance(value, (list, dict)):
                            # Skip complex data types
                            row += f"{'N/A':<15}"
                        else:
                            # Format numbers nicely
                            if metric == 'avg_answer_length':
                                value_str = f"{value:.1f}"
                            elif metric == 'error_rate':
                                value_str = f"{value:.1%}"
                            else:
                                value_str = f"{value:.3f}"
                            row += f"{value_str:<15}"
                    print(row)
        
        # Print summary statistics
        print("\n📈 SUMMARY:")
        print("-" * 50)
        
        # Find best performing model for each metric
        for metric in key_metrics:
            if metric == 'error_rate':
                best_value = float('inf')
                best_combo = None
                for combo, result in results.items():
                    if isinstance(result, dict) and "error" not in result:
                        value = result.get(metric, float('inf'))
                        if isinstance(value, (int, float)) and value < best_value:
                            best_value = value
                            best_combo = combo
            else:
                best_value = 0.0
                best_combo = None
                for combo, result in results.items():
                    if isinstance(result, dict) and "error" not in result:
                        value = result.get(metric, 0.0)
                        if isinstance(value, (int, float)) and value > best_value:
                            best_value = value
                            best_combo = combo
            
            if best_combo:
                metric_name = metric.replace('_', ' ').title()
                if metric == 'avg_answer_length':
                    print(f"🏆 Best {metric_name}: {best_value:.1f} words ({best_combo})")
                elif metric == 'error_rate':
                    print(f"🏆 Best {metric_name}: {best_value:.1%} ({best_combo})")
                else:
                    print(f"🏆 Best {metric_name}: {best_value:.3f} ({best_combo})")
        
        print("\n" + "="*120)
    
    def generate_comparison_charts(self, results: Dict[str, Dict[str, float]], 
                                 output_dir: str = ".") -> List[str]:
        """Generate comprehensive comparison charts"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        charts_dir = os.path.join(output_dir, f"charts_{timestamp}")
        os.makedirs(charts_dir, exist_ok=True)
        
        charts = []
        
        # Filter out non-metric entries from results
        metric_results = {k: v for k, v in results.items() 
                         if isinstance(v, dict) and not k.startswith('_')}
        
        if not metric_results:
            logging.warning("No valid metric results found for chart generation")
            return charts
        
        # Generate different chart types
        charts.append(self._create_metrics_bar_chart(metric_results, charts_dir))
        charts.append(self._create_new_metrics_chart(metric_results, charts_dir))
        charts.append(self._create_radar_chart(metric_results, charts_dir))
        charts.append(self._create_heatmap(metric_results, charts_dir))
        charts.append(self._create_qa_comparison_chart(results, charts_dir))
        
        logging.info(f"Generated {len(charts)} charts in {charts_dir}")
        return charts
    
    def _create_metrics_bar_chart(self, results: Dict[str, Dict[str, float]], 
                                charts_dir: str) -> str:
        """Create bar chart comparing meaningful metrics across combinations"""
        try:
            # Filter out combinations with errors
            valid_combinations = []
            for combo in results.keys():
                if isinstance(results[combo], dict) and "error" not in results[combo]:
                    valid_combinations.append(combo)
            
            if not valid_combinations:
                logging.warning("No valid combinations found for chart generation")
                return ""
            
            # Focus on meaningful metrics that show real differences
            metrics = [
                'avg_quality',
                'avg_correctness', 
                'avg_context_relevance',
                'avg_answer_length'
            ]
            
            # Create subplots
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('RAG Model Performance - Key Metrics', fontsize=16, fontweight='bold')
            
            # Color scheme for better visualization
            colors = ['#2E86AB', '#A23B72', '#F18F01']
            
            for i, metric in enumerate(metrics):
                ax = axes[i//2, i%2]
                values = []
                combo_labels = []
                
                for combo in valid_combinations:
                    value = results[combo].get(metric, 0)
                    # Ensure value is a scalar
                    if isinstance(value, (list, np.ndarray)):
                        value = float(value[0]) if len(value) > 0 else 0.0
                    else:
                        value = float(value)
                    values.append(value)
                    combo_labels.append(combo)
                
                if values:
                    bars = ax.bar(combo_labels, values, alpha=0.8, color=colors[:len(values)])
                    ax.set_title(f'{metric.replace("_", " ").title()}', fontweight='bold')
                    ax.set_ylabel('Score' if metric != 'avg_answer_length' else 'Words')
                    
                    # Rotate x-axis labels for better readability
                    ax.tick_params(axis='x', rotation=15)
                    
                    # Add value labels on bars
                    for bar, value in zip(bars, values):
                        if metric == 'avg_answer_length':
                            label = f'{value:.1f}'
                        else:
                            label = f'{value:.3f}'
                        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                               label, ha='center', va='bottom', fontweight='bold')
                    
                    # Add grid for better readability
                    ax.grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            chart_path = os.path.join(charts_dir, 'key_metrics_comparison.png')
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return chart_path
            
        except Exception as e:
            logging.error(f"Error creating metrics bar chart: {e}")
            return ""
    
    def _create_new_metrics_chart(self, results: Dict[str, Dict[str, float]], 
                                charts_dir: str) -> str:
        """Create chart for new meaningful metrics"""
        try:
            # Filter out combinations with errors
            valid_combinations = []
            for combo in results.keys():
                if isinstance(results[combo], dict) and "error" not in results[combo]:
                    valid_combinations.append(combo)
            
            if not valid_combinations:
                return ""
            
            # New metrics that show real differences
            metrics = [
                'avg_specificity',
                'avg_context_utilization', 
                'avg_coherence',
                'avg_context_coverage'
            ]
            
            # Create subplots
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('RAG Model Performance - Advanced Metrics', fontsize=16, fontweight='bold')
            
            # Color scheme
            colors = ['#2E86AB', '#A23B72', '#F18F01']
            
            for i, metric in enumerate(metrics):
                ax = axes[i//2, i%2]
                values = []
                combo_labels = []
                
                for combo in valid_combinations:
                    value = results[combo].get(metric, 0)
                    if isinstance(value, (list, np.ndarray)):
                        value = float(value[0]) if len(value) > 0 else 0.0
                    else:
                        value = float(value)
                    values.append(value)
                    combo_labels.append(combo)
                
                if values:
                    bars = ax.bar(combo_labels, values, alpha=0.8, color=colors[:len(values)])
                    
                    # Better metric names
                    metric_names = {
                        'avg_specificity': 'Answer Specificity',
                        'avg_context_utilization': 'Context Utilization',
                        'avg_coherence': 'Response Coherence',
                        'avg_context_coverage': 'Context Coverage'
                    }
                    ax.set_title(metric_names.get(metric, metric.replace("_", " ").title()), fontweight='bold')
                    ax.set_ylabel('Score')
                    ax.tick_params(axis='x', rotation=15)
                    
                    # Add value labels on bars
                    for bar, value in zip(bars, values):
                        label = f'{value:.3f}'
                        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                               label, ha='center', va='bottom', fontweight='bold')
                    
                    # Add grid
                    ax.grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            chart_path = os.path.join(charts_dir, 'advanced_metrics.png')
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return chart_path
            
        except Exception as e:
            logging.error(f"Error creating new metrics chart: {e}")
            return ""
    
    def _create_radar_chart(self, results: Dict[str, Dict[str, float]], 
                          charts_dir: str) -> str:
        """Create radar chart for comprehensive comparison"""
        try:
            # Select meaningful metrics for radar chart
            metrics = ['avg_quality', 'avg_correctness', 'avg_context_relevance', 'avg_specificity', 'avg_coherence']
            
            # Filter out combinations with errors
            valid_combinations = []
            for combo in results.keys():
                if isinstance(results[combo], dict) and "error" not in results[combo]:
                    valid_combinations.append(combo)
            
            if not valid_combinations:
                logging.warning("No valid combinations found for radar chart")
                return ""
            
            # Prepare data
            angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
            angles += angles[:1]  # Complete the circle
            
            fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
            
            for combo in valid_combinations:
                values = []
                for metric in metrics:
                    value = results[combo].get(metric, 0)
                    # Ensure value is a scalar
                    if isinstance(value, (list, np.ndarray)):
                        value = float(value[0]) if len(value) > 0 else 0.0
                    else:
                        value = float(value)
                    values.append(value)
                
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
            
        except Exception as e:
            logging.error(f"Error creating radar chart: {e}")
            return ""
    
    def _create_heatmap(self, results: Dict[str, Dict[str, float]], 
                       charts_dir: str) -> str:
        """Create heatmap of all metrics"""
        try:
            # Filter out combinations with errors
            valid_results = {}
            for combo, data in results.items():
                if isinstance(data, dict) and "error" not in data:
                    valid_results[combo] = data
            
            if not valid_results:
                logging.warning("No valid combinations found for heatmap")
                return ""
            
            # Convert to DataFrame
            df = pd.DataFrame(valid_results).T
            
            # Select numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                logging.warning("No numeric columns found for heatmap")
                return ""
            
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
            
        except Exception as e:
            logging.error(f"Error creating heatmap: {e}")
            return ""
    
    def _create_qa_comparison_chart(self, results: Dict[str, Dict[str, float]], 
                                  charts_dir: str) -> str:
        """Create a chart showing questions, context, and answers side by side for each combination"""
        try:
            # Get the responses and questions from results
            all_responses = results.get("_all_responses", {})
            test_questions = results.get("_test_questions", [])
            
            if not all_responses or not test_questions:
                logging.warning("No response data found for QA comparison chart")
                return ""
            
            # Filter out non-metric entries to get valid combinations
            valid_combinations = []
            for combo in results.keys():
                if isinstance(results[combo], dict) and not combo.startswith('_') and "error" not in results[combo]:
                    valid_combinations.append(combo)
            
            if len(valid_combinations) < 2:
                logging.warning("Need at least 2 valid combinations for comparison chart")
                return ""
            
            # Show first 5 questions for readability
            num_questions = min(5, len(test_questions))
            
            # Create figure with subplots for each question
            fig, axes = plt.subplots(num_questions, 1, figsize=(16, 4 * num_questions))
            if num_questions == 1:
                axes = [axes]
            
            fig.suptitle('Question-Context-Answer Comparison Across Model Combinations', 
                        fontsize=16, fontweight='bold', y=0.98)
            
            # Color scheme for combinations
            colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
            
            for q_idx in range(num_questions):
                ax = axes[q_idx]
                question = test_questions[q_idx]
                
                # Truncate question for display
                display_question = question[:80] + "..." if len(question) > 80 else question
                
                # Create text for each combination
                y_pos = 0.9
                ax.text(0.02, y_pos, f"Q{q_idx+1}: {display_question}", 
                       fontsize=12, fontweight='bold', transform=ax.transAxes,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
                
                y_pos -= 0.15
                
                for i, combo in enumerate(valid_combinations):
                    if combo in all_responses and q_idx < len(all_responses[combo]):
                        response = all_responses[combo][q_idx]
                        
                        # Get context and answer
                        context = response.get("context", [])
                        answer = response.get("answer", "No answer generated")
                        
                        # Format context for display
                        if context:
                            context_text = " ".join(context[:2])  # Show first 2 context pieces
                            if len(context) > 2:
                                context_text += "..."
                        else:
                            context_text = "No context retrieved"
                        
                        # Truncate for display
                        context_text = context_text[:100] + "..." if len(context_text) > 100 else context_text
                        answer_text = answer[:150] + "..." if len(answer) > 150 else answer
                        
                        # Display combination name
                        combo_display = combo.replace('_', ' ').title()
                        ax.text(0.02, y_pos, f"{combo_display}:", 
                               fontsize=11, fontweight='bold', color=colors[i % len(colors)],
                               transform=ax.transAxes)
                        y_pos -= 0.05
                        
                        # Display context
                        ax.text(0.05, y_pos, f"Context: {context_text}", 
                               fontsize=10, style='italic', transform=ax.transAxes,
                               bbox=dict(boxstyle="round,pad=0.2", facecolor="lightyellow", alpha=0.5))
                        y_pos -= 0.08
                        
                        # Display answer
                        ax.text(0.05, y_pos, f"Answer: {answer_text}", 
                               fontsize=10, transform=ax.transAxes,
                               bbox=dict(boxstyle="round,pad=0.2", facecolor="lightgreen", alpha=0.5))
                        y_pos -= 0.12
                
                # Remove axes
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.axis('off')
            
            plt.tight_layout()
            chart_path = os.path.join(charts_dir, 'qa_comparison.png')
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return chart_path
            
        except Exception as e:
            logging.error(f"Error creating QA comparison chart: {e}")
            return ""
    
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
                
                # Filter out non-metric entries
                metric_results = {k: v for k, v in results.items() 
                                if isinstance(v, dict) and not k.startswith('_')}
                
                for combo, metrics in metric_results.items():
                    f.write(f"\n{combo}:\n")
                    if isinstance(metrics, dict):
                        for metric, value in metrics.items():
                            if isinstance(value, (int, float)):
                                f.write(f"  {metric}: {value:.3f}\n")
                            else:
                                f.write(f"  {metric}: {value}\n")
                    else:
                        f.write(f"  {metrics}\n")
                
            logging.info(f"Answer comparison saved to {output_path}")
            return output_path
            
        except Exception as e:
            logging.error(f"Error saving answer comparison: {e}")
            return "" 
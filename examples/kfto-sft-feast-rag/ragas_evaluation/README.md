# RAGAS Evaluation Framework

A comprehensive evaluation framework for RAG (Retrieval Augmented Generation) models using RAGAS metrics and advanced custom evaluation criteria. This framework evaluates different combinations of question encoders and generators to assess RAG model performance with detailed quality metrics.

## 📁 Module Structure

```
ragas_evaluation/
├── __init__.py              # Package initialization
├── config.py                # Configuration management
├── dataset_loader.py        # Dataset loading utilities with caching and progress tracking
├── evaluator.py             # Main evaluation logic with enhanced metrics and QA comparison charts
├── utils.py                 # Utility functions and setup
├── main.py                  # Main orchestration script with CLI arguments
├── feast_rag_retriever.py   # Custom Feast RAG retriever implementation
├── test_dataset.py          # Test script for dataset loading
└── README.md               # This file
```

## 🚀 Quick Start

### Installation
```bash
cd examples/kfto-sft-feast-rag/ragas_evaluation

# Install dependencies
pip install -r requirements.txt

# Install tqdm for better progress bars
pip install tqdm

# Or install specific versions for CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

### Dataset Management

#### Download Natural Questions Dataset (One-time)
```bash
# Download to permanent cache directory
python dataset_loader.py download datasets/natural_questions

# Download specific split
python dataset_loader.py download datasets/natural_questions --split validation

# Test cached dataset
python dataset_loader.py test datasets/natural_questions
```

#### Use Cached Dataset for Evaluation
```bash
# Use cached dataset (fast)
python main.py --use-natural-questions --cache-dir datasets/natural_questions --nq-sample-percentage 0.02

# Use different sample percentages without re-downloading
python main.py --use-natural-questions --cache-dir datasets/natural_questions --nq-sample-percentage 0.05
python main.py --use-natural-questions --cache-dir datasets/natural_questions --nq-sample-percentage 0.1
```

### Run the Complete Evaluation
```bash
# Run all combinations with cached dataset
python main.py --use-natural-questions --cache-dir datasets/natural_questions --nq-sample-percentage 0.02

# Run specific combinations
python main.py --combinations 1 2 3 --cache-dir datasets/natural_questions

# Use curated questions (no download needed)
python main.py --combinations 2

# Specify output directory
python main.py --combinations 1 2 --output-dir my_results --cache-dir datasets/natural_questions

# Run with logging to file
python main.py --combinations 2 --output-dir debug_results --cache-dir datasets/natural_questions 2>&1 | tee evaluation.log
```

### Command Line Arguments
```bash
python main.py [OPTIONS]

Options:
  --checkpoint-dir PATH       Path to fine-tuned model checkpoint
  --feast-repo-path PATH      Path to Feast repository
  --use-natural-questions     Use Natural Questions dataset (~50GB download, takes 10-30 minutes)
  --nq-sample-percentage FLOAT Sample percentage for Natural Questions (default: 0.1)
  --max-questions INT         Maximum number of questions to evaluate (default: 100)
  --combinations [1|2|3|all]  Select combinations to test (default: all)
  --output-dir PATH           Output directory for results (default: ragas_evaluation_results)
  --cache-dir PATH            Directory to cache Natural Questions dataset (default: use default cache location)
```

### Dataset Loader Commands
```bash
# Download dataset
python dataset_loader.py download <cache_dir> [--split {train,validation}]

# Test dataset loading
python dataset_loader.py test <cache_dir> [--sample-percentage FLOAT] [--max-questions INT]

# Examples
python dataset_loader.py download datasets/natural_questions
python dataset_loader.py test datasets/natural_questions --sample-percentage 0.001 --max-questions 10
```

### Use as a Package
```python
from ragas_evaluation import RAGEvaluator, ModelConfig
from ragas_evaluation.dataset_loader import load_natural_questions_dataset

# Configure evaluation
config = ModelConfig(
    finetuned_checkpoint_dir="/opt/app-root/src/shared/fine_tuned_rag_model/inference_sft-rag-master-0",
    feast_repo_path="/opt/app-root/src/distributed-workloads/examples/kfto-sft-feast-rag/feature_repo",
    use_natural_questions=True,
    nq_sample_percentage=0.1,
    max_evaluation_questions=100
)

# Load dataset with caching
questions, answers = load_natural_questions_dataset(
    sample_percentage=0.1,
    max_questions=100,
    cache_dir="datasets/natural_questions"
)

# Run evaluation
evaluator = RAGEvaluator(config)
results = evaluator.run_comprehensive_evaluation(questions, answers, combinations=["1", "2"])

# Save results and generate charts
evaluator.save_results(results, "my_evaluation_results.json")
evaluator.generate_comparison_charts(results, output_dir="my_results")
evaluator.print_comparison_table(results)
```

## 📊 Modules Overview

### `config.py`
- **ModelConfig**: Centralized configuration for all evaluation parameters
- Model paths, device settings, evaluation parameters
- Dataset configuration options

### `dataset_loader.py`
- **load_natural_questions_dataset()**: Load and sample from Natural Questions dataset with caching and progress tracking
- **download_natural_questions_dataset()**: Download dataset to specified cache directory
- **load_curated_questions()**: Load curated factual questions as fallback
- **Progress tracking**: Built-in progress bars for downloads and processing
- **Caching support**: Persistent dataset storage for fast subsequent runs
- Smart filtering and sampling with reproducibility

### `evaluator.py` (Main Module)
- **RAGEvaluator**: Complete evaluation class with enhanced functionality
- Model loading and initialization (question encoders, generators)
- Custom Feast RAG retriever integration
- RAGAS metrics calculation with proper dataset formatting
- **Advanced fallback metrics** with question-type specific scoring
- **New custom metrics**: Answer specificity, context utilization, response coherence
- Comprehensive evaluation of 3 model combinations
- **QA Comparison Charts**: Visual comparison of questions, context, and answers across combinations
- Chart generation (bar charts, radar charts, heatmaps, QA comparison)
- Results reporting and comparison tables
- Answer comparison and analysis
- **Real-time quality indicators** during evaluation

### `utils.py`
- **setup_logging()**: Comprehensive logging configuration with output directory support
- Warning suppression for external libraries
- Error handling and setup utilities

### `main.py`
- **main()**: Complete orchestration with command-line argument support
- Configuration setup from CLI arguments
- Dataset loading with flexible options and caching
- Evaluation execution with selective combinations
- Results saving and visualization
- **Improved messaging**: Clear feedback about dataset loading and fallbacks

### `feast_rag_retriever.py`
- **FeastRAGRetriever**: Custom implementation for Feast RAG integration
- **FeastIndex**: Custom index implementation for feature store
- **FeastVectorStore**: Vector store implementation for document retrieval

## ⚙️ Configuration

### ModelConfig Options

```python
config = ModelConfig(
    # Paths
    finetuned_checkpoint_dir="/opt/app-root/src/shared/fine_tuned_rag_model/inference_sft-rag-master-0",
    feast_repo_path="/opt/app-root/src/distributed-workloads/examples/kfto-sft-feast-rag/feature_repo",
    
    # Model names
    original_qe_model="facebook/dpr-question_encoder-single-nq-base",
    original_gen_model="facebook/bart-large",
    
    # Device
    device="cuda",  # or "cpu"
    
    # Evaluation settings
    max_new_tokens=200,
    num_beams=1,
    do_sample=False,
    
    # Dataset settings
    use_natural_questions=True,
    nq_sample_percentage=0.1,
    max_evaluation_questions=100
)
```

## 📈 Evaluation Scenarios

The framework evaluates 3 model combinations:

1. **Original QE + Original Generator** (baseline with retrieval)
2. **Original QE + Fine-tuned Generator** (notebook configuration)
3. **Fine-tuned QE + Fine-tuned Generator** (full pipeline)

## 📊 Enhanced Metrics

### RAGAS Metrics
- **Context Recall**: How well the model retrieves relevant context
- **Context Precision**: How precise the retrieved context is

### Advanced Fallback Metrics
- **Error Rate**: Rate of failed responses
- **Average Quality**: Multi-factor quality score based on:
  - Answer length appropriateness
  - Key term matching between question and answer
  - Context utilization in answers
- **Average Correctness**: Question-type specific scoring:
  - **"Who" questions**: Person names, titles, roles detection
  - **"What" questions**: Descriptive content evaluation
  - **"When" questions**: Date and time period detection
  - **Default**: General answer quality assessment
- **Context Relevance**: Key term matching between questions and retrieved context
- **Context Coverage**: Amount of information available in retrieved context
- **Average Context Length**: Length of retrieved context
- **Average Answer Length**: Length of generated answers

### New Custom Metrics
- **Average Specificity**: How specific vs generic an answer is
- **Average Context Utilization**: How much of the retrieved context is actually used
- **Average Coherence**: How coherent and well-structured the answer is
- **Total Responses**: Number of processed responses
- **Successful Responses**: Number of non-error responses

## 🎯 Real-Time Quality Indicators

During evaluation, the framework provides real-time quality metrics:

```
📊 Quality metrics:
   Context relevance: 0.75 (3/4 key terms)
   Answer length: 15 words
   Context utilization: 0.60
```

## 📁 Output Files

The evaluation generates several output files with timestamps:

- **JSON Results**: `comprehensive_rag_evaluation_YYYYMMDD_HHMMSS.json`
- **Answer Comparison**: `answer_comparison_YYYYMMDD_HHMMSS.txt`
- **Charts Directory**: `charts_YYYYMMDD_HHMMSS/` containing:
  - `key_metrics_comparison.png` - Bar charts comparing key metrics
  - `new_metrics_comparison.png` - Bar charts for new custom metrics
  - `radar_chart.png` - Radar chart for comprehensive comparison
  - `metrics_heatmap.png` - Heatmap of all metrics
  - `qa_comparison.png` - **NEW**: Visual comparison of questions, context, and answers across combinations
- **Log File**: `ragas_evaluation_YYYYMMDD_HHMMSS.log`

## 🔧 Customization

### Adding New Metrics
1. Extend the `_calculate_fallback_metrics()` method in `evaluator.py`
2. Add new metric calculation logic
3. Update the comparison table generation

### Adding New Charts
1. Create new chart methods in `evaluator.py`
2. Add chart generation to `generate_comparison_charts()`
3. Update the main orchestration in `main.py`

### Adding New Datasets
1. Create new loading functions in `dataset_loader.py`
2. Update the dataset selection logic in `main.py`
3. Add configuration options in `config.py`

## 🐛 Troubleshooting

### Installation Issues

1. **CUDA Support**: For GPU acceleration, install PyTorch with CUDA support
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

2. **Progress Bars**: For better progress tracking, install tqdm
   ```bash
   pip install tqdm
   ```

3. **Feast Installation**: If Feast installation fails, try:
   ```bash
   pip install feast[redis]  # For Redis support
   # or
   pip install feast[postgres]  # For PostgreSQL support
   ```

4. **RAGAS Installation**: If RAGAS installation fails:
   ```bash
   pip install ragas[all]  # Install with all optional dependencies
   ```

### Runtime Issues

1. **Dataset Download**: Natural Questions dataset is large (~50GB)
   - Solution: Use caching with `--cache-dir` to avoid re-downloading
   - Solution: Use curated questions with `--use-natural-questions False`

2. **Disk Space**: Natural Questions dataset requires significant disk space
   - Solution: Use `--use-natural-questions False` for curated questions
   - Solution: Use smaller sample percentages

3. **Memory Issues**: Large models may cause OOM errors
   - Solution: Reduce `--max-questions` or use CPU

4. **Feast Connection**: Feast feature store connection issues
   - Solution: Check Feast configuration and network connectivity

5. **Model Loading**: Fine-tuned model loading failures
   - Solution: Verify model paths and fallback to original models

6. **RAGAS Errors**: RAGAS evaluation failures
   - Solution: Framework automatically falls back to enhanced custom metrics

7. **Import Errors**: If you get import errors for Feast components
   - Solution: Ensure you have the latest Feast version with RAG support
   ```bash
   pip install --upgrade feast
   ```

8. **Circular Import Errors**: If you encounter circular import issues
   - Solution: Clear Python cache and restart
   ```bash
   find . -name "*.pyc" -delete && find . -name "__pycache__" -type d -exec rm -rf {} +
   ```

9. **Dataset Loading Timeout**: If dataset loading times out
   - Solution: Use cached dataset with `--cache-dir`
   - Solution: Use curated questions instead

## 📝 Logging

The framework provides comprehensive logging:
- File logging with timestamps in specified output directory
- Console output for real-time monitoring
- Warning suppression for external libraries
- Error tracking and reporting
- Progress indicators for each evaluation step
- Real-time quality metrics during evaluation
- **Progress bars** for dataset downloads and processing

## 🎯 Key Features

- **Comprehensive Evaluation**: Tests 3 different model combinations
- **Enhanced Scoring**: Multi-dimensional quality assessment with question-type specific scoring
- **Real-time Monitoring**: Live quality indicators during evaluation
- **Robust Error Handling**: Graceful fallbacks when components fail
- **Detailed Progress Reporting**: Real-time status updates with quality metrics
- **Multiple Output Formats**: JSON, text, and visual charts
- **Flexible Configuration**: Command-line arguments and easy parameter modification
- **Modular Design**: Clean, maintainable code structure
- **Custom Feast Integration**: Tailored RAG retriever for optimal performance
- **Dataset Caching**: Persistent storage for fast subsequent runs
- **Progress Tracking**: Built-in progress bars for downloads and processing
- **QA Comparison Charts**: Visual comparison of questions, context, and answers
- **New Custom Metrics**: Answer specificity, context utilization, response coherence

## 🤝 Contributing

To extend the framework:

1. Follow the modular structure
2. Add proper error handling
3. Include logging for debugging
4. Update documentation
5. Add tests for new functionality
6. Maintain backward compatibility with CLI arguments

## 📋 Requirements

- Python 3.8+
- PyTorch
- Transformers
- RAGAS
- Feast (with RAG retriever support)
- Matplotlib
- Seaborn
- Pandas
- NumPy
- Datasets (Hugging Face)
- tqdm (for progress bars)

## 🔗 Dependencies

The framework uses a custom Feast RAG retriever implementation:
- `FeastRAGRetriever`: Custom RAG retriever with `generate_answer()` and `retrieve()` methods
- `FeastIndex`: Custom index for feature store integration
- `FeastVectorStore`: Vector store for document retrieval and similarity search

## 📊 Performance Tips

1. **Use Caching**: Download Natural Questions once and reuse with `--cache-dir`
2. **Sample Wisely**: Start with small percentages (0.01-0.02) for testing
3. **Monitor Progress**: Install tqdm for better progress tracking
4. **GPU Acceleration**: Use CUDA for faster model inference
5. **Memory Management**: Adjust `--max-questions` based on available memory 
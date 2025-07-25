# RAGAS Evaluation Framework

A comprehensive evaluation framework for RAG (Retrieval Augmented Generation) models using RAGAS metrics and custom evaluation criteria. This framework evaluates different combinations of question encoders and generators to assess RAG model performance.

## 📁 Module Structure

```
ragas_evaluation/
├── __init__.py              # Package initialization
├── config.py                # Configuration management
├── dataset_loader.py        # Dataset loading utilities
├── evaluator.py             # Main evaluation logic (all-in-one)
├── utils.py                 # Utility functions and setup
├── main.py                  # Main orchestration script
├── ragas_evaluation_script.py # Original monolithic script (backup)
└── README.md               # This file
```

## 🚀 Quick Start

### Installation
```bash
cd examples/kfto-sft-feast-rag/ragas_evaluation

# Install dependencies
pip install -r requirements.txt

# Or install specific versions for CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

### Run the Complete Evaluation
```bash
python main.py
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

# Load dataset
questions, answers = load_natural_questions_dataset(
    sample_percentage=0.1,
    max_questions=100
)

# Run evaluation
evaluator = RAGEvaluator(config)
results = evaluator.run_comprehensive_evaluation(questions, answers)

# Save results and generate charts
evaluator.save_results(results, "my_evaluation_results.json")
evaluator.generate_comparison_charts(results)
evaluator.print_comparison_table(results)
```

## 📊 Modules Overview

### `config.py`
- **ModelConfig**: Centralized configuration for all evaluation parameters
- Model paths, device settings, evaluation parameters
- Dataset configuration options

### `dataset_loader.py`
- **load_natural_questions_dataset()**: Load and sample from Natural Questions dataset with timeout handling
- **load_curated_questions()**: Load curated factual questions as fallback
- Smart filtering and sampling with reproducibility

### `evaluator.py` (Main Module)
- **RAGEvaluator**: Complete evaluation class with all functionality
- Model loading and initialization (question encoders, generators)
- Feast RAG retriever integration
- RAGAS metrics calculation
- Fallback metrics when RAGAS fails
- Comprehensive evaluation of 4 model combinations
- Chart generation (bar charts, radar charts, heatmaps)
- Results reporting and comparison tables
- Answer comparison and analysis

### `utils.py`
- **setup_logging()**: Comprehensive logging configuration
- Warning suppression for external libraries
- Error handling and setup utilities

### `main.py`
- **main()**: Complete orchestration of the evaluation workflow
- Configuration setup
- Dataset loading
- Evaluation execution
- Results saving and visualization

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

The framework evaluates 4 model combinations:

1. **Original QE + Original Generator** (baseline with retrieval)
2. **Fine-tuned Generator only** (no retrieval - test memorization)
3. **Fine-tuned QE + Original Generator** (test retrieval quality)
4. **Fine-tuned QE + Fine-tuned Generator** (full pipeline)

## 📊 Metrics

### RAGAS Metrics
- **Context Recall**: How well the model retrieves relevant context
- **Context Precision**: How precise the retrieved context is

### Fallback Metrics
- **Error Rate**: Rate of failed responses
- **Average Quality**: Heuristic quality score
- **Average Context Length**: Length of retrieved context
- **Total Responses**: Number of processed responses

## 📁 Output Files

The evaluation generates several output files with timestamps:

- **JSON Results**: `comprehensive_rag_evaluation_YYYYMMDD_HHMMSS.json`
- **Answer Comparison**: `answer_comparison_YYYYMMDD_HHMMSS.txt`
- **Charts Directory**: `charts_YYYYMMDD_HHMMSS/` containing:
  - `metrics_comparison.png` - Bar charts comparing metrics
  - `radar_chart.png` - Radar chart for comprehensive comparison
  - `metrics_heatmap.png` - Heatmap of all metrics
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

2. **Feast Installation**: If Feast installation fails, try:
   ```bash
   pip install feast[redis]  # For Redis support
   # or
   pip install feast[postgres]  # For PostgreSQL support
   ```

3. **RAGAS Installation**: If RAGAS installation fails:
   ```bash
   pip install ragas[all]  # Install with all optional dependencies
   ```

### Runtime Issues

1. **Disk Space**: Natural Questions dataset requires significant disk space
   - Solution: Use `use_natural_questions=False` for curated questions

2. **Memory Issues**: Large models may cause OOM errors
   - Solution: Reduce `max_evaluation_questions` or use CPU

3. **Feast Connection**: Feast feature store connection issues
   - Solution: Check Feast configuration and network connectivity

4. **Model Loading**: Fine-tuned model loading failures
   - Solution: Verify model paths and fallback to original models

5. **RAGAS Errors**: RAGAS evaluation failures
   - Solution: Framework automatically falls back to custom metrics

6. **Import Errors**: If you get import errors for Feast components
   - Solution: Ensure you have the latest Feast version with RAG support
   ```bash
   pip install --upgrade feast
   ```

## 📝 Logging

The framework provides comprehensive logging:
- File logging with timestamps
- Console output for real-time monitoring
- Warning suppression for external libraries
- Error tracking and reporting
- Progress indicators for each evaluation step

## 🎯 Key Features

- **Comprehensive Evaluation**: Tests 4 different model combinations
- **Robust Error Handling**: Graceful fallbacks when components fail
- **Detailed Progress Reporting**: Real-time status updates
- **Multiple Output Formats**: JSON, text, and visual charts
- **Flexible Configuration**: Easy to modify paths and parameters
- **Modular Design**: Clean, maintainable code structure

## 🤝 Contributing

To extend the framework:

1. Follow the modular structure
2. Add proper error handling
3. Include logging for debugging
4. Update documentation
5. Add tests for new functionality

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

## 🔗 Dependencies

The framework uses the official [Feast SDK](https://github.com/feast-dev/feast/blob/master/sdk/python/feast/rag_retriever.py) for RAG functionality:
- `FeastRAGRetriever`: Official Feast RAG retriever with `generate_answer()` method
- `FeastIndex`: Feast index for feature store integration 
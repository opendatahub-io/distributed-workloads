# RAGAS Evaluation Framework

A comprehensive evaluation framework for RAG (Retrieval Augmented Generation) models using RAGAS metrics and ground truth comparison.

## Quick Start

### Installation
```bash
cd examples/kfto-sft-feast-rag/ragas_evaluation
pip install -r requirements.txt
pip install tqdm
```

### Basic Usage

#### Download Natural Questions Dataset (One-time)
```bash
python dataset_loader.py download datasets/natural_questions
```

#### Run Evaluation
```bash
# Use cached dataset
python main.py --use-natural-questions --cache-dir datasets/natural_questions --nq-sample-percentage 0.02

# Use curated questions (no download needed)
python main.py --combinations 2

# Specify output directory
python main.py --combinations 1 2 --output-dir my_results
```

### Command Line Arguments
```bash
python main.py [OPTIONS]

Options:
  --use-natural-questions     Use Natural Questions dataset (~50GB download)
  --nq-sample-percentage FLOAT Sample percentage (default: 0.1)
  --max-questions INT         Maximum questions to evaluate (default: 100)
  --combinations [1|2|3|all]  Select combinations to test (default: all)
  --output-dir PATH           Output directory for results
  --cache-dir PATH            Directory to cache Natural Questions dataset
  --log-dir PATH              Directory for log files
```

## Evaluation Combinations

The framework evaluates 3 model combinations:

1. **Original QE + Original Generator** (baseline)
2. **Original QE + Fine-tuned Generator** (notebook config)
3. **Fine-tuned QE + Fine-tuned Generator** (full pipeline)

## Key Metrics

### RAGAS Metrics
- **Context Recall**: How well the model retrieves relevant context
- **Context Precision**: How precise the retrieved context is

### Ground Truth Metrics
- **Average Accuracy**: Semantic similarity with ground truth answers
- **Exact Match Rate**: Percentage of exact matches
- **Type Compatibility Rate**: How well answers match expected types

### Custom Metrics
- **Average Quality**: Multi-factor quality score
- **Context Relevance**: Key term matching between questions and context
- **Answer Specificity**: How specific vs generic answers are
- **Context Utilization**: How much retrieved context is used in answers

## Detailed Metric Calculations

### RAGAS Framework Metrics (Built-in)

These metrics are calculated using the official RAGAS framework (`ragas.metrics`):

#### Context Recall
**Source**: `ragas.metrics.context_recall`  
**Definition**: Measures how well the model retrieves relevant context that contains the ground truth answer.

**RAGAS Calculation**:
```python
from ragas.metrics import context_recall
# RAGAS automatically calculates this using their internal algorithms
# Our fallback implementation:
gt_words = set(ground_truth.lower().split())
context_words = set(retrieved_context.lower().split())
recall = len(gt_words.intersection(context_words)) / len(gt_words)
```

**Score Range**: 0.0 (no relevant context) to 1.0 (all ground truth terms found)

#### Context Precision
**Source**: `ragas.metrics.context_precision`  
**Definition**: Measures how precise and relevant the retrieved context is to the question.

**RAGAS Calculation**:
```python
from ragas.metrics import context_precision
# RAGAS automatically calculates this using their internal algorithms
# Our fallback implementation:
question_words = set(question.lower().split())
context_words = set(retrieved_context.lower().split())
precision = len(question_words.intersection(context_words)) / len(context_words)
```

**Score Range**: 0.0 (irrelevant context) to 1.0 (highly relevant context)

**Note**: If RAGAS evaluation fails, we fall back to manual calculation using word overlap methods.

### Custom Ground Truth Metrics

These metrics are custom implementations for comparing generated answers with ground truth:

#### Average Accuracy
**Source**: Custom implementation  
**Definition**: Semantic similarity between generated answers and ground truth answers.

**Custom Calculation**:
```python
def semantic_similarity(text1, text2):
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    return len(intersection) / len(union) if union else 0.0

# Average across all questions
```

**Score Range**: 0.0 (no similarity) to 1.0 (identical content)

#### Exact Match Rate
**Source**: Custom implementation  
**Definition**: Percentage of generated answers that exactly match ground truth answers.

**Custom Calculation**:
```python
exact_matches = sum(1 for gen, gt in zip(generated, ground_truth) 
                   if gen.lower().strip() == gt.lower().strip())
exact_match_rate = exact_matches / total_questions
```

**Score Range**: 0.0 (no exact matches) to 1.0 (all exact matches)

#### Type Compatibility Rate
**Source**: Custom implementation  
**Definition**: How well generated answers match expected answer types.

**Custom Calculation**:
```python
def check_type_compatibility(generated, expected_type):
    if expected_type == 'yes_no':
        yes_patterns = ['yes', 'true', 'correct', 'right']
        no_patterns = ['no', 'false', 'incorrect', 'wrong']
        return any(pattern in generated.lower() for pattern in yes_patterns + no_patterns)
    elif expected_type == 'short_text':
        return len(generated.split()) <= 10
    elif expected_type == 'long_text':
        return len(generated.split()) > 5
    return True

# Average across all questions
```

**Score Range**: 0.0 (no type compatibility) to 1.0 (perfect type compatibility)

### Custom Quality Metrics

These metrics are custom implementations for evaluating answer quality and context performance:

#### Average Quality
**Source**: Custom implementation  
**Definition**: Multi-dimensional quality score based on answer length, key term matching, and context utilization.

**Custom Calculation**:
```python
def calculate_quality(answer, question, context):
    base_score = 0.3
    
    # Key term matching (40% weight)
    key_terms = [word for word in question.lower().split() 
                if len(word) > 3 and word not in ['what', 'when', 'where', 'which', 'whose']]
    term_matches = sum(1 for term in key_terms if term in answer.lower())
    term_score = term_matches / len(key_terms) if key_terms else 0.5
    base_score += term_score * 0.4
    
    # Context utilization (30% weight)
    if context:
        context_words = set(context.lower().split())
        answer_words = set(answer.lower().split())
        context_utilization = len(answer_words.intersection(context_words)) / len(answer_words) if answer_words else 0
        base_score += context_utilization * 0.3
    
    return min(base_score, 1.0)
```

**Score Range**: 0.0 (poor quality) to 1.0 (excellent quality)

#### Context Relevance
**Source**: Custom implementation  
**Definition**: Key term matching between questions and retrieved context.

**Custom Calculation**:
```python
def context_relevance(question, context):
    key_terms = [word for word in question.lower().split() 
                if len(word) > 3 and word not in ['what', 'when', 'where', 'which', 'whose']]
    term_matches = sum(1 for term in key_terms if term in context.lower())
    relevance = term_matches / len(key_terms) if key_terms else 0.5
    
    # Bonus for exact phrase matches
    if any(term in context.lower() for term in key_terms):
        relevance += 0.2
    
    return min(relevance, 1.0)
```

**Score Range**: 0.0 (irrelevant context) to 1.0 (highly relevant context)

#### Answer Specificity
**Source**: Custom implementation  
**Definition**: How specific vs generic an answer is based on lexical diversity.

**Custom Calculation**:
```python
def answer_specificity(answer):
    words = answer.lower().split()
    unique_words = set(words)
    specificity = len(unique_words) / len(words) if words else 0.0
    
    # Bonus for technical terms
    technical_terms = ['temperature', 'pressure', 'chemical', 'molecular', 'atomic', 'biological']
    technical_bonus = sum(1 for word in words if word in technical_terms) * 0.1
    
    return min(specificity + technical_bonus, 1.0)
```

**Score Range**: 0.0 (very generic) to 1.0 (highly specific)

#### Context Utilization
**Source**: Custom implementation  
**Definition**: How much of the retrieved context is actually used in the generated answer.

**Custom Calculation**:
```python
def context_utilization(answer, context):
    context_words = set(context.lower().split())
    answer_words = set(answer.lower().split())
    
    if not context_words:
        return 0.0
    
    utilization = len(answer_words.intersection(context_words)) / len(context_words)
    return utilization
```

**Score Range**: 0.0 (no context used) to 1.0 (all context used)

### Custom Additional Metrics

These metrics are custom implementations for evaluating response characteristics:

#### Error Rate
**Source**: Custom implementation  
**Definition**: Percentage of responses that failed to generate or returned error messages.

**Custom Calculation**:
```python
error_count = sum(1 for answer in answers if answer.startswith("Error:") or "error" in answer.lower())
error_rate = error_count / total_questions
```

**Score Range**: 0.0 (no errors) to 1.0 (all responses failed)

#### Average Answer Length
**Source**: Custom implementation  
**Definition**: Average number of words in generated answers.

**Custom Calculation**:
```python
answer_lengths = [len(answer.split()) for answer in answers]
avg_answer_length = sum(answer_lengths) / len(answer_lengths)
```

**Score Range**: Varies based on dataset and model behavior

#### Context Coverage
**Source**: Custom implementation  
**Definition**: Amount of information available in retrieved context.

**Custom Calculation**:
```python
def context_coverage(context):
    total_length = sum(len(c) for c in context)
    if total_length < 50:
        return 0.2
    elif total_length < 200:
        return 0.5
    elif total_length < 500:
        return 0.8
    else:
        return 1.0
```

**Score Range**: 0.0 (minimal context) to 1.0 (comprehensive context)

### Metric Interpretation Guidelines

#### Score Ranges
- **0.0-0.3**: Poor performance
- **0.4-0.6**: Moderate performance  
- **0.7-0.8**: Good performance
- **0.9-1.0**: Excellent performance

#### Key Insights
- **High Context Utilization + Low Specificity**: Model copies context without adding value
- **Low Context Utilization + High Specificity**: Model generates original content but ignores retrieval
- **High Ground Truth Accuracy + High Type Compatibility**: Model generates answers that match expected format
- **Low Error Rate + High Quality**: Robust model performance

## Output Files

- **JSON Results**: `comprehensive_rag_evaluation_YYYYMMDD_HHMMSS.json`
- **Answer Comparison**: `answer_comparison_YYYYMMDD_HHMMSS.txt`
- **Charts**: `charts_YYYYMMDD_HHMMSS/` with visual comparisons
- **Log Files**: `ragas_evaluation_YYYYMMDD_HHMMSS.log`

## Troubleshooting

### Common Issues
1. **Dataset Download**: Use `--cache-dir` to avoid re-downloading
2. **Memory Issues**: Reduce `--max-questions` or use CPU
3. **Disk Space**: Use smaller sample percentages or curated questions
4. **Log Files**: Use `--log-dir` for custom log directory

### Dataset Commands
```bash
# Download dataset
python dataset_loader.py download datasets/natural_questions

# Test dataset
python dataset_loader.py test datasets/natural_questions
```

## Requirements

- Python 3.8+
- PyTorch, Transformers, RAGAS, Feast
- Matplotlib, Seaborn, Pandas, NumPy
- Datasets (Hugging Face), tqdm 
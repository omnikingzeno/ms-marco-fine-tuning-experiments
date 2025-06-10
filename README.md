# MS-MARCO Fine-tuning Experiments

This repository contains experiments for fine-tuning Sentence-BERT (SBERT) models on the MS-MARCO dataset using different strategies and approaches. The project evaluates various fine-tuning techniques including standard fine-tuning, LoRA (Low-Rank Adaptation), and hard negative mining to improve passage retrieval performance.

## Project Overview

The goal of this project is to systematically evaluate different fine-tuning strategies for sentence transformers on the MS-MARCO passage ranking dataset. We compare:

1. **Base Model**: `sentence-transformers/all-MiniLM-L6-v2` (no fine-tuning)
2. **Standard Fine-tuning**: Full model fine-tuning with triplet loss
3. **LoRA Fine-tuning**: Parameter-efficient fine-tuning using Low-Rank Adaptation
4. **Hard Negative Mining**: Fine-tuning with hard negatives from base model predictions

## Dataset

The experiments use the MS-MARCO passage ranking dataset:
- **Collection**: ~8.8M passages
- **Training Queries**: ~500K queries with relevance judgments
- **Dev Queries**: ~6.8K queries for evaluation
- **Hard Negatives**: Generated from top-200 base model predictions (ranks 50-200)

## Key Scripts and Components

### Data Processing
- **`process_train_queries.py`**: Generates top-200 similar passages for training queries using base model
- **`random_sample.py`**: Creates hard negative samples from base model predictions
- **`train_data_creator.py`**: Builds training triples (query, positive, hard negative) for fine-tuning
- **`visualization_data_creator.py`**: Creates data subset for embedding visualizations

### Model Training
- **`finetune.py`**: Main fine-tuning script supporting both standard and LoRA approaches
  - *Note: Contains commented sections for switching between standard fine-tuning and LoRA*
  - *Modify dataset paths and model configurations as needed*

### Evaluation
- **`eval_model.py`**: Evaluates fine-tuned models on dev set with efficient chunked processing
- **`inference_times.py`**: Benchmarks inference speed for standard models
- **`inference_times_lora.py`**: Benchmarks inference speed for LoRA models
- **`mrr_calculator.py`**: Calculates MRR@10 and MRR@100 metrics from results

### Infrastructure
- **`modal_offload.py`**: Modal.com functions for cloud-based training and evaluation
  - *Contains multiple function definitions - uncomment specific functions as needed*

## Fine-tuning Strategies

### 1. Standard Fine-tuning
- Full parameter updates using triplet loss
- Training data: Query-positive-negative triples
- Batch size: 1024, Epochs: 5, Learning rate: 2e-5

### 2. LoRA Fine-tuning
- Parameter-efficient approach updating only low-rank adapters
- LoRA rank: 16, Alpha: 32, Target modules: ["query", "value"]
- Significantly reduced memory requirements and training time

### 3. Hard Negative Mining
- Uses base model predictions (ranks 50-200) as hard negatives
- More challenging training examples compared to random negatives
- Improves model's ability to distinguish between relevant and irrelevant passages

## Usage Instructions

### 1. Data Preparation
```bash
# Generate base model predictions for training queries
python process_train_queries.py \
  --model sentence-transformers/all-MiniLM-L6-v2 \
  --filtered-queries filtered_queries.train.tsv \
  --collection collection.tsv \
  --output-dir ./results

# Create hard negative samples
python random_sample.py

# Build training triples
python train_data_creator.py \
  --filtered-queries filtered_queries.train.tsv \
  --qrels sorted_qrels.train.tsv \
  --random-samples random_samples.tsv \
  --collection collection.tsv \
  --output-dir ./datasets
```

### 2. Model Fine-tuning
```bash
# Standard fine-tuning (modify finetune.py configuration)
python finetune.py

# LoRA fine-tuning (uncomment LoRA sections in finetune.py)
python finetune.py
```

### 3. Model Evaluation
```bash
# Evaluate fine-tuned model
python eval_model.py \
  --model your-finetuned-model \
  --filtered-queries filtered_queries.dev.tsv \
  --collection collection.tsv \
  --output-dir ./evaluation-results

# Calculate MRR metrics
python mrr_calculator.py
```

### 4. Inference Benchmarking
```bash
# Benchmark standard model
python inference_times.py \
  --model sentence-transformers/all-MiniLM-L6-v2 \
  --filtered-queries queries.eval.tsv \
  --collection collection.tsv

# Benchmark LoRA model
python inference_times_lora.py \
  --model path-to-lora-model \
  --filtered-queries queries.eval.tsv \
  --collection collection.tsv
```

## Code Flexibility Notes

Several scripts contain commented code sections to support different experimental configurations:

- **`finetune.py`**: Contains both standard fine-tuning and LoRA implementations
- **`modal_offload.py`**: Multiple cloud functions - activate specific ones as needed
- **Training scripts**: Configurable parameters for different dataset sizes and model variants

Uncomment and modify these sections based on your specific experimental requirements.

## Performance Optimizations

The codebase includes several optimizations for large-scale processing:

- **Chunked Processing**: Collection processed in configurable chunks to manage memory
- **GPU Acceleration**: CUDA support for embedding computation and similarity calculations
- **Caching System**: Embeddings cached to disk to avoid recomputation
- **Efficient Data Formats**: Parquet files for intermediate results
- **Memory Management**: Aggressive cleanup and garbage collection

## Results Structure

Evaluation outputs include:
- **WANDB train loss plots**: Query-passage rankings with scores
- **UMAP Visualizations**: Experimental parameters and statistics
- **Timing Data**: Inference speed benchmarks
- **MRR Metrics**: MRR@10 and MRR@100 scores

## Requirements

Key dependencies:
- `sentence-transformers==4.1.0`
- `torch` (with CUDA support recommended)
- `transformers==4.41.0`
- `peft>=0.5.0` (for LoRA)
- `pandas`, `numpy`, `tqdm`
- `dask` (for large-scale data processing)
- `wandb` (for experiment tracking)

## Model Outputs

Trained models are available on Hugging Face Hub:
- `manupande21/all-MiniLM-L6-v2-finetuned-triplets`
- `manupande21/all-MiniLM-L6-v2-finetuned-triples_hard_negatives`
- `manupande21/all-MiniLM-L6-v2-LoRA-finetuned-1M`
- `manupande21/all-MiniLM-L6-v2-LoRA-finetuned-triplets_hard_negatives`

## License

This project is for research and educational purposes. Please ensure compliance with MS-MARCO dataset terms and conditions.

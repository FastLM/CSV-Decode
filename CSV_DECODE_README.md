# CSV-Decode: Certifiable Sub-Vocabulary Decoding

This repository implements **CSV-Decode**, a novel approach for efficient large language model inference using geometric upper bounds to construct small sub-vocabularies for each decoding step, enabling efficient sparse computation while maintaining dual correctness guarantees.

## Overview

CSV-Decode addresses the output layer bottleneck in large language models by:

1. **Geometric Upper Bounds**: Using centroid-plus-radius bounds to construct provably correct sub-vocabularies
2. **Exact Certification**: Providing exact top-k certification and ε-certified softmax approximations
3. **Adaptive Online Algorithm**: Dynamically expanding sub-vocabularies based on certification criteria
4. **Optimized Implementation**: Featuring sparse GEMV kernels, multi-GPU sharding, and CUDA Graph optimization

## Key Features

- **2-3x Speedup**: Achieves significant speedup over full vocabulary decoding
- **Provable Correctness**: Maintains exact top-k results and ε-bounded softmax approximations
- **Low Fallback Rates**: Typically <2% fallback to full vocabulary computation
- **Universal Compatibility**: Works with any LLM architecture without retraining
- **Energy Efficient**: Reduces energy consumption by ~52%

## Installation

1. Install dependencies:
```bash
sh install.sh
```

2. Update model paths in `src/util.py` (lines 31-38 and 49)

## Quick Start

### Basic Usage

Run CSV-Decode evaluation:
```bash
python benchmark/eval_csv_decode.py \
    --draft_model codellama-7b \
    --target_model codellama-70b \
    --eval_mode csv_decode \
    --use_csv_decode \
    --csv_num_clusters 2000 \
    --csv_epsilon 0.05
```

### Using the Evaluation Script

```bash
sh scripts/run_csv_decode.sh \
    --model codellama-7b \
    --target_model codellama-70b \
    --num_samples 5 \
    --csv_num_clusters 2000 \
    --csv_epsilon 0.05
```

## Architecture

### Core Components

1. **VocabularyClusterer** (`src/csv_decode.py`): Offline K-means clustering of vocabulary embeddings
2. **GeometricBounds**: Cauchy-Schwarz inequality derivations for cluster upper bounds
3. **CertificationMechanisms**: Top-k and ε-certified softmax approximation
4. **CSVDecodeEngine**: Main online algorithm implementation
5. **OptimizedSparseGEMV** (`src/sparse_gemv.py`): Efficient sparse matrix-vector multiplication

### Integration

CSV-Decode is integrated into the existing engine framework (`src/engine.py`) and can be used alongside other decoding methods:

- **csv_decode**: Pure CSV-Decode acceleration
- **small/large**: Baseline autoregressive decoding
- **sd/para_sd**: Speculative decoding methods

## Configuration Parameters

### CSV-Decode Specific Arguments

- `--use_csv_decode`: Enable CSV-Decode acceleration
- `--csv_num_clusters`: Number of clusters (default: 0.015 × vocab_size)
- `--csv_epsilon`: Softmax approximation tolerance (default: 0.05)
- `--csv_top_k`: Top-k for certification (default: 10)
- `--embedding_dim`: Embedding dimension (default: 4096)

### Performance Tuning

- **Cluster Count**: Higher counts provide tighter bounds but increase overhead
- **Epsilon**: Lower values provide better approximation but may increase fallback rates
- **Top-K**: Higher values improve certification but may reduce speedup

## Evaluation Results

Based on the paper's experimental results, CSV-Decode achieves:

- **2.67-4.95x speedup** over autoregressive baseline
- **1.16-1.27x improvement** over state-of-the-art speculative decoding
- **99.3% quality retention** with <2% fallback rates
- **52% energy reduction** compared to baseline

### Model Support

Tested on:
- Llama-2/3 (7B, 70B)
- CodeLlama (7B, 13B, 34B, 70B)
- Qwen2.5 (7B, 14B, 72B)
- DeepSeek-Coder (1.3B, 6.7B, 33B)
- Mistral-7B
- Yi-34B

## Implementation Details

### Geometric Bounds

The core insight uses Cauchy-Schwarz inequality:

```
ℓi(t) = ⟨Wi, ht⟩ + bi ≤ ⟨μc, ht⟩ + Rc∥ht∥2 + max_j∈c bj
```

Where:
- `μc`: Cluster centroid
- `Rc`: Cluster radius
- `ht`: Hidden state at step t

### Certification Mechanisms

1. **Top-K Certification**: Guarantees exact top-k results
2. **ε-Certified Softmax**: Bounds total variation distance ≤ ε

### Sparse GEMV Optimization

- **COO Format**: Row gathering + dense GEMV
- **Multi-level Tiling**: Optimized memory access patterns
- **Tensor Core Integration**: Mixed precision computation
- **CUDA Graph**: Reduced kernel launch overhead

## File Structure

```
CSV-Decode/
├── src/
│   ├── csv_decode.py          # Core CSV-Decode implementation
│   ├── sparse_gemv.py         # Optimized sparse GEMV kernels
│   ├── engine.py              # Main engine with CSV-Decode integration
│   ├── util.py                # Utilities and argument parsing
│   ├── kvcache.py             # KV cache implementation
│   └── kvcache4RC.py          # RC-specific KV cache
├── benchmark/
│   ├── eval_csv_decode.py     # CSV-Decode evaluation script
│   ├── eval_humaneval.py      # HumanEval evaluation
│   ├── eval_gsm8k.py          # GSM8K evaluation
│   ├── eval_mgsm.py           # MGSM evaluation
│   └── eval_mt_bench.py       # MT-Bench evaluation
├── scripts/
│   ├── run_csv_decode.sh      # CSV-Decode evaluation script
│   ├── run_ar.sh              # Autoregressive baseline
│   ├── run_sd.sh              # Speculative decoding
│   └── run_para_sd.sh         # Parallel speculative decoding
├── data/                      # Evaluation datasets
├── applications.py            # Demo application with UI
└── README.md                  # This file
```

## Usage Examples

### 1. Basic CSV-Decode Evaluation

```python
from src.engine import Decoding
from src.util import parse_arguments

# Parse arguments with CSV-Decode enabled
args = parse_arguments()
args.use_csv_decode = True
args.eval_mode = "csv_decode"

# Create evaluation instance
evaluator = Decoding(args)
evaluator.load_model()
evaluator.load_tokenizer()

# Run CSV-Decode
output = evaluator.csv_decode_sampling(input_ids)
```

### 2. Custom Configuration

```python
from src.csv_decode import create_csv_decode_config, CSVDecodeEngine

# Create custom configuration
config = create_csv_decode_config(
    vocab_size=128000,
    embedding_dim=4096,
    num_clusters=2000,
    epsilon=0.05
)

# Initialize engine
engine = CSVDecodeEngine(config)
engine.initialize_clusters(weight_matrix, bias_vector)

# Perform decoding step
logits, sub_vocab, is_certified, cert_type = engine.decode_step(
    hidden_state, weight_matrix, bias_vector
)
```

### 3. Performance Benchmarking

```python
from src.sparse_gemv import create_optimized_gemv

# Create optimized GEMV instance
gemv = create_optimized_gemv(device)

# Benchmark different kernels
results = gemv.benchmark_kernels(
    weight_matrix, hidden_state, token_indices, num_iterations=100
)

print(f"COO Kernel: {results['coo_time']:.3f}ms")
print(f"Tiled Kernel: {results['tiled_time']:.3f}ms")
print(f"Tensor Core: {results['tensor_core_time']:.3f}ms")
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce `csv_num_clusters` or `max_sub_vocab_size`
2. **Low Certification Rate**: Increase `csv_num_clusters` or adjust `csv_epsilon`
3. **High Fallback Rate**: Check cluster quality or reduce `csv_top_k`

### Performance Optimization

1. **Cluster Quality**: Ensure good vocabulary clustering with appropriate cluster count
2. **Memory Access**: Use optimized sparse GEMV kernels for your hardware
3. **Batch Processing**: Process multiple sequences together for better GPU utilization

## Citation

If you use CSV-Decode in your research, please cite:

```bibtex
@article{liu2024csvdecode,
  title={CSV-Decode: Certifiable Sub-Vocabulary Decoding for Efficient Large Language Model Inference},
  author={Liu, Dong and Yu, Yanxuan and Lengerich, Ben},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- Based on the paper "CSV-Decode: Certifiable Sub-Vocabulary Decoding for Efficient Large Language Model Inference"
- Built on top of the PEARL framework for parallel speculative decoding
- Uses scikit-learn for vocabulary clustering
- Optimized for NVIDIA A100/H100 GPUs with Tensor Core support

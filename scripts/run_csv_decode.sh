#!/bin/bash

# CSV-Decode Evaluation Script
# This script runs comprehensive evaluation of CSV-Decode performance

set -e

# Default parameters
MODEL="codellama-7b"
TARGET_MODEL="codellama-70b"
NUM_SAMPLES=5
MAX_TOKENS=512
TEMP=0.2
TOP_P=0.95
SEED=42

# CSV-Decode specific parameters
CSV_NUM_CLUSTERS=2000
CSV_EPSILON=0.05
CSV_TOP_K=10
EMBEDDING_DIM=4096

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --target_model)
            TARGET_MODEL="$2"
            shift 2
            ;;
        --num_samples)
            NUM_SAMPLES="$2"
            shift 2
            ;;
        --max_tokens)
            MAX_TOKENS="$2"
            shift 2
            ;;
        --csv_num_clusters)
            CSV_NUM_CLUSTERS="$2"
            shift 2
            ;;
        --csv_epsilon)
            CSV_EPSILON="$2"
            shift 2
            ;;
        --csv_top_k)
            CSV_TOP_K="$2"
            shift 2
            ;;
        --embedding_dim)
            EMBEDDING_DIM="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --model MODEL              Draft model name (default: codellama-7b)"
            echo "  --target_model MODEL       Target model name (default: codellama-70b)"
            echo "  --num_samples N            Number of samples per task (default: 5)"
            echo "  --max_tokens N             Maximum tokens to generate (default: 512)"
            echo "  --csv_num_clusters N       Number of clusters for CSV-Decode (default: 2000)"
            echo "  --csv_epsilon FLOAT        Softmax approximation tolerance (default: 0.05)"
            echo "  --csv_top_k N             Top-k for CSV-Decode certification (default: 10)"
            echo "  --embedding_dim N         Embedding dimension (default: 4096)"
            echo "  --help                    Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "CSV-Decode Evaluation"
echo "=========================================="
echo "Model: $MODEL"
echo "Target Model: $TARGET_MODEL"
echo "Number of Samples: $NUM_SAMPLES"
echo "Max Tokens: $MAX_TOKENS"
echo "CSV Num Clusters: $CSV_NUM_CLUSTERS"
echo "CSV Epsilon: $CSV_EPSILON"
echo "CSV Top-K: $CSV_TOP_K"
echo "Embedding Dim: $EMBEDDING_DIM"
echo "=========================================="

# Create experiment directory
EXP_NAME="csv_decode_${MODEL}_${TARGET_MODEL}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "exp/$EXP_NAME"

# Run CSV-Decode evaluation
echo "Running CSV-Decode evaluation..."
python benchmark/eval_csv_decode.py \
    --draft_model "$MODEL" \
    --target_model "$TARGET_MODEL" \
    --eval_mode csv_decode \
    --use_csv_decode \
    --num_samples_per_task "$NUM_SAMPLES" \
    --max_tokens "$MAX_TOKENS" \
    --temp "$TEMP" \
    --top_p "$TOP_P" \
    --seed "$SEED" \
    --csv_num_clusters "$CSV_NUM_CLUSTERS" \
    --csv_epsilon "$CSV_EPSILON" \
    --csv_top_k "$CSV_TOP_K" \
    --embedding_dim "$EMBEDDING_DIM" \
    --exp_name "$EXP_NAME"

echo "=========================================="
echo "CSV-Decode evaluation completed!"
echo "Results saved to: exp/$EXP_NAME/"
echo "=========================================="

# Run comparison with baseline autoregressive
echo "Running baseline autoregressive evaluation for comparison..."
python benchmark/eval_csv_decode.py \
    --draft_model "$MODEL" \
    --target_model "$TARGET_MODEL" \
    --eval_mode large \
    --num_samples_per_task "$NUM_SAMPLES" \
    --max_tokens "$MAX_TOKENS" \
    --temp "$TEMP" \
    --top_p "$TOP_P" \
    --seed "$SEED" \
    --exp_name "${EXP_NAME}_baseline"

echo "=========================================="
echo "Baseline evaluation completed!"
echo "Results saved to: exp/${EXP_NAME}_baseline/"
echo "=========================================="

# Generate comparison report
echo "Generating comparison report..."
python -c "
import json
import os

# Load results
csv_results_file = 'exp/$EXP_NAME/csv_decode_results.json'
baseline_results_file = 'exp/${EXP_NAME}_baseline/csv_decode_results.json'

if os.path.exists(csv_results_file) and os.path.exists(baseline_results_file):
    with open(csv_results_file, 'r') as f:
        csv_results = json.load(f)
    
    with open(baseline_results_file, 'r') as f:
        baseline_results = json.load(f)
    
    print('\\n' + '='*60)
    print('CSV-Decode vs Baseline Comparison Report')
    print('='*60)
    
    if 'comparison' in csv_results:
        comp = csv_results['comparison']
        print(f'Throughput Speedup: {comp[\"throughput_speedup\"]:.2f}x')
        print(f'Latency Reduction: {comp[\"latency_reduction\"]:.1%}')
        print(f'Quality Retention: {comp[\"quality_retention\"]:.1%}')
        print(f'Certification Rate: {comp[\"avg_certification_rate\"]:.1%}')
        print(f'Sub-vocab Ratio: {comp[\"avg_sub_vocab_ratio\"]:.1%}')
        print(f'Fallback Rate: {comp[\"avg_fallback_rate\"]:.1%}')
    
    print('\\nDetailed Performance Metrics:')
    csv_throughput = sum(csv_results['csv_decode']['throughput']) / len(csv_results['csv_decode']['throughput'])
    baseline_throughput = sum(baseline_results['baseline']['throughput']) / len(baseline_results['baseline']['throughput'])
    
    print(f'CSV-Decode Avg Throughput: {csv_throughput:.2f} tokens/s')
    print(f'Baseline Avg Throughput: {baseline_throughput:.2f} tokens/s')
    print(f'Speedup: {csv_throughput/baseline_throughput:.2f}x')
    
    print('='*60)
else:
    print('Results files not found. Please check the evaluation completed successfully.')
"

echo "Evaluation completed successfully!"

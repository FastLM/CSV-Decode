<div align="center"><h1>&nbsp;CSV-Decode: Certifiable Sub-Vocabulary Decoding for Efficient Large Language Model Inference</h1></div>

*News* 🔥
- [2025/10] **CSV-Decode**: Novel approach for efficient LLM inference using geometric upper bounds
- [2025/09] **2-4.95x speedup** over autoregressive baseline with provable correctness guarantees
- [2025/09] **compatibility** with LLM architecture without retraining



<br>

> TL; DR: we introduce **CSV-Decode** (Certifiable Sub-Vocabulary Decoding) to address the output layer bottleneck in Large Language Models (LLMs). CSV-Decode uses geometric upper bounds to construct small sub-vocabularies for each decoding step, enabling efficient sparse computation while maintaining dual correctness guarantees. In summary, our CSV-Decode is:
>
> - &#128293; **2.67-4.95x speedup** over autoregressive baseline across multiple models and tasks
> - **99.3% quality retention** with <2% fallback rates to full vocabulary computation
> - **52% energy reduction** compared to baseline inference
> - **Provably correct**: Exact top-k certification and ε-certified softmax approximations
> - **Training-free**: Works with any pre-trained LLM without retraining
> - **Universal compatibility**: Supports all major LLM architectures (Llama, CodeLlama, Qwen, DeepSeek, etc.)
> - **Advanced optimizations**: Sparse GEMV kernels, multi-GPU sharding, and CUDA Graph integration

<br>

<!-- Using HTML to center the abstract -->

---

<br>

<div class="columns is-centered has-text-centered">
    <div class="column is-four-fifths">
        <h2>Overview of CSV-Decode</h2>
        <div class="content has-text-justified">
        </div>
    </div>
</div>

**CSV-Decode** addresses the fundamental bottleneck in LLM inference: the expensive output layer computation over large vocabularies. Our key insight is that for most decoding steps, only a small subset of the vocabulary actually contributes meaningfully to the final output distribution.

### Core Innovation: Geometric Upper Bounds

CSV-Decode leverages geometric reasoning to identify which tokens can be safely omitted from computation:

1. **Vocabulary Clustering**: Offline K-means clustering groups semantically similar tokens
2. **Cauchy-Schwarz Bounds**: Derives tight upper bounds on logits for entire clusters
3. **Certification Mechanisms**: Provides exact top-k certification and ε-certified softmax approximations
4. **Adaptive Algorithm**: Dynamically expands sub-vocabularies based on certification criteria

### Mathematical Foundation

For any token i in cluster c, we bound the logit using Cauchy-Schwarz inequality:

```
ℓi(t) = ⟨Wi, ht⟩ + bi ≤ ⟨μc, ht⟩ + Rc∥ht∥2 + max_j∈c bj
```

Where:
- `μc`: Cluster centroid
- `Rc`: Cluster radius  
- `ht`: Hidden state at step t

This enables us to skip entire clusters without computing individual token logits, while maintaining provable correctness guarantees.


## preparation

Follow the instructions below to prepare for reproducing the results in the paper.

1. experimental environment: `sh install.sh` will install the necessary packages in the project.
2. code changes: changes the code `src/util.py` line 31-38 and line 49, to fill in your model paths and data paths.



## reproduction

All the running scripts, including scripts for auto-regress decoding, vanilla speculative decoding, parallel speculative decoding, comparison, ablation studies and case studies. These scripts can be directly executed for reproduction.

```shell
sh scripts/run_para_sd.sh
```



## Examples

You can try CSV-Decode with the following commands:

**Basic CSV-Decode evaluation:**
```shell
python benchmark/eval_csv_decode.py \
    --eval_mode csv_decode \
    --use_csv_decode \
    --csv_num_clusters 2000 \
    --csv_epsilon 0.05 \
    --draft_model codellama-7b \
    --target_model codellama-70b
```

**Automated CSV-Decode evaluation script:**
```shell
sh scripts/run_csv_decode.sh \
    --model codellama-7b \
    --target_model codellama-70b \
    --num_samples 5 \
    --csv_num_clusters 2000 \
    --csv_epsilon 0.05
```

**CSV-Decode with custom parameters:**
```shell
python benchmark/eval_csv_decode.py \
    --eval_mode csv_decode \
    --use_csv_decode \
    --csv_num_clusters 3000 \
    --csv_epsilon 0.01 \
    --csv_top_k 20 \
    --embedding_dim 4096 \
    --draft_model codellama-7b \
    --target_model codellama-70b
```

## With UI
We have provided a suggested web interface, which you can use by running the following command. 

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes 2 applications.py --eval_mode csv_decode --use_csv_decode -n 1  -e applications --draft_model codellama-7b --target_model codellama-70b --max_tokens 1024 --temp 0
```

## CSV-Decode Configuration

### Key Parameters

- `--use_csv_decode`: Enable CSV-Decode acceleration
- `--csv_num_clusters`: Number of clusters (default: 0.015 × vocab_size)
- `--csv_epsilon`: Softmax approximation tolerance (default: 0.05)
- `--csv_top_k`: Top-k for certification (default: 10)
- `--embedding_dim`: Embedding dimension (default: 4096)

### Performance Tuning

- **Cluster Count**: Higher counts provide tighter bounds but increase overhead
- **Epsilon**: Lower values provide better approximation but may increase fallback rates
- **Top-K**: Higher values improve certification but may reduce speedup

### Supported Models

CSV-Decode has been tested on:
- Llama-2/3 (7B, 70B)
- CodeLlama (7B, 13B, 34B, 70B)
- Qwen2.5 (7B, 14B, 72B)
- DeepSeek-Coder (1.3B, 6.7B, 33B)
- Mistral-7B
- Yi-34B

## Implementation Notes (Reproducibility)

1. **Softmax variants (Hierarchical & Adaptive Softmax)**  
   The paper compares CSV-Decode to Hierarchical Softmax [Mikolov et al.] and Adaptive Softmax [Grave et al.]. Implementations of these baselines are in **`src/softmax_variants.py`**: `HierarchicalSoftmax` and `AdaptiveSoftmax`, with factory helpers `build_hierarchical_softmax` and `build_adaptive_softmax`. They are provided for ablation and Table II–style comparisons; CSV-Decode itself does not use them.

2. **Logit computation (no simulation)**  
   Logits in the decode step are computed from the real output layer: **ℓ_i = ⟨W_i, h_t⟩ + b_i** (paper Eq. 2). In `src/csv_decode.py`, `expand_sub_vocabulary` takes `embedding_matrix`, `bias_vector`, and `hidden_state`, and uses **sparse GEMV** (`gathered_rows = W[indices]; logits = gathered_rows @ h_t + bias[indices]`). There is no placeholder or random logit simulation; the previous `torch.randn` placeholder has been removed.

3. **Sparse GEMV kernels and CUDA Graph**  
   - **Sparse GEMV**: Implemented in **`src/sparse_gemv.py`** (COO row-gather + GEMV, multi-level tiling, Tensor Core path). The inference loop in **`src/csv_decode.py`** uses this when you pass a `sparse_gemv_backend` to `CSVDecodeEngine(..., sparse_gemv_backend=create_optimized_gemv(device))`; otherwise it uses the same formula via PyTorch `matmul` and indexing.  
   - **CUDA Graph**: **`src/cuda_graph.py`** provides `CUDAGraphCapture` for capturing/replaying fixed-shape sparse GEMV to reduce kernel launch overhead (paper Section IV). Integration is optional; use `use_cuda_graph_for_sparse_gemv` or wire `CUDAGraphCapture` into your decode loop for maximum throughput.

## FAQ

1. AttributeError: 'list' object has no attribute 'get_seq_length'.

In latest `transformers` (version >= 4.49.0), all the past_key_values are class of `DynamicCache` instead of `tuple`. Hence you should change the error line of code from `past_key_values[0][0].shape[2]` to `past_key_values.get_seq_length()`. We have fixed some bugs within the code. If you find any bug, feel free to raise an issue.

2. Unexpected generations, such as meaningless text.

This issue may be directly caused due to precision overflow. You can add `.to(torch.float32)` to solve this issue. (Such as Line 187 of `src/engine.py`)

3. CSV-Decode: CUDA Out of Memory.

This issue may occur with large vocabulary sizes. Try reducing `--csv_num_clusters` or `--max_sub_vocab_size` in the configuration.

4. CSV-Decode: Low Certification Rate.

If certification rates are low, try increasing `--csv_num_clusters` or adjusting `--csv_epsilon` to a higher value (e.g., 0.1).

5. CSV-Decode: High Fallback Rate.

High fallback rates indicate loose bounds. Check cluster quality or reduce `--csv_top_k` for better performance.

6. CSV-Decode Performance Optimization.

For optimal performance:
- Ensure good vocabulary clustering with appropriate cluster count
- Use optimized sparse GEMV kernels for your hardware
- Process multiple sequences together for better GPU utilization


<div class="columns is-centered has-text-centered">
    <div class="column is-four-fifths">
        <h2>Citation</h2>
        <div class="content has-text-justified">
        </div>
    </div>
</div>


If you find our work useful your research, please cite our paper:

```
@article{liu2025csvdecode,
      title={CSV-Decode: Certifiable Sub-Vocabulary Decoding for Efficient Large Language Model Inference}, 
      author={Dong Liu and Yanxuan Yu and Ben Lengerich},
      year={2025},
      eprint={2511.21702},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2511.21702}, 
}
```

## Acknowledgments

This codebase is built upon the **PEARL** framework. We gratefully acknowledge the PEARL team for providing the foundation that made this CSV-Decode implementation possible.

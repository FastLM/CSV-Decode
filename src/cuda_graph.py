"""
CUDA Graph integration for CSV-Decode (Section IV of paper).

CUDA Graphs capture a sequence of kernels and replay them with reduced launch
overhead (T_launch ~3-5µs -> T_graph <1µs). This module provides capture/replay
for the sparse GEMV and bound-computation steps used in the CSV-Decode inference loop.
"""

import torch
from typing import Optional, List, Callable, Any


class CUDAGraphCapture:
    """
    Capture and replay a fixed-shape CSV-Decode sub-step (e.g. one sparse GEMV)
    to reduce kernel launch overhead. Use when sub-vocabulary size is stable.
    """

    def __init__(
        self,
        device: torch.device,
        embedding_dim: int,
        max_sub_vocab_chunk: int = 4096,
    ):
        self.device = device
        self.embedding_dim = embedding_dim
        self.max_sub_vocab_chunk = max_sub_vocab_chunk
        self._graph: Optional[torch.cuda.CUDAGraph] = None
        self._static_weight: Optional[torch.Tensor] = None
        self._static_hidden: Optional[torch.Tensor] = None
        self._static_bias: Optional[torch.Tensor] = None
        self._static_out: Optional[torch.Tensor] = None
        self._captured = False

    def capture(
        self,
        weight_chunk: torch.Tensor,
        hidden: torch.Tensor,
        bias_chunk: torch.Tensor,
        out: torch.Tensor,
    ) -> None:
        """Capture the GEMV: out = weight_chunk @ hidden + bias_chunk."""
        if self._graph is not None:
            return
        self._graph = torch.cuda.CUDAGraph()
        self._static_weight = weight_chunk
        self._static_hidden = hidden
        self._static_bias = bias_chunk
        self._static_out = out

        with torch.cuda.graph(self._graph):
            self._static_out.copy_(
                torch.matmul(self._static_weight, self._static_hidden) + self._static_bias
            )
        self._captured = True

    def replay(
        self,
        weight_chunk: torch.Tensor,
        hidden: torch.Tensor,
        bias_chunk: torch.Tensor,
    ) -> torch.Tensor:
        """Replay the captured graph; copies inputs into static buffers and replays."""
        if not self._captured or self._graph is None or self._static_weight is None or self._static_out is None:
            return torch.matmul(weight_chunk, hidden) + bias_chunk
        self._static_weight.copy_(weight_chunk)
        if self._static_hidden is not None:
            self._static_hidden.copy_(hidden)
        if self._static_bias is not None:
            self._static_bias.copy_(bias_chunk)
        self._graph.replay()
        return self._static_out.clone()


def capture_csv_decode_step_graph(
    device: torch.device,
    vocab_size: int,
    embedding_dim: int,
    num_clusters: int,
    max_sub_vocab_size: int,
) -> Optional[torch.cuda.CUDAGraph]:
    """
    Optional: capture a full decode step when shapes are fixed (e.g. benchmark mode).
    Returns a CUDAGraph if capture is successful; otherwise None.
    CSV-Decode's dynamic cluster expansion means the main loop is often not
    fully capturable; this helper is for the fixed-shape sparse GEMV portion.
    """
    if not torch.cuda.is_available():
        return None
    try:
        # Placeholder: actual capture would run one decode step with fixed inputs
        return None
    except Exception:
        return None


def use_cuda_graph_for_sparse_gemv(
    weight_matrix: torch.Tensor,
    hidden_state: torch.Tensor,
    token_indices: List[int],
    bias_vector: Optional[torch.Tensor],
    graph_capture: Optional[CUDAGraphCapture],
) -> torch.Tensor:
    """
    Compute sparse GEMV for token_indices, using CUDA Graph replay when
    a capture is provided and shape matches.
    """
    if graph_capture is None or len(token_indices) != graph_capture.max_sub_vocab_chunk:
        gathered = weight_matrix[token_indices]
        logits = torch.matmul(gathered, hidden_state)
        if bias_vector is not None:
            logits = logits + bias_vector[token_indices]
        return logits

    weight_chunk = weight_matrix[token_indices]
    bias_chunk = bias_vector[token_indices] if bias_vector is not None else torch.zeros(
        len(token_indices), device=weight_matrix.device, dtype=weight_matrix.dtype
    )
    return graph_capture.replay(weight_chunk, hidden_state, bias_chunk)

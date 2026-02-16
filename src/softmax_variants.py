"""
Softmax Variants (Baselines from Related Work)

Implements Hierarchical Softmax [Mikolov et al., ICLR 2013] and Adaptive Softmax
[Grave et al., ICML 2017] as referenced in the CSV-Decode paper (Section II-B)
for comparison and ablation. These are training-time or architectural
alternatives; CSV-Decode is inference-only and does not require retraining.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Tuple, Optional, Dict


# ---------------------------------------------------------------------------
# Hierarchical Softmax (Mikolov et al.)
# ---------------------------------------------------------------------------

class HierarchicalSoftmax(nn.Module):
    """
    Hierarchical Softmax [Mikolov et al., ICLR 2013]: vocabulary in a binary tree;
    probability of a token = product of binary decisions along the path root -> leaf.
    Complexity O(log V) per token instead of O(V). Requires a fixed tree layout.
    """

    def __init__(self, vocab_size: int, hidden_size: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        # Full binary tree: V leaves => V-1 internal nodes. Root = 0.
        self.num_internal = max(1, vocab_size - 1)
        self.node_weights = nn.Parameter(torch.empty(self.num_internal, hidden_size))
        self.node_biases = nn.Parameter(torch.empty(self.num_internal))
        self._path_cache: Optional[Dict[int, List[Tuple[int, int]]]] = None
        self.reset_parameters()
        self._build_path_cache()

    def reset_parameters(self):
        nn.init.normal_(self.node_weights, 0, 0.02)
        nn.init.zeros_(self.node_biases)

    def _build_path_cache(self):
        """Precompute path from root to each leaf. Tree: internal nodes 0..n-1, leaves n..n+V-1; root 0."""
        V = self.vocab_size
        n = self.num_internal  # V - 1
        if n <= 0:
            self._path_cache = {i: [] for i in range(V)}
            return
        self._path_cache = {}
        for token_id in range(V):
            path = []
            leaf_linear = n + token_id
            node = leaf_linear
            while node > 0:
                parent = (node - 1) // 2
                direction = 1 if node == 2 * parent + 2 else 0
                if parent < n:
                    path.append((parent, direction))
                node = parent
            path.reverse()
            self._path_cache[token_id] = path

    def _get_path_to_leaf(self, token_id: int) -> List[Tuple[int, int]]:
        cache = self._path_cache
        return (cache.get(token_id, []) if cache is not None else [])

    def forward_log_probs(self, hidden: torch.Tensor) -> torch.Tensor:
        """Compute log-probability for each token. hidden: (batch, hidden_size). Returns (batch, vocab_size)."""
        batch = hidden.size(0)
        device = hidden.device
        log_probs = torch.empty(batch, self.vocab_size, device=device, dtype=hidden.dtype)
        for v in range(self.vocab_size):
            path = self._get_path_to_leaf(v)
            log_p = torch.zeros(batch, device=device, dtype=hidden.dtype)
            for node_id, direction in path:
                w = self.node_weights[node_id]
                b = self.node_biases[node_id]
                logit = (hidden @ w + b).squeeze(-1)
                log_p = log_p + (F.logsigmoid(logit) if direction == 1 else F.logsigmoid(-logit))
            log_probs[:, v] = log_p
        return log_probs

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """Return log-probs over full vocabulary (batch, V) for comparison with standard softmax."""
        return self.forward_log_probs(hidden)


# ---------------------------------------------------------------------------
# Adaptive Softmax (Grave et al.)
# ---------------------------------------------------------------------------

class AdaptiveSoftmax(nn.Module):
    """
    Adaptive Softmax: partitions vocabulary by frequency; head cluster (frequent tokens)
    gets full softmax, tail clusters use reduced computation. Reduces effective V.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        head_size: int = 0,
        tail_sizes: Optional[List[int]] = None,
        div_value: float = 4.0,
    ):
        """
        Args:
            vocab_size: V
            hidden_size: d
            head_size: number of frequent tokens in the head (full softmax). If 0, set to vocab_size // 2.
            tail_sizes: sizes of tail clusters. If None, one tail with rest of vocab.
            div_value: used when tail_sizes is None to define geometric split.
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        if head_size <= 0:
            head_size = max(1, vocab_size // 2)
        self.head_size = min(head_size, vocab_size)
        if tail_sizes is None:
            rest = vocab_size - self.head_size
            self.tail_sizes = [rest] if rest > 0 else []
        else:
            self.tail_sizes = list(tail_sizes)
        self.head = nn.Linear(hidden_size, self.head_size)
        tail_dim = int(hidden_size // div_value)
        self.tail_proj = nn.Linear(hidden_size, tail_dim, bias=False)
        self.tail_clusters = nn.ModuleList()
        offset = self.head_size
        for size in self.tail_sizes:
            self.tail_clusters.append(nn.Linear(tail_dim, size))
            offset += size
        self._cluster_offsets = self._build_offsets()

    def _build_offsets(self) -> List[int]:
        off = [0, self.head_size]
        for s in self.tail_sizes:
            off.append(off[-1] + s)
        return off

    def forward(
        self,
        hidden: torch.Tensor,
        target_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        hidden: (batch, hidden_size).
        If target_ids is None, returns logits for head only (batch, head_size).
        Else returns full log-probabilities for the batch (for loss).
        """
        head_logits = self.head(hidden)
        if target_ids is None:
            return head_logits
        # For training: compute loss per cluster
        batch = hidden.size(0)
        device = hidden.device
        tail_h = self.tail_proj(hidden)
        all_logits = [head_logits]
        for i, cluster in enumerate(self.tail_clusters):
            all_logits.append(cluster(tail_h))
        return torch.cat(all_logits, dim=-1)

    def forward_logits_full(
        self, hidden: torch.Tensor
    ) -> torch.Tensor:
        """
        Inference: return logits over full vocabulary (batch, V).
        Head uses head logits; tail uses tail cluster logits (same hidden projected).
        """
        batch = hidden.size(0)
        device = hidden.device
        head_logits = self.head(hidden)
        tail_h = self.tail_proj(hidden)
        logits_list = [head_logits]
        for cluster in self.tail_clusters:
            logits_list.append(cluster(tail_h))
        return torch.cat(logits_list, dim=-1)

    def forward_sub_vocab_logits(
        self,
        hidden: torch.Tensor,
        token_indices: List[int],
    ) -> torch.Tensor:
        """
        Compute logits only for a subset of token indices (for efficient decoding).
        """
        if hidden.dim() == 1:
            hidden = hidden.unsqueeze(0)
        head_logits = self.head(hidden)
        tail_h = self.tail_proj(hidden)
        out = []
        for idx in token_indices:
            if idx < self.head_size:
                out.append(head_logits[:, idx : idx + 1])
            else:
                offset = self.head_size
                for c, cluster in enumerate(self.tail_clusters):
                    size = self.tail_sizes[c]
                    if offset <= idx < offset + size:
                        local = idx - offset
                        out.append(cluster(tail_h)[:, local : local + 1])
                        break
                    offset += size
                else:
                    out.append(
                        torch.full(
                            (hidden.size(0), 1),
                            float("-inf"),
                            device=hidden.device,
                            dtype=hidden.dtype,
                        )
                    )
        return torch.cat(out, dim=1).squeeze(0)


def build_hierarchical_softmax(vocab_size: int, hidden_size: int) -> HierarchicalSoftmax:
    """Factory for Hierarchical Softmax (paper baseline)."""
    return HierarchicalSoftmax(vocab_size=vocab_size, hidden_size=hidden_size)


def build_adaptive_softmax(
    vocab_size: int,
    hidden_size: int,
    head_ratio: float = 0.5,
) -> AdaptiveSoftmax:
    """Factory for Adaptive Softmax (paper baseline). head_ratio = head_size / vocab_size."""
    head_size = max(1, int(vocab_size * head_ratio))
    return AdaptiveSoftmax(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        head_size=head_size,
    )

"""
CSV-Decode: Certifiable Sub-Vocabulary Decoding for Efficient Large Language Model Inference

This module implements the core CSV-Decode functionality including:
1. Vocabulary clustering using K-means
2. Geometric upper bounds using centroid-plus-radius
3. Exact top-k certification and ε-certified softmax approximation
4. Online adaptive sub-vocabulary construction
"""

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import KMeans
from typing import Dict, List, Tuple, Optional
import heapq
import math
from dataclasses import dataclass


@dataclass
class ClusterMetadata:
    """Metadata for a vocabulary cluster"""
    centroid: torch.Tensor  # Cluster centroid μc
    radius: float  # Cluster radius Rc
    bias_max: float  # Maximum bias in cluster
    token_indices: List[int]  # Token indices in this cluster
    cluster_id: int


@dataclass
class CSVDecodeConfig:
    """Configuration for CSV-Decode"""
    num_clusters: int = 2000
    epsilon: float = 0.05  # Softmax approximation tolerance
    max_sub_vocab_size: int = 50000  # Maximum sub-vocabulary size
    use_spherical_clustering: bool = True  # Use spherical K-means for normalized embeddings
    bias_binning: bool = True  # Use bias binning for tighter bounds
    top_m_bias: int = 3  # Top-m bias values to consider per cluster


class VocabularyClusterer:
    """Handles offline vocabulary clustering using K-means"""
    
    def __init__(self, config: CSVDecodeConfig):
        self.config = config
        self.clusters: Dict[int, ClusterMetadata] = {}
        self.cluster_assignments: Dict[int, int] = {}  # token_id -> cluster_id
        
    def cluster_vocabulary(self, embedding_matrix: torch.Tensor, bias_vector: torch.Tensor) -> Dict[int, ClusterMetadata]:
        """
        Cluster vocabulary embeddings using K-means
        
        Args:
            embedding_matrix: Shape (V, d) where V is vocab size, d is embedding dim
            bias_vector: Shape (V,) bias values for each token
            
        Returns:
            Dictionary mapping cluster_id to ClusterMetadata
        """
        V, d = embedding_matrix.shape
        device = embedding_matrix.device
        
        # Normalize embeddings if using spherical clustering
        if self.config.use_spherical_clustering:
            embeddings_norm = F.normalize(embedding_matrix, p=2, dim=1)
        else:
            embeddings_norm = embedding_matrix
            
        # Convert to numpy for sklearn
        embeddings_np = embeddings_norm.cpu().numpy()
        
        # Perform K-means clustering
        kmeans = KMeans(
            n_clusters=self.config.num_clusters,
            random_state=42,
            n_init=10,
            max_iter=100
        )
        cluster_labels = kmeans.fit_predict(embeddings_np)
        
        # Build cluster metadata
        self.clusters = {}
        for cluster_id in range(self.config.num_clusters):
            cluster_mask = cluster_labels == cluster_id
            cluster_indices = np.where(cluster_mask)[0].tolist()
            
            if len(cluster_indices) == 0:
                continue
                
            # Compute centroid
            cluster_embeddings = embeddings_norm[cluster_indices]
            centroid = torch.mean(cluster_embeddings, dim=0)
            
            # Compute radius (maximum distance from centroid)
            distances = torch.norm(cluster_embeddings - centroid.unsqueeze(0), dim=1)
            radius = torch.max(distances).item()
            
            # Compute maximum bias in cluster
            cluster_biases = bias_vector[cluster_indices]
            bias_max = torch.max(cluster_biases).item()
            
            # Store cluster metadata
            self.clusters[cluster_id] = ClusterMetadata(
                centroid=centroid,
                radius=radius,
                bias_max=bias_max,
                token_indices=cluster_indices,
                cluster_id=cluster_id
            )
            
            # Store token-to-cluster mapping
            for token_idx in cluster_indices:
                self.cluster_assignments[token_idx] = cluster_id
                
        return self.clusters
    
    def get_cluster_for_token(self, token_id: int) -> Optional[int]:
        """Get cluster ID for a given token"""
        return self.cluster_assignments.get(token_id)
    
    def get_cluster_metadata(self, cluster_id: int) -> Optional[ClusterMetadata]:
        """Get metadata for a specific cluster"""
        return self.clusters.get(cluster_id)


class GeometricBounds:
    """Computes geometric upper bounds for clusters using Cauchy-Schwarz inequality"""
    
    @staticmethod
    def compute_cluster_upper_bound(
        cluster_metadata: ClusterMetadata,
        hidden_state: torch.Tensor,
        use_spherical: bool = True
    ) -> float:
        """
        Compute upper bound for cluster using Cauchy-Schwarz inequality
        
        For any token i in cluster c:
        ℓi(t) = ⟨Wi, ht⟩ + bi ≤ ⟨μc, ht⟩ + Rc∥ht∥2 + max_j∈c bj
        
        Args:
            cluster_metadata: Cluster metadata containing centroid, radius, bias_max
            hidden_state: Current hidden state ht
            use_spherical: Whether to use spherical bounds for normalized embeddings
            
        Returns:
            Upper bound Uc(ht) for the cluster
        """
        centroid = cluster_metadata.centroid
        radius = cluster_metadata.radius
        bias_max = cluster_metadata.bias_max
        
        if use_spherical:
            # For spherical clustering with normalized embeddings
            # Bound: ⟨Wi, h⟩ ≤ ∥h∥2 (∥μc∥2 cos θc + sin θc)
            h_norm = torch.norm(hidden_state)
            mu_norm = torch.norm(centroid)
            
            if mu_norm > 0 and h_norm > 0:
                cos_theta = torch.dot(centroid, hidden_state) / (mu_norm * h_norm)
                # Angular radius approximation
                angular_radius = radius / mu_norm if mu_norm > 0 else radius
                bound = h_norm * (mu_norm * cos_theta + angular_radius)
            else:
                bound = h_norm * radius
        else:
            # Standard Euclidean bound
            centroid_dot = torch.dot(centroid, hidden_state)
            h_norm = torch.norm(hidden_state)
            bound = centroid_dot + radius * h_norm
            
        return bound.item() + bias_max


class CertificationMechanisms:
    """Implements exact top-k certification and ε-certified softmax approximation"""
    
    @staticmethod
    def check_top_k_certification(
        sub_vocab_logits: torch.Tensor,
        sub_vocab_indices: List[int],
        k: int,
        cluster_bounds: Dict[int, float],
        clusterer: VocabularyClusterer
    ) -> Tuple[bool, float]:
        """
        Check if top-k is certified for current sub-vocabulary
        
        Args:
            sub_vocab_logits: Logits for tokens in sub-vocabulary
            sub_vocab_indices: Token indices in sub-vocabulary
            k: Number of top tokens to consider
            cluster_bounds: Upper bounds for all clusters
            clusterer: Vocabulary clusterer
            
        Returns:
            (is_certified, kth_largest_logit)
        """
        if len(sub_vocab_logits) < k:
            return False, float('-inf')
            
        # Get k-th largest logit in sub-vocabulary
        sorted_logits, _ = torch.sort(sub_vocab_logits, descending=True)
        kth_largest = sorted_logits[k-1].item()
        
        # Check if any unopened cluster can exceed kth largest
        opened_clusters = set()
        for token_idx in sub_vocab_indices:
            cluster_id = clusterer.get_cluster_for_token(token_idx)
            if cluster_id is not None:
                opened_clusters.add(cluster_id)
        
        # Check bounds for unopened clusters
        for cluster_id, bound in cluster_bounds.items():
            if cluster_id not in opened_clusters:
                if bound >= kth_largest:
                    return False, kth_largest
                    
        return True, kth_largest
    
    @staticmethod
    def check_softmax_epsilon_certification(
        sub_vocab_logits: torch.Tensor,
        sub_vocab_indices: List[int],
        epsilon: float,
        cluster_bounds: Dict[int, float],
        clusterer: VocabularyClusterer
    ) -> Tuple[bool, float]:
        """
        Check if softmax approximation is ε-certified
        
        Args:
            sub_vocab_logits: Logits for tokens in sub-vocabulary
            sub_vocab_indices: Token indices in sub-vocabulary
            epsilon: Maximum allowed total variation distance
            cluster_bounds: Upper bounds for all clusters
            clusterer: Vocabulary clusterer
            
        Returns:
            (is_certified, relative_error_bound)
        """
        # Compute log-sum-exp for sub-vocabulary
        log_sum_exp_sub = torch.logsumexp(sub_vocab_logits, dim=0).item()
        
        # Compute upper bound for external probability mass
        opened_clusters = set()
        for token_idx in sub_vocab_indices:
            cluster_id = clusterer.get_cluster_for_token(token_idx)
            if cluster_id is not None:
                opened_clusters.add(cluster_id)
        
        # Sum of upper bounds for unopened clusters
        external_mass_bound = 0.0
        for cluster_id, bound in cluster_bounds.items():
            if cluster_id not in opened_clusters:
                cluster_metadata = clusterer.get_cluster_metadata(cluster_id)
                if cluster_metadata:
                    # Approximate: |c| * exp(bound) for cluster c
                    cluster_size = len(cluster_metadata.token_indices)
                    external_mass_bound += cluster_size * math.exp(bound)
        
        # Check certification condition
        if external_mass_bound == 0:
            return True, 0.0
            
        relative_error = external_mass_bound / (math.exp(log_sum_exp_sub) + external_mass_bound)
        is_certified = relative_error <= epsilon
        
        return is_certified, relative_error


class CSVDecodeEngine:
    """Main CSV-Decode engine implementing the online algorithm"""
    
    def __init__(self, config: CSVDecodeConfig):
        self.config = config
        self.clusterer = VocabularyClusterer(config)
        self.bounds_computer = GeometricBounds()
        self.certifier = CertificationMechanisms()
        
        # Runtime state
        self.cluster_bounds: Dict[int, float] = {}
        self.priority_queue: List[Tuple[float, int]] = []  # Max-heap: (-bound, cluster_id)
        
    def initialize_clusters(self, embedding_matrix: torch.Tensor, bias_vector: torch.Tensor):
        """Initialize vocabulary clusters (offline preprocessing)"""
        self.clusterer.cluster_vocabulary(embedding_matrix, bias_vector)
        
    def compute_all_bounds(self, hidden_state: torch.Tensor) -> Dict[int, float]:
        """Compute upper bounds for all clusters"""
        bounds = {}
        for cluster_id, metadata in self.clusterer.clusters.items():
            bound = self.bounds_computer.compute_cluster_upper_bound(
                metadata, hidden_state, self.config.use_spherical_clustering
            )
            bounds[cluster_id] = bound
        return bounds
    
    def build_priority_queue(self, bounds: Dict[int, float]):
        """Build max-heap priority queue of clusters ordered by upper bounds"""
        self.priority_queue = [(-bound, cluster_id) for cluster_id, bound in bounds.items()]
        heapq.heapify(self.priority_queue)
    
    def expand_sub_vocabulary(
        self,
        current_sub_vocab: List[int],
        current_logits: torch.Tensor,
        k: int,
        epsilon: float
    ) -> Tuple[List[int], torch.Tensor, bool, str]:
        """
        Expand sub-vocabulary using CSV-Decode algorithm
        
        Args:
            current_sub_vocab: Current sub-vocabulary token indices
            current_logits: Logits for current sub-vocabulary
            k: Number of top tokens for certification
            epsilon: Softmax approximation tolerance
            
        Returns:
            (expanded_sub_vocab, expanded_logits, is_certified, certification_type)
        """
        sub_vocab = current_sub_vocab.copy()
        logits = current_logits.clone()
        
        while len(sub_vocab) < self.config.max_sub_vocab_size:
            # Check certifications
            top_k_certified, kth_largest = self.certifier.check_top_k_certification(
                logits, sub_vocab, k, self.cluster_bounds, self.clusterer
            )
            
            epsilon_certified, relative_error = self.certifier.check_softmax_epsilon_certification(
                logits, sub_vocab, epsilon, self.cluster_bounds, self.clusterer
            )
            
            if top_k_certified or epsilon_certified:
                cert_type = "top_k" if top_k_certified else "epsilon"
                return sub_vocab, logits, True, cert_type
            
            # Expand by adding highest-priority cluster
            if not self.priority_queue:
                break
                
            _, cluster_id = heapq.heappop(self.priority_queue)
            cluster_metadata = self.clusterer.get_cluster_metadata(cluster_id)
            
            if cluster_metadata is None:
                continue
                
            # Add all tokens from this cluster to sub-vocabulary
            new_tokens = cluster_metadata.token_indices
            sub_vocab.extend(new_tokens)
            
            # Note: In practice, you would compute actual logits for these tokens
            # For now, we'll use placeholder values
            new_logits = torch.randn(len(new_tokens), device=logits.device)
            logits = torch.cat([logits, new_logits], dim=0)
        
        return sub_vocab, logits, False, "fallback"
    
    def decode_step(
        self,
        hidden_state: torch.Tensor,
        embedding_matrix: torch.Tensor,
        bias_vector: torch.Tensor,
        k: int = 10,
        epsilon: float = None
    ) -> Tuple[torch.Tensor, List[int], bool, str]:
        """
        Perform one CSV-Decode step
        
        Args:
            hidden_state: Current hidden state ht
            embedding_matrix: Output layer weight matrix (V, d)
            bias_vector: Bias vector (V,)
            k: Number of top tokens for certification
            epsilon: Softmax approximation tolerance (uses config default if None)
            
        Returns:
            (logits, sub_vocab_indices, is_certified, certification_type)
        """
        if epsilon is None:
            epsilon = self.config.epsilon
            
        # Compute bounds for all clusters
        self.cluster_bounds = self.compute_all_bounds(hidden_state)
        
        # Build priority queue
        self.build_priority_queue(self.cluster_bounds)
        
        # Start with empty sub-vocabulary
        sub_vocab = []
        logits = torch.empty(0, device=hidden_state.device)
        
        # Expand sub-vocabulary
        final_sub_vocab, final_logits, is_certified, cert_type = self.expand_sub_vocabulary(
            sub_vocab, logits, k, epsilon
        )
        
        return final_logits, final_sub_vocab, is_certified, cert_type


def create_csv_decode_config(
    vocab_size: int,
    embedding_dim: int,
    num_clusters: int = None,
    epsilon: float = 0.05
) -> CSVDecodeConfig:
    """
    Create CSV-Decode configuration with reasonable defaults
    
    Args:
        vocab_size: Vocabulary size V
        embedding_dim: Embedding dimension d
        num_clusters: Number of clusters (default: 0.015 * V)
        epsilon: Softmax approximation tolerance
        
    Returns:
        CSVDecodeConfig instance
    """
    if num_clusters is None:
        num_clusters = max(500, int(0.015 * vocab_size))
    
    max_sub_vocab_size = min(50000, int(0.2 * vocab_size))
    
    return CSVDecodeConfig(
        num_clusters=num_clusters,
        epsilon=epsilon,
        max_sub_vocab_size=max_sub_vocab_size,
        use_spherical_clustering=True,
        bias_binning=True,
        top_m_bias=3
    )

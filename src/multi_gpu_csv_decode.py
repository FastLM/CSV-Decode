"""
Multi-GPU Support for CSV-Decode

This module implements distributed CSV-Decode across multiple GPUs using NCCL communication
and intelligent cluster sharding strategies.
"""

import torch
import torch.distributed as dist
from typing import Dict, List, Tuple, Optional
import heapq
import math


class MultiGPUSharding:
    """Handles multi-GPU sharding for CSV-Decode clusters"""
    
    def __init__(self, num_gpus: int, device_ids: List[int]):
        self.num_gpus = num_gpus
        self.device_ids = device_ids
        self.cluster_assignments: Dict[int, int] = {}  # cluster_id -> gpu_id
        self.gpu_clusters: Dict[int, List[int]] = {i: [] for i in range(num_gpus)}
        
    def assign_clusters_round_robin(self, cluster_ids: List[int]):
        """Assign clusters to GPUs using round-robin strategy"""
        for i, cluster_id in enumerate(cluster_ids):
            gpu_id = i % self.num_gpus
            self.cluster_assignments[cluster_id] = gpu_id
            self.gpu_clusters[gpu_id].append(cluster_id)
    
    def assign_clusters_by_size(self, cluster_metadata: Dict[int, 'ClusterMetadata']):
        """Assign clusters to GPUs balancing by cluster size"""
        # Sort clusters by size (descending)
        sorted_clusters = sorted(
            cluster_metadata.items(),
            key=lambda x: len(x[1].token_indices),
            reverse=True
        )
        
        # Assign largest clusters first to balance load
        gpu_loads = [0] * self.num_gpus
        
        for cluster_id, metadata in sorted_clusters:
            # Assign to GPU with minimum load
            min_gpu = min(range(self.num_gpus), key=lambda i: gpu_loads[i])
            self.cluster_assignments[cluster_id] = min_gpu
            self.gpu_clusters[min_gpu].append(cluster_id)
            gpu_loads[min_gpu] += len(metadata.token_indices)
    
    def assign_clusters_by_semantic_similarity(self, cluster_metadata: Dict[int, 'ClusterMetadata']):
        """Assign semantically similar clusters to the same GPU"""
        # This is a simplified version - in practice, you'd use more sophisticated
        # semantic similarity measures
        
        # Group clusters by centroid similarity
        centroids = [(cid, metadata.centroid) for cid, metadata in cluster_metadata.items()]
        
        # Simple clustering of centroids
        semantic_groups = []
        used_clusters = set()
        
        for cluster_id, centroid in centroids:
            if cluster_id in used_clusters:
                continue
                
            group = [cluster_id]
            used_clusters.add(cluster_id)
            
            # Find similar clusters
            for other_id, other_centroid in centroids:
                if other_id in used_clusters:
                    continue
                    
                # Compute cosine similarity
                similarity = torch.cosine_similarity(
                    centroid.unsqueeze(0), other_centroid.unsqueeze(0)
                ).item()
                
                if similarity > 0.8:  # Threshold for semantic similarity
                    group.append(other_id)
                    used_clusters.add(other_id)
            
            semantic_groups.append(group)
        
        # Assign groups to GPUs
        for i, group in enumerate(semantic_groups):
            gpu_id = i % self.num_gpus
            for cluster_id in group:
                self.cluster_assignments[cluster_id] = gpu_id
                self.gpu_clusters[gpu_id].append(cluster_id)
    
    def get_clusters_for_gpu(self, gpu_id: int) -> List[int]:
        """Get list of cluster IDs assigned to a specific GPU"""
        return self.gpu_clusters.get(gpu_id, [])
    
    def get_gpu_for_cluster(self, cluster_id: int) -> Optional[int]:
        """Get GPU ID for a specific cluster"""
        return self.cluster_assignments.get(cluster_id)


class NCCLCommunication:
    """Handles NCCL communication for multi-GPU CSV-Decode"""
    
    def __init__(self, device_ids: List[int]):
        self.device_ids = device_ids
        self.num_gpus = len(device_ids)
        
    def initialize_distributed(self, backend: str = 'nccl'):
        """Initialize distributed communication"""
        if not dist.is_initialized():
            dist.init_process_group(backend=backend)
    
    def gather_cluster_bounds(self, local_bounds: Dict[int, float]) -> Dict[int, float]:
        """Gather cluster bounds from all GPUs"""
        if not dist.is_initialized():
            return local_bounds
            
        # Convert to tensors for communication
        all_bounds = {}
        
        # Gather from all processes
        gathered_data = [None] * self.num_gpus
        dist.all_gather_object(gathered_data, local_bounds)
        
        # Merge all bounds
        for bounds_dict in gathered_data:
            if bounds_dict is not None:
                all_bounds.update(bounds_dict)
        
        return all_bounds
    
    def gather_logits(self, local_logits: torch.Tensor, local_indices: List[int]) -> Tuple[torch.Tensor, List[int]]:
        """Gather logits and indices from all GPUs"""
        if not dist.is_initialized():
            return local_logits, local_indices
            
        # Gather logits
        gathered_logits = [None] * self.num_gpus
        dist.all_gather_object(gathered_logits, local_logits)
        
        # Gather indices
        gathered_indices = [None] * self.num_gpus
        dist.all_gather_object(gathered_indices, local_indices)
        
        # Concatenate results
        all_logits = torch.cat([logits for logits in gathered_logits if logits is not None])
        all_indices = []
        for indices in gathered_indices:
            if indices is not None:
                all_indices.extend(indices)
        
        return all_logits, all_indices
    
    def broadcast_priority_queue(self, priority_queue: List[Tuple[float, int]]) -> List[Tuple[float, int]]:
        """Broadcast priority queue to all GPUs"""
        if not dist.is_initialized():
            return priority_queue
            
        # Broadcast from rank 0
        dist.broadcast_object_list([priority_queue], src=0)
        return priority_queue[0] if priority_queue else []


class DistributedCSVDecodeEngine:
    """Distributed CSV-Decode engine for multi-GPU inference"""
    
    def __init__(self, config, device_ids: List[int]):
        self.config = config
        self.device_ids = device_ids
        self.num_gpus = len(device_ids)
        
        # Initialize components
        self.sharding = MultiGPUSharding(self.num_gpus, device_ids)
        self.communication = NCCLCommunication(device_ids)
        
        # Per-GPU engines
        self.gpu_engines = {}
        self.clusterer = None
        
    def initialize_distributed(self):
        """Initialize distributed training"""
        self.communication.initialize_distributed()
        
    def initialize_clusters_distributed(self, embedding_matrix: torch.Tensor, bias_vector: torch.Tensor):
        """Initialize clusters with distributed processing"""
        # Only rank 0 does clustering
        if dist.get_rank() == 0:
            from .csv_decode import VocabularyClusterer
            clusterer = VocabularyClusterer(self.config)
            clusters = clusterer.cluster_vocabulary(embedding_matrix, bias_vector)
            
            # Assign clusters to GPUs
            self.sharding.assign_clusters_by_size(clusters)
            
            # Broadcast cluster assignments
            dist.broadcast_object_list([self.sharding.cluster_assignments], src=0)
        else:
            # Receive cluster assignments
            cluster_assignments = [None]
            dist.broadcast_object_list(cluster_assignments, src=0)
            self.sharding.cluster_assignments = cluster_assignments[0]
        
        # Initialize per-GPU engines
        for gpu_id in self.device_ids:
            from .csv_decode import CSVDecodeEngine
            engine = CSVDecodeEngine(self.config)
            engine.clusterer.clusters = {
                cid: metadata for cid, metadata in clusters.items()
                if self.sharding.get_gpu_for_cluster(cid) == gpu_id
            }
            self.gpu_engines[gpu_id] = engine
    
    def compute_bounds_distributed(self, hidden_state: torch.Tensor) -> Dict[int, float]:
        """Compute cluster bounds across all GPUs"""
        local_bounds = {}
        
        # Each GPU computes bounds for its assigned clusters
        for gpu_id, engine in self.gpu_engines.items():
            if torch.cuda.current_device() == gpu_id:
                gpu_bounds = engine.compute_all_bounds(hidden_state)
                local_bounds.update(gpu_bounds)
        
        # Gather bounds from all GPUs
        all_bounds = self.communication.gather_cluster_bounds(local_bounds)
        return all_bounds
    
    def decode_step_distributed(
        self,
        hidden_state: torch.Tensor,
        weight_matrix: torch.Tensor,
        bias_vector: torch.Tensor,
        k: int = 10,
        epsilon: float = None
    ) -> Tuple[torch.Tensor, List[int], bool, str]:
        """Perform distributed CSV-Decode step"""
        if epsilon is None:
            epsilon = self.config.epsilon
            
        # Compute bounds across all GPUs
        cluster_bounds = self.compute_bounds_distributed(hidden_state)
        
        # Build priority queue (on rank 0)
        if dist.get_rank() == 0:
            priority_queue = [(-bound, cluster_id) for cluster_id, bound in cluster_bounds.items()]
            heapq.heapify(priority_queue)
        else:
            priority_queue = []
        
        # Broadcast priority queue
        priority_queue = self.communication.broadcast_priority_queue(priority_queue)
        
        # Each GPU processes its assigned clusters
        local_logits = torch.empty(0, device=hidden_state.device)
        local_indices = []
        
        for gpu_id, engine in self.gpu_engines.items():
            if torch.cuda.current_device() == gpu_id:
                # Process clusters assigned to this GPU
                assigned_clusters = self.sharding.get_clusters_for_gpu(gpu_id)
                
                for cluster_id in assigned_clusters:
                    cluster_metadata = engine.clusterer.get_cluster_metadata(cluster_id)
                    if cluster_metadata:
                        # Compute logits for this cluster
                        cluster_logits = torch.matmul(
                            weight_matrix[cluster_metadata.token_indices],
                            hidden_state
                        ) + bias_vector[cluster_metadata.token_indices]
                        
                        local_logits = torch.cat([local_logits, cluster_logits])
                        local_indices.extend(cluster_metadata.token_indices)
        
        # Gather results from all GPUs
        all_logits, all_indices = self.communication.gather_logits(local_logits, local_indices)
        
        # Certification (on rank 0)
        if dist.get_rank() == 0:
            from .csv_decode import CertificationMechanisms
            is_certified, cert_type = CertificationMechanisms.check_adaptive_certification(
                all_logits, all_indices, k, epsilon, cluster_bounds, 
                self.gpu_engines[0].clusterer, context_length=0
            )
        else:
            is_certified, cert_type = False, "none"
        
        # Broadcast certification result
        cert_result = [None]
        if dist.get_rank() == 0:
            cert_result = [(is_certified, cert_type)]
        dist.broadcast_object_list(cert_result, src=0)
        is_certified, cert_type = cert_result[0]
        
        return all_logits, all_indices, is_certified, cert_type


def create_distributed_csv_decode(config, device_ids: List[int]) -> DistributedCSVDecodeEngine:
    """Factory function to create distributed CSV-Decode engine"""
    return DistributedCSVDecodeEngine(config, device_ids)

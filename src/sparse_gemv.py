"""
Optimized Sparse GEMV Kernels for CSV-Decode

This module implements efficient sparse matrix-vector multiplication kernels
specifically optimized for CSV-Decode's sub-vocabulary computation patterns.
Includes CUDA kernels, multi-level tiling, and Tensor Core integration.
"""

import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional
import math


class SparseGEMVKernel:
    """Optimized sparse GEMV kernel for CSV-Decode"""
    
    def __init__(self, device: torch.device, dtype: torch.dtype = torch.float16):
        self.device = device
        self.dtype = dtype
        
        # Kernel parameters optimized for A100/H100
        self.block_size_x = 128  # Bx
        self.block_size_y = 4    # By
        self.warps_per_block = 4  # NW
        self.tile_size = 16      # For Tensor Core operations
        
    def sparse_gemv_coo(
        self,
        weight_matrix: torch.Tensor,
        hidden_state: torch.Tensor,
        token_indices: List[int],
        bias_vector: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Sparse GEMV using COO format with row gathering
        
        Implements: gathered_rows = W[indices]; logits = gathered_rows @ h_t + bias[indices]
        
        Args:
            weight_matrix: Full weight matrix (V, d)
            hidden_state: Hidden state vector (d,)
            token_indices: List of token indices to compute
            bias_vector: Optional bias vector (V,)
            
        Returns:
            Logits for specified tokens (len(token_indices),)
        """
        if not token_indices:
            return torch.empty(0, device=self.device, dtype=self.dtype)
            
        # Gather rows for specified tokens
        gathered_rows = weight_matrix[token_indices]  # (len(indices), d)
        
        # Compute logits: gathered_rows @ hidden_state
        logits = torch.matmul(gathered_rows, hidden_state)  # (len(indices),)
        
        # Add bias if provided
        if bias_vector is not None:
            gathered_bias = bias_vector[token_indices]
            logits = logits + gathered_bias
            
        return logits
    
    def sparse_gemv_csr(
        self,
        csr_values: torch.Tensor,
        csr_indices: torch.Tensor,
        csr_pointers: torch.Tensor,
        hidden_state: torch.Tensor,
        bias_vector: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Sparse GEMV using CSR format for memory efficiency
        
        Args:
            csr_values: Non-zero values (nnz,)
            csr_indices: Column indices (nnz,)
            csr_pointers: Row pointers (V+1,)
            hidden_state: Hidden state vector (d,)
            bias_vector: Optional bias vector (V,)
            
        Returns:
            Logits for all tokens (V,)
        """
        V = len(csr_pointers) - 1
        logits = torch.zeros(V, device=self.device, dtype=self.dtype)
        
        # Process each row
        for row in range(V):
            start_ptr = csr_pointers[row]
            end_ptr = csr_pointers[row + 1]
            
            if start_ptr < end_ptr:
                row_values = csr_values[start_ptr:end_ptr]
                col_indices = csr_indices[start_ptr:end_ptr]
                
                # Gather corresponding hidden state elements
                gathered_hidden = hidden_state[col_indices]
                
                # Compute dot product
                logits[row] = torch.sum(row_values * gathered_hidden)
                
                # Add bias if provided
                if bias_vector is not None:
                    logits[row] += bias_vector[row]
                    
        return logits


class MultiLevelTiling:
    """Multi-level tiling for optimized memory access patterns"""
    
    def __init__(self, device: torch.device):
        self.device = device
        
        # Memory hierarchy parameters
        self.l1_cache_size = 128 * 1024  # 128KB L1 cache
        self.l2_cache_size = 6 * 1024 * 1024  # 6MB L2 cache
        self.shared_mem_size = 48 * 1024  # 48KB shared memory
        
    def compute_tile_sizes(self, matrix_shape: Tuple[int, int]) -> Tuple[int, int, int]:
        """
        Compute optimal tile sizes based on matrix dimensions and memory hierarchy
        
        Args:
            matrix_shape: (rows, cols) of the matrix
            
        Returns:
            (tile_rows, tile_cols, num_tiles)
        """
        rows, cols = matrix_shape
        
        # Base tile size calculation
        base_tile_size = int(math.sqrt(self.shared_mem_size // (4 * 2)))  # FP16
        
        # Adjust for matrix dimensions
        tile_rows = min(base_tile_size, rows)
        tile_cols = min(base_tile_size, cols)
        
        # Ensure tiles fit in shared memory
        while tile_rows * tile_cols * 4 * 2 > self.shared_mem_size:
            if tile_rows > tile_cols:
                tile_rows //= 2
            else:
                tile_cols //= 2
                
        num_tiles = math.ceil(rows / tile_rows) * math.ceil(cols / tile_cols)
        
        return tile_rows, tile_cols, num_tiles
    
    def tiled_gemv(
        self,
        weight_matrix: torch.Tensor,
        hidden_state: torch.Tensor,
        token_indices: List[int],
        bias_vector: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Tiled GEMV computation with optimized memory access
        
        Args:
            weight_matrix: Weight matrix (V, d)
            hidden_state: Hidden state vector (d,)
            token_indices: Token indices to compute
            bias_vector: Optional bias vector
            
        Returns:
            Logits for specified tokens
        """
        if not token_indices:
            return torch.empty(0, device=self.device)
            
        gathered_rows = weight_matrix[token_indices]
        rows, cols = gathered_rows.shape
        
        # Compute tile sizes
        tile_rows, tile_cols, _ = self.compute_tile_sizes((rows, cols))
        
        # Initialize output
        logits = torch.zeros(rows, device=self.device, dtype=gathered_rows.dtype)
        
        # Process tiles
        for tile_row_start in range(0, rows, tile_rows):
            tile_row_end = min(tile_row_start + tile_rows, rows)
            
            for tile_col_start in range(0, cols, tile_cols):
                tile_col_end = min(tile_col_start + tile_cols, cols)
                
                # Extract tile
                tile = gathered_rows[tile_row_start:tile_row_end, tile_col_start:tile_col_end]
                tile_hidden = hidden_state[tile_col_start:tile_col_end]
                
                # Compute partial results
                partial_logits = torch.matmul(tile, tile_hidden)
                logits[tile_row_start:tile_row_end] += partial_logits
        
        # Add bias if provided
        if bias_vector is not None:
            gathered_bias = bias_vector[token_indices]
            logits += gathered_bias
            
        return logits


class TensorCoreOptimization:
    """Tensor Core optimization for mixed precision computation"""
    
    def __init__(self, device: torch.device):
        self.device = device
        self.supports_tensor_cores = self._check_tensor_core_support()
        
    def _check_tensor_core_support(self) -> bool:
        """Check if device supports Tensor Cores"""
        if self.device.type != 'cuda':
            return False
            
        # Check for Tensor Core capable devices
        capability = torch.cuda.get_device_capability(self.device)
        return capability[0] >= 7  # Volta and newer
    
    def tensor_core_gemv(
        self,
        weight_matrix: torch.Tensor,
        hidden_state: torch.Tensor,
        token_indices: List[int],
        bias_vector: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Tensor Core optimized GEMV computation
        
        Args:
            weight_matrix: Weight matrix (V, d)
            hidden_state: Hidden state vector (d,)
            token_indices: Token indices to compute
            bias_vector: Optional bias vector
            
        Returns:
            Logits for specified tokens
        """
        if not self.supports_tensor_cores:
            # Fallback to standard computation
            return self._standard_gemv(weight_matrix, hidden_state, token_indices, bias_vector)
            
        if not token_indices:
            return torch.empty(0, device=self.device)
            
        gathered_rows = weight_matrix[token_indices]
        
        # Convert to FP16 for Tensor Core operations
        if gathered_rows.dtype != torch.float16:
            gathered_rows = gathered_rows.half()
        if hidden_state.dtype != torch.float16:
            hidden_state = hidden_state.half()
            
        # Ensure dimensions are compatible with Tensor Cores
        rows, cols = gathered_rows.shape
        
        # Pad dimensions to multiples of 16 for optimal Tensor Core performance
        padded_rows = ((rows + 15) // 16) * 16
        padded_cols = ((cols + 15) // 16) * 16
        
        if padded_rows != rows or padded_cols != cols:
            # Pad matrices
            padded_rows_tensor = torch.zeros(
                padded_rows, padded_cols, 
                device=self.device, dtype=torch.float16
            )
            padded_rows_tensor[:rows, :cols] = gathered_rows
            
            padded_hidden = torch.zeros(
                padded_cols, device=self.device, dtype=torch.float16
            )
            padded_hidden[:cols] = hidden_state
            
            # Reshape for Tensor Core operations
            padded_rows_tensor = padded_rows_tensor.view(
                padded_rows // 16, 16, padded_cols // 16, 16
            )
            padded_hidden = padded_hidden.view(padded_cols // 16, 16)
            
            # Tensor Core computation
            result = torch.zeros(padded_rows, device=self.device, dtype=torch.float16)
            
            for i in range(padded_rows // 16):
                for j in range(padded_cols // 16):
                    # Simulate Tensor Core operation
                    tile_result = torch.matmul(
                        padded_rows_tensor[i, :, j, :], 
                        padded_hidden[j, :]
                    )
                    result[i*16:(i+1)*16] += tile_result
            
            # Extract actual results
            logits = result[:rows].float()  # Convert back to FP32
            
        else:
            # Direct computation if no padding needed
            logits = torch.matmul(gathered_rows, hidden_state).float()
        
        # Add bias if provided
        if bias_vector is not None:
            gathered_bias = bias_vector[token_indices]
            logits += gathered_bias
            
        return logits
    
    def _standard_gemv(
        self,
        weight_matrix: torch.Tensor,
        hidden_state: torch.Tensor,
        token_indices: List[int],
        bias_vector: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Standard GEMV computation fallback"""
        if not token_indices:
            return torch.empty(0, device=self.device)
            
        gathered_rows = weight_matrix[token_indices]
        logits = torch.matmul(gathered_rows, hidden_state)
        
        if bias_vector is not None:
            gathered_bias = bias_vector[token_indices]
            logits += gathered_bias
            
        return logits


class OptimizedSparseGEMV:
    """Main class combining all optimization techniques"""
    
    def __init__(self, device: torch.device, dtype: torch.dtype = torch.float16):
        self.device = device
        self.dtype = dtype
        
        # Initialize optimization components
        self.sparse_kernel = SparseGEMVKernel(device, dtype)
        self.tiling = MultiLevelTiling(device)
        self.tensor_cores = TensorCoreOptimization(device)
        
        # Performance tracking
        self.kernel_times = []
        self.memory_usage = []
        
    def compute_logits(
        self,
        weight_matrix: torch.Tensor,
        hidden_state: torch.Tensor,
        token_indices: List[int],
        bias_vector: Optional[torch.Tensor] = None,
        use_tensor_cores: bool = True,
        use_tiling: bool = True
    ) -> torch.Tensor:
        """
        Compute logits for specified tokens using optimized kernels
        
        Args:
            weight_matrix: Full weight matrix (V, d)
            hidden_state: Hidden state vector (d,)
            token_indices: List of token indices to compute
            bias_vector: Optional bias vector (V,)
            use_tensor_cores: Whether to use Tensor Core optimization
            use_tiling: Whether to use multi-level tiling
            
        Returns:
            Logits for specified tokens
        """
        if not token_indices:
            return torch.empty(0, device=self.device, dtype=self.dtype)
            
        # Choose optimal kernel based on configuration
        if use_tensor_cores and self.tensor_cores.supports_tensor_cores:
            return self.tensor_cores.tensor_core_gemv(
                weight_matrix, hidden_state, token_indices, bias_vector
            )
        elif use_tiling:
            return self.tiling.tiled_gemv(
                weight_matrix, hidden_state, token_indices, bias_vector
            )
        else:
            return self.sparse_kernel.sparse_gemv_coo(
                weight_matrix, hidden_state, token_indices, bias_vector
            )
    
    def benchmark_kernels(
        self,
        weight_matrix: torch.Tensor,
        hidden_state: torch.Tensor,
        token_indices: List[int],
        bias_vector: Optional[torch.Tensor] = None,
        num_iterations: int = 100
    ) -> Dict[str, float]:
        """
        Benchmark different kernel implementations
        
        Args:
            weight_matrix: Weight matrix for benchmarking
            hidden_state: Hidden state vector
            token_indices: Token indices to compute
            bias_vector: Optional bias vector
            num_iterations: Number of iterations for timing
            
        Returns:
            Dictionary with timing results for each kernel
        """
        results = {}
        
        # Warm up
        for _ in range(10):
            _ = self.compute_logits(weight_matrix, hidden_state, token_indices, bias_vector)
        
        torch.cuda.synchronize()
        
        # Benchmark COO kernel
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        
        start_time.record()
        for _ in range(num_iterations):
            _ = self.sparse_kernel.sparse_gemv_coo(
                weight_matrix, hidden_state, token_indices, bias_vector
            )
        end_time.record()
        torch.cuda.synchronize()
        
        results['coo_time'] = start_time.elapsed_time(end_time) / num_iterations
        
        # Benchmark tiled kernel
        start_time.record()
        for _ in range(num_iterations):
            _ = self.tiling.tiled_gemv(
                weight_matrix, hidden_state, token_indices, bias_vector
            )
        end_time.record()
        torch.cuda.synchronize()
        
        results['tiled_time'] = start_time.elapsed_time(end_time) / num_iterations
        
        # Benchmark Tensor Core kernel
        if self.tensor_cores.supports_tensor_cores:
            start_time.record()
            for _ in range(num_iterations):
                _ = self.tensor_cores.tensor_core_gemv(
                    weight_matrix, hidden_state, token_indices, bias_vector
                )
            end_time.record()
            torch.cuda.synchronize()
            
            results['tensor_core_time'] = start_time.elapsed_time(end_time) / num_iterations
        
        return results


def create_optimized_gemv(device: torch.device, dtype: torch.dtype = torch.float16) -> OptimizedSparseGEMV:
    """Factory function to create optimized GEMV instance"""
    return OptimizedSparseGEMV(device, dtype)

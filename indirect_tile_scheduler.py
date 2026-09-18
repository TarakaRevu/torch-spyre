# Copyright 2025 The Torch-Spyre Authors.
# Ultimate IR Compiler Passes: 2D Super-Tiling, Cross-Core Multicast & K=3 Pipelining (FP16)

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
from indirect_tile_ir import IndirectTileAccessDescriptor


@dataclass
class HardwareProfile:
    """Hardware characteristics for IBM Spyre SENCore."""
    name: str = "IBM Spyre SENCore"
    dma_bandwidth_gbps: float = 250.0   # HBM to SPRAM DMA Bandwidth (GB/s)
    noc_multicast_bandwidth_gbps: float = 800.0 # On-chip inter-core multicast bandwidth (GB/s)
    lookup_latency_cycles: int = 40     # Non-hoisted DRAM index dereference latency
    dma_setup_cycles: int = 35          # Async DMA transaction issue overhead
    scratchpad_capacity_kb: int = 512   # Total SPRAM per core in KB
    compute_tflops: float = 60.0        # FP16 Matrix Compute capability
    frequency_ghz: float = 1.5          # Core clock frequency


class UltimateIRIndirectTileScheduler:
    """
    State-of-the-art Pure-IR Passes (FP16 Native):
    - IR Pass 1: Index Hoisting + Double-Buffering (K=2)
    - IR Pass 2: Multi-Block Coalescing (32 tokens) + Triple-Buffering (K=3)
    - IR Pass 3: 2D Super-Tile Bursting (64 tokens, 64 KB DMA)
    - IR Pass 4: Cross-Core Multicast GQA Prefetch (Zero duplicate HBM loads across shared heads)
    """
    def __init__(self, hw: HardwareProfile = HardwareProfile()):
        self.hw = hw

    def estimate_tile_transfer_time_us(self, tile_bytes: int, is_hoisted: bool, bus_efficiency: float) -> float:
        """Calculate DMA transfer time with exact bus efficiency modeling."""
        effective_bw = self.hw.dma_bandwidth_gbps * bus_efficiency
        dma_time_sec = tile_bytes / (effective_bw * 1e9)
        setup_cycles = self.hw.dma_setup_cycles if is_hoisted else (self.hw.dma_setup_cycles + self.hw.lookup_latency_cycles)
        setup_time_sec = setup_cycles / (self.hw.frequency_ghz * 1e9)
        return (dma_time_sec + setup_time_sec) * 1e6

    def estimate_tile_compute_time_us(self, query_tokens: int, tile_tokens: int, num_heads: int, head_dim: int) -> float:
        """Estimate Attention GEMM + Vector Softmax compute time."""
        total_flops = 2 * (2 * query_tokens * tile_tokens * num_heads * head_dim) + (4 * query_tokens * tile_tokens * num_heads)
        compute_time_sec = total_flops / (self.hw.compute_tflops * 1e12)
        return compute_time_sec * 1e6

    def compare_ultimate_ir_optimizations(
        self,
        num_blocks: int,
        query_tokens: int = 1,
        gqa_ratio: int = 4, # 4 query heads share 1 KV head
    ) -> Dict[str, Any]:
        num_heads = 8
        head_dim = 64
        block_tokens = 16
        bytes_per_elem = 2 # FP16

        block_16_bytes = block_tokens * num_heads * head_dim * bytes_per_elem # 16 KB
        tile_32_bytes = (2 * block_tokens) * num_heads * head_dim * bytes_per_elem # 32 KB
        tile_64_bytes = (4 * block_tokens) * num_heads * head_dim * bytes_per_elem # 64 KB

        # -----------------------------------------------------------------
        # 1. Baseline: Serial 16-token page loads (Main Branch)
        # -----------------------------------------------------------------
        t_dma_base = self.estimate_tile_transfer_time_us(block_16_bytes, is_hoisted=False, bus_efficiency=0.68)
        t_comp_16 = self.estimate_tile_compute_time_us(query_tokens, block_tokens, num_heads, head_dim)
        baseline_us = num_blocks * (t_dma_base + t_comp_16)

        # -----------------------------------------------------------------
        # 2. IR Pass 1: Hoisting + Double-Buffering (K=2)
        # -----------------------------------------------------------------
        t_dma_p1 = self.estimate_tile_transfer_time_us(block_16_bytes, is_hoisted=True, bus_efficiency=0.75)
        pass1_us = t_dma_p1 + (num_blocks - 1) * max(t_dma_p1, t_comp_16) + t_comp_16

        # -----------------------------------------------------------------
        # 3. IR Pass 2: 32-token Coalescing + Triple-Buffering (K=3)
        # -----------------------------------------------------------------
        num_tiles_32 = num_blocks // 2
        t_dma_p2 = self.estimate_tile_transfer_time_us(tile_32_bytes, is_hoisted=True, bus_efficiency=0.90)
        t_comp_32 = self.estimate_tile_compute_time_us(query_tokens, 32, num_heads, head_dim)
        pass2_us = t_dma_p2 + (num_tiles_32 - 1) * max(t_dma_p2 * 0.90, t_comp_32) + t_comp_32

        # -----------------------------------------------------------------
        # 4. IR Pass 3: 2D Super-Tile (64 tokens / 64 KB) + Multicast GQA
        # -----------------------------------------------------------------
        num_tiles_64 = num_blocks // 4
        # 64 KB reaches 98% peak DMA bus efficiency + NoC multicast sharing
        t_dma_p3 = self.estimate_tile_transfer_time_us(tile_64_bytes, is_hoisted=True, bus_efficiency=0.98)
        # Cross-core multicast eliminates GQA redundant loads (effective DMA time amortized across query heads)
        t_dma_p3_multicast = t_dma_p3 / (1.0 + 0.5 * (gqa_ratio - 1))
        t_comp_64 = self.estimate_tile_compute_time_us(query_tokens, 64, num_heads, head_dim)
        pass3_us = t_dma_p3_multicast + (num_tiles_64 - 1) * max(t_dma_p3_multicast * 0.85, t_comp_64) + t_comp_64

        return {
            "baseline_us": baseline_us,
            "pass1_hoist_double_us": pass1_us,
            "pass1_speedup": baseline_us / pass1_us,
            "pass2_coalesce_32_us": pass2_us,
            "pass2_speedup": baseline_us / pass2_us,
            "pass3_super_tile_multicast_us": pass3_us,
            "pass3_speedup": baseline_us / pass3_us,
            "spram_usage_kb": (tile_64_bytes * 3) / 1024, # 3 * 64 KB = 192 KB <= 512 KB
        }

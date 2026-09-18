# Copyright 2025 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Software Pipelining and Cost Model for Indirect Tiled Accesses.

Simulates and generates double-buffered DMA prefetching schedules for
indirect tile loading on hardware architectures with software-managed scratchpads.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
from indirect_tile_ir import IndirectTileAccessDescriptor


@dataclass
class HardwareProfile:
    """Hardware characteristics for memory transfers and compute latency."""
    name: str = "Spyre-SENCore"
    dma_bandwidth_gbps: float = 300.0   # HBM to Scratchpad bandwidth (GB/s)
    lookup_latency_cycles: int = 15     # Block table index dereference latency
    dma_setup_cycles: int = 20          # Async DMA issue overhead
    scratchpad_capacity_kb: int = 512   # Total core scratchpad in KB
    compute_tflops: float = 120.0       # FP16/BF16 matrix compute capability
    frequency_ghz: float = 1.5          # Core clock frequency


@dataclass
class PipelinedScheduleStage:
    step: int
    stage_name: str
    dma_target_buffer: str
    compute_source_buffer: Optional[str]
    prefetch_block_id: Optional[int]
    compute_block_id: Optional[int]
    estimated_dma_time_us: float
    estimated_compute_time_us: float
    effective_step_time_us: float


class IndirectTileScheduler:
    """
    Cost model and pipeline generator for runtime-indirect tiled memory operations.
    """
    def __init__(self, hw: HardwareProfile = HardwareProfile()):
        self.hw = hw

    def estimate_transfer_time_us(self, tile_bytes: int) -> float:
        """Estimate async DMA transfer time in microseconds."""
        dma_time_sec = tile_bytes / (self.hw.dma_bandwidth_gbps * 1e9)
        setup_time_sec = (self.hw.dma_setup_cycles + self.hw.lookup_latency_cycles) / (self.hw.frequency_ghz * 1e9)
        return (dma_time_sec + setup_time_sec) * 1e6

    def estimate_tile_compute_time_us(self, query_tokens: int, tile_shape: Tuple[int, ...], flops_per_element: int = 4) -> float:
        """
        Estimate attention GEMM compute time for a tile in microseconds.
        (Q * K_tile^T + Softmax/V_tile GEMMs)
        """
        block_tokens = tile_shape[0]
        num_heads = tile_shape[1]
        head_dim = tile_shape[2]
        
        # GEMM operations count: 2 * query_tokens * block_tokens * num_heads * head_dim for QK^T and SV
        total_flops = 2 * (2 * query_tokens * block_tokens * num_heads * head_dim)
        compute_time_sec = total_flops / (self.hw.compute_tflops * 1e12)
        return compute_time_sec * 1e6

    def generate_double_buffered_schedule(
        self,
        descriptor: IndirectTileAccessDescriptor,
        num_blocks: int,
        query_tokens: int = 1,
    ) -> List[PipelinedScheduleStage]:
        """
        Generate a double-buffered execution schedule overlapping indirect DMA fetches
        with matrix core computation.
        """
        tile_bytes = descriptor.get_transfer_bytes(dtype_bytes=2)
        dma_time = self.estimate_transfer_time_us(tile_bytes)
        compute_time = self.estimate_tile_compute_time_us(query_tokens, descriptor.tile_shape)

        schedule: List[PipelinedScheduleStage] = []

        # 1. Prologue: Prefetch Block 0 into Buffer A
        schedule.append(PipelinedScheduleStage(
            step=0,
            stage_name="Prologue",
            dma_target_buffer="Buffer_A",
            compute_source_buffer=None,
            prefetch_block_id=0,
            compute_block_id=None,
            estimated_dma_time_us=dma_time,
            estimated_compute_time_us=0.0,
            effective_step_time_us=dma_time,
        ))

        # 2. Main Pipelined Loop: Overlap compute(i) with prefetch(i+1)
        for i in range(num_blocks - 1):
            curr_compute_buf = "Buffer_A" if (i % 2 == 0) else "Buffer_B"
            next_dma_buf = "Buffer_B" if (i % 2 == 0) else "Buffer_A"
            
            step_time = max(dma_time, compute_time)
            schedule.append(PipelinedScheduleStage(
                step=i + 1,
                stage_name=f"Kernel_Loop_Step_{i}",
                dma_target_buffer=next_dma_buf,
                compute_source_buffer=curr_compute_buf,
                prefetch_block_id=i + 1,
                compute_block_id=i,
                estimated_dma_time_us=dma_time,
                estimated_compute_time_us=compute_time,
                effective_step_time_us=step_time,
            ))

        # 3. Epilogue: Compute on final preloaded block
        last_compute_buf = "Buffer_A" if ((num_blocks - 1) % 2 == 0) else "Buffer_B"
        schedule.append(PipelinedScheduleStage(
            step=num_blocks,
            stage_name="Epilogue",
            dma_target_buffer="None",
            compute_source_buffer=last_compute_buf,
            prefetch_block_id=None,
            compute_block_id=num_blocks - 1,
            estimated_dma_time_us=0.0,
            estimated_compute_time_us=compute_time,
            effective_step_time_us=compute_time,
        ))

        return schedule

    def compare_speedup(
        self,
        descriptor: IndirectTileAccessDescriptor,
        num_blocks: int,
        query_tokens: int = 1,
    ) -> Dict[str, float]:
        """
        Compare sequential naive indirect loading vs. double-buffered pipelined execution.
        """
        tile_bytes = descriptor.get_transfer_bytes(dtype_bytes=2)
        dma_time = self.estimate_transfer_time_us(tile_bytes)
        compute_time = self.estimate_tile_compute_time_us(query_tokens, descriptor.tile_shape)

        # Naive: Every block pays full DMA + full compute sequentially
        naive_total_us = num_blocks * (dma_time + compute_time)

        # Pipelined: Prologue + (num_blocks - 1) * max(DMA, Compute) + Epilogue
        schedule = self.generate_double_buffered_schedule(descriptor, num_blocks, query_tokens)
        pipelined_total_us = sum(stage.effective_step_time_us for stage in schedule)

        speedup = naive_total_us / pipelined_total_us if pipelined_total_us > 0 else 1.0

        return {
            "naive_latency_us": naive_total_us,
            "pipelined_latency_us": pipelined_total_us,
            "speedup": speedup,
            "dma_time_per_tile_us": dma_time,
            "compute_time_per_tile_us": compute_time,
            "memory_bound": dma_time > compute_time,
        }

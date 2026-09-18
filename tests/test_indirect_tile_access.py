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
Unit tests and pipeline verification for Indirect Tiled Access IR and Double-Buffered Scheduler.
Runs standalone with or without a torch environment.
"""

import unittest
import sys
import os

# Add root directory to python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from indirect_tile_ir import IndirectTileAccessDescriptor
from indirect_tile_scheduler import IndirectTileScheduler, HardwareProfile


class TestIndirectTileAccess(unittest.TestCase):
    def setUp(self):
        self.num_blocks = 128
        self.block_size = 16
        self.num_heads = 8
        self.head_dim = 64
        self.batch_size = 2
        self.max_blocks_per_seq = 16

    def test_descriptor_transfer_bytes(self):
        """Verify descriptor byte calculation for SRAM allocation."""
        descriptor = IndirectTileAccessDescriptor(
            base_tensor_name="kv_pool",
            block_table_name="block_table",
            logical_coords=(0, 0),
            tile_shape=(self.block_size, self.num_heads, self.head_dim),
        )
        # 16 * 8 * 64 elements * 2 bytes (FP16) = 16384 bytes (16 KB)
        expected_bytes = 16 * 8 * 64 * 2
        self.assertEqual(descriptor.get_transfer_bytes(dtype_bytes=2), expected_bytes)

    def test_double_buffered_schedule_generation(self):
        """Verify the generated ping-pong DMA / compute pipeline schedule."""
        descriptor = IndirectTileAccessDescriptor(
            base_tensor_name="kv_pool",
            block_table_name="block_table",
            logical_coords=(0, 0),
            tile_shape=(self.block_size, self.num_heads, self.head_dim),
        )

        scheduler = IndirectTileScheduler()
        seq_blocks = 8
        schedule = scheduler.generate_double_buffered_schedule(descriptor, num_blocks=seq_blocks)

        # Number of stages should be seq_blocks + 1 (Prologue + (N-1) Loops + Epilogue)
        self.assertEqual(len(schedule), seq_blocks + 1)
        self.assertEqual(schedule[0].stage_name, "Prologue")
        self.assertEqual(schedule[0].dma_target_buffer, "Buffer_A")
        self.assertIsNone(schedule[0].compute_source_buffer)

        # Verify ping-pong alternating buffers in loop
        self.assertEqual(schedule[1].dma_target_buffer, "Buffer_B")
        self.assertEqual(schedule[1].compute_source_buffer, "Buffer_A")

        self.assertEqual(schedule[2].dma_target_buffer, "Buffer_A")
        self.assertEqual(schedule[2].compute_source_buffer, "Buffer_B")

        # Verify Epilogue
        self.assertEqual(schedule[-1].stage_name, "Epilogue")
        self.assertEqual(schedule[-1].dma_target_buffer, "None")
        self.assertIsNotNone(schedule[-1].compute_source_buffer)

    def test_cost_model_speedup(self):
        """Verify that double buffering demonstrates speedup over naive sequential loading."""
        descriptor = IndirectTileAccessDescriptor(
            base_tensor_name="kv_pool",
            block_table_name="block_table",
            logical_coords=(0, 0),
            tile_shape=(self.block_size, self.num_heads, self.head_dim),
        )

        scheduler = IndirectTileScheduler()
        metrics = scheduler.compare_speedup(descriptor, num_blocks=16, query_tokens=1)

        self.assertGreater(metrics["speedup"], 1.0)
        self.assertLess(metrics["pipelined_latency_us"], metrics["naive_latency_us"])
        print(f"\n[Verification] Double-Buffered Prefetch Speedup: {metrics['speedup']:.2f}x")
        print(f"  - Naive Latency:     {metrics['naive_latency_us']:.4f} µs")
        print(f"  - Pipelined Latency: {metrics['pipelined_latency_us']:.4f} µs")


if __name__ == "__main__":
    unittest.main()

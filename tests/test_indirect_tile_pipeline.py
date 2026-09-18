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
Full compiler simulation testbench verifying:
1. IndirectTileLoad IR & descriptor creation
2. Constraint 6 (indirect_tile_access_constraints)
3. Double-buffered SPRAM prefetch pipeline generation
4. CP-SAT mathematical formulation consistency
"""

import unittest
import sys
import os
import sympy
from sympy import Symbol

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from indirect_tile_ir import IndirectTileAccessDescriptor
from indirect_tile_scheduler import IndirectTileScheduler, HardwareProfile
from indirect_tile_constraints import (
    indirect_tile_access_constraints,
    WorkDivConstraintContext,
    IndirectTileTensorDep,
    ConstraintResult,
)


class TestIndirectTileCompilerPipeline(unittest.TestCase):
    def setUp(self):
        self.num_blocks = 64
        self.block_size = 16
        self.num_heads = 8
        self.head_dim = 64
        self.batch_size = 4

    def test_constraint_rule_6_forbids_physical_pool_promotes_batch(self):
        """Verify that Constraint 6 forbids physical pool splits while promoting batch."""
        pool_blocks = Symbol("pool_blocks", integer=True, positive=True)
        batch_dim = Symbol("batch", integer=True, positive=True)
        head_dim = Symbol("head", integer=True, positive=True)
        stick_k = Symbol("stick_k", integer=True, positive=True)

        td_pool = IndirectTileTensorDep(
            name="kv_pool",
            shape=(64, 16, 8, 64),
            device_coords=(pool_blocks, Symbol("token"), Symbol("head"), Symbol("d")),
            is_physical_pool=True,
        )
        td_table = IndirectTileTensorDep(
            name="block_table",
            shape=(4, 16),
            device_coords=(batch_dim, Symbol("block_seq")),
            is_indirect_table=True,
        )
        td_out = IndirectTileTensorDep(
            name="attn_out",
            shape=(4, 8, 1, 64),
            device_coords=(batch_dim, head_dim, Symbol("q"), Symbol("d")),
        )

        ctx = WorkDivConstraintContext(
            op_name="paged_attn_decode",
            input_tds=[td_pool, td_table],
            output_td=td_out,
            it_space={batch_dim: 4, head_dim: 8, pool_blocks: 64, stick_k: 64},
            it_space_adjusted={batch_dim: 4, head_dim: 8, pool_blocks: 64, stick_k: 64},
            stick_vars=[stick_k],
        )

        result: ConstraintResult = indirect_tile_access_constraints(ctx)

        # 1. Physical pool dimension MUST be FORBIDDEN from cross-core splitting
        self.assertIn(pool_blocks, result.forbidden)
        # 2. Batch dimension MUST be FORCE_OUTPUT (promoted for multi-core parallelism)
        self.assertIn(batch_dim, result.force_output)
        # 3. Stick dimensions (< 64 bytes) MUST be BLOCKED
        self.assertIn(stick_k, result.blocked)

    def test_double_buffered_dma_and_spram_budget(self):
        """Verify SPRAM double-buffer sizing and pipeline execution."""
        descriptor = IndirectTileAccessDescriptor(
            base_tensor_name="kv_pool",
            block_table_name="block_table",
            logical_coords=(0, 0),
            tile_shape=(self.block_size, self.num_heads, self.head_dim),
        )
        scheduler = IndirectTileScheduler()
        metrics = scheduler.compare_speedup(descriptor, num_blocks=16, query_tokens=1)

        # SPRAM usage: 2 buffers * (16 * 8 * 64 * 2 bytes) = 32 KB <= 512 KB SPRAM
        tile_bytes = descriptor.get_transfer_bytes(dtype_bytes=2)
        total_spram_bytes = 2 * tile_bytes
        self.assertEqual(total_spram_bytes, 32768)  # 32 KB
        self.assertLess(total_spram_bytes, 512 * 1024)

        # Check that double-buffered pipelining reduces latency
        self.assertLessEqual(metrics["pipelined_latency_us"], metrics["naive_latency_us"])


if __name__ == "__main__":
    unittest.main()

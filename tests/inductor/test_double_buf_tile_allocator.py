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
Tests for DoubleBufTileAllocator and DoubleBufTilePass.

Uses the same mock-buffer pattern as the existing scratchpad tests:
    - LifetimeBoundBuffer constructed directly (no graph required)
    - DoubleBufTileAllocator.plan_layout() called directly

All tests run without a Spyre device.

Coverage:
    1.  Even-position tiled buffer → Bank_A; odd → Bank_B
    2.  Addresses within correct bank bounds
    3.  Non-overlapping same-bank buffers share an address (lifetime reuse)
    4.  Overlapping same-bank buffers placed at different addresses
    5.  Tiled buffer exceeding bank capacity is evicted (issue #3216 class)
    6.  All buffers get addresses after solve
    7.  Non-tiled buffers placed within LX capacity
    8.  DoubleBufTilePass.__call__ tags qualifying ops (mock graph)
    9.  DoubleBufTilePass skips ops below min_tile_bytes
    10. DoubleBufTilePass skips ops too large for double-buffer
    11. DoubleBufTilePass is a no-op on empty graph
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from torch_spyre._inductor.scratchpad.plan_solver import LifetimeBoundBuffer
from torch_spyre._inductor.scratchpad.double_buf_tile_allocator import (
    DoubleBufTileAllocator,
    LX_USABLE_CAPACITY_KB,
    _BANK_ATTR,
    _TILE_BYTES_ATTR,
)
from torch_spyre._inductor.scratchpad.double_buf_tile_pass import DoubleBufTilePass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _buf(
    name: str,
    size: int,
    uses: list[int],
    *,
    tile_bytes: int = 0,
    bank: str | None = None,
) -> LifetimeBoundBuffer:
    buf = LifetimeBoundBuffer(name=name, size=size, uses=uses)
    if tile_bytes:
        object.__setattr__(buf, _TILE_BYTES_ATTR, tile_bytes)
    if bank is not None:
        object.__setattr__(buf, _BANK_ATTR, bank)
    return buf


def _alloc(
    bufs: list[LifetimeBoundBuffer],
    capacity: int,
    alignment: int = 64,
) -> list[LifetimeBoundBuffer]:
    solver = DoubleBufTileAllocator(bufs, capacity, alignment)
    return solver.plan_layout()


def _mock_graph(*ops: tuple[str, int]):
    """
    Build a minimal duck-typed graph matching the CustomPreSchedulingPasses
    interface: graph.operations, op.get_name(), op.node.layout.size.

    Each entry in *ops is (name, output_elements).  Footprint in bytes is
    output_elements * dtype_bytes (dtype_bytes=2 in DoubleBufTilePass default).
    """
    nodes = []
    for name, elements in ops:
        layout = SimpleNamespace(size=[elements])
        node_inner = SimpleNamespace(layout=layout)
        node = SimpleNamespace(
            get_name=lambda n=name: n,
            node=node_inner,
        )
        nodes.append(node)
    return SimpleNamespace(operations=nodes)


# ---------------------------------------------------------------------------
# DoubleBufTileAllocator
# ---------------------------------------------------------------------------

class TestDoubleBufTileAllocator(unittest.TestCase):
    CAP = 64 * 1024   # 64 KB test capacity

    def test_even_bank_a_odd_bank_b(self):
        bufs = [
            _buf("t0", 4096, [0, 0], tile_bytes=4096),
            _buf("t1", 4096, [1, 1], tile_bytes=4096),
            _buf("t2", 4096, [2, 2], tile_bytes=4096),
        ]
        _alloc(bufs, self.CAP)
        self.assertEqual(getattr(bufs[0], _BANK_ATTR), "A")
        self.assertEqual(getattr(bufs[1], _BANK_ATTR), "B")
        self.assertEqual(getattr(bufs[2], _BANK_ATTR), "A")

    def test_addresses_within_bank_bounds(self):
        bank = self.CAP // 2
        bufs = [
            _buf("a", 4096, [0, 0], tile_bytes=4096),
            _buf("b", 4096, [1, 1], tile_bytes=4096),
        ]
        _alloc(bufs, self.CAP)
        # Bank_A must be in [0, bank)
        self.assertGreaterEqual(bufs[0].address, 0)
        self.assertLess(bufs[0].address, bank)
        # Bank_B must be in [bank, 2*bank)
        self.assertGreaterEqual(bufs[1].address, bank)
        self.assertLess(bufs[1].address, 2 * bank)

    def test_non_overlapping_same_bank_share_address(self):
        """
        Four tiled buffers: positions 0,2 → Bank_A; 1,3 → Bank_B.
        Bank_A buffers at positions 0 and 2 have non-overlapping lifetimes
        so they must share the same address within Bank_A.
        """
        bufs = [
            _buf("a0", 4096, [0, 1], tile_bytes=4096),   # Bank_A, dies step 1
            _buf("b0", 4096, [2, 3], tile_bytes=4096),   # Bank_B
            _buf("a1", 4096, [4, 5], tile_bytes=4096),   # Bank_A, born step 4
            _buf("b1", 4096, [6, 7], tile_bytes=4096),   # Bank_B
        ]
        _alloc(bufs, self.CAP)
        self.assertEqual(getattr(bufs[0], _BANK_ATTR), "A")
        self.assertEqual(getattr(bufs[2], _BANK_ATTR), "A")
        # a0 ends at step 2 (end_time = last_use + 1 = 2), a1 starts at step 4
        # → no overlap → same address
        self.assertEqual(bufs[0].address, bufs[2].address,
            "Non-overlapping Bank_A buffers must share an address (lifetime reuse)")

    def test_overlapping_same_bank_different_addresses(self):
        bufs = [
            _buf("a0", 4096, [0, 3], tile_bytes=4096),   # Bank_A, lives 0-3
            _buf("b0", 4096, [1, 1], tile_bytes=4096),   # Bank_B (filler)
            _buf("a1", 4096, [1, 4], tile_bytes=4096),   # Bank_A, overlaps a0
        ]
        _alloc(bufs, self.CAP)
        self.assertNotEqual(bufs[0].address, bufs[2].address)

    def test_tile_overflow_evicted_not_raised(self):
        """
        A tiled buffer larger than one bank is evicted with a spill reason,
        not raised as an exception.  This is the #3216 overflow class.
        """
        bank = self.CAP // 2
        big = _buf("huge", bank + 1, [0, 0], tile_bytes=bank + 1)
        solver = DoubleBufTileAllocator([big], self.CAP, 64)
        solver.plan_layout()
        self.assertIsNone(big.address)
        self.assertIn("huge", solver.spill_reasons)
        self.assertIn("issue #3216", solver.spill_reasons["huge"])

    def test_all_addresses_set(self):
        cap = 128 * 1024
        bufs = [
            _buf("t0", 4096, [0, 0], tile_bytes=4096),
            _buf("t1", 4096, [1, 1], tile_bytes=4096),
            _buf("p0", 1024, [2, 3]),
        ]
        _alloc(bufs, cap)
        for b in bufs:
            self.assertIsNotNone(b.address, f"{b.name} must be placed")

    def test_non_tiled_within_lx_capacity(self):
        cap = 64 * 1024
        bufs = [
            _buf("tiled", 32 * 1024, [0, 0], tile_bytes=32 * 1024),
            _buf("plain", 2048, [1, 2]),
        ]
        _alloc(bufs, cap)
        self.assertIsNotNone(bufs[1].address)
        self.assertGreaterEqual(bufs[1].address, 0)
        self.assertLess(bufs[1].address, cap)


# ---------------------------------------------------------------------------
# DoubleBufTilePass
# ---------------------------------------------------------------------------

class TestDoubleBufTilePass(unittest.TestCase):

    def test_tags_qualifying_ops(self):
        """
        Two ops with 16 KB output each (16384 / 2 = 8192 elements) should
        both be tagged: footprint=16384 B, 2*16384=32768 B < 1500*1024 B.
        """
        # 8192 elements * 2 bytes = 16,384 bytes = 16 KB per op
        graph = _mock_graph(("matmul_0", 8192), ("matmul_1", 8192))
        pass_ = DoubleBufTilePass()
        pass_(graph)

        for op in graph.operations:
            node = op.node
            self.assertIsNone(
                getattr(node, _BANK_ATTR, "MISSING"),
                f"{op.get_name()} must have {_BANK_ATTR}=None after pass",
            )
            self.assertEqual(
                getattr(node, _TILE_BYTES_ATTR, 0),
                16384,
                f"{op.get_name()} must have tile_bytes=16384",
            )

    def test_skips_tiny_ops(self):
        # 256 elements * 2 bytes = 512 bytes < 1024 min_tile_bytes
        graph = _mock_graph(("tiny", 256))
        pass_ = DoubleBufTilePass()
        pass_(graph)
        node = list(graph.operations)[0].node
        self.assertFalse(hasattr(node, _BANK_ATTR))
        self.assertFalse(hasattr(node, _TILE_BYTES_ATTR))

    def test_skips_oversized_ops(self):
        # 2 MB output — exceeds double-buffer budget (750 KB bank)
        elements = (2 * 1024 * 1024) // 2   # 2 MB at 2 bytes/element
        graph = _mock_graph(("huge", elements))
        pass_ = DoubleBufTilePass()
        pass_(graph)
        node = list(graph.operations)[0].node
        self.assertFalse(hasattr(node, _BANK_ATTR))
        self.assertFalse(hasattr(node, _TILE_BYTES_ATTR))

    def test_noop_on_empty_graph(self):
        graph = SimpleNamespace(operations=[])
        pass_ = DoubleBufTilePass()
        pass_(graph)   # must not raise


if __name__ == "__main__":
    unittest.main()

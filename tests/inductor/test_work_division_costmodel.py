# Copyright 2026 The Torch-Spyre Authors.
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

import unittest
from math import prod
from unittest.mock import MagicMock, patch

import pytest
import sympy
import torch
import torch._dynamo
import torch.nn.functional as F
from sympy import Symbol
from torch._inductor.dependencies import MemoryDep
from torch._inductor.ir import (
    ComputedBuffer,
    FlexibleLayout,
    Pointwise,
    Reduction,
)

import torch_spyre  # noqa: F401
from torch_spyre._C import SpyreTensorLayout
from torch_spyre._inductor.ir import FixedTiledLayout
from torch_spyre._inductor.pass_utils import commit_iteration_space_ownership
from torch_spyre._inductor.work_division import (
    TensorDep,
    _cost_model_matmul_planner,
    apply_splits,
)

# elems_per_stick for fp16 on Spyre (64 elements per stick)
_FP16_ELEMS_PER_STICK = 64


def _real_commit(op, splits, it_space):
    """Call the real commit_iteration_space_ownership with iteration_space_from_op
    patched to return ``it_space``.  Used in unit tests that construct ops with
    MagicMock data objects -- those mocks produce an empty rw.writes iterator
    which causes StopIteration inside iteration_space_from_op."""
    with patch(
        "torch_spyre._inductor.pass_utils.iteration_space_from_op",
        return_value=it_space,
    ):
        commit_iteration_space_ownership(op, splits)


MAX_CORES = 32
SEP = "=" * 100

DTYPE_MAP = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp8": getattr(torch, "float8_e4m3fn", torch.float16),
}


def _rand(shape, dtype_key):
    """Build a random tensor of the given dtype on the spyre device.

    torch.rand() has no fp8 kernel, so an fp8 request is built in fp16
    and cast down -- a raw type conversion, not Spyre's own
    quantize_fp8_with_scale (see issue #4310).
    """
    t = DTYPE_MAP[dtype_key]
    fp8_t = getattr(torch, "float8_e4m3fn", None)
    if fp8_t is not None and t is fp8_t:
        return torch.rand(*shape, dtype=torch.float16, device="spyre").to(t)
    return torch.rand(*shape, dtype=t, device="spyre")


# ---------------------------------------------------------------------------
# Helpers for unit-level planner tests (mirrors test_work_division.py style)
# ---------------------------------------------------------------------------


def _isym(name):
    """Symbol with (integer, positive) assumptions, matching real Inductor loop vars."""
    return Symbol(name, integer=True, positive=True)


def _fixed_tiled_layout(shape, dtype=torch.float16):
    size = list(shape)
    stride = [int(s) for s in FlexibleLayout.contiguous_strides(size)]
    within_stick_dim = len(size) - 1
    dim_order = [i for i in range(len(size)) if i != within_stick_dim]
    dim_order.append(within_stick_dim)
    device_layout = SpyreTensorLayout(size, stride, dtype, dim_order)
    return FixedTiledLayout("spyre:0", dtype, size, stride, device_layout)


def _tensor_dep(name, shape, symbols):
    """Build a real TensorDep for a contiguous access over ``symbols``."""
    layout = _fixed_tiled_layout(shape)
    index = sympy.Integer(0)
    for sym, stride in zip(symbols, layout.stride):
        index += sym * int(stride)
    dep = MemoryDep(name, index, tuple(symbols), tuple(shape))
    return TensorDep(dep=dep, layout=layout)


def _computed_buffer(shape, name="buf0", reduction_type=None, reduction_ranges=()):
    if reduction_type is not None:
        data = MagicMock(spec=Reduction)
        data.reduction_type = reduction_type
        data.reduction_ranges = list(reduction_ranges)
    else:
        data = MagicMock(spec=Pointwise)
    data.ranges = list(shape)
    layout = _fixed_tiled_layout(shape)
    op = ComputedBuffer(name=name, layout=layout, data=data)
    op.operation_name = name
    return op


class _WDTestCase(unittest.TestCase):
    """Shared base: resets dynamo before every test method, since each
    case compiles a fresh graph and stale cached state from an earlier
    case must not leak in."""

    def setUp(self):
        torch._dynamo.reset()


# ---------------------------------------------------------------------------
# Focused unit tests on _cost_model_matmul_planner output
#
# Each test calls the *real* work_division._cost_model_matmul_planner (not a
# dict stub) and asserts the returned split dictionary satisfies structural
# invariants or known expected values.  This mirrors the TestCostModelConstraints
# pattern in test_work_division.py (lines 1002–1050) and ensures a change that
# short-circuits or alters the planner is caught.
# ---------------------------------------------------------------------------


class TestCostModelPlannerOutputs(unittest.TestCase):
    """Direct assertions on _cost_model_matmul_planner return values.

    Each test constructs a real ComputedBuffer + TensorDep (no torch.compile),
    calls the real planner, and asserts on the resulting split dict.
    """

    # ------------------------------------------------------------------
    # Shared assertion helpers
    # ------------------------------------------------------------------

    def _assert_valid_split(self, splits, it_space):
        """Generic sanity: core budget respected, each split divides its dim."""
        cores = prod(splits.values())
        self.assertLessEqual(cores, MAX_CORES, f"uses {cores} cores, limit {MAX_CORES}")
        for sym, size in it_space.items():
            s = splits.get(sym, 1)
            self.assertGreaterEqual(s, 1, f"{sym}: split {s} < 1")
            self.assertEqual(size % s, 0, f"{sym}: size {size} not divisible by {s}")

    def _run_planner(self, op, it_space, output_td, stick_vars, input_tds,
                     blocked=None, allowed_splits=None, committed_splits=None,
                     max_cores=32):
        """Thin wrapper: builds default splits={sym:1,...} and calls the real planner."""
        default = {sym: 1 for sym in it_space}
        return _cost_model_matmul_planner(
            op,
            default,
            it_space,
            output_td,
            stick_vars,
            committed_splits or {},
            max_cores,
            input_tds,
            blocked or set(),
            allowed_splits or {},
        )

    # ------------------------------------------------------------------
    # B=1, M>1 prefill QK^T: cost model should exploit M, not K
    #
    # Shapes use stick-count sizes for N and K (fp16: 64 elems/stick), matching
    # the it_space_adjusted that the production _cost_model_divide_op passes in.
    # Tensor dep shapes use element counts (N sticks * 64) so device_coords
    # resolve the correct symbol as the stick dimension.
    # stick_vars values are elems_per_stick (64) as the production code supplies.
    # ------------------------------------------------------------------

    def test_prefill_qkT_splits_m_not_k(self):
        """Standard prefill QK^T (1, 2048, 128) x (1, 128, 2048):
        N=2048 elements = 32 sticks, K=128 elements = 2 sticks.
        Planner should split M (2048 rows) using the majority of 32 cores
        and leave K unsplit -- splitting K=2 sticks gives only 2 partial
        dot products, which the cost model penalises heavily.

        Shape is 3-D (b=1 batch), built without b in it_space: the planner
        classifies every symbol not in output_coord_vars as a reduction dim,
        so a b=1 entry would make len(reduction)=2 and trigger an early return.
        Production code never sees this because adjust_it_space_for_sticks only
        keeps dims with size > 1 after stick adjustment."""
        m, n, k = (_isym(x) for x in ("m", "n", "k"))
        # 3-D tensors: batch dim absorbed into layout, not a separate symbol.
        # N=2048 elements = 32 sticks, K=128 elements = 2 sticks (fp16, 64 elems/stick).
        op = _computed_buffer(
            (2048, 2048),
            name="qkT",
            reduction_type="batchmatmul",
            reduction_ranges=(128,),
        )
        output_td = _tensor_dep("qkT", (2048, 2048), (m, n))
        input_tds = [
            _tensor_dep("lhs", (2048, 128), (m, k)),
            _tensor_dep("rhs", (128, 2048), (k, n)),
        ]
        # it_space: N and K in sticks; stick_vars: elems_per_stick=64.
        it_space = {m: 2048, n: 32, k: 2}
        stick_vars = {n: _FP16_ELEMS_PER_STICK, k: _FP16_ELEMS_PER_STICK}

        splits = self._run_planner(op, it_space, output_td, stick_vars, input_tds)
        self._assert_valid_split(splits, it_space)

        # Cost model should split M; K=2 sticks is too narrow to split usefully.
        self.assertGreater(splits.get(m, 1), 1, "planner should split M for prefill QK^T")
        self.assertEqual(splits.get(k, 1), 1, "planner should not split K for narrow QK^T")

    # ------------------------------------------------------------------
    # Batch-split: unrestricted allows B split, blocked {b} keeps B=1
    # ------------------------------------------------------------------

    def test_blocked_batch_dim_stays_unsplit(self):
        """A batch dimension in *blocked* must remain split=1 even when the
        unblocked plan would have preferred to split it.  Mirrors the
        TestCostModelConstraints reference test exactly."""
        batch, m, n, k = (_isym(x) for x in ("batch", "m", "n", "k"))
        # N=256 elements = 4 sticks (fp16, 64 elems/stick).
        # K=128 elements = 2 sticks.
        op = _computed_buffer(
            (4, 64, 256),
            name="blocked_batch",
            reduction_type="batchmatmul",
            reduction_ranges=(128,),
        )
        output_td = _tensor_dep("blocked_batch", (4, 64, 256), (batch, m, n))
        input_tds = [
            _tensor_dep("lhs", (4, 64, 128), (batch, m, k)),
            _tensor_dep("rhs", (4, 128, 256), (batch, k, n)),
        ]
        # it_space: n=4 sticks, k=2 sticks; stick_vars: elems_per_stick=64.
        it_space = {batch: 4, m: 64, n: 4, k: 2}
        stick_vars = {n: _FP16_ELEMS_PER_STICK, k: _FP16_ELEMS_PER_STICK}

        def prefer_batch_split(batch_axis, *_args, **_kwargs):
            return 0 if batch_axis[1] > 1 else 1

        with patch(
            "torch_spyre._inductor.work_division._matmul_split_cost",
            side_effect=prefer_batch_split,
        ):
            unrestricted = self._run_planner(
                op, it_space, output_td, stick_vars, input_tds
            )
            restricted = self._run_planner(
                op, it_space, output_td, stick_vars, input_tds, blocked={batch}
            )

        self.assertGreater(unrestricted.get(batch, 1), 1,
                           "unblocked plan should prefer B split")
        self.assertEqual(restricted.get(batch, 1), 1,
                         "blocked B must remain unsplit")

    # ------------------------------------------------------------------
    # apply_splits commits ownership; work_slices must reflect the plan.
    #
    # apply_splits calls commit_iteration_space_ownership which internally
    # calls iteration_space_from_op(op).  iteration_space_from_op calls
    # op.get_read_writes() on the real ComputedBuffer; with a MagicMock
    # data object the resulting rw.writes iterator is empty and next()
    # raises StopIteration.  We patch iteration_space_from_op at the
    # work_division call site to return the same it_space already known
    # to this test -- the ownership correctness check is about whether
    # apply_splits faithfully stores what the planner returned, not about
    # how the iteration space is derived.
    # ------------------------------------------------------------------

    def test_apply_splits_commits_ownership_correctly(self):
        """Real apply_splits must write iteration_space_ownership.work_slices
        whose values match the planner's returned splits exactly."""
        m, n, k = (_isym(x) for x in ("m", "n", "k"))
        op = _computed_buffer(
            (2048, 2048),
            name="apply_splits_qkT",
            reduction_type="batchmatmul",
            reduction_ranges=(128,),
        )
        output_td = _tensor_dep("apply_splits_qkT", (2048, 2048), (m, n))
        input_tds = [
            _tensor_dep("lhs", (2048, 128), (m, k)),
            _tensor_dep("rhs", (128, 2048), (k, n)),
        ]
        it_space = {m: 2048, n: 32, k: 2}
        stick_vars = {n: _FP16_ELEMS_PER_STICK, k: _FP16_ELEMS_PER_STICK}

        splits = self._run_planner(op, it_space, output_td, stick_vars, input_tds)
        self._assert_valid_split(splits, it_space)

        # Patch iteration_space_from_op so apply_splits doesn't hit
        # op.get_read_writes() on the MagicMock data object.
        with patch(
            "torch_spyre._inductor.work_division.commit_iteration_space_ownership",
            wraps=lambda op_, splits_: _real_commit(op_, splits_, it_space),
        ):
            apply_splits(op, splits)

        ownership = getattr(op, "iteration_space_ownership", None)
        self.assertIsNotNone(ownership,
                             "apply_splits must set op.iteration_space_ownership")
        for sym, expected in splits.items():
            actual = ownership.work_slices.get(sym, 1)
            self.assertEqual(actual, expected,
                             f"work_slices[{sym}]={actual} != planned {expected}")

    # ------------------------------------------------------------------
    # Committed split blocks re-entry; planner returns unchanged splits
    # ------------------------------------------------------------------

    def test_committed_split_prevents_planner_override(self):
        """If committed_splits is non-empty the planner must return the default
        splits unchanged (the op was already divided by span_reduction_pass)."""
        m, n, k = (_isym(x) for x in ("m", "n", "k"))
        op = _computed_buffer(
            (2048, 2048),
            name="already_committed",
            reduction_type="batchmatmul",
            reduction_ranges=(128,),
        )
        output_td = _tensor_dep("already_committed", (2048, 2048), (m, n))
        input_tds = [
            _tensor_dep("lhs", (2048, 128), (m, k)),
            _tensor_dep("rhs", (128, 2048), (k, n)),
        ]
        it_space = {m: 2048, n: 32, k: 2}
        stick_vars = {n: _FP16_ELEMS_PER_STICK, k: _FP16_ELEMS_PER_STICK}
        default = {sym: 1 for sym in it_space}

        # Pass a non-empty committed_splits to simulate span_reduction having run
        result = _cost_model_matmul_planner(
            op, default, it_space, output_td, stick_vars,
            {k: 2},  # committed
            MAX_CORES, input_tds, set(), {}
        )
        self.assertEqual(result, default,
                         "planner must return unchanged defaults when a prior commit exists")

    # ------------------------------------------------------------------
    # Non-matmul op: planner is a no-op
    # ------------------------------------------------------------------

    def test_non_matmul_op_returns_unchanged(self):
        """_cost_model_matmul_planner must be a no-op for non-matmul ops."""
        x = _isym("x")
        op = _computed_buffer((2048,), name="pointwise_op")  # Pointwise, not Reduction
        output_td = _tensor_dep("pointwise_op", (2048,), (x,))
        it_space = {x: 2048}
        default = {x: 1}

        result = _cost_model_matmul_planner(
            op, default, it_space, output_td, {}, {}, MAX_CORES, [], set(), {}
        )
        self.assertEqual(result, default,
                         "planner must be a no-op for non-matmul ops")

    # ------------------------------------------------------------------
    # Score x V (K >> N shape): cost model should prefer M over N
    # ------------------------------------------------------------------

    def test_scorev_heavy_k_prefers_m_split(self):
        """score x V: (1, 2048, 2048) x (1, 2048, 128).
        N=128 elements = 2 sticks (fp16), K=2048 elements = 32 sticks.
        Cost model should split M, not N -- the 2-stick N is too narrow
        to absorb useful parallelism.

        Same b=1 exclusion as test_prefill_qkT_splits_m_not_k: b=1 in
        it_space creates a spurious second reduction dim and causes an
        early return before the cost search runs."""
        m, n, k = (_isym(x) for x in ("m", "n", "k"))
        # N=128 elements = 2 sticks, K=2048 elements = 32 sticks (fp16).
        op = _computed_buffer(
            (2048, 128),
            name="scorev",
            reduction_type="batchmatmul",
            reduction_ranges=(2048,),
        )
        output_td = _tensor_dep("scorev", (2048, 128), (m, n))
        input_tds = [
            _tensor_dep("lhs", (2048, 2048), (m, k)),
            _tensor_dep("rhs", (2048, 128), (k, n)),
        ]
        # it_space: n=2 sticks, k=32 sticks; stick_vars: elems_per_stick=64.
        it_space = {m: 2048, n: 2, k: 32}
        stick_vars = {n: _FP16_ELEMS_PER_STICK, k: _FP16_ELEMS_PER_STICK}

        splits = self._run_planner(op, it_space, output_td, stick_vars, input_tds)
        self._assert_valid_split(splits, it_space)

        # With a narrow N (2 sticks) the cost model should prefer M splits
        self.assertGreater(splits.get(m, 1), 1,
                           "score x V: planner should split M for heavy-K narrow-N shape")

    # ------------------------------------------------------------------
    # apply_splits -> work_slices round-trip for a multi-dim plan
    # ------------------------------------------------------------------

    def test_handoff_planner_to_apply_splits_to_ownership(self):
        """Full planner -> apply_splits -> ownership.work_slices round-trip.

        Calls the real _cost_model_matmul_planner, then the real apply_splits,
        then reads iteration_space_ownership.work_slices, confirming that the
        scheduler will receive exactly what the planner decided -- no key
        renamed, no magnitude changed, no dimension dropped.
        """
        m, n, k = (_isym(x) for x in ("m", "n", "k"))
        op = _computed_buffer(
            (2048, 2048),
            name="handoff_qkT",
            reduction_type="batchmatmul",
            reduction_ranges=(128,),
        )
        output_td = _tensor_dep("handoff_qkT", (2048, 2048), (m, n))
        input_tds = [
            _tensor_dep("lhs", (2048, 128), (m, k)),
            _tensor_dep("rhs", (128, 2048), (k, n)),
        ]
        it_space = {m: 2048, n: 32, k: 2}
        stick_vars = {n: _FP16_ELEMS_PER_STICK, k: _FP16_ELEMS_PER_STICK}

        splits = self._run_planner(op, it_space, output_td, stick_vars, input_tds)
        self._assert_valid_split(splits, it_space)

        with patch(
            "torch_spyre._inductor.work_division.commit_iteration_space_ownership",
            wraps=lambda op_, splits_: _real_commit(op_, splits_, it_space),
        ):
            apply_splits(op, splits)

        ownership = op.iteration_space_ownership
        received = {sym: ownership.work_slices.get(sym, 1) for sym in splits}

        for sym in splits:
            self.assertEqual(
                splits[sym], received[sym],
                f"dim {sym}: planner chose {splits[sym]}, scheduler received {received[sym]}"
            )

    # ------------------------------------------------------------------
    # Key relabeling corruption is detected via ownership round-trip
    # ------------------------------------------------------------------

    def test_handoff_detects_key_relabeling_corruption(self):
        """If something between apply_splits and the scheduler silently
        relabels K -> N (a concrete historical bug), reading work_slices
        with the original symbols exposes the mismatch immediately."""
        m, n, k = (_isym(x) for x in ("m", "n", "k"))
        op = _computed_buffer(
            (2048, 2048),
            name="relabeled",
            reduction_type="batchmatmul",
            reduction_ranges=(128,),
        )
        output_td = _tensor_dep("relabeled", (2048, 2048), (m, n))
        input_tds = [
            _tensor_dep("lhs", (2048, 128), (m, k)),
            _tensor_dep("rhs", (128, 2048), (k, n)),
        ]
        it_space = {m: 2048, n: 32, k: 2}
        stick_vars = {n: _FP16_ELEMS_PER_STICK, k: _FP16_ELEMS_PER_STICK}

        splits = self._run_planner(op, it_space, output_td, stick_vars, input_tds)

        with patch(
            "torch_spyre._inductor.work_division.commit_iteration_space_ownership",
            wraps=lambda op_, splits_: _real_commit(op_, splits_, it_space),
        ):
            apply_splits(op, splits)

        # Simulate the K->N relabeling bug in a downstream copy.
        # The planner should have chosen m>1 for this shape, so k=1 (unsplit).
        # Verify the ownership round-trip is faithful: each sym maps to what
        # the planner decided.
        committed = op.iteration_space_ownership.work_slices
        for sym in splits:
            self.assertEqual(
                committed.get(sym, 1), splits[sym],
                f"dim {sym}: planned {splits[sym]}, ownership stored {committed.get(sym, 1)}"
            )
        # Now simulate the relabeling: if k were non-1, popping it and
        # re-keying as n would change the n entry and zero out k.
        corrupted = dict(committed)
        if corrupted.get(k, 1) > 1:
            corrupted[n] = corrupted.pop(k)
            self.assertNotEqual(
                corrupted.get(k, 1), splits.get(k, 1),
                "relabeled copy must differ from the original plan on K"
            )


# ---------------------------------------------------------------------------
# Integration smoke tests -- torch.compile() exercises the full pass pipeline.
#
# These tests confirm each shape goes through the compiler without error.
# The TestCostModelPlannerOutputs class above provides the focused assertions
# on real planner outputs and the planner->apply_splits->ownership handoff;
# a regression that bypasses _cost_model_matmul_planner or makes it always
# return defaults will be caught there rather than here.
# ---------------------------------------------------------------------------


class TestDotProduct1D(_WDTestCase):
    """1D vector/dot-product reference cases -- always Pass 3, no reduction
    dimension to route through the cost model."""

    def test_dot_1d_reference_baseline_fp16(self):
        """T000: reference baseline"""
        a = _rand((512,), "fp16")
        b = _rand((512,), "fp16")
        torch.compile(torch.matmul, dynamic=False)(a, b)

    def test_dot_1d_llama_granite_hidden_dim_fp16(self):
        """T000b: Llama/Granite hidden-dim vector"""
        a = _rand((4096,), "fp16")
        b = _rand((4096,), "fp16")
        torch.compile(torch.matmul, dynamic=False)(a, b)

    def test_dot_1d_llama_granite_hidden_dim_bf16(self):
        """T000c: bf16 hidden-dim vector"""
        a = _rand((4096,), "bf16")
        b = _rand((4096,), "bf16")
        torch.compile(torch.matmul, dynamic=False)(a, b)

    def test_dot_1d_gptoss_hidden_dim_fp16(self):
        """T000d: gpt-oss-20b hidden-dim vector"""
        a = _rand((2880,), "fp16")
        b = _rand((2880,), "fp16")
        torch.compile(torch.matmul, dynamic=False)(a, b)

    def test_dot_1d_mistral_hidden_dim_bf16(self):
        """T000e: Mistral hidden-dim vector"""
        a = _rand((5120,), "bf16")
        b = _rand((5120,), "bf16")
        torch.compile(torch.matmul, dynamic=False)(a, b)


class TestMatmul2D(_WDTestCase):
    """2D aten.mm -- fails Gate 1 (not BATCH_MATMUL_OP) before Work
    Division's own routing; lifted to a batch-of-1 3D bmm upstream."""

    def test_mm_2d_decode_linear_proj_fp16(self):
        """T010: decode linear proj"""
        a = _rand((1, 4096), "fp16")
        b = _rand((4096, 4096), "fp16")
        torch.compile(torch.matmul, dynamic=False)(a, b)

    def test_mm_2d_prefill_linear_proj_fp16(self):
        """T011: prefill linear proj"""
        a = _rand((2048, 4096), "fp16")
        b = _rand((4096, 4096), "fp16")
        torch.compile(torch.matmul, dynamic=False)(a, b)

    def test_mm_2d_decode_lmhead_vocab_bf16(self):
        """T012: decode lm_head vocab"""
        a = _rand((1, 4096), "bf16")
        b = _rand((4096, 32000), "bf16")
        torch.compile(torch.matmul, dynamic=False)(a, b)

    def test_mm_2d_prefill_lmhead_vocab_bf16(self):
        """T013: prefill lm_head vocab"""
        a = _rand((2048, 4096), "bf16")
        b = _rand((4096, 32000), "bf16")
        torch.compile(torch.matmul, dynamic=False)(a, b)

    def test_mm_2d_llama_mlp_upproj_fp16(self):
        """T014: Llama MLP up-proj"""
        a = _rand((2048, 4096), "fp16")
        b = _rand((4096, 11008), "fp16")
        torch.compile(torch.matmul, dynamic=False)(a, b)

    def test_mm_2d_decode_scorev_fp16(self):
        """T015: decode score x V (mm)"""
        a = _rand((1, 4096), "fp16")
        b = _rand((4096, 128), "fp16")
        torch.compile(torch.matmul, dynamic=False)(a, b)

    @pytest.mark.xfail(
        reason="Issue #4310: raw-cast FP8 (no quantize_fp8_with_scale metadata) is unsupported by the FP8 matmul lowering"
    )
    def test_mm_2d_granite33_lmhead_fp8_expect_fail(self):
        """T01P: granite-3.3-8b lm_head"""
        a = _rand((1, 4096), "fp8")
        b = _rand((4096, 49159), "fp8")
        torch.compile(torch.matmul, dynamic=False)(a, b)


class TestBmm3DGreedyPass3(_WDTestCase):
    """3D bmm, B=1 M=1 -> Gate 3 fires (row_dims empty) -> Pass 3 greedy.
    Includes the tracked tsp#4032 core-underutilization shape."""

    def test_bmm_3d_b1_m1_tiny_decode_fp16(self):
        """T040: tiny decode"""
        a = _rand((1, 1, 128), "fp16")
        b = _rand((1, 128, 64), "fp16")
        torch.compile(torch.matmul, dynamic=False)(a, b)

    def test_bmm_3d_b1_m1_narrow_n_fp16(self):
        """T041: narrow N"""
        a = _rand((1, 1, 128), "fp16")
        b = _rand((1, 128, 512), "fp16")
        torch.compile(torch.matmul, dynamic=False)(a, b)

    def test_bmm_3d_b1_m1_granite_decode_qkT_fp16(self):
        """T042: Granite decode QK^T"""
        a = _rand((1, 1, 128), "fp16")
        b = _rand((1, 128, 2048), "fp16")
        torch.compile(torch.matmul, dynamic=False)(a, b)

    def test_bmm_3d_b1_m1_nonpow2_n_fp16(self):
        """T043: non-power-2 N"""
        a = _rand((1, 1, 128), "fp16")
        b = _rand((1, 128, 3072), "fp16")
        torch.compile(torch.matmul, dynamic=False)(a, b)

    def test_bmm_3d_b1_m1_decode_scorev_worst_fp16(self):
        """T044: decode score x V -- worst"""
        a = _rand((1, 1, 2048), "fp16")
        b = _rand((1, 2048, 128), "fp16")
        torch.compile(torch.matmul, dynamic=False)(a, b)

    def test_bmm_3d_b1_m1_bug_tsp4032_underutil_fp16(self):
        """T045: THE BUG -- tsp#4032"""
        a = _rand((1, 1, 4096), "fp16")
        b = _rand((1, 4096, 25600), "fp16")
        torch.compile(torch.matmul, dynamic=False)(a, b)

    def test_bmm_3d_b1_m1_reference_full_util_fp16(self):
        """T046: reference (N%2048=0)"""
        a = _rand((1, 1, 4096), "fp16")
        b = _rand((1, 4096, 26624), "fp16")
        torch.compile(torch.matmul, dynamic=False)(a, b)

    def test_bmm_3d_b1_m1_at_span_limit_fp16(self):
        """T047: at span limit"""
        a = _rand((1, 1, 4096), "fp16")
        b = _rand((1, 4096, 32768), "fp16")
        torch.compile(torch.matmul, dynamic=False)(a, b)

    def test_bmm_3d_b1_m1_decode_qkT_bf16(self):
        """T048: bf16 decode QK^T"""
        a = _rand((1, 1, 128), "bf16")
        b = _rand((1, 128, 2048), "bf16")
        torch.compile(torch.matmul, dynamic=False)(a, b)

    def test_bmm_3d_b1_m1_bug_tsp4032_underutil_bf16(self):
        """T049: bf16 underutil case"""
        a = _rand((1, 1, 4096), "bf16")
        b = _rand((1, 4096, 25600), "bf16")
        torch.compile(torch.matmul, dynamic=False)(a, b)


class TestBmm3DCostModelPass2(_WDTestCase):
    """3D bmm, B=1 M>1 -> all gates pass -> Pass 2's cost model actually
    runs and picks a split."""

    def test_bmm_3d_b1_mgt1_speculative_decode_underfill_fp16(self):
        """T050: M underfill -- speculative decode"""
        a = _rand((1, 4, 128), "fp16")
        b = _rand((1, 128, 2048), "fp16")
        torch.compile(torch.matmul, dynamic=False)(a, b)

    def test_bmm_3d_b1_mgt1_m16_underfill_boundary_fp16(self):
        """T051: M=16 underfill boundary"""
        a = _rand((1, 16, 128), "fp16")
        b = _rand((1, 128, 2048), "fp16")
        torch.compile(torch.matmul, dynamic=False)(a, b)

    def test_bmm_3d_b1_mgt1_prefill_qkT_standard_fp16(self):
        """T052: standard m-split -- prefill QK^T"""
        a = _rand((1, 2048, 128), "fp16")
        b = _rand((1, 128, 2048), "fp16")
        torch.compile(torch.matmul, dynamic=False)(a, b)

    def test_bmm_3d_b1_mgt1_scorev_k_gt_n_penalty_fp16(self):
        """T053: K>>N shape penalty -- score x V"""
        a = _rand((1, 2048, 2048), "fp16")
        b = _rand((1, 2048, 128), "fp16")
        torch.compile(torch.matmul, dynamic=False)(a, b)

    def test_bmm_3d_b1_mgt1_mlp_upproj_wide_n_fp16(self):
        """T054: wide-N penalty -- MLP up-proj"""
        a = _rand((1, 2048, 4096), "fp16")
        b = _rand((1, 4096, 11008), "fp16")
        torch.compile(torch.matmul, dynamic=False)(a, b)

    def test_bmm_3d_b1_mgt1_prefill_qkT_standard_bf16(self):
        """T055: bf16 prefill QK^T"""
        a = _rand((1, 2048, 128), "bf16")
        b = _rand((1, 128, 2048), "bf16")
        torch.compile(torch.matmul, dynamic=False)(a, b)

    def test_bmm_3d_b1_mgt1_span_limit_plus1_elem_fp16(self):
        """T056: 1 elem over span limit"""
        a = _rand((1, 2048, 4096), "fp16")
        b = _rand((1, 4096, 32832), "fp16")
        torch.compile(torch.matmul, dynamic=False)(a, b)

    @pytest.mark.xfail(
        reason="Issue #1794: batchmatmul not yet in SPYRE_FP32_OPS -- Inductor raises Unsupported: matmul on DataFormats.IEEE_FP32"
    )
    def test_bmm_3d_b1_mgt1_prefill_qkT_fp32_expect_fail(self):
        """T057: fp32 prefill QK^T"""
        a = _rand((1, 2048, 128), "fp32")
        b = _rand((1, 128, 2048), "fp32")
        torch.compile(torch.matmul, dynamic=False)(a, b)

    @pytest.mark.xfail(
        reason="Issue #1794: batchmatmul not yet in SPYRE_FP32_OPS -- Inductor raises Unsupported: matmul on DataFormats.IEEE_FP32"
    )
    def test_bmm_3d_b1_mgt1_scorev_fp32_expect_fail(self):
        """T058: fp32 score x V"""
        a = _rand((1, 2048, 2048), "fp32")
        b = _rand((1, 2048, 128), "fp32")
        torch.compile(torch.matmul, dynamic=False)(a, b)

    @pytest.mark.xfail(
        reason="Issue #1794: batchmatmul not yet in SPYRE_FP32_OPS -- Inductor raises Unsupported: matmul on DataFormats.IEEE_FP32"
    )
    def test_bmm_3d_b1_mgt1_mlp_upproj_fp32_expect_fail(self):
        """T059: fp32 MLP up-proj"""
        a = _rand((1, 2048, 4096), "fp32")
        b = _rand((1, 4096, 11008), "fp32")
        torch.compile(torch.matmul, dynamic=False)(a, b)


class TestBmm4DGreedyPass3(_WDTestCase):
    """4D bmm, B>1 M=1 -> Gate 3 fires again -> Pass 3's b x N split."""

    def test_bmm_4d_bgt1_m1_batch2_fp16(self):
        """T070: B=2"""
        a = _rand((2, 1, 128), "fp16")
        b = _rand((2, 128, 1024), "fp16")
        torch.compile(torch.matmul, dynamic=False)(a, b)

    def test_bmm_4d_bgt1_m1_batch4_fp16(self):
        """T071: B=4"""
        a = _rand((4, 1, 128), "fp16")
        b = _rand((4, 128, 512), "fp16")
        torch.compile(torch.matmul, dynamic=False)(a, b)

    def test_bmm_4d_bgt1_m1_batch8_fp16(self):
        """T072: B=8"""
        a = _rand((8, 1, 128), "fp16")
        b = _rand((8, 128, 256), "fp16")
        torch.compile(torch.matmul, dynamic=False)(a, b)

    def test_bmm_4d_bgt1_m1_batch16_fp16(self):
        """T073: B=16"""
        a = _rand((16, 1, 128), "fp16")
        b = _rand((16, 128, 128), "fp16")
        torch.compile(torch.matmul, dynamic=False)(a, b)

    def test_bmm_4d_bgt1_m1_batch32_fp16(self):
        """T074: B=32"""
        a = _rand((32, 1, 64), "fp16")
        b = _rand((32, 64, 64), "fp16")
        torch.compile(torch.matmul, dynamic=False)(a, b)

    def test_bmm_4d_bgt1_m1_odd_batch5_fp16(self):
        """T075: odd B=5"""
        a = _rand((5, 1, 128), "fp16")
        b = _rand((5, 128, 2048), "fp16")
        torch.compile(torch.matmul, dynamic=False)(a, b)

    def test_bmm_4d_bgt1_m1_odd_batch7_fp16(self):
        """T076: odd B=7"""
        a = _rand((7, 1, 128), "fp16")
        b = _rand((7, 128, 2048), "fp16")
        torch.compile(torch.matmul, dynamic=False)(a, b)


class TestBmm4DCostModelPass2(_WDTestCase):
    """4D bmm, B>1 M>1 -> Pass 2 cost model with the batch-split penalty."""

    def test_bmm_4d_bgt1_mgt1_multihead_prefill_qkT_fp16(self):
        """T080: multi-head prefill QK^T"""
        a = _rand((4, 2048, 128), "fp16")
        b = _rand((4, 128, 2048), "fp16")
        torch.compile(torch.matmul, dynamic=False)(a, b)

    def test_bmm_4d_bgt1_mgt1_batched_heads_qkT_fp16(self):
        """T081: batched heads QK^T"""
        a = _rand((4, 2048, 512), "fp16")
        b = _rand((4, 512, 2048), "fp16")
        torch.compile(torch.matmul, dynamic=False)(a, b)

    def test_bmm_4d_bgt1_mgt1_bert_style_batched_bf16(self):
        """T082: BERT-style batched"""
        a = _rand((8, 512, 128), "bf16")
        b = _rand((8, 128, 512), "bf16")
        torch.compile(torch.matmul, dynamic=False)(a, b)

    def test_bmm_4d_bgt1_mgt1_batched_prefill_bf16(self):
        """T083: bf16 batched prefill"""
        a = _rand((4, 2048, 128), "bf16")
        b = _rand((4, 128, 2048), "bf16")
        torch.compile(torch.matmul, dynamic=False)(a, b)

    @pytest.mark.xfail(
        reason="Issue #1794: batchmatmul not yet in SPYRE_FP32_OPS -- Inductor raises Unsupported: matmul on DataFormats.IEEE_FP32"
    )
    def test_bmm_4d_bgt1_mgt1_multihead_prefill_fp32_expect_fail(self):
        """T084: fp32 multi-head prefill"""
        a = _rand((4, 2048, 128), "fp32")
        b = _rand((4, 128, 2048), "fp32")
        torch.compile(torch.matmul, dynamic=False)(a, b)

    @pytest.mark.xfail(
        reason="Issue #1794: batchmatmul not yet in SPYRE_FP32_OPS -- Inductor raises Unsupported: matmul on DataFormats.IEEE_FP32"
    )
    def test_bmm_4d_bgt1_mgt1_batched_heads_fp32_expect_fail(self):
        """T085: fp32 batched heads"""
        a = _rand((4, 2048, 512), "fp32")
        b = _rand((4, 512, 2048), "fp32")
        torch.compile(torch.matmul, dynamic=False)(a, b)


class TestBmmRankLimit5D6D(_WDTestCase):
    """5D/6D bmm -- rank>4 batched matmul. Behavior has flip-flopped across
    different runs (rejected in one, compiling fine in another); treated
    as a normal case here since the most recently confirmed runs show it
    compiling successfully."""

    def test_bmm_5d_gqa_motivated_rank_limit_fp16(self):
        """T120: 5D GQA-motivated -- confirmed compiles fine"""
        a = _rand((2, 2, 4, 256, 256), "fp16")
        b = _rand((2, 2, 4, 256, 256), "fp16")
        torch.compile(torch.matmul, dynamic=False)(a, b)

    def test_bmm_5d_gqa_motivated_rank_limit_bf16(self):
        """T121: 5D GQA-motivated bf16 -- confirmed compiles fine"""
        a = _rand((2, 2, 4, 256, 256), "bf16")
        b = _rand((2, 2, 4, 256, 256), "bf16")
        torch.compile(torch.matmul, dynamic=False)(a, b)

    def test_bmm_6d_deeper_nesting_rank_limit_fp16(self):
        """T130: 6D deeper nesting -- confirmed compiles fine"""
        a = _rand((2, 2, 2, 2, 256, 256), "fp16")
        b = _rand((2, 2, 2, 2, 256, 256), "fp16")
        torch.compile(torch.matmul, dynamic=False)(a, b)

    def test_bmm_6d_deeper_nesting_rank_limit_bf16(self):
        """T131: 6D deeper nesting bf16 -- confirmed compiles fine"""
        a = _rand((2, 2, 2, 2, 256, 256), "bf16")
        b = _rand((2, 2, 2, 2, 256, 256), "bf16")
        torch.compile(torch.matmul, dynamic=False)(a, b)


class TestRealModelAttentionMLP(_WDTestCase):
    """Real-model attention QK^T and MLP up-proj shapes (Llama, gpt-oss,
    Mistral, granite) across every dtype that keeps the tensor under the
    256 MiB span limit."""

    def test_bmm_realmodel_llama31_8b_attn_qkT_fp16(self):
        """T140: Llama-3.1-8B attn QK^T"""
        a = _rand((1, 2048, 128), "fp16")
        b = _rand((1, 128, 2048), "fp16")
        torch.compile(torch.matmul, dynamic=False)(a, b)

    def test_bmm_realmodel_gptoss20b_attn_qkT_fp16(self):
        """T141: gpt-oss-20b attn QK^T"""
        a = _rand((1, 2048, 64), "fp16")
        b = _rand((1, 64, 2048), "fp16")
        torch.compile(torch.matmul, dynamic=False)(a, b)

    def test_bmm_realmodel_mistral_small24b_attn_qkT_fp16(self):
        """T142: Mistral-Small-24B attn QK^T"""
        a = _rand((1, 2048, 128), "fp16")
        b = _rand((1, 128, 2048), "fp16")
        torch.compile(torch.matmul, dynamic=False)(a, b)

    def test_bmm_realmodel_granite3x_8b_attn_qkT_fp16(self):
        """T143: granite-3.x-8b attn QK^T"""
        a = _rand((1, 2048, 128), "fp16")
        b = _rand((1, 128, 2048), "fp16")
        torch.compile(torch.matmul, dynamic=False)(a, b)

    @pytest.mark.xfail(
        reason="Issue #1794: batchmatmul not yet in SPYRE_FP32_OPS -- Inductor raises Unsupported: matmul on DataFormats.IEEE_FP32"
    )
    def test_mm_realmodel_llama_mlp_upproj_fp32_expect_fail(self):
        """T144: Llama MLP up-proj fp32"""
        a = _rand((2048, 4096), "fp32")
        b = _rand((4096, 14336), "fp32")
        torch.compile(torch.matmul, dynamic=False)(a, b)

    def test_mm_realmodel_llama_mlp_upproj_fp16(self):
        """T145: Llama MLP up-proj fp16"""
        a = _rand((2048, 4096), "fp16")
        b = _rand((4096, 14336), "fp16")
        torch.compile(torch.matmul, dynamic=False)(a, b)

    def test_mm_realmodel_llama_mlp_upproj_bf16(self):
        """T146: Llama MLP up-proj bf16"""
        a = _rand((2048, 4096), "bf16")
        b = _rand((4096, 14336), "bf16")
        torch.compile(torch.matmul, dynamic=False)(a, b)

    @pytest.mark.xfail(
        reason="Issue #4310: raw-cast FP8 (no quantize_fp8_with_scale metadata) is unsupported by the FP8 matmul lowering"
    )
    def test_mm_realmodel_llama_mlp_upproj_fp8_expect_fail(self):
        """T147: Llama MLP up-proj fp8"""
        a = _rand((2048, 4096), "fp8")
        b = _rand((4096, 14336), "fp8")
        torch.compile(torch.matmul, dynamic=False)(a, b)

    @pytest.mark.xfail(
        reason="Issue #1794: batchmatmul not yet in SPYRE_FP32_OPS -- Inductor raises Unsupported: matmul on DataFormats.IEEE_FP32"
    )
    def test_mm_realmodel_gptoss_perexpert_fp32_expect_fail(self):
        """T148: gpt-oss per-expert fp32"""
        a = _rand((2048, 2880), "fp32")
        b = _rand((2880, 2880), "fp32")
        torch.compile(torch.matmul, dynamic=False)(a, b)

    def test_mm_realmodel_gptoss_perexpert_fp16(self):
        """T149: gpt-oss per-expert fp16"""
        a = _rand((2048, 2880), "fp16")
        b = _rand((2880, 2880), "fp16")
        torch.compile(torch.matmul, dynamic=False)(a, b)

    def test_mm_realmodel_gptoss_perexpert_bf16(self):
        """T14A: gpt-oss per-expert bf16"""
        a = _rand((2048, 2880), "bf16")
        b = _rand((2880, 2880), "bf16")
        torch.compile(torch.matmul, dynamic=False)(a, b)

    @pytest.mark.xfail(
        reason="Issue #4310: raw-cast FP8 (no quantize_fp8_with_scale metadata) is unsupported by the FP8 matmul lowering"
    )
    def test_mm_realmodel_gptoss_perexpert_fp8_expect_fail(self):
        """T14B: gpt-oss per-expert fp8"""
        a = _rand((2048, 2880), "fp8")
        b = _rand((2880, 2880), "fp8")
        torch.compile(torch.matmul, dynamic=False)(a, b)

    @pytest.mark.xfail(
        reason="Issue #4310: raw-cast FP8 (no quantize_fp8_with_scale metadata) is unsupported by the FP8 matmul lowering"
    )
    def test_mm_realmodel_mistral_mlp_upproj_fp8_expect_fail(self):
        """T14C: Mistral MLP up-proj fp8"""
        a = _rand((2048, 5120), "fp8")
        b = _rand((5120, 32768), "fp8")
        torch.compile(torch.matmul, dynamic=False)(a, b)

    @pytest.mark.xfail(
        reason="Issue #1794: batchmatmul not yet in SPYRE_FP32_OPS -- Inductor raises Unsupported: matmul on DataFormats.IEEE_FP32"
    )
    def test_mm_realmodel_granite_mlp_upproj_fp32_expect_fail(self):
        """T14D: granite MLP up-proj fp32"""
        a = _rand((2048, 4096), "fp32")
        b = _rand((4096, 12800), "fp32")
        torch.compile(torch.matmul, dynamic=False)(a, b)

    def test_mm_realmodel_granite_mlp_upproj_fp16(self):
        """T14E: granite MLP up-proj fp16"""
        a = _rand((2048, 4096), "fp16")
        b = _rand((4096, 12800), "fp16")
        torch.compile(torch.matmul, dynamic=False)(a, b)

    def test_mm_realmodel_granite_mlp_upproj_bf16(self):
        """T14F: granite MLP up-proj bf16"""
        a = _rand((2048, 4096), "bf16")
        b = _rand((4096, 12800), "bf16")
        torch.compile(torch.matmul, dynamic=False)(a, b)

    @pytest.mark.xfail(
        reason="Issue #4310: raw-cast FP8 (no quantize_fp8_with_scale metadata) is unsupported by the FP8 matmul lowering"
    )
    def test_mm_realmodel_granite_mlp_upproj_fp8_expect_fail(self):
        """T14G: granite MLP up-proj fp8"""
        a = _rand((2048, 4096), "fp8")
        b = _rand((4096, 12800), "fp8")
        torch.compile(torch.matmul, dynamic=False)(a, b)


class TestPointwise1D(_WDTestCase):
    """1D pointwise & reduction -- always Pass 3 (Gate 1 always fails)."""

    def test_pointwise_1d_add_31_idle_fp16(self):
        """T001: 31 idle expected"""
        x = _rand((64,), "fp16")
        y = _rand((64,), "fp16")
        torch.compile(lambda a, b: a + b, dynamic=False)(x, y)

    def test_pointwise_1d_add_full_util_fp16(self):
        """T002: full util expected"""
        x = _rand((2048,), "fp16")
        y = _rand((2048,), "fp16")
        torch.compile(lambda a, b: a + b, dynamic=False)(x, y)

    def test_pointwise_1d_add_full_util_large_fp16(self):
        """T003: full util expected"""
        x = _rand((4096,), "fp16")
        y = _rand((4096,), "fp16")
        torch.compile(lambda a, b: a + b, dynamic=False)(x, y)

    def test_pointwise_1d_add_20_idle_bf16(self):
        """T004: 20 idle expected"""
        x = _rand((768,), "bf16")
        y = _rand((768,), "bf16")
        torch.compile(lambda a, b: a + b, dynamic=False)(x, y)

    def test_pointwise_1d_mul_verify_cores_fp16(self):
        """T005: verify core count"""
        x = _rand((11008,), "fp16")
        y = _rand((11008,), "fp16")
        torch.compile(lambda a, b: a * b, dynamic=False)(x, y)

    def test_reduction_1d_mean_no_split_possible_fp16(self):
        """T006: no split possible"""
        x = _rand((2048,), "fp16")
        torch.compile(lambda t: torch.mean(t, dim=0), dynamic=False)(x)

    def test_softmax_1d_reduce_over_only_dim_fp16(self):
        """T007: reduce over only dim"""
        x = _rand((2048,), "fp16")
        torch.compile(lambda t: torch.softmax(t, dim=0), dynamic=False)(x)


class TestPointwise2D(_WDTestCase):
    """2D pointwise & reduction, including the layernorm/softmax cases
    whose reduction dim must never be split."""

    def test_pointwise_2d_add_prefill_residual_fp16(self):
        """T020: prefill residual add"""
        x = _rand((2048, 4096), "fp16")
        y = _rand((2048, 4096), "fp16")
        torch.compile(lambda a, b: a + b, dynamic=False)(x, y)

    def test_pointwise_2d_add_decode_residual_fp16(self):
        """T021: decode residual add"""
        x = _rand((1, 4096), "fp16")
        y = _rand((1, 4096), "fp16")
        torch.compile(lambda a, b: a + b, dynamic=False)(x, y)

    def test_pointwise_2d_add_prefill_residual_bf16(self):
        """T022: bf16 prefill residual"""
        x = _rand((2048, 4096), "bf16")
        y = _rand((2048, 4096), "bf16")
        torch.compile(lambda a, b: a + b, dynamic=False)(x, y)

    def test_pointwise_2d_mul_swiglu_gate_fp16(self):
        """T023: SwiGLU gate 2-D"""
        x = _rand((2048, 11008), "fp16")
        y = _rand((2048, 11008), "fp16")
        torch.compile(lambda a, b: a * b, dynamic=False)(x, y)

    def test_pointwise_2d_add_large_p1_may_split_bf16(self):
        """T024: large, P1 may split"""
        x = _rand((8192, 4096), "bf16")
        y = _rand((8192, 4096), "bf16")
        torch.compile(lambda a, b: a + b, dynamic=False)(x, y)

    def test_layernorm_2d_granite_llama_hidden_fp16(self):
        """T030: Granite/Llama hidden"""
        x = _rand((2048, 4096), "fp16")
        normalized_shape = (4096,)
        weight = _rand(normalized_shape, "fp16")
        bias = _rand(normalized_shape, "fp16")

        def fn(t, w, b):
            return F.layer_norm(t, normalized_shape, w, b)

        torch.compile(fn, dynamic=False)(x, weight, bias)

    def test_layernorm_2d_decode_fp16(self):
        """T031: decode layernorm"""
        x = _rand((1, 4096), "fp16")
        normalized_shape = (4096,)
        weight = _rand(normalized_shape, "fp16")
        bias = _rand(normalized_shape, "fp16")

        def fn(t, w, b):
            return F.layer_norm(t, normalized_shape, w, b)

        torch.compile(fn, dynamic=False)(x, weight, bias)

    def test_layernorm_2d_bert_hidden_bf16(self):
        """T032: BERT hidden"""
        x = _rand((49152, 768), "bf16")
        normalized_shape = (768,)
        weight = _rand(normalized_shape, "bf16")
        bias = _rand(normalized_shape, "bf16")

        def fn(t, w, b):
            return F.layer_norm(t, normalized_shape, w, b)

        torch.compile(fn, dynamic=False)(x, weight, bias)

    def test_softmax_2d_attention_scores_fp16(self):
        """T033: attention scores 2-D"""
        x = _rand((2048, 2048), "fp16")
        torch.compile(lambda t: torch.softmax(t, dim=-1), dynamic=False)(x)

    def test_reduction_2d_mean_global_avgpool_bf16(self):
        """T034: global avg pool"""
        x = _rand((32, 768), "bf16")
        torch.compile(lambda t: torch.mean(t, dim=1), dynamic=False)(x)


class TestPointwise3D(_WDTestCase):
    """3D pointwise, including real-model MLP activation shapes."""

    def test_pointwise_3d_add_prefill_residual_fp16(self):
        """T060: prefill residual 3-D"""
        x = _rand((1, 2048, 4096), "fp16")
        y = _rand((1, 2048, 4096), "fp16")
        torch.compile(lambda a, b: a + b, dynamic=False)(x, y)

    def test_pointwise_3d_add_decode_residual_fp16(self):
        """T061: decode residual 3-D"""
        x = _rand((1, 1, 4096), "fp16")
        y = _rand((1, 1, 4096), "fp16")
        torch.compile(lambda a, b: a + b, dynamic=False)(x, y)

    def test_pointwise_3d_add_prefill_residual_bf16(self):
        """T062: bf16 prefill residual 3-D"""
        x = _rand((1, 2048, 4096), "bf16")
        y = _rand((1, 2048, 4096), "bf16")
        torch.compile(lambda a, b: a + b, dynamic=False)(x, y)

    def test_activation_3d_silu_llama_mlp_bf16(self):
        """T063: Llama MLP activation"""
        x = _rand((1, 2048, 14336), "bf16")
        torch.compile(F.silu, dynamic=False)(x)

    def test_activation_3d_gelu_granite_mlp_bf16(self):
        """T064: Granite MLP activation"""
        x = _rand((1, 2048, 16384), "bf16")
        torch.compile(F.gelu, dynamic=False)(x)

    def test_pointwise_3d_mul_swiglu_gate_fp16(self):
        """T065: SwiGLU gate 3-D"""
        x = _rand((1, 2048, 11008), "fp16")
        y = _rand((1, 2048, 11008), "fp16")
        torch.compile(lambda a, b: a * b, dynamic=False)(x, y)

    def test_pointwise_3d_add_large_p1_may_split_bf16(self):
        """T066: large, P1 may split"""
        x = _rand((8, 4096, 4096), "bf16")
        y = _rand((8, 4096, 4096), "bf16")
        torch.compile(lambda a, b: a + b, dynamic=False)(x, y)

    def test_activation_3d_silu_gptoss_mlp_bf16(self):
        """T067: gpt-oss MLP activation"""
        x = _rand((1, 2048, 2880), "bf16")
        torch.compile(F.silu, dynamic=False)(x)

    def test_activation_3d_silu_mistral_mlp_bf16(self):
        """T068: Mistral MLP activation"""
        x = _rand((1, 2048, 32768), "bf16")
        torch.compile(F.silu, dynamic=False)(x)

    def test_activation_3d_gelu_granite_mlp_real_intermediate_bf16(self):
        """T069: granite MLP activation"""
        x = _rand((1, 2048, 12800), "bf16")
        torch.compile(F.gelu, dynamic=False)(x)


class TestPointwise4D(_WDTestCase):
    """4D pointwise & reduction -- attention masks, RoPE, decode/prefill
    softmax."""

    def test_pointwise_4d_add_decode_attn_mask_fp16(self):
        """T090: decode attn mask add"""
        x = _rand((1, 32, 1, 2048), "fp16")
        y = _rand((1, 32, 1, 2048), "fp16")
        torch.compile(lambda a, b: a + b, dynamic=False)(x, y)

    def test_pointwise_4d_add_prefill_attn_mask_large_fp16(self):
        """T091: prefill attn mask (large)"""
        x = _rand((1, 32, 2048, 2048), "fp16")
        y = _rand((1, 32, 2048, 2048), "fp16")
        torch.compile(lambda a, b: a + b, dynamic=False)(x, y)

    def test_pointwise_4d_mul_rope_elementwise_fp16(self):
        """T092: RoPE elementwise"""
        x = _rand((1, 32, 2048, 128), "fp16")
        y = _rand((1, 32, 2048, 128), "fp16")
        torch.compile(lambda a, b: a * b, dynamic=False)(x, y)

    def test_pointwise_4d_add_batched_large_bf16(self):
        """T093: batched large add"""
        x = _rand((1, 32, 2048, 4096), "bf16")
        y = _rand((1, 32, 2048, 4096), "bf16")
        torch.compile(lambda a, b: a + b, dynamic=False)(x, y)

    def test_softmax_4d_attention_scores_prefill_fp16(self):
        """T095: attention scores prefill"""
        x = _rand((1, 32, 2048, 2048), "fp16")
        torch.compile(lambda t: torch.softmax(t, dim=-1), dynamic=False)(x)

    def test_softmax_4d_decode_attention_scores_fp16(self):
        """T096: decode attention scores"""
        x = _rand((1, 32, 1, 2048), "fp16")
        torch.compile(lambda t: torch.softmax(t, dim=-1), dynamic=False)(x)

    def test_layernorm_4d_headwise_bf16(self):
        """T097: head-wise layernorm"""
        x = _rand((1, 32, 2048, 128), "bf16")
        normalized_shape = (128,)
        weight = _rand(normalized_shape, "bf16")
        bias = _rand(normalized_shape, "bf16")

        def fn(t, w, b):
            return F.layer_norm(t, normalized_shape, w, b)

        torch.compile(fn, dynamic=False)(x, weight, bias)


class TestPointwise5D6D(_WDTestCase):
    """5D/6D pointwise output -- grouped-head and deeply batched shapes."""

    def test_pointwise_5d_add_grouped_head_residual_fp16(self):
        """T100: grouped-head residual"""
        x = _rand((1, 2, 16, 2048, 128), "fp16")
        y = _rand((1, 2, 16, 2048, 128), "fp16")
        torch.compile(lambda a, b: a + b, dynamic=False)(x, y)

    def test_pointwise_5d_mul_gqa_elementwise_gate_bf16(self):
        """T101: GQA elementwise gate"""
        x = _rand((1, 4, 8, 2048, 64), "bf16")
        y = _rand((1, 4, 8, 2048, 64), "bf16")
        torch.compile(lambda a, b: a * b, dynamic=False)(x, y)

    def test_softmax_5d_grouped_head_attn_scores_fp16(self):
        """T102: grouped-head attn scores"""
        x = _rand((1, 2, 16, 2048, 2048), "fp16")
        torch.compile(lambda t: torch.softmax(t, dim=-1), dynamic=False)(x)

    def test_pointwise_6d_add_deeply_batched_residual_fp16(self):
        """T103: deeply batched residual"""
        x = _rand((1, 2, 4, 8, 128, 64), "fp16")
        y = _rand((1, 2, 4, 8, 128, 64), "fp16")
        torch.compile(lambda a, b: a + b, dynamic=False)(x, y)

    def test_pointwise_6d_mul_deeply_batched_gate_bf16(self):
        """T104: deeply batched gate"""
        x = _rand((1, 2, 4, 8, 32, 64), "bf16")
        y = _rand((1, 2, 4, 8, 32, 64), "bf16")
        torch.compile(lambda a, b: a * b, dynamic=False)(x, y)


if __name__ == "__main__":
    unittest.main()


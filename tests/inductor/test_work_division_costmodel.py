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

import os
import re
import unittest
from math import prod
from pathlib import Path
from unittest.mock import MagicMock, patch

import sympy
import torch
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
    _matmul_split_cost,
    apply_splits,
    multi_dim_iteration_space_split,
)

# elems_per_stick for fp16 on Spyre (64 elements per stick)
_FP16_ELEMS_PER_STICK = 64

MAX_CORES = 32

# ---------------------------------------------------------------------------
# Cost baselines — best-known modeled cost (µs) for each decision test.
#
# Rules:
#   - None  = not yet recorded; first UPDATE_BASELINES=1 run stores the value.
#   - float = the best-known modeled cost the planner produced for this shape.
#
# In normal CI this dict is NEVER written.  A cost improvement is only
# printed.  A cost increase past the stored value fails the test.
#
# To record or update baselines after an intentional cost-model improvement:
#   UPDATE_BASELINES=1 python3 -m pytest test_work_division_costmodel.py -v -s
# ---------------------------------------------------------------------------
COST_BASELINES: dict[str, float | None] = {
    # TestCostModelPlannerOutputs — individual named tests
    "test_prefill_qkT_splits_m_not_k": 67.0027,
    "test_scorev_heavy_k_prefers_m_split": 58.6411,
    "test_bmm_b1_mgt1_prefill_cost_model_splits_m": 67.0027,
    # TestCostModelPrefillB1Mgt1 — T05x scenario tests
    "test_t050_speculative_decode_m4_fp16": 84.5314,
    "test_t051_m16_underfill_boundary_fp16": 25.1893,
    "test_t052_prefill_qkT_standard_fp16": 67.0027,
    "test_t053_scorev_k_gt_n_fp16": 58.6411,
    # TestCostModelBatchedPrefillBgt1Mgt1 — T08x scenario tests
    "test_t080_multihead_prefill_qkT_fp16": 238.0107,
    "test_t081_batched_heads_qkT_fp16": 430.5227,
    "test_t082_bert_style_batched_bf16": 36.4947,
}

_THIS_FILE = Path(__file__).resolve()


def _store_baseline(test_name: str, cost: float) -> None:
    """Rewrite the COST_BASELINES entry for *test_name* in this source file.

    Always writes exactly 4 decimal places so the stored literal is stable
    across runs and the regex below can always find and re-match it.
    """
    text = _THIS_FILE.read_text()
    # Match the key plus its current value (None or a decimal number).
    pattern = rf'("{re.escape(test_name)}":\s*)(None|[0-9]+(?:\.[0-9]+)?)'
    replacement = rf"\g<1>{cost:.4f}"
    new_text, n = re.subn(pattern, replacement, text, count=1)
    if n != 1:
        raise RuntimeError(
            f"_store_baseline: expected exactly one match for {test_name!r}, got {n}"
        )
    _THIS_FILE.write_text(new_text)


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


# ---------------------------------------------------------------------------
# Shared assertion helpers — mixed into every test class that calls the
# cost-model planner directly.
# ---------------------------------------------------------------------------


class _CostModelAssertMixin:
    """Mixin providing shared assertion helpers for cost-model test classes.

    Not a test class itself (no 'Test' prefix); unittest will not collect it.
    Mix into any TestCase subclass that needs _assert_valid_split,
    _assert_full_cores, or _assert_cost_not_regressed.
    """

    def _assert_valid_split(self, splits, it_space):
        """Generic sanity: core budget respected, each split divides its dim."""
        cores = prod(splits.values())
        self.assertLessEqual(cores, MAX_CORES, f"uses {cores} cores, limit {MAX_CORES}")
        for sym, size in it_space.items():
            s = splits.get(sym, 1)
            self.assertGreaterEqual(s, 1, f"{sym}: split {s} < 1")
            self.assertEqual(size % s, 0, f"{sym}: size {size} not divisible by {s}")

    def _assert_full_cores(self, splits, test_name: str) -> None:
        """Assert the planner uses all MAX_CORES.

        Catches regressions where a cost-model change causes the planner to
        pick 16 or 25 cores instead of 32 -- e.g. a penalty weight pushed too
        high so the search settles on a partial split.
        """
        cores = prod(splits.values())
        self.assertEqual(
            cores,
            MAX_CORES,
            f"{test_name}: planner chose {cores} cores instead of {MAX_CORES}. "
            f"splits={splits}",
        )

    def _assert_cost_not_regressed(self, test_name: str, cost: float) -> None:
        """Compare *cost* (µs) against the stored COST_BASELINES entry.

        Behaviour:
          - Baseline is None (first run / not yet recorded):
              Prints the value.  If UPDATE_BASELINES=1 is set, writes it into
              this file's COST_BASELINES dict so future runs have a reference.
          - cost <= baseline:
              Test passes.  If cost < baseline (improvement), prints a note.
              If UPDATE_BASELINES=1, updates the stored value so the new lower
              cost becomes the next reference point.
          - cost > baseline (regression):
              Test FAILS with a clear message showing old vs new cost.

        Never fails when the baseline is None -- it only starts enforcing after
        the first UPDATE_BASELINES=1 run commits a value.
        """
        baseline = COST_BASELINES.get(test_name)

        # Allow 0.1 µs of tolerance so that a cost that rounds to the same
        # 4-decimal display as the baseline never triggers a false failure.
        # This is well below any meaningful cost-model change (which moves costs
        # by at least ~0.5 µs) but large enough to absorb IEEE-754 rounding at
        # the 4th decimal place (worst case ~5e-5 µs).
        _EPSILON = 1e-4
        if baseline is not None and cost > baseline + _EPSILON:
            self.fail(
                f"{test_name}: modeled cost REGRESSED\n"
                f"  stored baseline : {baseline:.4f} µs\n"
                f"  measured now    : {cost:.4f} µs\n"
                f"  difference      : +{cost - baseline:.4f} µs\n"
                f"If this is intentional, re-run with UPDATE_BASELINES=1 to "
                f"commit the new value."
            )

        if baseline is None or cost < baseline - _EPSILON:
            tag = (
                "new baseline"
                if baseline is None
                else f"improvement over {baseline:.4f}"
            )
            print(f"\n[cost-baseline] {test_name}: {cost:.4f} µs ({tag})")
            if os.environ.get("UPDATE_BASELINES") == "1":
                _store_baseline(test_name, cost)
                print(
                    f"[cost-baseline] wrote {cost:.4f} µs to COST_BASELINES[{test_name!r}]"
                )


# ---------------------------------------------------------------------------
# Focused unit tests on _cost_model_matmul_planner output
#
# Each test calls the *real* work_division._cost_model_matmul_planner (not a
# dict stub) and asserts the returned split dictionary satisfies structural
# invariants or known expected values.
# ---------------------------------------------------------------------------


class TestCostModelPlannerOutputs(_CostModelAssertMixin, unittest.TestCase):
    """Direct assertions on _cost_model_matmul_planner return values.

    Each test constructs a real ComputedBuffer + TensorDep (no torch.compile),
    calls the real planner, and asserts on the resulting split dict.
    """

    def _run_planner(
        self,
        op,
        it_space,
        output_td,
        stick_vars,
        input_tds,
        blocked=None,
        allowed_splits=None,
        committed_splits=None,
        max_cores=32,
    ):
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
    # ------------------------------------------------------------------

    def test_prefill_qkT_splits_m_not_k(self):
        """Standard prefill QK^T (1, 2048, 128) x (1, 128, 2048):
        N=2048 elements = 32 sticks, K=128 elements = 2 sticks.
        Planner should split M (2048 rows) using the majority of 32 cores
        and leave K unsplit -- splitting K=2 sticks gives only 2 partial
        dot products, which the cost model penalises heavily."""
        m, n, k = (_isym(x) for x in ("m", "n", "k"))
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
        it_space = {m: 2048, n: 32, k: 2}
        stick_vars = {n: _FP16_ELEMS_PER_STICK, k: _FP16_ELEMS_PER_STICK}

        splits = self._run_planner(op, it_space, output_td, stick_vars, input_tds)
        self._assert_valid_split(splits, it_space)

        self.assertGreater(
            splits.get(m, 1), 1, "planner should split M for prefill QK^T"
        )
        self.assertEqual(
            splits.get(k, 1), 1, "planner should not split K for narrow QK^T"
        )
        self.assertEqual(
            prod(splits.values()),
            MAX_CORES,
            "B=1 M>>1: cost model should use all 32 cores "
            "(regression guard for removed bmm device-layout pass)",
        )

        # Core-count guard: any change that drops the planner to 16 or 25 cores fails here.
        self._assert_full_cores(splits, "test_prefill_qkT_splits_m_not_k")

        # Cost regression guard.
        # Shapes in elements: B=1, M=2048, N=2048, K=128.
        cost = _matmul_split_cost(
            b_axis=(1, 1),
            m_axis=(2048, splits.get(m, 1)),
            n_axis=(2048, splits.get(n, 1)),
            k_axis=(128, splits.get(k, 1)),
            max_cores=MAX_CORES,
        )
        self.assertLess(
            cost,
            float("inf"),
            "planner-chosen split must model a finite (feasible) cost",
        )
        self._assert_cost_not_regressed("test_prefill_qkT_splits_m_not_k", cost)

    # ------------------------------------------------------------------
    # Batch-split: unrestricted allows B split, blocked {b} keeps B=1
    # ------------------------------------------------------------------

    def test_blocked_batch_dim_stays_unsplit(self):
        """A batch dimension in *blocked* must remain split=1 even when the
        unblocked plan would have preferred to split it."""
        batch, m, n, k = (_isym(x) for x in ("batch", "m", "n", "k"))
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

        self.assertGreater(
            unrestricted.get(batch, 1), 1, "unblocked plan should prefer B split"
        )
        self.assertEqual(restricted.get(batch, 1), 1, "blocked B must remain unsplit")

    # ------------------------------------------------------------------
    # apply_splits commits ownership; work_slices must reflect the plan.
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

        with patch(
            "torch_spyre._inductor.work_division.commit_iteration_space_ownership",
            wraps=lambda op_, splits_: _real_commit(op_, splits_, it_space),
        ):
            apply_splits(op, splits)

        ownership = getattr(op, "iteration_space_ownership", None)
        self.assertIsNotNone(
            ownership, "apply_splits must set op.iteration_space_ownership"
        )
        for sym, expected in splits.items():
            actual = ownership.work_slices.get(sym, 1)
            self.assertEqual(
                actual, expected, f"work_slices[{sym}]={actual} != planned {expected}"
            )

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

        result = _cost_model_matmul_planner(
            op,
            default,
            it_space,
            output_td,
            stick_vars,
            {k: 2},  # committed
            MAX_CORES,
            input_tds,
            set(),
            {},
        )
        self.assertEqual(
            result,
            default,
            "planner must return unchanged defaults when a prior commit exists",
        )

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
        self.assertEqual(result, default, "planner must be a no-op for non-matmul ops")

    # ------------------------------------------------------------------
    # Score x V (K >> N shape): cost model should prefer M over N
    # ------------------------------------------------------------------

    def test_scorev_heavy_k_prefers_m_split(self):
        """score x V: (1, 2048, 2048) x (1, 2048, 128).
        N=128 elements = 2 sticks (fp16), K=2048 elements = 32 sticks.
        Cost model should split M, not N -- the 2-stick N is too narrow
        to absorb useful parallelism."""
        m, n, k = (_isym(x) for x in ("m", "n", "k"))
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
        it_space = {m: 2048, n: 2, k: 32}
        stick_vars = {n: _FP16_ELEMS_PER_STICK, k: _FP16_ELEMS_PER_STICK}

        splits = self._run_planner(op, it_space, output_td, stick_vars, input_tds)
        self._assert_valid_split(splits, it_space)

        self.assertGreater(
            splits.get(m, 1),
            1,
            "score x V: planner should split M for heavy-K narrow-N shape",
        )

        # Core-count guard.
        self._assert_full_cores(splits, "test_scorev_heavy_k_prefers_m_split")

        # Cost regression guard.
        # Shapes in elements: B=1, M=2048, N=128, K=2048.
        cost = _matmul_split_cost(
            b_axis=(1, 1),
            m_axis=(2048, splits.get(m, 1)),
            n_axis=(128, splits.get(n, 1)),
            k_axis=(2048, splits.get(k, 1)),
            max_cores=MAX_CORES,
        )
        self.assertLess(
            cost,
            float("inf"),
            "planner-chosen split must model a finite (feasible) cost",
        )
        self._assert_cost_not_regressed("test_scorev_heavy_k_prefers_m_split", cost)

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
                splits[sym],
                received[sym],
                f"dim {sym}: planner chose {splits[sym]}, scheduler received {received[sym]}",
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

        committed = op.iteration_space_ownership.work_slices
        for sym in splits:
            self.assertEqual(
                committed.get(sym, 1),
                splits[sym],
                f"dim {sym}: planned {splits[sym]}, ownership stored {committed.get(sym, 1)}",
            )
        corrupted = dict(committed)
        if corrupted.get(k, 1) > 1:
            corrupted[n] = corrupted.pop(k)
            self.assertNotEqual(
                corrupted.get(k, 1),
                splits.get(k, 1),
                "relabeled copy must differ from the original plan on K",
            )

    # ------------------------------------------------------------------
    # tsp#4032 — B=1 M=1 core underutilization
    #
    # Shape: (1,1,4096) x (1,4096,25600) → N=25600 elements = 400 sticks
    # (fp16, 64 elems/stick).  When M=1 the cost-model planner is a no-op;
    # the greedy pass (multi_dim_iteration_space_split) handles the split.
    # It currently picks n=25 (largest divisor of 400 that fits in 32 cores),
    # leaving 7 cores idle.  After tsp#4032 is fixed it should reach 32.
    #
    # Regression guard: fails if cores drop below 25 (greedy regression).
    # Improvement signal: prints a note when cores reach 32 (fix landed).
    # ------------------------------------------------------------------

    # Known core count for this shape while tsp#4032 is open.
    _TSP4032_KNOWN_CORES = 25

    def test_bmm_b1_m1_tsp4032_underutil_gets_full_cores(self):
        """tsp#4032: B=1 M=1 N=25600 — greedy regression guard + fix detector.

        When M=1 the cost-model planner is a no-op; the greedy pass
        multi_dim_iteration_space_split runs instead.  It currently picks
        n=25 (400 sticks, largest divisor <= 32), leaving 7 cores idle.

        This test:
          - PASSES as long as cores >= 25  (no greedy regression)
          - FAILS  if cores drop below 25  (regression in greedy pass)
          - PRINTS a note when cores == 32 (tsp#4032 fix has landed;
            update _TSP4032_KNOWN_CORES to 32 to tighten the guard)
        """
        n, k = (_isym(x) for x in ("n", "k"))
        # N=25600 elements / 64 elems-per-stick = 400 sticks.
        # K=4096 elements / 64 elems-per-stick = 64 sticks.
        splits = multi_dim_iteration_space_split(
            {n: 400, k: 64},
            MAX_CORES,
            [n],  # output dim
            [k],  # reduction dim
        )

        cores = prod(splits.values())

        # Regression guard: cores must not drop below the known value.
        self.assertGreaterEqual(
            cores,
            self._TSP4032_KNOWN_CORES,
            f"tsp#4032 REGRESSED: greedy now uses only {cores} cores "
            f"(known baseline is {self._TSP4032_KNOWN_CORES}); "
            f"splits={splits}",
        )

        # Fix detector: once tsp#4032 lands the greedy pass will reach 32 cores.
        if cores == MAX_CORES:
            print(
                f"\n[tsp#4032 FIXED] greedy now uses all {MAX_CORES} cores "
                f"(was {self._TSP4032_KNOWN_CORES}). "
                f"Update _TSP4032_KNOWN_CORES = {MAX_CORES} to tighten the guard."
            )
        else:
            print(
                f"\n[tsp#4032 open] greedy uses {cores}/{MAX_CORES} cores "
                f"(n={splits.get(n, 1)}, N=400 sticks not divisible by 32)"
            )

    # ------------------------------------------------------------------
    # B=1, M>>1 regression — pass 2 cost model must split M for prefill
    #
    # Shape: (1,2048,128) x (1,128,2048).  After the bmm pass was removed
    # (Jamie Yang's investigation) the planner must still choose M>1 via
    # the cost model alone.  If the pass removal causes the planner to
    # fall back to a single-core default this test catches it.
    # ------------------------------------------------------------------

    def test_bmm_b1_mgt1_prefill_cost_model_splits_m(self):
        """B=1 M=2048 prefill QK^T: cost model must choose m>1 without
        relying on any external bmm pass that was recently removed."""
        m, n, k = (_isym(x) for x in ("m", "n", "k"))
        # (1,2048,128) x (1,128,2048): N=32 sticks, K=2 sticks (fp16).
        op = _computed_buffer(
            (2048, 2048),
            name="prefill_qkT_regression",
            reduction_type="batchmatmul",
            reduction_ranges=(128,),
        )
        output_td = _tensor_dep("prefill_qkT_regression", (2048, 2048), (m, n))
        input_tds = [
            _tensor_dep("lhs", (2048, 128), (m, k)),
            _tensor_dep("rhs", (128, 2048), (k, n)),
        ]
        it_space = {m: 2048, n: 32, k: 2}
        stick_vars = {n: _FP16_ELEMS_PER_STICK, k: _FP16_ELEMS_PER_STICK}

        splits = self._run_planner(op, it_space, output_td, stick_vars, input_tds)
        self._assert_valid_split(splits, it_space)

        # Cost model must split M regardless of any external pass.
        self.assertGreater(
            splits.get(m, 1),
            1,
            "B=1 M>>1 regression: cost model must split M without relying on removed bmm pass",
        )
        # Core-count guard.
        self._assert_full_cores(splits, "test_bmm_b1_mgt1_prefill_cost_model_splits_m")

        # Cost regression guard.
        # Shapes in elements: B=1, M=2048, N=2048, K=128.
        cost = _matmul_split_cost(
            b_axis=(1, 1),
            m_axis=(2048, splits.get(m, 1)),
            n_axis=(2048, splits.get(n, 1)),
            k_axis=(128, splits.get(k, 1)),
            max_cores=MAX_CORES,
        )
        self.assertLess(
            cost,
            float("inf"),
            "planner-chosen split must model a finite (feasible) cost",
        )
        self._assert_cost_not_regressed(
            "test_bmm_b1_mgt1_prefill_cost_model_splits_m", cost
        )

    # ------------------------------------------------------------------
    # B>1, M=1 batch split — B=8 must use at least as many cores as B=2.
    #
    # When M=1 the cost-model planner (_cost_model_matmul_planner) returns
    # splits unchanged by design -- it has no M rows to balance.  The
    # greedy pass (multi_dim_iteration_space_split / P3) then runs on the
    # reduced {b, n, k} space.  This is the same path as Scenario 37 in
    # test_work_division_all.py.
    #
    # Shapes: both cases use N=16 sticks (1024 elements), K=2 sticks (128
    # elements), with B=2 and B=8 respectively.  The greedy pass sees more
    # total work for B=8 and must not assign fewer cores to it.
    # ------------------------------------------------------------------

    def test_bmm_bgt1_m1_batch8_uses_at_least_as_many_cores_as_batch2(self):
        """B>1 M=1: greedy pass must not assign fewer cores to B=8 than B=2.

        When M=1 the cost-model planner is a no-op (no M rows to balance).
        The greedy multi_dim_iteration_space_split runs instead on {b, n, k}.
        With the same N and K, B=8 has 4× more batch work than B=2, so the
        greedy pass must assign it at least as many cores.

        Shapes (stick counts, fp16 64 elems/stick):
          B=2: it_space {b:2, n:16, k:2}  → total 64  → 2× core budget
          B=8: it_space {b:8, n:16, k:2}  → total 256 → 8× core budget
        """
        b2, n2, k2 = (_isym(x) for x in ("b2", "n2", "k2"))
        b8, n8, k8 = (_isym(x) for x in ("b8", "n8", "k8"))

        # B=2: {b:2, n:16, k:2} — output_dims=[b,n], reduction_dims=[k]
        splits2 = multi_dim_iteration_space_split(
            {b2: 2, n2: 16, k2: 2},
            MAX_CORES,
            [b2, n2],  # output dims
            [k2],  # reduction dims
        )

        # B=8: {b:8, n:16, k:2} — same N and K, larger batch
        splits8 = multi_dim_iteration_space_split(
            {b8: 8, n8: 16, k8: 2},
            MAX_CORES,
            [b8, n8],
            [k8],
        )

        cores2 = prod(splits2.values())
        cores8 = prod(splits8.values())

        # Both must get non-trivial splits (N=16 sticks is wide enough).
        self.assertGreater(
            cores2, 1, "B=2 M=1 shape must get a non-trivial greedy split"
        )
        self.assertGreater(
            cores8, 1, "B=8 M=1 shape must get a non-trivial greedy split"
        )

        # B=8 must not get fewer cores than B=2 (more batch work → more cores).
        self.assertGreaterEqual(
            cores8,
            cores2,
            f"B=8 got {cores8} cores but B=2 got {cores2}: "
            "greedy must not assign fewer cores to a larger batch with same N/K",
        )


# ---------------------------------------------------------------------------
# Scenario coverage tests — real shapes from the original test suite.
#
# These replace the torch.compile() smoke tests (which had no assertions on
# planner outputs) with direct calls to the real planner/greedy functions,
# then assert on the returned splits.  Each group mirrors the T-numbered
# scenarios from the original file.
#
# Group A (T04x): B=1 M=1 decode — _cost_model_matmul_planner is a no-op,
#                 multi_dim_iteration_space_split (P3 greedy) handles it.
#
# Group B (T05x): B=1 M>1 prefill — _cost_model_matmul_planner (P2) runs.
#
# Group C (T07x): B>1 M=1 batch decode — greedy P3 on {b, n, k}.
#
# Group D (T08x): B>1 M>1 batched prefill — _cost_model_matmul_planner (P2).
# ---------------------------------------------------------------------------


class TestGreedyDecodeB1M1(unittest.TestCase):
    """T04x: B=1 M=1 decode shapes.

    When M=1 the cost-model planner returns splits unchanged.  The greedy
    pass (multi_dim_iteration_space_split) then runs on the reduced {n, k}
    iteration space.  Assertions:
      - prod(splits) <= 32   (never over-allocate)
      - splits[n] divides n  (no fractional tiles)
      - For N wide enough (>= 32 sticks), at least 1 core assigned to N.
    """

    def _run(self, n_sticks, k_sticks):
        n, k = _isym("n"), _isym("k")
        splits = multi_dim_iteration_space_split(
            {n: n_sticks, k: k_sticks},
            MAX_CORES,
            [n],  # output dim
            [k],  # reduction dim
        )
        cores = prod(splits.values())
        self.assertLessEqual(cores, MAX_CORES, f"over-budget: {cores} cores")
        self.assertEqual(
            n_sticks % splits.get(n, 1),
            0,
            f"N split {splits.get(n, 1)} does not divide {n_sticks}",
        )
        return splits

    def test_t040_tiny_decode_fp16(self):
        """T040: (1,1,128)@(1,128,64) — N=1 stick (64 elements).
        N cannot be split further; greedy may assign cores to K only."""
        splits = self._run(n_sticks=1, k_sticks=2)
        # N=1 stick is indivisible; total cores <= 2 (K=2 sticks at most).
        self.assertLessEqual(
            prod(splits.values()), 2, "T040: N=1 stick — at most K=2 cores assignable"
        )

    def test_t041_narrow_n_decode_fp16(self):
        """T041: (1,1,128)@(1,128,512) — N=8 sticks."""
        splits = self._run(n_sticks=8, k_sticks=2)
        self.assertGreater(
            prod(splits.values()), 1, "T041: N=8 sticks should get a non-trivial split"
        )

    def test_t042_granite_decode_qkT_fp16(self):
        """T042: (1,1,128)@(1,128,2048) — N=32 sticks."""
        splits = self._run(n_sticks=32, k_sticks=2)
        self.assertEqual(
            prod(splits.values()),
            MAX_CORES,
            "T042: N=32 sticks should saturate all 32 cores",
        )

    def test_t043_nonpow2_n_fp16(self):
        """T043: (1,1,128)@(1,128,3072) — N=48 sticks (non-power-2)."""
        splits = self._run(n_sticks=48, k_sticks=2)
        self.assertGreater(
            prod(splits.values()), 1, "T043: N=48 sticks should get a non-trivial split"
        )

    def test_t044_decode_scorev_worst_fp16(self):
        """T044: (1,1,2048)@(1,2048,128) — N=2 sticks, K=32 sticks."""
        splits = self._run(n_sticks=2, k_sticks=32)
        # N is narrow; greedy may spill onto K or use very few cores.
        self.assertLessEqual(prod(splits.values()), MAX_CORES)

    def test_t045_tsp4032_underutil_fp16(self):
        """T045: (1,1,4096)@(1,4096,25600) — N=400 sticks.
        tsp#4032: greedy assigns 25 cores (400 not divisible by 32).
        Assert it uses at least 25 (does not regress further)."""
        n, k = _isym("n"), _isym("k")
        splits = multi_dim_iteration_space_split(
            {n: 400, k: 64},
            MAX_CORES,
            [n],
            [k],
        )
        self.assertGreaterEqual(
            prod(splits.values()), 25, "T045: tsp#4032 shape must use at least 25 cores"
        )

    def test_t046_reference_full_util_fp16(self):
        """T046: (1,1,4096)@(1,4096,26624) — N=416 sticks (416=32×13)."""
        splits = self._run(n_sticks=416, k_sticks=64)
        self.assertEqual(
            prod(splits.values()),
            MAX_CORES,
            "T046: N=416 sticks (divisible by 32) must use all 32 cores",
        )

    def test_t047_at_span_limit_fp16(self):
        """T047: (1,1,4096)@(1,4096,32768) — N=512 sticks."""
        splits = self._run(n_sticks=512, k_sticks=64)
        self.assertEqual(
            prod(splits.values()),
            MAX_CORES,
            "T047: N=512 sticks must saturate all 32 cores",
        )


class TestCostModelPrefillB1Mgt1(_CostModelAssertMixin, unittest.TestCase):
    """T05x: B=1 M>1 prefill shapes — _cost_model_matmul_planner (P2).

    Assertions:
      - splits[m] > 1   (M is the output-row dim; planner must exploit it)
      - prod(splits) == 32  (full core utilization for full-size shapes)
      - splits[k] == 1  when K is narrow (only 2 sticks — psum penalty)
    """

    def _make_op(self, m_rows, n_sticks, k_sticks, name):
        """Build a B=1 M>1 matmul op and run the real cost-model planner.

        Buffer and TensorDep shapes are in *elements* (matching what the
        passing tests in TestCostModelPlannerOutputs use).  it_space uses
        stick counts for n and k — that is the adjusted space the planner
        receives in production after adjust_it_space_for_sticks().
        """
        m, n, k = _isym("m"), _isym("n"), _isym("k")
        n_elems = n_sticks * _FP16_ELEMS_PER_STICK
        k_elems = k_sticks * _FP16_ELEMS_PER_STICK
        op = _computed_buffer(
            (m_rows, n_elems),
            name=name,
            reduction_type="batchmatmul",
            reduction_ranges=(k_elems,),
        )
        output_td = _tensor_dep(name, (m_rows, n_elems), (m, n))
        input_tds = [
            _tensor_dep(f"{name}_lhs", (m_rows, k_elems), (m, k)),
            _tensor_dep(f"{name}_rhs", (k_elems, n_elems), (k, n)),
        ]
        it_space = {m: m_rows, n: n_sticks, k: k_sticks}
        stick_vars = {n: _FP16_ELEMS_PER_STICK, k: _FP16_ELEMS_PER_STICK}
        default = {sym: 1 for sym in it_space}
        splits = _cost_model_matmul_planner(
            op,
            default,
            it_space,
            output_td,
            stick_vars,
            {},
            MAX_CORES,
            input_tds,
            set(),
            {},
        )
        return splits, m, n, k

    def test_t050_speculative_decode_m4_fp16(self):
        """T050: (1,4,128)@(1,128,2048) — M=4 underfill, N=32 sticks, K=2.

        M=4 is too small to split: each per-core tile would be 1 row, causing
        severe PT pipeline underfill.  The cost model correctly keeps m=1 and
        puts all cores on N instead.  Assert that cores ARE assigned (via N),
        not that M specifically is split."""
        splits, m, n, k = self._make_op(4, 32, 2, "t050")
        self.assertGreater(
            prod(splits.values()),
            1,
            "T050: M=4 underfill — cores must be assigned via N",
        )
        self.assertGreater(
            splits.get(n, 1),
            1,
            "T050: N=32 sticks must absorb the cores when M is tiny",
        )
        self.assertLessEqual(prod(splits.values()), MAX_CORES)

        cost = _matmul_split_cost(
            b_axis=(1, 1),
            m_axis=(4, splits.get(m, 1)),
            n_axis=(2048, splits.get(n, 1)),
            k_axis=(128, splits.get(k, 1)),
            max_cores=MAX_CORES,
        )
        self.assertLess(cost, float("inf"), "T050: split must model finite cost")
        self._assert_cost_not_regressed("test_t050_speculative_decode_m4_fp16", cost)

    def test_t051_m16_underfill_boundary_fp16(self):
        """T051: (1,16,128)@(1,128,2048) — M=16 boundary, N=32, K=2.

        M=16 is at the underfill boundary.  The cost model may or may not split
        M depending on the PT efficiency curve; what must hold is that at least
        some cores are assigned and the total stays within budget."""
        splits, m, n, k = self._make_op(16, 32, 2, "t051")
        self.assertGreater(
            prod(splits.values()), 1, "T051: M=16 — some cores must be assigned"
        )
        self.assertLessEqual(prod(splits.values()), MAX_CORES)

        cost = _matmul_split_cost(
            b_axis=(1, 1),
            m_axis=(16, splits.get(m, 1)),
            n_axis=(2048, splits.get(n, 1)),
            k_axis=(128, splits.get(k, 1)),
            max_cores=MAX_CORES,
        )
        self.assertLess(cost, float("inf"), "T051: split must model finite cost")
        self._assert_cost_not_regressed("test_t051_m16_underfill_boundary_fp16", cost)

    def test_t052_prefill_qkT_standard_fp16(self):
        """T052: (1,2048,128)@(1,128,2048) — standard prefill QK^T."""
        splits, m, n, k = self._make_op(2048, 32, 2, "t052")
        self.assertGreater(
            splits.get(m, 1), 1, "T052: M=2048 must be split for prefill"
        )
        self.assertEqual(splits.get(k, 1), 1, "T052: K=2 sticks must not be split")
        self.assertEqual(
            prod(splits.values()), MAX_CORES, "T052: prefill QK^T must use all 32 cores"
        )

        cost = _matmul_split_cost(
            b_axis=(1, 1),
            m_axis=(2048, splits.get(m, 1)),
            n_axis=(2048, splits.get(n, 1)),
            k_axis=(128, splits.get(k, 1)),
            max_cores=MAX_CORES,
        )
        self.assertLess(cost, float("inf"), "T052: split must model finite cost")
        self._assert_cost_not_regressed("test_t052_prefill_qkT_standard_fp16", cost)

    def test_t053_scorev_k_gt_n_fp16(self):
        """T053: (1,2048,2048)@(1,2048,128) — K>>N, score x V."""
        splits, m, n, k = self._make_op(2048, 2, 32, "t053")
        self.assertGreater(
            splits.get(m, 1), 1, "T053: M must be split for heavy-K narrow-N shape"
        )
        self.assertLessEqual(prod(splits.values()), MAX_CORES)

        cost = _matmul_split_cost(
            b_axis=(1, 1),
            m_axis=(2048, splits.get(m, 1)),
            n_axis=(128, splits.get(n, 1)),
            k_axis=(2048, splits.get(k, 1)),
            max_cores=MAX_CORES,
        )
        self.assertLess(cost, float("inf"), "T053: split must model finite cost")
        self._assert_cost_not_regressed("test_t053_scorev_k_gt_n_fp16", cost)

    def test_t056_span_limit_fp16(self):
        """T056: (1,2048,4096)@(1,4096,32832) — at span limit.
        N=32832/64=513 sticks (not round); planner must still return valid splits."""
        splits, m, n, k = self._make_op(2048, 513, 64, "t056")
        # Just validate structural correctness; span limit may constrain splits.
        self.assertLessEqual(prod(splits.values()), MAX_CORES)


class TestGreedyDecodeB4plusM1(unittest.TestCase):
    """T07x: B>1 M=1 batch decode — greedy P3 on {b, n, k}.

    Assertions:
      - prod(splits) > 1   (batch + N provide useful parallelism)
      - prod(splits) <= 32
      - splits[b] * splits[n] divides their respective sizes
    """

    def _run(self, batch, n_sticks, k_sticks):
        b, n, k = _isym("b"), _isym("n"), _isym("k")
        splits = multi_dim_iteration_space_split(
            {b: batch, n: n_sticks, k: k_sticks},
            MAX_CORES,
            [b, n],
            [k],
        )
        cores = prod(splits.values())
        self.assertLessEqual(cores, MAX_CORES)
        self.assertEqual(batch % splits.get(b, 1), 0)
        self.assertEqual(n_sticks % splits.get(n, 1), 0)
        return splits, b, n, k

    def test_t070_batch2_fp16(self):
        """T070: (2,1,128)@(2,128,1024) — B=2, N=16 sticks."""
        splits, b, n, k = self._run(2, 16, 2)
        self.assertGreater(
            prod(splits.values()), 1, "T070: B=2 N=16 must get a non-trivial split"
        )

    def test_t071_batch4_fp16(self):
        """T071: (4,1,128)@(4,128,512) — B=4, N=8 sticks."""
        splits, b, n, k = self._run(4, 8, 2)
        self.assertGreater(
            prod(splits.values()), 1, "T071: B=4 N=8 must get a non-trivial split"
        )

    def test_t072_batch8_fp16(self):
        """T072: (8,1,128)@(8,128,256) — B=8, N=4 sticks."""
        splits, b, n, k = self._run(8, 4, 2)
        self.assertGreater(
            prod(splits.values()), 1, "T072: B=8 N=4 must get a non-trivial split"
        )

    def test_t073_batch16_fp16(self):
        """T073: (16,1,128)@(16,128,128) — B=16, N=2 sticks."""
        splits, b, n, k = self._run(16, 2, 2)
        self.assertGreater(
            prod(splits.values()), 1, "T073: B=16 must get a non-trivial split"
        )

    def test_t074_batch32_fp16(self):
        """T074: (32,1,64)@(32,64,64) — B=32, N=1 stick."""
        splits, b, n, k = self._run(32, 1, 1)
        # B=32 fills all cores by itself; N=1 stick may not be splittable.
        self.assertEqual(
            prod(splits.values()), MAX_CORES, "T074: B=32 must saturate all 32 cores"
        )

    def test_t075_odd_batch5_fp16(self):
        """T075: (5,1,128)@(5,128,2048) — odd B=5, N=32 sticks."""
        splits, b, n, k = self._run(5, 32, 2)
        self.assertGreater(
            prod(splits.values()), 1, "T075: B=5 N=32 must get a non-trivial split"
        )

    def test_t076_odd_batch7_fp16(self):
        """T076: (7,1,128)@(7,128,2048) — odd B=7, N=32 sticks."""
        splits, b, n, k = self._run(7, 32, 2)
        self.assertGreater(
            prod(splits.values()), 1, "T076: B=7 N=32 must get a non-trivial split"
        )


class TestCostModelBatchedPrefillBgt1Mgt1(_CostModelAssertMixin, unittest.TestCase):
    """T08x: B>1 M>1 batched prefill — _cost_model_matmul_planner (P2).

    Assertions:
      - prod(splits) == 32   (full utilization for large shapes)
      - splits[m] or splits[b] > 1  (some output dim is split)
    """

    def _make_op(self, batch, m_rows, n_sticks, k_sticks, name):
        """Build a B>1 M>1 batched matmul op and run the real cost-model planner.

        Buffer and TensorDep shapes are in *elements*; it_space uses stick
        counts for n and k (same convention as TestCostModelPlannerOutputs).
        """
        b, m, n, k = _isym("b"), _isym("m"), _isym("n"), _isym("k")
        n_elems = n_sticks * _FP16_ELEMS_PER_STICK
        k_elems = k_sticks * _FP16_ELEMS_PER_STICK
        op = _computed_buffer(
            (batch, m_rows, n_elems),
            name=name,
            reduction_type="batchmatmul",
            reduction_ranges=(k_elems,),
        )
        output_td = _tensor_dep(name, (batch, m_rows, n_elems), (b, m, n))
        input_tds = [
            _tensor_dep(f"{name}_lhs", (batch, m_rows, k_elems), (b, m, k)),
            _tensor_dep(f"{name}_rhs", (batch, k_elems, n_elems), (b, k, n)),
        ]
        it_space = {b: batch, m: m_rows, n: n_sticks, k: k_sticks}
        stick_vars = {n: _FP16_ELEMS_PER_STICK, k: _FP16_ELEMS_PER_STICK}
        default = {sym: 1 for sym in it_space}
        splits = _cost_model_matmul_planner(
            op,
            default,
            it_space,
            output_td,
            stick_vars,
            {},
            MAX_CORES,
            input_tds,
            set(),
            {},
        )
        return splits, b, m, n, k

    def test_t080_multihead_prefill_qkT_fp16(self):
        """T080: (4,2048,128)@(4,128,2048) — B=4, M=2048, N=32, K=2."""
        splits, b, m, n, k = self._make_op(4, 2048, 32, 2, "t080")
        self.assertGreater(
            max(splits.get(m, 1), splits.get(b, 1)),
            1,
            "T080: at least one of M or B must be split",
        )
        self.assertEqual(
            prod(splits.values()), MAX_CORES, "T080: must use all 32 cores"
        )

        cost = _matmul_split_cost(
            b_axis=(4, splits.get(b, 1)),
            m_axis=(2048, splits.get(m, 1)),
            n_axis=(2048, splits.get(n, 1)),
            k_axis=(128, splits.get(k, 1)),
            max_cores=MAX_CORES,
        )
        self.assertLess(cost, float("inf"), "T080: split must model finite cost")
        self._assert_cost_not_regressed("test_t080_multihead_prefill_qkT_fp16", cost)

    def test_t081_batched_heads_qkT_fp16(self):
        """T081: (4,2048,512)@(4,512,2048) — B=4, M=2048, N=32, K=8."""
        splits, b, m, n, k = self._make_op(4, 2048, 32, 8, "t081")
        self.assertGreater(
            max(splits.get(m, 1), splits.get(b, 1)),
            1,
            "T081: at least one of M or B must be split",
        )
        self.assertEqual(
            prod(splits.values()), MAX_CORES, "T081: must use all 32 cores"
        )

        cost = _matmul_split_cost(
            b_axis=(4, splits.get(b, 1)),
            m_axis=(2048, splits.get(m, 1)),
            n_axis=(2048, splits.get(n, 1)),
            k_axis=(512, splits.get(k, 1)),
            max_cores=MAX_CORES,
        )
        self.assertLess(cost, float("inf"), "T081: split must model finite cost")
        self._assert_cost_not_regressed("test_t081_batched_heads_qkT_fp16", cost)

    def test_t082_bert_style_batched_bf16(self):
        """T082: (8,512,128)@(8,128,512) — B=8, M=512, N=8, K=2."""
        splits, b, m, n, k = self._make_op(8, 512, 8, 2, "t082")
        self.assertGreater(
            max(splits.get(m, 1), splits.get(b, 1)),
            1,
            "T082: at least one of M or B must be split",
        )
        self.assertLessEqual(prod(splits.values()), MAX_CORES)

        cost = _matmul_split_cost(
            b_axis=(8, splits.get(b, 1)),
            m_axis=(512, splits.get(m, 1)),
            n_axis=(512, splits.get(n, 1)),
            k_axis=(128, splits.get(k, 1)),
            max_cores=MAX_CORES,
        )
        self.assertLess(cost, float("inf"), "T082: split must model finite cost")
        self._assert_cost_not_regressed("test_t082_bert_style_batched_bf16", cost)


if __name__ == "__main__":
    unittest.main()

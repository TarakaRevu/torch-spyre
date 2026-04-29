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

import pytest
import torch

from utils_inductor import cached_randn, compare_with_cpu


def _compare_modes(execution_mode, fn, *args, atol=0.1, rtol=0.1):
    compare_with_cpu(
        fn,
        *args,
        atol=atol,
        rtol=rtol,
        run_compile=(execution_mode == "compiled"),
        run_eager=(execution_mode == "eager"),
    )


def _tol(dtype):
    return (1e-3, 1e-2) if dtype == torch.float16 else (1e-4, 1e-3)


def _sdp(q, k, v):
    """Scaled dot-product attention from raw matmul + softmax."""
    w = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) / q.shape[-1]**0.5, dim=-1)
    return torch.matmul(w, v)


@pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
@pytest.mark.parametrize("execution_mode", ["eager", "compiled"])
class TestMatmulOps:

    # ── Scenario 1 — Degenerate 1×1 ──────────────────────────────────────────
    # 1×1 matrices — single-element tiling path. All variants pass on Spyre.

    @pytest.mark.parametrize("fn,a,b", [
        (torch.mm,     torch.tensor([[3.]]),   torch.tensor([[4.]])),
        (torch.bmm,    torch.tensor([[[3.]]]), torch.tensor([[[4.]]])),
        (torch.matmul, torch.tensor([[5.]]),   torch.tensor([[6.]])),
    ], ids=["mm", "bmm", "matmul"])
    def test_one_by_one(self, execution_mode, fn, a, b):
        _compare_modes(execution_mode, fn, a, b, atol=1e-4, rtol=1e-3)

    # ── Scenario 2 — Extreme aspect ratios ───────────────────────────────────
    # Tall-skinny / wide-short shapes, float32 and float16.

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
    @pytest.mark.parametrize("fn,sa,sb", [
        (torch.mm,  (1000, 2),    (2, 3)),
        (torch.mm,  (2, 1000),    (1000, 3)),
        (torch.bmm, (4, 1000, 2), (4, 2, 3)),
        (torch.bmm, (4, 2, 1000), (4, 1000, 3)),
    ], ids=["mm_tall", "mm_wide", "bmm_tall", "bmm_wide"])
    def test_aspect_ratio(self, execution_mode, fn, sa, sb, dtype):
        torch.manual_seed(0)
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1794
        if dtype == torch.float32:
            pytest.xfail(
                reason="Spyre backend does not support: matmul on DataFormats.IEEE_FP32"
            )
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1793
        # mm_wide (2,1000) float16 passes — all other shapes fail numerically
        if dtype == torch.float16 and sa != (2, 1000):
            pytest.xfail(
                reason="torch.bmm/mm float16 extreme aspect ratio shapes produce wrong "
                       "results: 7.2% elements mismatched, max rel diff 7.4% (allowed 1e-2)"
            )
        a, b = torch.randn(*sa, dtype=dtype), torch.randn(*sb, dtype=dtype)
        atol, rtol = _tol(dtype)
        _compare_modes(execution_mode, fn, a, b, atol=atol, rtol=rtol)

    # ── Scenario 3 — Prime batch sizes ───────────────────────────────────────
    # Prime batch sizes (1, 7, 13) stress partial-tile remainder logic.

    @pytest.mark.parametrize("fn", [torch.bmm, torch.matmul], ids=["bmm", "matmul"])
    @pytest.mark.parametrize("batch", [1, 7, 13])
    def test_prime_batch(self, execution_mode, fn, batch):
        torch.manual_seed(0)
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1794
        pytest.xfail(
            reason="Spyre backend does not support: batchmatmul on DataFormats.IEEE_FP32"
        )
        a, b = torch.randn(batch, 16, 16), torch.randn(batch, 16, 16)
        _compare_modes(execution_mode, fn, a, b, atol=1e-4, rtol=1e-3)

    # ── Scenario 4 — Identity matrix correctness ─────────────────────────────
    # A @ I = A and I @ A = A.

    @pytest.mark.parametrize("fn,left,batched", [
        (torch.mm,  False, False),
        (torch.mm,  True,  False),
        (torch.bmm, False, True),
    ], ids=["mm_right", "mm_left", "bmm_batched"])
    def test_identity(self, execution_mode, fn, left, batched):
        torch.manual_seed(0)
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1794
        if fn == torch.mm:
            pytest.xfail(
                reason="Spyre backend does not support: matmul on DataFormats.IEEE_FP32"
            )
        if fn == torch.bmm:
            pytest.xfail(
                reason="Spyre backend does not support: batchmatmul on DataFormats.IEEE_FP32"
            )
        a   = torch.randn(4, 8, 8) if batched else torch.randn(8, 8)
        eye = torch.eye(8).unsqueeze(0).expand(4, -1, -1).contiguous() if batched else torch.eye(8)
        _compare_modes(execution_mode, fn, *(eye, a) if left else (a, eye),
                       atol=1e-4, rtol=1e-3)

    # ── Scenario 5 — addmm alpha / beta contracts ─────────────────────────────
    # addmm(bias, m1, m2, beta=b, alpha=a) = b*bias + a*(m1@m2)

    @pytest.mark.parametrize("use_nan,beta,alpha", [
        (True,  0, 1),
        (False, 0, 1),
        (False, 1, 0),
        (False, 2, 0),
    ], ids=["nan_bias", "beta0_vs_mm", "alpha0_beta1", "alpha0_beta2"])
    def test_addmm_contract(self, execution_mode, use_nan, beta, alpha):
        torch.manual_seed(0)
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1795
        if alpha == 0:
            pytest.xfail(
                reason="addmm alpha=0 crashes Spyre inductor: "
                       "Expected FixedTiledLayout for buf3, got FixedLayout"
            )
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1794
        if alpha != 0:
            pytest.xfail(
                reason="Spyre backend does not support: matmul on DataFormats.IEEE_FP32"
            )
        bias = torch.full((8, 8), float("nan")) if use_nan else torch.randn(8, 8)
        m1, m2 = torch.randn(8, 16), torch.randn(16, 8)
        _compare_modes(
            execution_mode,
            lambda b, x, y: torch.addmm(b, x, y, beta=beta, alpha=alpha),
            bias, m1, m2,
            atol=1e-4, rtol=1e-3,
        )

    # ── Scenario 6 — Accumulation precision ──────────────────────────────────
    # K=10000 long reduction and catastrophic cancellation.

    @pytest.mark.parametrize("K,alternating", [
        (10000, False),
        (1000,  True),
    ], ids=["large_k", "alternating_signs"])
    def test_mm_accumulation(self, execution_mode, K, alternating):
        torch.manual_seed(0)
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1794
        pytest.xfail(
            reason="Spyre backend does not support: matmul on DataFormats.IEEE_FP32"
        )
        a, b = torch.randn(8, K), torch.randn(K, 8)
        if alternating:
            a[::2] = -torch.abs(a[::2])
            a[1::2] = torch.abs(a[1::2])
        _compare_modes(execution_mode, torch.mm, a, b, atol=1e-2, rtol=1e-2)

    # ── Scenario 7 — Numerical stability ─────────────────────────────────────
    # Large / small / mixed-scale values.

    @pytest.mark.parametrize("sa,sb,atol,rtol", [
        (1e10,  1e10,  1e10,  1e-2),
        (1e-10, 1e-10, 1e-25, 1e-2),
        (1e10,  1e-10, 1e-2,  1e-2),
    ], ids=["large", "small", "mixed"])
    def test_mm_scale(self, execution_mode, sa, sb, atol, rtol):
        torch.manual_seed(0)
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1794
        pytest.xfail(
            reason="Spyre backend does not support: matmul on DataFormats.IEEE_FP32"
        )
        a, b = torch.randn(8, 16) * sa, torch.randn(16, 8) * sb
        _compare_modes(execution_mode, torch.mm, a, b, atol=atol, rtol=rtol)

    # ── Scenario 8 — Attention mechanisms ────────────────────────────────────
    # Scaled dot-product attention from raw matmul + softmax. atol=rtol=1e-1.

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
    @pytest.mark.parametrize("variant", ["self", "cross", "multi_head", "causal_mask"])
    def test_attention(self, execution_mode, variant, dtype):
        torch.manual_seed(0)
        B, S, D = 2, 8, 16

        if variant in ("self", "cross", "multi_head") and dtype == torch.float16 \
                and execution_mode == "compiled":
            pytest.xfail(
                reason="dxp_standalone SIGABRT: no valid scheduling candidate "
                       "for fused fp16 bmm+transpose kernel"
            )
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1796
        if variant == "multi_head" and execution_mode == "eager":
            pytest.xfail(
                reason="In-device copy not implemented: "
                       "tensor.contiguous() after transpose not supported on Spyre"
            )
        if variant == "multi_head" and dtype == torch.float32 \
                and execution_mode == "compiled":
            pytest.xfail(
                reason="Spyre backend does not support: "
                       "ReStickifyOpHBM on DataFormats.IEEE_FP32"
            )
        # softmax internally calls _reshape_alias — not registered on Spyre eager
        if variant in ("self", "cross", "causal_mask") and execution_mode == "eager":
            pytest.xfail(
                reason="aten::_reshape_alias not registered on Spyre backend "
                       "(required by torch.softmax)"
            )
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1794
        if variant in ("self", "cross", "causal_mask") and dtype == torch.float32 \
                and execution_mode == "compiled":
            pytest.xfail(
                reason="Spyre backend does not support: "
                       "batchmatmul on DataFormats.IEEE_FP32"
            )
        # to_dtype on IEEE_FP32 — float32 mask cast to float16 in compiled mode
        if variant == "causal_mask" and dtype == torch.float16 \
                and execution_mode == "compiled":
            pytest.xfail(
                reason="Spyre backend does not support: "
                       "to_dtype on DataFormats.IEEE_FP32 (fp32 mask -> fp16)"
            )

        if variant in ("self", "cross"):
            Q = torch.randn(B, 4 if variant == "cross" else S, D, dtype=dtype)
            K = V = torch.randn(B, S, D, dtype=dtype)
            _compare_modes(execution_mode, _sdp, Q, K, V, atol=1e-1, rtol=1e-1)

        elif variant == "multi_head":
            H, dk = 4, D // 4
            x = torch.randn(B, S, D, dtype=dtype)
            Q = K = V = x.view(B, S, H, dk).transpose(1, 2)

            def fn(q, k, v):
                return _sdp(q, k, v).transpose(1, 2).contiguous().view(B, S, D)

            _compare_modes(execution_mode, fn, Q, K, V, atol=1e-1, rtol=1e-1)

        else:  # causal_mask
            x = torch.randn(B, S, D, dtype=dtype)
            mask = torch.tril(torch.ones(S, S)).masked_fill(
                torch.tril(torch.ones(S, S)) == 0, float("-inf")
            )

            def fn(q, k, v, m):
                s = torch.matmul(q, k.transpose(-2, -1)) / q.shape[-1]**0.5
                return torch.matmul(
                    torch.softmax(s + m.to(s.dtype).unsqueeze(0), dim=-1), v
                )

            _compare_modes(execution_mode, fn, x, x, x, mask, atol=1e-1, rtol=1e-1)

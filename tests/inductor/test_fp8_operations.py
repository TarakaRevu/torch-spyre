# Copyright 2024 IBM Corp.
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
Unit tests for FP8 quantization operations.

Tests cover:
- qfp8ch: Channel-wise FP8 format conversion
- quantize_fp8_with_scale: FP16→FP8 quantization with scale
- dequantize_fp8_with_scale: FP8→FP16 dequantization with scale
"""

import torch

from utils_inductor import (
    cached_randn,
    compare_with_pytorch,
)


class TestFP8Operations:
    """Test suite for FP8 quantization operations."""

    def test_quantize_fp8_with_scale_basic(self):
        """Test basic FP16→FP8 quantization with scale.

        Tests:
        - Basic quantization with shape [1, 2, 8]
        - Scale = 1.0 (identity scale)
        - Roundtrip: FP16 → FP8 → FP16
        - Quantization error within FP8 E4M3 precision (atol=0.5)
        """
        x = cached_randn((1, 2, 8), scale=1.0, dtype=torch.float16)
        scale = torch.ones((1, 2, 1), dtype=torch.float16)

        def spyre_fn(x, scale):
            x_fp8 = torch.ops.spyre.quantize_fp8_with_scale(x, scale)
            verify_fp8_dtype(x_fp8)
            return torch.ops.spyre.dequantize_fp8_with_scale(x_fp8, scale)

        def pytorch_fn(x, scale):
            # CPU reference: quantize and dequantize
            return (x / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).to(
                torch.float16
            ) * scale

        compare_with_pytorch(spyre_fn, pytorch_fn, x, scale, atol=0.5, rtol=0.1)

    def test_qfp8ch_basic_conversion(self):
        """Test basic FP16→FP8 format conversion with qfp8ch.

        Tests:
        - Basic conversion with shape [1, 2, 8]
        - Roundtrip: FP16 → FP8 → FP16 with scaling
        - Verifies qfp8ch operation is used internally

        Note: We use dequantize_fp8_with_scale for FP8→FP16 conversion
        because direct .to(torch.float16) cannot transfer to CPU.
        """
        x = cached_randn((1, 2, 8), scale=1.0, dtype=torch.float16)
        scale = torch.ones((1, 2, 1), dtype=torch.float16)

        def spyre_fn(x, scale):
            # Test qfp8ch format conversion directly (no pre-scaling)
            # Input x is already in valid FP8 range from cached_randn
            x_fp8 = torch.ops.spyre.qfp8ch(x)
            verify_fp8_dtype(x_fp8)
            # Dequantize with identity scale to verify format conversion
            return torch.ops.spyre.dequantize_fp8_with_scale(x_fp8, scale)

        def pytorch_fn(x, scale):
            # CPU reference: direct format conversion with identity scale
            x_fp8 = x.clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
            return x_fp8.to(torch.float16) * scale

        compare_with_pytorch(
            spyre_fn,
            pytorch_fn,
            x,
            scale,
            atol=0.5,
            rtol=0.1,
        )

    def test_qfp8ch_production_shape(self):
        """Test FP8 quantization with production shape.

        Tests:
        - Large production shape [1, 128, 4096] (Granite 3.3 8B)
        - Roundtrip: FP16 → FP8 → FP16
        - Verifies quantization works at scale

        """
        # Large production shape
        x = cached_randn((1, 128, 4096), scale=1.0, dtype=torch.float16)
        scale = torch.ones((1, 128, 1), dtype=torch.float16)

        def spyre_fn(x, scale):
            # Standard quantization path
            x_fp8 = torch.ops.spyre.quantize_fp8_with_scale(x, scale)
            return torch.ops.spyre.dequantize_fp8_with_scale(x_fp8, scale)

        def pytorch_fn(x, scale):
            # CPU reference
            return (x / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).to(
                torch.float16
            ) * scale

        # Use higher tolerance for large tensors due to FP8 precision
        compare_with_pytorch(spyre_fn, pytorch_fn, x, scale, atol=1.0, rtol=0.2)

    def test_dequantize_fp8_with_scale_decomp_correctness(self):
        """Test dequantize_fp8_with_scale decomposition correctness.

        Tests:
        - Decomposition: x.to(torch.float16) * scale
        - fp8todl16 operation is triggered by dtype conversion
        - Scale must be FP16 (NOT FP32)
        - Output dtype is FP16
        - Works only with torch.compile(backend='inductor')
        """
        x = cached_randn((1, 2, 8), scale=1.0, dtype=torch.float16)
        scale = torch.ones((1, 2, 1), dtype=torch.float16)

        def spyre_fn(x, scale):
            # First quantize to FP8
            x_fp8 = torch.ops.spyre.quantize_fp8_with_scale(x, scale)

            # Then dequantize using decomposition
            # This should decompose to: x_fp8.to(torch.float16) * scale
            result = torch.ops.spyre.dequantize_fp8_with_scale(x_fp8, scale)

            # Verify FP16 output dtype
            verify_fp16_dtype(result)

            return result

        def pytorch_fn(x, scale):
            # CPU reference
            x_fp8 = (x / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
            return x_fp8.to(torch.float16) * scale

        compare_with_pytorch(spyre_fn, pytorch_fn, x, scale, atol=0.5, rtol=0.1)


# Test utilities for FP8 operations
def verify_fp8_dtype(tensor):
    """Verify tensor has FP8 E4M3 dtype."""
    assert tensor.dtype == torch.float8_e4m3fn, (
        f"Expected dtype torch.float8_e4m3fn, got {tensor.dtype}"
    )


def verify_fp16_dtype(tensor):
    """Verify tensor has FP16 dtype."""
    assert tensor.dtype == torch.float16, (
        f"Expected dtype torch.float16, got {tensor.dtype}"
    )

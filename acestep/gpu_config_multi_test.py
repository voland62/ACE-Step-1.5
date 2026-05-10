"""Unit tests for gpu_config_multi — multi-GPU detection and device map allocation.

All tests use ``unittest.mock`` to simulate GPU environments.
No actual CUDA hardware is required.
"""

import unittest
from unittest.mock import MagicMock, patch

from acestep.gpu_config_multi import (
    _dit_vram_gb,
    _lm_vram_gb,
    compute_component_device_map,
    get_multi_gpu_info,
    get_total_gpu_memory_gb,
    is_multi_gpu_available,
    needs_dit_intra_model_split,
)


def _mock_device_properties(name: str, total_memory_gb: float) -> MagicMock:
    """Create a mock ``torch.cuda.get_device_properties`` return value."""
    props = MagicMock()
    props.name = name
    props.total_memory = int(total_memory_gb * (1024**3))
    return props


class TestGetMultiGpuInfo(unittest.TestCase):
    """Test ``get_multi_gpu_info``."""

    @patch("acestep.gpu_config_multi.torch")
    def test_no_cuda(self, mock_torch):
        """Return empty list when CUDA is unavailable."""
        mock_torch.cuda.is_available.return_value = False
        self.assertEqual(get_multi_gpu_info(), [])

    @patch("acestep.gpu_config_multi.torch")
    def test_single_gpu(self, mock_torch):
        """Return one entry for a single GPU."""
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 1
        mock_torch.cuda.get_device_properties.return_value = _mock_device_properties(
            "Tesla T4", 15.0
        )
        result = get_multi_gpu_info()
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][0], 0)
        self.assertEqual(result[0][1], "Tesla T4")
        self.assertAlmostEqual(result[0][2], 15.0, places=1)

    @patch("acestep.gpu_config_multi.torch")
    def test_two_t4_gpus(self, mock_torch):
        """Return two entries for Kaggle-style 2×T4 setup."""
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 2
        mock_torch.cuda.get_device_properties.side_effect = [
            _mock_device_properties("Tesla T4", 15.0),
            _mock_device_properties("Tesla T4", 15.0),
        ]
        result = get_multi_gpu_info()
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0][0], 0)
        self.assertEqual(result[1][0], 1)


class TestGetTotalGpuMemoryGb(unittest.TestCase):
    """Test ``get_total_gpu_memory_gb``."""

    @patch("acestep.gpu_config_multi.get_multi_gpu_info")
    def test_sums_two_gpus(self, mock_info):
        """Total is the sum of individual GPU VRAM values."""
        mock_info.return_value = [
            (0, "T4", 15.0),
            (1, "T4", 15.0),
        ]
        self.assertAlmostEqual(get_total_gpu_memory_gb(), 30.0)

    @patch("acestep.gpu_config_multi.get_multi_gpu_info")
    def test_no_gpus(self, mock_info):
        """Returns 0.0 when no GPUs are available."""
        mock_info.return_value = []
        self.assertAlmostEqual(get_total_gpu_memory_gb(), 0.0)


class TestIsMultiGpuAvailable(unittest.TestCase):
    """Test ``is_multi_gpu_available``."""

    @patch("acestep.gpu_config_multi.torch")
    def test_two_gpus_returns_true(self, mock_torch):
        """True when ≥2 GPUs detected."""
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 2
        self.assertTrue(is_multi_gpu_available())

    @patch("acestep.gpu_config_multi.torch")
    def test_one_gpu_returns_false(self, mock_torch):
        """False for a single GPU."""
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 1
        self.assertFalse(is_multi_gpu_available())

    @patch("acestep.gpu_config_multi.torch")
    def test_no_cuda_returns_false(self, mock_torch):
        """False when CUDA is unavailable."""
        mock_torch.cuda.is_available.return_value = False
        self.assertFalse(is_multi_gpu_available())


class TestNeedsDitIntraModelSplit(unittest.TestCase):
    """Test ``needs_dit_intra_model_split``."""

    def test_2b_dit_fits_on_t4(self):
        """2B turbo DiT (~4.7 GB + companions) fits on a 15 GB T4."""
        self.assertFalse(needs_dit_intra_model_split([15.0, 15.0], "turbo"))

    def test_xl_dit_fits_on_t4(self):
        """XL DiT (~9.0 GB + companions) fits on a 15 GB T4."""
        self.assertFalse(needs_dit_intra_model_split([15.0, 15.0], "xl_turbo"))

    def test_xl_dit_needs_split_on_8gb(self):
        """XL DiT + companions (~11.5 GB) doesn't fit on an 8 GB GPU."""
        self.assertTrue(needs_dit_intra_model_split([8.0, 8.0], "xl_turbo"))

    def test_turbo_dit_fits_on_8gb(self):
        """2B turbo (~7.2 GB total) fits on 8 GB."""
        self.assertFalse(needs_dit_intra_model_split([8.0, 8.0], "turbo"))


class TestComputeComponentDeviceMap(unittest.TestCase):
    """Test ``compute_component_device_map`` allocation strategy."""

    def test_single_gpu_returns_none(self):
        """Single GPU returns None (use existing single-GPU path)."""
        result = compute_component_device_map([15.0], "turbo", "1.7B")
        self.assertIsNone(result)

    def test_inter_model_2b_turbo_2xt4(self):
        """2B turbo on 2×T4 → inter-model: DiT on :0, LM on :1."""
        result = compute_component_device_map([15.0, 15.0], "turbo", "1.7B")
        self.assertIsNotNone(result)
        # DiT group all on same GPU
        self.assertEqual(result["dit"], result["vae"])
        self.assertEqual(result["dit"], result["text_encoder"])
        # LM on a different GPU
        self.assertNotEqual(result["dit"], result["lm"])
        # Not intra-model
        self.assertNotEqual(result["dit"], "auto")

    def test_inter_model_no_lm(self):
        """Without LM, LM device defaults to DiT device."""
        result = compute_component_device_map([15.0, 15.0], "turbo", "")
        self.assertIsNotNone(result)
        self.assertEqual(result["dit"], result["lm"])

    def test_xl_dit_on_8gb_triggers_intra_split(self):
        """XL DiT on 2×8 GB → intra-model split (dit='auto')."""
        result = compute_component_device_map([8.0, 8.0], "xl_turbo", "0.6B")
        self.assertIsNotNone(result)
        self.assertEqual(result["dit"], "auto")

    def test_xl_dit_on_t4_inter_model(self):
        """XL DiT on 2×T4 (15 GB) → fits on one GPU, inter-model."""
        result = compute_component_device_map([15.0, 15.0], "xl_turbo", "1.7B")
        self.assertIsNotNone(result)
        self.assertNotEqual(result["dit"], "auto")
        self.assertNotEqual(result["dit"], result["lm"])

    def test_asymmetric_gpus(self):
        """Asymmetric GPUs: DiT goes to larger GPU."""
        result = compute_component_device_map([8.0, 16.0], "turbo", "1.7B")
        self.assertIsNotNone(result)
        # DiT should be on the 16 GB GPU (index 1)
        self.assertEqual(result["dit"], "cuda:1")
        self.assertEqual(result["lm"], "cuda:0")


class TestVramHelpers(unittest.TestCase):
    """Test internal VRAM calculation helpers."""

    def test_dit_vram_turbo(self):
        """Turbo DiT returns known VRAM value."""
        self.assertAlmostEqual(_dit_vram_gb("turbo"), 4.7)

    def test_dit_vram_xl_turbo(self):
        """XL turbo DiT returns known VRAM value."""
        self.assertAlmostEqual(_dit_vram_gb("xl_turbo"), 9.0)

    def test_dit_vram_unknown_falls_back(self):
        """Unknown dit type falls back to turbo."""
        self.assertAlmostEqual(_dit_vram_gb("unknown"), 4.7)

    def test_lm_vram_1_7b(self):
        """1.7B LM returns weights + KV cache."""
        expected = 3.4 + 1.0  # weights + kv_cache_4k
        self.assertAlmostEqual(_lm_vram_gb("1.7B"), expected)

    def test_lm_vram_unknown(self):
        """Unknown LM size returns 0.0."""
        self.assertAlmostEqual(_lm_vram_gb("unknown"), 0.0)


if __name__ == "__main__":
    unittest.main()

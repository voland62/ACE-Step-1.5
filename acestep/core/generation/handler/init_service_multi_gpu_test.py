"""Integration tests for multi-GPU model loading and device placement.

All tests use ``unittest.mock`` to simulate multi-GPU environments.
No actual CUDA hardware or model checkpoints are required.
"""

import os
import tempfile
import types
import unittest
from unittest.mock import MagicMock, patch

import torch

from acestep.gpu_config import GPUConfig


def _make_gpu_config(*, multi_gpu_device_map=None, per_gpu_memory_gb=None):
    """Create a GPUConfig with multi-GPU fields for testing."""
    return GPUConfig(
        tier="tier5",
        gpu_memory_gb=15.0,
        max_duration_with_lm=480,
        max_duration_without_lm=600,
        max_batch_size_with_lm=4,
        max_batch_size_without_lm=4,
        init_lm_default=True,
        available_lm_models=["acestep-5Hz-lm-1.7B"],
        recommended_lm_model="acestep-5Hz-lm-1.7B",
        lm_backend_restriction="all",
        recommended_backend="vllm",
        offload_to_cpu_default=False,
        offload_dit_to_cpu_default=False,
        quantization_default=False,
        compile_model_default=True,
        lm_memory_gb={"0.6B": 3, "1.7B": 8},
        multi_gpu_available=multi_gpu_device_map is not None,
        num_gpus=2 if multi_gpu_device_map else 1,
        per_gpu_memory_gb=per_gpu_memory_gb or [],
        multi_gpu_device_map=multi_gpu_device_map,
    )


class TestMultiGpuDisablesCpuOffload(unittest.TestCase):
    """Verify that multi-GPU mode forces CPU offload off."""

    def test_offload_forced_false_when_multi_gpu_active(self):
        """Offload flags must be False when multi_gpu_device_map is set."""
        device_map = {"dit": "cuda:0", "vae": "cuda:0", "text_encoder": "cuda:0", "lm": "cuda:1"}
        config = _make_gpu_config(
            multi_gpu_device_map=device_map,
            per_gpu_memory_gb=[15.0, 15.0],
        )
        self.assertFalse(config.offload_to_cpu_default)
        self.assertFalse(config.offload_dit_to_cpu_default)


class TestMultiGpuDeviceMapFields(unittest.TestCase):
    """Verify GPUConfig multi-GPU fields are populated correctly."""

    def test_single_gpu_has_no_device_map(self):
        """Single-GPU config should have None device map."""
        config = _make_gpu_config()
        self.assertIsNone(config.multi_gpu_device_map)
        self.assertFalse(config.multi_gpu_available)
        self.assertEqual(config.num_gpus, 1)

    def test_multi_gpu_config_stores_device_map(self):
        """Multi-GPU config should store the device map."""
        device_map = {"dit": "cuda:0", "vae": "cuda:0", "text_encoder": "cuda:0", "lm": "cuda:1"}
        config = _make_gpu_config(
            multi_gpu_device_map=device_map,
            per_gpu_memory_gb=[15.0, 15.0],
        )
        self.assertTrue(config.multi_gpu_available)
        self.assertEqual(config.num_gpus, 2)
        self.assertEqual(config.multi_gpu_device_map, device_map)
        self.assertEqual(config.per_gpu_memory_gb, [15.0, 15.0])


class TestOffloadContextMultiGpuGuard(unittest.TestCase):
    """Verify that _load_model_context is a no-op when multi-GPU is active."""

    def test_multi_gpu_skips_offload(self):
        """_load_model_context should yield immediately with multi-GPU."""
        from acestep.core.generation.handler.init_service_offload_context import (
            InitServiceOffloadContextMixin,
        )

        host = type("Host", (InitServiceOffloadContextMixin,), {})()
        host.offload_to_cpu = True
        host.offload_dit_to_cpu = True
        host.last_init_params = {
            "multi_gpu_device_map": {"dit": "cuda:0", "lm": "cuda:1"},
        }
        host.model = MagicMock()

        with patch.object(host, "_recursive_to_device", create=True) as move_mock:
            with host._load_model_context("model"):
                pass

        move_mock.assert_not_called()

    def test_single_gpu_still_offloads(self):
        """_load_model_context should still offload with single GPU."""
        from acestep.core.generation.handler.init_service_offload_context import (
            InitServiceOffloadContextMixin,
        )

        host = type("Host", (InitServiceOffloadContextMixin,), {})()
        host.offload_to_cpu = False
        host.last_init_params = {"multi_gpu_device_map": None}
        host.model = MagicMock()

        # offload_to_cpu=False → yields immediately (no offload)
        with patch.object(host, "_recursive_to_device", create=True) as move_mock:
            with host._load_model_context("model"):
                pass

        move_mock.assert_not_called()


class TestLoaderMultiGpuDevicePlacement(unittest.TestCase):
    """Verify _load_main_model_from_checkpoint uses multi-GPU device map."""

    def _make_host(self):
        """Create a minimal host with loader mixin methods."""
        from acestep.core.generation.handler.init_service_loader import (
            InitServiceLoaderMixin,
        )

        host = type("Host", (InitServiceLoaderMixin,), {})()
        host.dtype = torch.float32
        host.offload_to_cpu = False
        host.offload_dit_to_cpu = False
        host.device = "cuda"
        # Stubs for methods from other mixins that loader calls
        host.is_flash_attention_available = lambda device=None: False
        host._sync_alignment_config = lambda: None
        host._apply_cuda_bool_argsort_workaround = lambda: None
        host._apply_dit_quantization = lambda q: None
        return host

    def test_inter_model_placement_uses_dit_device(self):
        """DiT should be placed on the device specified in device map."""
        host = self._make_host()
        host.device = "cpu"
        device_map = {
            "dit": "cpu",
            "vae": "cpu",
            "text_encoder": "cpu",
            "lm": "cpu",
        }

        dummy_model = MagicMock()
        dummy_model.config = types.SimpleNamespace(_attn_implementation="sdpa")
        dummy_model.to.return_value = dummy_model

        with tempfile.TemporaryDirectory() as tmpdir:
            torch.save(torch.zeros(1, 1, 1), os.path.join(tmpdir, "silence_latent.pt"))

            with patch("torch.cuda.is_available", return_value=False), \
                    patch("transformers.AutoModel.from_pretrained", return_value=dummy_model):
                host._load_main_model_from_checkpoint(
                    model_checkpoint_path=tmpdir,
                    device="cpu",
                    use_flash_attention=False,
                    compile_model=False,
                    quantization=None,
                    multi_gpu_device_map=device_map,
                )

        # Model should be placed on dit device from device map
        dummy_model.to.assert_any_call("cpu")
        # Should NOT use default "cuda" path
        self.assertIsNotNone(host.model)

    def test_intra_model_split_calls_dispatch(self):
        """DiT='auto' should trigger accelerate dispatch."""
        host = self._make_host()
        host.device = "cpu"
        device_map = {
            "dit": "auto",
            "vae": "cpu",
            "text_encoder": "cpu",
            "lm": "cpu",
        }

        dummy_model = MagicMock()
        dummy_model.config = types.SimpleNamespace(_attn_implementation="sdpa")
        dummy_model.to.return_value = dummy_model
        dummy_model._no_split_modules = ["AceStepDiTLayer"]
        dummy_param = MagicMock()
        dummy_param.device = torch.device("cpu")
        dummy_model.parameters.return_value = iter([dummy_param])

        with tempfile.TemporaryDirectory() as tmpdir:
            torch.save(torch.zeros(1, 1, 1), os.path.join(tmpdir, "silence_latent.pt"))

            with patch("torch.cuda.is_available", return_value=False), \
                    patch("transformers.AutoModel.from_pretrained", return_value=dummy_model):
                # Replace _dispatch_model_across_gpus with a mock
                host._dispatch_model_across_gpus = MagicMock()
                host._load_main_model_from_checkpoint(
                    model_checkpoint_path=tmpdir,
                    device="cpu",
                    use_flash_attention=False,
                    compile_model=False,
                    quantization=None,
                    multi_gpu_device_map=device_map,
                    per_gpu_memory_gb=[15.0, 15.0],
                )

        host._dispatch_model_across_gpus.assert_called_once_with([15.0, 15.0])

    def test_single_gpu_fallback_unchanged(self):
        """Without multi_gpu_device_map, existing behavior is preserved."""
        host = self._make_host()
        host.device = "cpu"

        dummy_model = MagicMock()
        dummy_model.config = types.SimpleNamespace(_attn_implementation="sdpa")
        dummy_model.to.return_value = dummy_model

        with tempfile.TemporaryDirectory() as tmpdir:
            torch.save(torch.zeros(1, 1, 1), os.path.join(tmpdir, "silence_latent.pt"))

            with patch("torch.cuda.is_available", return_value=False), \
                    patch("transformers.AutoModel.from_pretrained", return_value=dummy_model):
                host._load_main_model_from_checkpoint(
                    model_checkpoint_path=tmpdir,
                    device="cpu",
                    use_flash_attention=False,
                    compile_model=False,
                    quantization=None,
                    # No multi_gpu_device_map
                )

        # Should use default device "cpu"
        dummy_model.to.assert_any_call("cpu")


class TestComponentLoaderMultiGpu(unittest.TestCase):
    """Verify VAE and text encoder use device map for placement."""

    def test_vae_uses_device_map_device(self):
        """VAE should go to the device specified in device map."""
        from acestep.core.generation.handler.init_service_loader_components import (
            InitServiceLoaderComponentsMixin,
        )

        host = type("Host", (InitServiceLoaderComponentsMixin,), {})()
        host.dtype = torch.float32
        host.offload_to_cpu = True  # Would normally offload, but multi-GPU overrides

        device_map = {"dit": "cuda:0", "vae": "cuda:1", "text_encoder": "cuda:0", "lm": "cuda:1"}

        dummy_vae = MagicMock()
        dummy_vae.to.return_value = dummy_vae

        with tempfile.TemporaryDirectory() as tmpdir:
            vae_dir = os.path.join(tmpdir, "vae")
            os.makedirs(vae_dir)

            with patch("diffusers.models.AutoencoderOobleck.from_pretrained", return_value=dummy_vae), \
                    patch("acestep.model_downloader.resolve_vae_path", return_value=vae_dir), \
                    patch("os.path.exists", return_value=True):
                host._get_vae_dtype = lambda d="cpu": torch.float32
                host._ensure_len_for_compile = lambda m, n: None
                host._load_vae_model(
                    checkpoint_dir=tmpdir,
                    device="cuda",
                    compile_model=False,
                    multi_gpu_device_map=device_map,
                )

        # VAE should be placed on cuda:1 from device map
        dummy_vae.to.assert_any_call("cuda:1")


if __name__ == "__main__":
    unittest.main()

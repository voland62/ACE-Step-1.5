"""Checkpoint and model-loading helpers for service initialization."""

import importlib
import os
from typing import Optional

import torch
from loguru import logger

from acestep import gpu_config
from .init_service_loader_components import InitServiceLoaderComponentsMixin


class InitServiceLoaderMixin(InitServiceLoaderComponentsMixin):
    """Helpers for heavy model component loading."""

    def _cuda_supports_bool_argsort(self) -> bool:
        """Return whether CUDA argsort supports bool tensors on the active device."""
        if not torch.cuda.is_available():
            return True
        target_device = str(getattr(self, "device", "cuda"))
        if not target_device.startswith("cuda"):
            target_device = "cuda"
        try:
            mask_cat = torch.tensor([[True, False]], device=target_device)
            _ = mask_cat.argsort(dim=1, descending=True, stable=True)
            return True
        except RuntimeError as exc:
            logger.debug(
                "[_cuda_supports_bool_argsort] Treating CUDA bool argsort probe failure as unsupported: {}",
                exc,
            )
            return False

    def _apply_cuda_bool_argsort_workaround(self) -> None:
        """Patch dynamic model helpers when bool argsort is unsupported on CUDA."""
        target_device = str(getattr(self, "device", ""))
        if not target_device.startswith("cuda"):
            return
        if self._cuda_supports_bool_argsort():
            return

        model_module_name = getattr(self.model.__class__, "__module__", "")
        if not model_module_name:
            return

        try:
            model_module = importlib.import_module(model_module_name)
        except Exception as exc:
            logger.warning(
                "[initialize_service] Failed to import model module for CUDA bool-argsort workaround: {}",
                exc,
            )
            return

        original_pack_sequences = getattr(model_module, "pack_sequences", None)
        if original_pack_sequences is None:
            return
        if getattr(original_pack_sequences, "__acestep_bool_argsort_patched__", False):
            return

        def _pack_sequences_cuda_compat(hidden1, hidden2, mask1, mask2):
            # ``pack_sequences`` only needs sortable integer-like masks here; keep
            # truthy/falsey semantics while avoiding CUDA bool argsort failures.
            if isinstance(mask1, torch.Tensor) and mask1.is_cuda and mask1.dtype == torch.bool:
                mask1 = mask1.to(torch.int32)
            if isinstance(mask2, torch.Tensor) and mask2.is_cuda and mask2.dtype == torch.bool:
                mask2 = mask2.to(torch.int32)
            return original_pack_sequences(hidden1, hidden2, mask1, mask2)

        _pack_sequences_cuda_compat.__acestep_bool_argsort_patched__ = True
        setattr(model_module, "pack_sequences", _pack_sequences_cuda_compat)
        logger.warning(
            "[initialize_service] Applied CUDA bool-argsort workaround to {}.pack_sequences",
            model_module_name,
        )

    @staticmethod
    def _build_quantization_config(quantization: str):
        """Return a torchao quantization config object for the requested mode."""
        if quantization == "int8_weight_only":
            from torchao.quantization import Int8WeightOnlyConfig
            return Int8WeightOnlyConfig()
        if quantization == "fp8_weight_only":
            from torchao.quantization import Float8WeightOnlyConfig
            return Float8WeightOnlyConfig()
        if quantization == "w8a8_dynamic":
            from torchao.quantization import Int8DynamicActivationInt8WeightConfig, MappingType
            return Int8DynamicActivationInt8WeightConfig(act_mapping_type=MappingType.ASYMMETRIC)
        raise ValueError(f"Unsupported quantization type: {quantization}")

    def _apply_dit_quantization(self, quantization: Optional[str]) -> None:
        """Apply torchao quantization to DiT linear layers when requested."""
        if quantization is None:
            return
        from torchao.quantization import quantize_
        from torchao.quantization.quant_api import _is_linear

        quant_config = self._build_quantization_config(quantization)
        def _dit_filter_fn(module, fqn):
            """Keep only decoder-side DiT linear layers and exclude tokenizers."""
            if not _is_linear(module, fqn):
                return False
            parts = fqn.split(".")
            if not parts or parts[0] != "decoder":
                return False
            for part in parts:
                if part in ("tokenizer", "detokenizer"):
                    return False
            return True

        quantize_(self.model, quant_config, filter_fn=_dit_filter_fn)
        logger.info(f"[initialize_service] DiT quantized with: {quantization}")

    def _load_main_model_from_checkpoint(
        self,
        *,
        model_checkpoint_path: str,
        device: str,
        use_flash_attention: bool,
        compile_model: bool,
        quantization: Optional[str],
        multi_gpu_device_map: Optional[dict] = None,
        per_gpu_memory_gb: Optional[list] = None,
    ) -> str:
        """Load DiT, apply compile/quantization options, and return selected attention backend.

        Args:
            model_checkpoint_path: Path to the DiT checkpoint directory.
            device: Target runtime device (e.g. ``"cuda"``, ``"cuda:0"``).
            use_flash_attention: Whether to prefer flash attention.
            compile_model: Whether to apply ``torch.compile``.
            quantization: Quantization mode string or ``None``.
            multi_gpu_device_map: Optional component→device map from
                ``gpu_config_multi.compute_component_device_map``.
                When ``dit`` is ``"auto"``, accelerate dispatch is used.
            per_gpu_memory_gb: Per-GPU VRAM list (needed for accelerate
                ``max_memory`` when using intra-model split).
        """
        from transformers import AutoModel

        if not os.path.exists(model_checkpoint_path):
            raise FileNotFoundError(f"ACE-Step V1.5 checkpoint not found at {model_checkpoint_path}")

        # Determine actual DiT target device from multi-GPU map
        dit_device = device
        use_accelerate_dispatch = False
        if multi_gpu_device_map is not None:
            dit_target = multi_gpu_device_map.get("dit", device)
            if dit_target == "auto":
                use_accelerate_dispatch = True
                logger.info(
                    "[initialize_service] Multi-GPU: DiT will use accelerate "
                    "intra-model split (device_map='auto')"
                )
            else:
                dit_device = dit_target
                logger.info(
                    "[initialize_service] Multi-GPU: DiT assigned to {}",
                    dit_device,
                )

        if torch.cuda.is_available():
            if getattr(self, "model", None) is not None:
                del self.model
                self.model = None
            torch.cuda.empty_cache()
            try:
                torch.cuda.synchronize()
            except RuntimeError as exc:
                logger.warning(
                    "[initialize_service] cuda.synchronize() failed during pre-load cleanup: {}. "
                    "Continuing with fresh load attempt.",
                    exc,
                )

        if use_flash_attention and self.is_flash_attention_available(dit_device):
            attn_implementation = "flash_attention_2"
        elif dit_device.startswith("cuda") and not gpu_config.cuda_supports_bfloat16():
            # Check if using float32 (manual override or future auto-detection)
            if getattr(self, "dtype", None) == torch.float32:
                logger.info(
                    "[initialize_service] float32 detected on Pre-Ampere CUDA: "
                    "using SDPA (eager attention not needed for float32)."
                )
                attn_implementation = "sdpa"
            else:
                # Pre-Ampere GPUs in float16 can overflow in SDPA's fused softmax
                # with longer sequences, producing NaN/Inf latents.
                # Eager attention upcasts to float32 for softmax, avoiding overflow.
                logger.info(
                    "[initialize_service] Pre-Ampere CUDA detected: using eager "
                    "attention for float16 numerical stability."
                )
                attn_implementation = "eager"
        else:
            if use_flash_attention:
                logger.warning(
                    f"[initialize_service] Flash attention requested but unavailable for device={dit_device}. "
                    "Falling back to SDPA."
                )
            attn_implementation = "sdpa"

        attn_candidates = [attn_implementation]
        if "sdpa" not in attn_candidates:
            attn_candidates.append("sdpa")
        if "eager" not in attn_candidates:
            attn_candidates.append("eager")

        last_attn_error = None
        self.model = None
        for candidate in attn_candidates:
            try:
                logger.info(f"[initialize_service] Attempting to load model with attention implementation: {candidate}")
                self.model = AutoModel.from_pretrained(
                    model_checkpoint_path,
                    trust_remote_code=True,
                    attn_implementation=candidate,
                    dtype=self.dtype,
                )
                attn_implementation = candidate
                break
            except Exception as exc:
                last_attn_error = exc
                logger.warning(f"[initialize_service] Failed to load model with {candidate}: {exc}")

        if self.model is None:
            raise RuntimeError(
                f"Failed to load model with attention implementations {attn_candidates}: {last_attn_error}"
            ) from last_attn_error

        self.model.config._attn_implementation = attn_implementation
        self.config = self.model.config
        self._sync_alignment_config()
        self._apply_cuda_bool_argsort_workaround()

        # --- Device placement ---
        if use_accelerate_dispatch:
            # Intra-model split: use accelerate to spread DiT across GPUs
            self._dispatch_model_across_gpus(per_gpu_memory_gb or [])
        elif multi_gpu_device_map is not None:
            # Inter-model: DiT goes to a specific GPU
            self.model = self.model.to(dit_device).to(self.dtype)
        elif not self.offload_to_cpu:
            self.model = self.model.to(device).to(self.dtype)
        elif not self.offload_dit_to_cpu:
            logger.info(f"[initialize_service] Keeping main model on {device} (persistent)")
            self.model = self.model.to(device).to(self.dtype)
        else:
            self.model = self.model.to("cpu").to(self.dtype)
        self.model.eval()

        if compile_model and not use_accelerate_dispatch:
            self._ensure_len_for_compile(self.model, "model")
            self.model = torch.compile(self.model)
        self._apply_dit_quantization(quantization)

        silence_latent_path = os.path.join(model_checkpoint_path, "silence_latent.pt")
        if not os.path.exists(silence_latent_path):
            raise FileNotFoundError(f"Silence latent not found at {silence_latent_path}")
        self.silence_latent = torch.load(silence_latent_path, weights_only=True).transpose(1, 2)
        # For multi-GPU, silence latent goes to the DiT's device
        if use_accelerate_dispatch:
            # When dispatched, tensors follow the model's first parameter device
            try:
                first_param_device = next(self.model.parameters()).device
                silence_latent_device = first_param_device
            except StopIteration:
                silence_latent_device = device
        elif multi_gpu_device_map is not None:
            silence_latent_device = dit_device
        else:
            silence_latent_device = "cpu" if self.offload_to_cpu and self.offload_dit_to_cpu else device
        self.silence_latent = self.silence_latent.to(silence_latent_device).to(self.dtype)
        return attn_implementation

    def _dispatch_model_across_gpus(self, per_gpu_memory_gb: list) -> None:
        """Use accelerate to split the DiT model across available GPUs.

        Args:
            per_gpu_memory_gb: Available VRAM per GPU in GB.
        """
        from accelerate import dispatch_model, infer_auto_device_map

        # Build max_memory dict for accelerate
        max_memory = {}
        for i, mem_gb in enumerate(per_gpu_memory_gb):
            # Reserve 1 GB per GPU for other models and safety margin
            usable = max(0.5, mem_gb - 1.0)
            max_memory[i] = f"{usable:.1f}GiB"

        # Get no_split_module_classes from the model if available
        no_split = getattr(self.model, "_no_split_modules", None) or []

        self.model = self.model.to(self.dtype)
        device_map = infer_auto_device_map(
            self.model,
            max_memory=max_memory,
            no_split_module_classes=no_split,
        )
        dispatch_model(self.model, device_map=device_map)

        logger.info(
            "[initialize_service] DiT dispatched across GPUs via accelerate. "
            "Device map summary: {}",
            {k: str(v) for k, v in device_map.items()
             if not k.startswith("decoder.dit_decoder.layers.")},
        )

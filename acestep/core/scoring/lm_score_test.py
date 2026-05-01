"""Tests for LM scoring memory cleanup helpers."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from acestep.core.scoring.lm_score import _offload_cached_hf_scoring_model


class LmScoreMemoryCleanupTests(unittest.TestCase):
    """Verify auxiliary HF scoring models do not stay on accelerator memory."""

    def test_offloads_vllm_hf_scoring_model_to_cpu(self):
        """vLLM scoring uses a separate HF model that should be released."""
        model = MagicMock()
        llm_handler = SimpleNamespace(llm_backend="vllm", _hf_model_for_scoring=model)

        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.empty_cache") as empty_cache_mock,
        ):
            _offload_cached_hf_scoring_model(llm_handler)

        model.to.assert_called_once_with("cpu")
        empty_cache_mock.assert_called_once()

    def test_does_not_offload_pt_backend_model(self):
        """PyTorch backend scoring reuses the active LM model."""
        model = MagicMock()
        llm_handler = SimpleNamespace(llm_backend="pt", _hf_model_for_scoring=model)

        _offload_cached_hf_scoring_model(llm_handler)

        model.to.assert_not_called()


if __name__ == "__main__":
    unittest.main()

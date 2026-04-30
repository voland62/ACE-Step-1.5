"""Unit tests for the flow-edit dispatch helper (#1156 PR-B).

Regression coverage for codex round-1 P1 + 2×P2 findings:
* device-mismatch on CUDA / MPS / XPU (token IDs not moved off CPU)
* dispatch returning raw embeddings instead of ``prepare_condition``
  outputs (breaks downstream auto-LRC / DiT alignment scoring)
* LM Phase 1 still ran in edit mode (replaced src_audio with codes)
"""

import unittest

import torch

from acestep.core.generation.handler.service_generate_flow_edit import (
    dispatch_flow_edit,
)
from acestep.core.generation.handler._flow_edit_dispatch_test_support import (
    FakeHandler,
    make_edit_ctx,
    make_payload,
)


class DispatchFlowEditTests(unittest.TestCase):

    def test_calls_flowedit_with_paired_conditions(self):
        handler = FakeHandler()
        payload = make_payload()
        dispatch_flow_edit(
            handler, payload=payload, generate_kwargs={"infer_steps": 4},
            seed_param=42, edit_ctx=make_edit_ctx(),
        )
        self.assertEqual(handler.model.flowedit_generate_audio.call_count, 1)
        call_kwargs = handler.model.flowedit_generate_audio.call_args.kwargs
        # Source side comes from payload.
        self.assertIs(call_kwargs["text_hidden_states"], payload["text_hidden_states"])
        self.assertIs(call_kwargs["src_latents"], payload["src_latents"])
        # Target side was built fresh.
        self.assertIsNotNone(call_kwargs["target_text_hidden_states"])
        self.assertIsNotNone(call_kwargs["target_lyric_hidden_states"])
        # Window params propagated.
        self.assertEqual(call_kwargs["edit_n_min"], 0.2)
        self.assertEqual(call_kwargs["edit_n_max"], 0.8)
        self.assertEqual(call_kwargs["edit_n_avg"], 1)
        self.assertEqual(call_kwargs["seed"], 42)

    def test_returns_prepare_condition_tensors_not_raw_embeddings(self):
        """Regression for codex P2 round-1 finding."""
        handler = FakeHandler()
        outputs, enc_hs, enc_am, ctx = dispatch_flow_edit(
            handler, payload=make_payload(), generate_kwargs={"infer_steps": 4},
            seed_param=None, edit_ctx=make_edit_ctx(),
        )
        self.assertIn("target_latents", outputs)
        # Sentinel-7 tensors come from the mocked prepare_condition.
        self.assertIs(enc_hs, handler.prepared_enc_hs)
        self.assertIs(enc_am, handler.prepared_enc_am)
        self.assertIs(ctx, handler.prepared_ctx)

    def test_missing_flowedit_method_raises(self):
        handler = FakeHandler(model_has_flowedit=False)
        with self.assertRaises(RuntimeError) as cm:
            dispatch_flow_edit(
                handler, payload=make_payload(),
                generate_kwargs={"infer_steps": 4}, seed_param=None,
                edit_ctx=make_edit_ctx(),
            )
        self.assertIn("flowedit_generate_audio", str(cm.exception))

    def test_token_ids_moved_to_handler_device(self):
        """Regression for codex P1 round-1 finding."""
        handler = FakeHandler()
        captured_device = []

        def _capture(ids):
            captured_device.append(ids.device)
            return torch.zeros(ids.shape[0], ids.shape[1], 16)

        handler.infer_text_embeddings = _capture
        handler.infer_lyric_embeddings = _capture
        dispatch_flow_edit(
            handler, payload=make_payload(), generate_kwargs={"infer_steps": 4},
            seed_param=None, edit_ctx=make_edit_ctx(),
        )
        self.assertEqual(len(captured_device), 2)
        for d in captured_device:
            self.assertEqual(d, handler.device)

    def test_default_window_params_when_missing(self):
        handler = FakeHandler()
        dispatch_flow_edit(
            handler, payload=make_payload(), generate_kwargs={"infer_steps": 4},
            seed_param=None, edit_ctx={"task_type": "edit"},
        )
        kwargs = handler.model.flowedit_generate_audio.call_args.kwargs
        self.assertEqual(kwargs["edit_n_min"], 0.0)
        self.assertEqual(kwargs["edit_n_max"], 1.0)
        self.assertEqual(kwargs["edit_n_avg"], 1)

    def test_dict_metas_parsed_before_tokenization(self):
        """Regression for codex P2 round-3 finding.

        Pre-fix the dispatch passed raw metas (often dicts from
        ``prepare_batch_data``) straight into
        ``_prepare_text_conditioning_inputs``, so target prompts got a
        ``{'bpm': 120}`` repr instead of the parsed ``- bpm: 120`` block
        the source path produced.  Post-fix the dispatch calls
        ``handler._parse_metas`` first, matching the source pipeline.
        """
        handler = FakeHandler()
        edit_ctx = make_edit_ctx()
        # Real handler hands us a list of dicts here.
        edit_ctx["metas"] = [{"bpm": 120, "key": "C minor"}]
        dispatch_flow_edit(
            handler, payload=make_payload(), generate_kwargs={"infer_steps": 4},
            seed_param=None, edit_ctx=edit_ctx,
        )
        self.assertTrue(any(
            isinstance(m, str) and m.startswith("PARSED:")
            for m in handler.captured_parsed_metas
        ), f"target metas were not parsed: {handler.captured_parsed_metas}")

    def test_retake_seed_forwarded_from_generate_kwargs(self):
        """Regression for codex P2 round-2 finding.

        Pre-fix the dispatch only forwarded the main ``seed`` and dropped
        ``retake_seed`` from ``generate_kwargs``, so retake variation /
        reproducibility silently fell back to the main seed under
        ``task_type="edit"``.
        """
        handler = FakeHandler()
        dispatch_flow_edit(
            handler, payload=make_payload(),
            generate_kwargs={"infer_steps": 4, "retake_seed": [99, 100]},
            seed_param=42, edit_ctx=make_edit_ctx(),
        )
        kwargs = handler.model.flowedit_generate_audio.call_args.kwargs
        self.assertEqual(kwargs["seed"], 42)
        self.assertEqual(kwargs["retake_seed"], [99, 100])


class EditSkipsLmPhaseTests(unittest.TestCase):
    """Verify ``inference.generate_music`` skips LM Phase 1 for ``edit``."""

    def test_edit_in_skip_lm_tasks_set(self):
        from pathlib import Path
        src = (Path(__file__).resolve().parents[3] / "inference.py").read_text()
        self.assertIn(
            'skip_lm_tasks = {"cover", "cover-nofsq", "repaint", "extract", "edit"}',
            src,
            "edit must be in skip_lm_tasks so LM Phase 1 doesn't replace src_audio",
        )


if __name__ == "__main__":
    unittest.main()

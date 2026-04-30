"""Unit tests for the flow-edit dispatch helper (#1156 PR-B).

Verifies that:
* ``dispatch_flow_edit`` invokes ``model.flowedit_generate_audio`` with
  the right paired-condition kwargs (source from payload, target built
  from ``edit_ctx``).
* Models without ``flowedit_generate_audio`` raise a clear error.
"""

import unittest
from contextlib import contextmanager
from unittest.mock import MagicMock

import torch

from acestep.core.generation.handler.service_generate_flow_edit import (
    dispatch_flow_edit,
)


class _FakeHandler:
    """Minimal handler stand-in that exposes the surface ``dispatch_flow_edit`` needs."""

    def __init__(self, model_has_flowedit: bool = True):
        self.model = MagicMock()
        if not model_has_flowedit:
            del self.model.flowedit_generate_audio
        else:
            self.model.flowedit_generate_audio.return_value = {
                "target_latents": torch.zeros(1, 8, 16),
                "time_costs": {},
            }
        self.silence_latent = torch.zeros(1, 8, 16)

    @contextmanager
    def _load_model_context(self, name):
        yield

    def _prepare_text_conditioning_inputs(self, *, batch_size, instructions,
                                          captions, lyrics, parsed_metas,
                                          vocal_languages, audio_cover_strength):
        # Return fake padded tensors matching the contract.
        seq = 4
        return (
            ["fake-text-input"] * batch_size,
            torch.zeros(batch_size, seq, dtype=torch.long),
            torch.ones(batch_size, seq),
            torch.zeros(batch_size, seq, dtype=torch.long),
            torch.ones(batch_size, seq),
            None, None,
        )

    def infer_text_embeddings(self, ids):
        return torch.zeros(ids.shape[0], ids.shape[1], 16)

    def infer_lyric_embeddings(self, ids):
        return torch.zeros(ids.shape[0], ids.shape[1], 16)


def _make_payload(bsz: int = 1, seq: int = 8, ch: int = 16):
    return {
        "src_latents": torch.randn(bsz, seq, ch),
        "text_hidden_states": torch.randn(bsz, 4, ch),
        "text_attention_mask": torch.ones(bsz, 4),
        "lyric_hidden_states": torch.randn(bsz, 4, ch),
        "lyric_attention_mask": torch.ones(bsz, 4),
        "refer_audio_acoustic_hidden_states_packed": torch.zeros(0, 4, ch),
        "refer_audio_order_mask": torch.zeros(0, dtype=torch.long),
        "chunk_mask": torch.ones(bsz, seq, ch),
        "is_covers": torch.zeros(bsz, dtype=torch.long),
        "precomputed_lm_hints_25Hz": None,
    }


def _make_edit_ctx(target_caption="my new caption", target_lyrics="new lyrics"):
    return {
        "task_type": "edit",
        "edit_target_caption": target_caption,
        "edit_target_lyrics": target_lyrics,
        "vocal_languages": ["en"],
        "metas": [""],
        "instructions": ["Fill the audio semantic mask based on the given conditions:"],
        "edit_n_min": 0.2,
        "edit_n_max": 0.8,
        "edit_n_avg": 1,
    }


class DispatchFlowEditTests(unittest.TestCase):

    def test_calls_flowedit_with_paired_conditions(self):
        handler = _FakeHandler()
        payload = _make_payload()
        edit_ctx = _make_edit_ctx()

        dispatch_flow_edit(
            handler, payload=payload, generate_kwargs={"infer_steps": 4},
            seed_param=42, edit_ctx=edit_ctx,
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

    def test_returns_4_tuple_with_source_encoder_state(self):
        handler = _FakeHandler()
        payload = _make_payload()
        out = dispatch_flow_edit(
            handler, payload=payload, generate_kwargs={"infer_steps": 4},
            seed_param=None, edit_ctx=_make_edit_ctx(),
        )
        outputs, enc_hs, enc_am, ctx = out
        self.assertIn("target_latents", outputs)
        # Source encoder state passed through unchanged so downstream
        # metadata persistence still sees the source-side view.
        self.assertIs(enc_hs, payload["text_hidden_states"])
        self.assertIs(enc_am, payload["text_attention_mask"])
        self.assertIs(ctx, payload["src_latents"])

    def test_missing_flowedit_method_raises(self):
        handler = _FakeHandler(model_has_flowedit=False)
        with self.assertRaises(RuntimeError) as cm:
            dispatch_flow_edit(
                handler, payload=_make_payload(),
                generate_kwargs={"infer_steps": 4}, seed_param=None,
                edit_ctx=_make_edit_ctx(),
            )
        self.assertIn("flowedit_generate_audio", str(cm.exception))

    def test_default_window_params_when_missing(self):
        handler = _FakeHandler()
        # Empty edit_ctx (other than task_type) should default to full window.
        out = dispatch_flow_edit(
            handler, payload=_make_payload(), generate_kwargs={"infer_steps": 4},
            seed_param=None, edit_ctx={"task_type": "edit"},
        )
        kwargs = handler.model.flowedit_generate_audio.call_args.kwargs
        self.assertEqual(kwargs["edit_n_min"], 0.0)
        self.assertEqual(kwargs["edit_n_max"], 1.0)
        self.assertEqual(kwargs["edit_n_avg"], 1)


if __name__ == "__main__":
    unittest.main()

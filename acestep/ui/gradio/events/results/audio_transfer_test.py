"""Tests for result-audio transfer helpers."""

import unittest

from acestep.ui.gradio.events.results.audio_transfer import send_audio_to_repaint


class SendAudioToRepaintTests(unittest.TestCase):
    """Verify generated-source repaint transfer behavior."""

    def test_send_to_repaint_resets_seed_to_random(self):
        """Generated-source repaint should not reuse the source generation seed."""
        updates = send_audio_to_repaint(
            audio_file="/tmp/generated.wav",
            lm_metadata={"lyrics": "new words", "caption": "new caption"},
            current_lyrics="old words",
            current_caption="old caption",
            current_mode="Custom",
            llm_handler=None,
        )

        self.assertEqual("/tmp/generated.wav", updates[0])
        self.assertEqual(True, updates[6]["value"])
        self.assertEqual("-1", updates[7]["value"])


if __name__ == "__main__":
    unittest.main()

"""Retake (variation-generation) controls for the generation tab.

Split out from ``generation_tab_secondary_controls.py`` to keep that
module under the 200 LOC cap defined in AGENTS.md.
"""

from typing import Any

import gradio as gr


def build_retake_controls() -> dict[str, Any]:
    """Create retake controls — controllable variation generation (issue #1155).

    Retake mixes a fresh independent noise draw into the main initial noise
    via a variance-preserving sin/cos blend.  ``variance=0`` is a no-op;
    higher values drift the output further from the seeded baseline.

    Args:
        None.

    Returns:
        Component map with the retake accordion and its inputs.
    """

    with gr.Accordion("Retake (variation generation)", open=False) as retake_accordion:
        retake_variance = gr.Slider(
            label="Retake Variance",
            minimum=0.0,
            maximum=1.0,
            step=0.01,
            value=0.0,
            info="0=no retake (baseline). Low (0.05–0.15) = subtle variation; high (0.5+) = stronger drift.",
        )
        retake_seed = gr.Textbox(
            label="Retake Seed",
            value="",
            placeholder="Leave empty for random; integer to reproduce a variation",
            info="Independent seed for the retake noise. Recorded in metadata when variance > 0.",
        )
    return {
        "retake_accordion": retake_accordion,
        "retake_variance": retake_variance,
        "retake_seed": retake_seed,
    }

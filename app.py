#!/usr/bin/env python3
"""
Gradio Web UI for Electrical Schematic Detection (SAM3).

Two-tab design:
  Tab 1 — Single Image Debug:  inspect one image in detail.
  Tab 2 — Batch Analysis:      full dataset evaluation + metrics.

Launch:
    python3 app.py  →  http://localhost:7860
"""

from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import gradio as gr

from src.config import (
    ANNOTATIONS_PATH,
    INPUT_DIR,
    OUTPUT_DIR,
    OUTPUT_IMAGES_DIR,
    OUTPUT_JSON_PATH,
    REFERENCES_DIR,
)
from src.inference_engine import InferenceEngine
from src.metrics import compute_metrics_for_ui

matplotlib.use("Agg")

# ============================================================
# Shared State
# ============================================================

engine = InferenceEngine()


# ============================================================
# Tab 1 — Single Image Debug  (logic)
# ============================================================

def predict_single_image_logic(image_name: str):
    """
    Analyze one image → (annotated_image, json_data, debug_log).
    """
    if not image_name:
        return None, [], "⚠️ No image selected."

    image_path = INPUT_DIR / image_name
    if not image_path.exists():
        return None, [{"error": f"Not found: {image_path}"}], f"❌ {image_path}"

    try:
        annotated_rgb, detections, debug_log = engine.process_single_image(
            image_path,
        )
        return annotated_rgb, detections, debug_log
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        return None, [{"error": str(e)}], f"❌ {e}\n{tb}"


# ============================================================
# Tab 2 — Batch Analysis  (logic)
# ============================================================

def run_batch_logic(skip_inference: bool):
    """
    Generator: yields (log, metrics_df, cm_figure, gallery, ref_gallery).

    Iterates over InferenceEngine.process_batch() generator so log messages
    stream to the UI in real-time after each image.
    """
    logs: List[str] = []
    NO_CHANGE = gr.update()

    def log(msg: str):
        logs.append(msg)

    def stream():
        """Current log text + no-change for other outputs."""
        return "\n".join(logs), NO_CHANGE, NO_CHANGE, NO_CHANGE, NO_CHANGE

    # ── Skip inference path ──
    if skip_inference:
        if not OUTPUT_JSON_PATH.exists():
            logs.append("❌ No existing results.json found.")
            logs.append("   Uncheck 'Skip Inference' to run the pipeline.")
            yield "\n".join(logs), pd.DataFrame(), None, [], []
            return
        logs.append(f"📂 Loading existing results from {OUTPUT_JSON_PATH}")
        yield stream()
    else:
        # ── Full inference: clear stale outputs, then stream ──
        logs.append("🚀 Running full inference pipeline...")
        yield "\n".join(logs), pd.DataFrame(), None, [], []

        try:
            for _ in engine.process_batch(log=log):
                yield stream()  # push log updates after each image
        except Exception as e:
            import traceback
            logs.append(f"❌ {e}")
            logs.append(traceback.format_exc())
            yield "\n".join(logs), pd.DataFrame(), None, [], []
            return

    # ── Post-processing ──
    try:
        logs.append("\n📊 Computing metrics...")
        yield stream()

        df, fig = compute_metrics_for_ui(ANNOTATIONS_PATH, OUTPUT_JSON_PATH)
        logs.append("✅ Metrics computed")
        yield stream()

        gallery = engine.load_output_gallery()
        logs.append(f"🖼️  Loaded {len(gallery)} output images")
        yield stream()

        ref_gallery = engine.load_reference_gallery()
        logs.append("✅ Done!")

        # Final yield: populate ALL outputs
        yield "\n".join(logs), df, fig, gallery, ref_gallery

    except Exception as e:
        import traceback
        logs.append(f"❌ {e}")
        logs.append(traceback.format_exc())
        yield "\n".join(logs), pd.DataFrame(), None, [], []


# ============================================================
# UI Layout
# ============================================================

def build_ui() -> gr.Blocks:
    with gr.Blocks(
        title="Electrical Schematic Detection (SAM3)",
    ) as demo:

        gr.Markdown(
            "# ⚡ Electrical Schematic Detection (SAM3)\n"
            "Visual-prompt detection of resistors, capacitors, inductors "
            "and other components."
        )

        # ── Tab 1: Single Image Debug ─────────────────────────
        with gr.Tab("🔍 Single Image Debug"):
            with gr.Row():
                # Left Column — 70 %
                with gr.Column(scale=7):
                    image_dropdown = gr.Dropdown(
                        label="Select Image",
                        choices=engine.list_input_images(),
                        interactive=True,
                    )
                    analyze_btn = gr.Button(
                        "🔍 Analyze This Image",
                        variant="primary",
                    )
                    result_image = gr.Image(
                        label="Visual Result",
                        type="numpy",
                        interactive=False,
                        height=500,
                    )

                # Right Column — 30 %
                with gr.Column(scale=3):
                    result_json = gr.JSON(
                        label="Raw Detections",
                    )
                    debug_log = gr.Textbox(
                        label="Debug Log",
                        lines=12,
                        max_lines=20,
                        interactive=False,
                        autoscroll=True,
                    )

            analyze_btn.click(
                fn=predict_single_image_logic,
                inputs=[image_dropdown],
                outputs=[result_image, result_json, debug_log],
            )

        # ── Tab 2: Batch Analysis — Single-View Dashboard ────
        with gr.Tab("📊 Batch Analysis & Metrics"):
            # Controls Row
            with gr.Row():
                skip_checkbox = gr.Checkbox(
                    label="Skip Inference (load existing results.json)",
                    value=True,
                )
                batch_btn = gr.Button(
                    "📊 Run Batch Evaluation",
                    variant="primary",
                )

            # Dashboard: 2-Column Split
            with gr.Row(equal_height=False):
                # ── Left Column: Analytics (scale 4) ──────────
                with gr.Column(scale=4):
                    metrics_table = gr.Dataframe(
                        label="Precision / Recall / F1-Score",
                        interactive=False,
                        wrap=True,
                    )
                    cm_plot = gr.Plot(
                        label="Confusion Matrix",
                    )
                    batch_log = gr.Textbox(
                        label="Execution Logs",
                        lines=6,
                        max_lines=6,
                        interactive=False,
                        autoscroll=True,
                    )

                # ── Right Column: Visuals (scale 6) ───────────
                with gr.Column(scale=6):
                    output_gallery = gr.Gallery(
                        label="All Output Images",
                        columns=3,
                        height=500,
                        object_fit="contain",
                        preview=True,
                    )
                    ref_gallery = gr.Gallery(
                        label="Reference Mosaics",
                        columns=3,
                        height=200,
                        object_fit="contain",
                    )

            batch_btn.click(
                fn=run_batch_logic,
                inputs=[skip_checkbox],
                outputs=[
                    batch_log,
                    metrics_table,
                    cm_plot,
                    output_gallery,
                    ref_gallery,
                ],
            )

    return demo


# ============================================================
# Entry
# ============================================================

if __name__ == "__main__":
    print("Starting Gradio UI...", flush=True)
    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        theme=gr.themes.Soft(),
    )

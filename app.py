"""
Transcriber – local audio transcription with speaker diarization.

Usage:
    python app.py
Then open http://localhost:7860 in your browser.
"""

from __future__ import annotations

import os
import tempfile
import traceback

import gradio as gr
from dotenv import load_dotenv

from src.pipeline import TranscriptionPipeline

# ── config ────────────────────────────────────────────────────────────────────

load_dotenv()
HF_TOKEN: str | None = os.getenv("HF_TOKEN")

MODELS: dict[str, str] = {
    "large-v3  ·  best quality": "mlx-community/whisper-large-v3-mlx",
    "large-v3-turbo  ·  fast + quality": "mlx-community/whisper-large-v3-turbo",
    "medium  ·  lightweight, fastest": "mlx-community/whisper-medium-mlx",
}
DEFAULT_MODEL = "large-v3  ·  best quality"

LANGUAGES: dict[str, str | None] = {
    "Auto": None,
    "Russian": "ru",
    "English": "en",
    "German": "de",
    "French": "fr",
    "Spanish": "es",
    "Italian": "it",
    "Japanese": "ja",
    "Chinese": "zh",
    "Portuguese": "pt",
    "Korean": "ko",
}

# ── pipeline cache ────────────────────────────────────────────────────────────

_pipelines: dict[str, TranscriptionPipeline] = {}


def _get_pipeline(model_label: str) -> TranscriptionPipeline:
    model_id = MODELS[model_label]
    if model_id not in _pipelines:
        _pipelines[model_id] = TranscriptionPipeline(
            hf_token=HF_TOKEN,
            whisper_model=model_id,
        )
    return _pipelines[model_id]


# ── display helper ────────────────────────────────────────────────────────────

def _fmt_display(segments: list[dict]) -> str:
    if not segments:
        return "(transcript is empty)"
    lines: list[str] = []
    for seg in segments:
        m_s = int(seg["start"] // 60)
        s_s = int(seg["start"] % 60)
        m_e = int(seg["end"] // 60)
        s_e = int(seg["end"] % 60)
        ts = f"{m_s:02d}:{s_s:02d} – {m_e:02d}:{s_e:02d}"
        lines.append(f"[{ts}]  {seg['speaker']}\n{seg['text']}\n")
    return "\n".join(lines)


# ── main callback ─────────────────────────────────────────────────────────────

def _on_transcribe(audio_path, model_label, language_label, num_speakers, use_llm):
    """
    Synchronous handler. Gradio's queue runs this in a thread automatically,
    keeping the event loop free for WebSocket heartbeats.
    Returns (transcript_text, file_path).
    """
    if audio_path is None:
        return "⚠️  Please upload an audio file.", gr.update(value=None, visible=False)

    if not HF_TOKEN:
        return (
            "⚠️  HF_TOKEN not found.\n"
            "Add it to .env in the project root:\nHF_TOKEN=hf_..."
        ), gr.update(value=None, visible=False)

    try:
        language = LANGUAGES.get(language_label)
        n_spk = int(num_speakers) if num_speakers > 0 else None

        pipe = _get_pipeline(model_label)
        segments = pipe.run(
            audio_path,
            language=language,
            num_speakers=n_spk,
            llm_postprocess=use_llm,
        )
        txt = _fmt_display(segments)

        # Write transcript to a temp file for download
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", prefix="transcript_",
            delete=False, encoding="utf-8",
        )
        tmp.write(txt)
        tmp.close()

        return txt, gr.update(value=tmp.name, visible=True)

    except Exception as exc:
        tb = traceback.format_exc()
        return f"❌  Error:\n\n{exc}\n\n{tb}", gr.update(value=None, visible=False)


# ── UI ────────────────────────────────────────────────────────────────────────

CSS = """
#header     { text-align: center; margin-bottom: 8px; }
#transcript { font-family: monospace; font-size: 13px; }
#run-btn    { min-height: 48px; font-size: 16px; }
"""

JS_INIT = """
() => {
    const PATCH = {
        'Перетащите аудио сюда': 'Drop audio file here',
        'Нажмите для загрузки':  'Click to upload',
        '- или -': '— or —',
        'или': 'or',
    };
    function patchI18n() {
        const w = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
        while (w.nextNode()) {
            const k = w.currentNode.textContent.trim();
            if (PATCH[k]) w.currentNode.textContent = PATCH[k];
        }
    }
    setTimeout(patchI18n, 300);
    setTimeout(patchI18n, 1500);
}
"""


with gr.Blocks(title="Transcriber") as demo:

    gr.Markdown(
        "# 🎙 Transcriber\n"
        "Local transcription with speaker diarization",
        elem_id="header",
    )

    with gr.Row():
        # ── left column: controls ─────────────────────────────────────────
        with gr.Column(scale=1, min_width=280):
            audio_input = gr.Audio(
                label="Audio file",
                type="filepath",
                sources=["upload"],
            )
            model_dd = gr.Dropdown(
                label="Whisper model",
                choices=list(MODELS.keys()),
                value=DEFAULT_MODEL,
            )
            with gr.Row():
                lang_dd = gr.Dropdown(
                    label="Language",
                    choices=list(LANGUAGES.keys()),
                    value="Auto",
                )
                spk_slider = gr.Slider(
                    label="Speakers (0 = auto)",
                    minimum=0, maximum=10, step=1, value=0,
                )
            llm_toggle = gr.Checkbox(
                label="LLM post-processing",
                value=True,
                info="Fix names, terms, punctuation with AI",
            )
            run_btn = gr.Button(
                "▶  Transcribe",
                variant="primary",
                elem_id="run-btn",
            )

        # ── right column: output ──────────────────────────────────────────
        with gr.Column(scale=2):
            output_box = gr.Textbox(
                label="Transcript",
                lines=22, max_lines=60,
                placeholder="Results will appear here…",
                elem_id="transcript",
            )
            download_file = gr.File(
                label="Download transcript",
                visible=False,
                interactive=False,
            )

    # ── wiring ────────────────────────────────────────────────────────────

    run_btn.click(
        fn=_on_transcribe,
        inputs=[audio_input, model_dd, lang_dd, spk_slider, llm_toggle],
        outputs=[output_box, download_file],
    )


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    demo.queue()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        theme=gr.themes.Soft(),
        css=CSS,
        js=JS_INIT,
    )

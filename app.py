"""
Transcriber – local audio transcription with speaker diarization.
Runs on Apple Silicon via MLX (Whisper) + pyannote.audio (diarization).

Usage:
    python app.py
Then open http://localhost:7860 in your browser.
"""

from __future__ import annotations

import os
import traceback

import gradio as gr
from dotenv import load_dotenv

from src.export import to_srt, to_txt
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

def _on_transcribe(audio_path, model_label, language_label, num_speakers):
    """
    Synchronous handler. Gradio's queue runs this in a thread automatically,
    keeping the event loop free for WebSocket heartbeats.
    """
    if audio_path is None:
        return "⚠️  Please upload an audio file.", "", ""

    if not HF_TOKEN:
        return (
            "⚠️  HF_TOKEN not found.\n"
            "Add it to .env in the project root:\nHF_TOKEN=hf_...",
            "", "",
        )

    try:
        language = LANGUAGES.get(language_label)
        n_spk = int(num_speakers) if num_speakers > 0 else None

        pipe = _get_pipeline(model_label)
        segments = pipe.run(audio_path, language=language, num_speakers=n_spk)

        display = _fmt_display(segments)
        txt = to_txt(segments)
        srt = to_srt(segments)
        return display, txt, srt

    except Exception as exc:
        tb = traceback.format_exc()
        return f"❌  Error:\n\n{exc}\n\n{tb}", "", ""


# ── UI ────────────────────────────────────────────────────────────────────────

CSS = """
#header     { text-align: center; margin-bottom: 8px; }
#transcript { font-family: monospace; font-size: 13px; }
#run-btn    { min-height: 48px; font-size: 16px; }
#action-row { gap: 6px; align-items: center; justify-content: flex-end; }
.icon-btn   { min-width: 36px !important; width: 36px !important;
              padding: 0 !important; font-size: 16px !important;
              flex: none !important; }
"""

# Replace Gradio's locale-dependent upload strings with English.
JS_INIT = """
() => {
    const PATCH = {
        'Перетащите аудио сюда': 'Drop audio file here',
        'Нажмите для загрузки':  'Click to upload',
        '- или -': '— or —',
        'или': 'or',
    };
    function go() {
        const w = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
        while (w.nextNode()) {
            const k = w.currentNode.textContent.trim();
            if (PATCH[k]) w.currentNode.textContent = PATCH[k];
        }
    }
    setTimeout(go, 300);
    setTimeout(go, 1500);
    setTimeout(go, 4000);
}
"""

# Copy transcript text to clipboard (pure client-side).
JS_COPY = """
(txt) => {
    if (!txt) return txt;
    navigator.clipboard.writeText(txt).catch(() => {});
    return txt;
}
"""

# Download text content as a file (pure client-side, no server round-trip).
JS_DOWNLOAD_TXT = """
(txt) => {
    if (!txt) return txt;
    const blob = new Blob([txt], {type: 'text/plain;charset=utf-8'});
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'transcript.txt';
    a.click();
    URL.revokeObjectURL(a.href);
    return txt;
}
"""

with gr.Blocks(title="Transcriber") as demo:

    # Hidden state: stores raw TXT export for copy/download buttons.
    txt_state = gr.State("")
    srt_state = gr.State("")

    gr.Markdown(
        "# 🎙 Transcriber\n"
        "Local transcription with speaker diarization  \n"
        "*(MLX-Whisper + pyannote.audio · Apple Silicon)*",
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
            with gr.Row(elem_id="action-row"):
                copy_btn = gr.Button("⧉", variant="secondary", size="sm",
                                     elem_classes=["icon-btn"])
                dl_btn = gr.Button("⬇", variant="secondary", size="sm",
                                   elem_classes=["icon-btn"])

    # ── wiring ────────────────────────────────────────────────────────────

    run_btn.click(
        fn=_on_transcribe,
        inputs=[audio_input, model_dd, lang_dd, spk_slider],
        outputs=[output_box, txt_state, srt_state],
    )

    # Pure JS actions — no server calls, instant response.
    copy_btn.click(fn=None, inputs=[txt_state], outputs=[txt_state], js=JS_COPY)
    dl_btn.click(fn=None, inputs=[txt_state], outputs=[txt_state], js=JS_DOWNLOAD_TXT)


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

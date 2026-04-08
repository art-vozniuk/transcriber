"""
Transcriber – local audio transcription with speaker diarization.

Usage:
    python app.py
Then open http://localhost:7860 in your browser.
"""

from __future__ import annotations

from html import escape
import os
import queue
import tempfile
import threading
import time
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


def _fmt_stage_seconds(seconds: float) -> str:
    return f"{seconds:.1f}s"


def _fmt_total_elapsed(seconds: float) -> str:
    whole = max(0, int(seconds))
    hours, remainder = divmod(whole, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _loading_suffix(seconds: float) -> str:
    return "." * ((int(seconds * 2) % 3) + 1)


def _finalize_stage(completed: list[dict], current: dict | None, finished_at: float) -> None:
    if not current:
        return
    completed.append(
        {
            "title": current["title"],
            "detail": current["detail"],
            "duration": max(0.0, finished_at - current["started_at"]),
        }
    )
    del completed[:-6]


def _render_status_panel(
    completed: list[dict],
    current: dict | None,
    total_elapsed: float,
    mode: str = "idle",
    message: str = "",
) -> str:
    history_items = completed[-7:]
    log_html: list[str] = []

    if history_items:
        total_history = len(history_items)
        for idx, entry in enumerate(history_items):
            opacity = 0.22 + 0.66 * ((idx + 1) / total_history)
            log_html.append(
                f"""
                <div class="status-entry status-entry--past" style="opacity:{opacity:.2f}">
                    <div class="status-entry__title">
                        {escape(entry["title"])}
                        <span class="status-entry__time">({_fmt_stage_seconds(entry["duration"])})</span>
                    </div>
                    <div class="status-entry__detail">{escape(entry["detail"])}</div>
                </div>
                """
            )

    if mode == "running" and current:
        stage_elapsed = max(0.0, total_elapsed - current["started_at_offset"])
        log_html.append(
            f"""
            <div class="status-entry status-entry--current">
                <div class="status-entry__title">
                    <span class="status-entry__live-dot"></span>
                    {escape(current["title"])}
                    <span class="status-entry__time">({_fmt_stage_seconds(stage_elapsed)})</span>
                </div>
                <div class="status-entry__detail">{escape(current["detail"])}{escape(_loading_suffix(stage_elapsed))}</div>
            </div>
            """
        )
    elif mode == "error":
        log_html.append(
            f"""
            <div class="status-entry status-entry--error">
                <div class="status-entry__title">
                    Run failed
                    <span class="status-entry__time">({_fmt_total_elapsed(total_elapsed)})</span>
                </div>
                <div class="status-entry__detail">{escape(message or "Something went wrong during processing.")}</div>
            </div>
            """
        )
    elif not log_html:
        log_html.append(
            """
            <div class="status-empty">
                Recent pipeline activity will appear here.
            </div>
            """
        )

    return f"""
    <div class="status-shell">
        <div class="status-history-wrap">
            <div class="status-history-fade"></div>
            <div class="status-history">
                {''.join(log_html)}
            </div>
        </div>
        <div class="status-footer">Total {_fmt_total_elapsed(total_elapsed)}</div>
    </div>
    """


# ── main callback ─────────────────────────────────────────────────────────────

def _on_transcribe(
    audio_path,
    model_label,
    language_label,
    num_speakers,
    use_llm,
):
    run_started_at = time.monotonic()

    if audio_path is None:
        yield (
            _render_status_panel(
                [],
                None,
                0.0,
                mode="error",
                message="Please upload an audio file before starting.",
            ),
            "",
            gr.update(value=None, visible=False),
        )
        return

    if not HF_TOKEN:
        yield (
            _render_status_panel(
                [],
                None,
                0.0,
                mode="error",
                message="HF_TOKEN was not found. Add HF_TOKEN=hf_... to .env first.",
            ),
            "",
            gr.update(value=None, visible=False),
        )
        return

    language = LANGUAGES.get(language_label)
    n_spk = int(num_speakers) if num_speakers > 0 else None

    events: queue.Queue = queue.Queue()
    completed: list[dict] = []
    current = {
        "key": "setup",
        "title": "Job setup",
        "detail": "Preparing the transcription request",
        "started_at": run_started_at,
        "started_at_offset": 0.0,
    }
    mode = "running"
    panel_message = ""
    transcript_value = ""
    download_value = gr.update(value=None, visible=False)

    def emit_status(stage_key: str, title: str, detail: str) -> None:
        events.put(
            {
                "type": "status",
                "stage_key": stage_key,
                "title": title,
                "detail": detail,
            }
        )

    def worker() -> None:
        try:
            emit_status(
                "setup",
                "Job setup",
                (
                    f"Whisper {MODELS[model_label]} | "
                    f"language {language or 'auto'} | "
                    f"speakers {n_spk or 'auto'} | "
                    f"LLM cleanup {'on' if use_llm else 'off'}"
                ),
            )
            pipe = _get_pipeline(model_label)
            segments = pipe.run(
                audio_path,
                language=language,
                num_speakers=n_spk,
                llm_postprocess=use_llm,
                on_status=emit_status,
            )
            txt = _fmt_display(segments)

            tmp = tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".txt",
                prefix="transcript_",
                delete=False,
                encoding="utf-8",
            )
            tmp.write(txt)
            tmp.close()

            events.put(
                {
                    "type": "result",
                    "text": txt,
                    "file": tmp.name,
                    "summary": f"Transcript completed with {len(segments)} segments.",
                }
            )
        except Exception as exc:
            events.put(
                {
                    "type": "error",
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )

    threading.Thread(target=worker, daemon=True).start()

    yield (
        _render_status_panel(completed, current, 0.0, mode=mode),
        transcript_value,
        download_value,
    )

    finished = False
    last_render_at = time.monotonic()

    while True:
        processed_event = False

        while True:
            timeout = 0.25 if not processed_event else 0.0
            try:
                event = events.get(timeout=timeout)
            except queue.Empty:
                break

            processed_event = True
            now = time.monotonic()

            if event["type"] == "status":
                if current and current["key"] != event["stage_key"]:
                    _finalize_stage(completed, current, now)
                    current = {
                        "key": event["stage_key"],
                        "title": event["title"],
                        "detail": event["detail"],
                        "started_at": now,
                        "started_at_offset": now - run_started_at,
                    }
                elif current:
                    current["title"] = event["title"]
                    current["detail"] = event["detail"]
                else:
                    current = {
                        "key": event["stage_key"],
                        "title": event["title"],
                        "detail": event["detail"],
                        "started_at": now,
                        "started_at_offset": now - run_started_at,
                    }

            elif event["type"] == "result":
                if current:
                    _finalize_stage(completed, current, now)
                    current = None
                mode = "done"
                panel_message = event["summary"]
                transcript_value = event["text"]
                download_value = gr.update(value=event["file"], visible=True)
                finished = True

            elif event["type"] == "error":
                if current:
                    _finalize_stage(completed, current, now)
                    current = None
                mode = "error"
                panel_message = event["message"]
                transcript_value = (
                    f"Error:\n\n{event['message']}\n\n{event['traceback']}"
                )
                download_value = gr.update(value=None, visible=False)
                finished = True

        now = time.monotonic()
        if processed_event or now - last_render_at >= 0.25:
            yield (
                _render_status_panel(
                    completed,
                    current,
                    now - run_started_at,
                    mode=mode,
                    message=panel_message,
                ),
                transcript_value,
                download_value,
            )
            last_render_at = now

        if finished and events.empty():
            break


# ── UI ────────────────────────────────────────────────────────────────────────

CSS = """
#header {
    text-align: center;
    margin-bottom: 8px;
}

#run-btn {
    min-height: 48px;
    font-size: 16px;
}

#status-log-panel {
    margin-bottom: 14px;
}

.status-shell {
    position: relative;
    min-height: 240px;
    display: grid;
    grid-template-rows: minmax(0, 1fr) auto;
    gap: 10px;
    padding: 18px 20px 14px;
    border-radius: 24px;
    background:
        radial-gradient(circle at top right, rgba(72, 156, 255, 0.18), transparent 34%),
        radial-gradient(circle at bottom left, rgba(42, 198, 155, 0.14), transparent 32%),
        linear-gradient(180deg, #ffffff 0%, #f4f8fc 100%);
    border: 1px solid rgba(25, 46, 77, 0.08);
    box-shadow: 0 18px 40px rgba(19, 41, 75, 0.08);
    overflow: hidden;
}

.status-history-wrap {
    position: relative;
    min-height: 200px;
    overflow: hidden;
}

.status-history-fade {
    position: absolute;
    inset: 0 0 auto 0;
    height: 64px;
    background: linear-gradient(180deg, rgba(244, 248, 252, 1) 0%, rgba(244, 248, 252, 0) 100%);
    pointer-events: none;
    z-index: 2;
}

.status-history {
    position: absolute;
    inset: 0;
    display: flex;
    flex-direction: column;
    justify-content: flex-end;
    gap: 12px;
}

.status-entry {
    transform: translateY(0);
    animation: status-slide-in 220ms ease-out;
}

.status-entry--current {
    padding: 2px 0 0;
}

.status-entry--error .status-entry__title,
.status-entry--error .status-entry__detail {
    color: #b34638;
}

.status-entry__title {
    color: #38506b;
    font-size: 14px;
    font-weight: 650;
    letter-spacing: 0.01em;
    display: flex;
    align-items: center;
    gap: 8px;
}

.status-entry__detail {
    margin-top: 3px;
    color: #6e7f94;
    font-size: 13px;
    line-height: 1.5;
}

.status-entry__time {
    color: #8293a7;
    font-weight: 500;
}

.status-entry__live-dot {
    width: 9px;
    height: 9px;
    border-radius: 999px;
    background: #2d6cdf;
    flex: 0 0 auto;
    animation: status-pulse 1.2s ease-in-out infinite;
}

.status-empty {
    color: #7b8ca0;
    font-size: 13px;
    line-height: 1.5;
}

.status-footer {
    justify-self: end;
    color: #71849b;
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.04em;
}

#transcript textarea,
#transcript {
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
    font-size: 13px;
}

@keyframes status-pulse {
    0% { transform: scale(0.92); opacity: 0.72; }
    50% { transform: scale(1.16); opacity: 1; }
    100% { transform: scale(0.92); opacity: 0.72; }
}

@keyframes status-slide-in {
    from {
        opacity: 0;
        transform: translateY(8px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}
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
                value=False,
                info="Optional. First run downloads ~4 GB; long recordings can take much longer with AI cleanup enabled.",
            )
            run_btn = gr.Button(
                "▶  Transcribe",
                variant="primary",
                elem_id="run-btn",
            )

        # ── right column: output ──────────────────────────────────────────
        with gr.Column(scale=2):
            status_panel = gr.HTML(
                value=_render_status_panel([], None, 0.0),
                label="Pipeline log",
                show_label=True,
                elem_id="status-log-panel",
            )
            output_box = gr.Textbox(
                label="Transcript",
                lines=18,
                max_lines=60,
                placeholder="Transcript will appear here when processing is complete…",
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
        outputs=[status_panel, output_box, download_file],
        show_progress="hidden",
        stream_every=0.2,
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

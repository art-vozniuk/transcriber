# 🎙 Transcriber

Local audio transcription with speaker diarization, optimized for Apple Silicon.

- **ASR** — [MLX-Whisper](https://github.com/ml-explore/mlx-examples) (large-v3 / turbo / medium) via Apple's MLX framework
- **Diarization** — [pyannote.audio 4.x](https://github.com/pyannote/pyannote-audio) (who spoke when)
- **UI** — [Gradio](https://gradio.app) drag-and-drop web interface

Output format per segment:
```
[00:05 – 00:12]  SPEAKER_00
Hello, how are you?

[00:13 – 00:20]  SPEAKER_01
I'm doing great, thanks!
```
Exports to `.txt` and `.srt`.

---

## Requirements

- macOS with Apple Silicon (M1 / M2 / M3 / M4)
- Python 3.11+
- [uv](https://github.com/astral-sh/uv)
- A free [HuggingFace](https://huggingface.co) account + token

> No `ffmpeg` required — audio decoding is handled entirely by `torchaudio`.

---

## Setup

**1. Clone and install dependencies**

```bash
git clone https://github.com/your-username/transcriber.git
cd transcriber
uv venv --python 3.11
uv pip install -r requirements.txt
```

**2. Accept pyannote model terms** *(one-time, free)*

Log in to HuggingFace and accept the terms for both:
- https://huggingface.co/pyannote/speaker-diarization-3.1
- https://huggingface.co/pyannote/segmentation-3.0

**3. Add your HuggingFace token**

```bash
echo "HF_TOKEN=hf_your_token_here" > .env
```

**4. Run**

```bash
source .venv/bin/activate
python app.py
```

Open [http://localhost:7860](http://localhost:7860) in your browser.

---

## Models

| Label | HF repo | Notes |
|---|---|---|
| large-v3 · best quality | `mlx-community/whisper-large-v3-mlx` | ~3 GB, highest accuracy |
| large-v3-turbo · fast + quality | `mlx-community/whisper-large-v3-turbo` | ~1.6 GB, good balance |
| medium · lightweight, fastest | `mlx-community/whisper-medium-mlx` | ~0.8 GB, fastest |

Models are downloaded automatically on first use and cached by HuggingFace Hub.

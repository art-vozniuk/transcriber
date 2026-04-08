# 🎙 Transcriber

<img width="1515" height="910" alt="image" src="https://github.com/user-attachments/assets/041ee605-3038-438e-b830-f76c49233b77" />

Local audio transcription with speaker diarization, optimized for Apple Silicon.

- **ASR** — [MLX-Whisper](https://github.com/ml-explore/mlx-examples) (large-v3 / turbo / medium) via Apple's MLX framework
- **Diarization** — [pyannote.audio 4.x](https://github.com/pyannote/pyannote-audio) (who spoke when)
- **VAD** — [Silero VAD](https://github.com/snakers4/silero-vad) pre-filtering for cleaner transcription
- **LLM post-processing** — optional local LLM (Qwen2.5-7B via MLX) to fix punctuation, names, and domain terms
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
uv sync
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
uv run python app.py
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

---

## LLM post-processing

Enable the **"LLM post-processing"** checkbox in the UI to run a local LLM after transcription. It is disabled by default because the first run downloads about 4 GB and long recordings become much slower. This fixes:

- Punctuation (commas, periods, question marks)
- Capitalization of names and proper nouns
- Broken dashes (e.g. `some -thing` → `some-thing`)
- Domain-specific term recognition (via glossary)

The LLM model (`Qwen2.5-7B-Instruct-4bit`, ~4 GB) is downloaded automatically on first use.

### Custom LLM prompt

Create a file `llm_prompt.txt` in the project root to customize the LLM system prompt. If this file is missing or empty, a sensible default prompt is used.

Example `llm_prompt.txt`:
```
You are a transcript post-processor.
Fix ONLY recognition errors — do not rephrase or rewrite.

RULES:
1. Fix punctuation: add commas, periods, question marks, dashes
2. Fix capitalization of names and proper nouns
3. Map misheard words to glossary terms when they sound similar
4. NEVER delete or skip any word from the original text
5. Output ONLY the corrected text, no explanations
```

### Glossary

Create a file `glossary.txt` in the project root with domain-specific terms. These are used in two ways:

1. **Whisper's `initial_prompt`** — terms are injected into Whisper's decoder context, increasing their recognition probability
2. **LLM post-processing** — terms are passed to the LLM so it can map misheard words to correct ones

Example `glossary.txt`:
```
ACME Corp, ProjectX (internal tool), Staging, Production, Pipeline,
John Smith, Jane Doe, API Gateway, ETL Pipeline
```

> Both `glossary.txt` and `llm_prompt.txt` are in `.gitignore` — they contain project-specific context and should not be committed.

---

## Pipeline architecture

```
Audio file
  │
  ├─► Silero VAD ─► speech regions (filters silence/noise)
  │
  ├─► MLX-Whisper ─► per-chunk transcription with word timestamps
  │     (uses glossary in initial_prompt)
  │
  ├─► Hallucination filter ─► removes garbage segments
  │
  ├─► pyannote.audio ─► speaker turns
  │
  ├─► Word-level diarization ─► assigns speaker per word
  │     (with flickering smoothing)
  │
  ├─► Segment merging ─► groups words into readable segments
  │     (max 30s / 50 words per segment)
  │
  └─► [optional] LLM post-processing ─► fixes punctuation, names, terms
```

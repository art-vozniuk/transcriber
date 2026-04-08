from __future__ import annotations

import os
import tempfile

import numpy as np
import torch
import torchaudio
import mlx_whisper
from pyannote.audio import Pipeline as PyannotePipeline

SAMPLE_RATE = 16_000

# ── helpers ───────────────────────────────────────────────────────────────────

def _best_device() -> torch.device:
    """Return the best available torch device (MPS > CPU)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    """Return the duration of the overlap between two time intervals."""
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def _load_audio_np(audio_path: str, target_sr: int = 16_000) -> np.ndarray:
    """
    Load any audio file via torchaudio and return a float32 numpy array
    at *target_sr* Hz (mono).  No ffmpeg required.
    """
    waveform, sr = torchaudio.load(audio_path)

    # Mix down to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample to target sample rate
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)

    # Shape: (1, samples) → (samples,), dtype float32
    return waveform.squeeze(0).numpy().astype(np.float32)


def _to_wav16k(audio_path: str) -> str:
    """
    Save a 16 kHz mono WAV to a temporary file (used by pyannote).
    Returns the path; the caller is responsible for deletion.
    """
    waveform_np = _load_audio_np(audio_path)
    waveform_t = torch.from_numpy(waveform_np).unsqueeze(0)  # (1, samples)
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    torchaudio.save(tmp.name, waveform_t, 16_000)
    return tmp.name


# ── main class ────────────────────────────────────────────────────────────────

class TranscriptionPipeline:
    """
    End-to-end pipeline: MLX-Whisper (ASR) + pyannote.audio (diarization).

    Usage
    -----
    pipe = TranscriptionPipeline(hf_token="hf_...")
    segments = pipe.run("audio.mp3")
    # [{"start": 0.0, "end": 5.2, "text": "Hello", "speaker": "SPEAKER_00"}, ...]
    """

    DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"

    def __init__(
        self,
        hf_token: str,
        whisper_model: str = "mlx-community/whisper-large-v3-mlx",
        glossary_path: str = "glossary.txt",
        llm_prompt_path: str = "llm_prompt.txt",
    ) -> None:
        self.hf_token = hf_token
        self.whisper_model = whisper_model
        self.glossary_path = glossary_path
        self.llm_prompt_path = llm_prompt_path
        self._diarizer: PyannotePipeline | None = None

    # ── VAD ─────────────────────────────────────────────────────────────────────

    def _get_vad_segments(
        self, audio: np.ndarray, sr: int = SAMPLE_RATE
    ) -> list[dict]:
        """
        Use Silero VAD to find speech regions.
        Returns list of {"start": float, "end": float} in seconds.
        """
        model, utils = torch.hub.load(
            "snakers4/silero-vad", "silero_vad", trust_repo=True
        )
        get_speech_timestamps = utils[0]

        wav = torch.from_numpy(audio)
        stamps = get_speech_timestamps(
            wav, model,
            sampling_rate=sr,
            threshold=0.35,
            min_speech_duration_ms=250,
            max_speech_duration_s=15,
            min_silence_duration_ms=200,
            speech_pad_ms=200,
        )
        return [
            {"start": s["start"] / sr, "end": s["end"] / sr}
            for s in stamps
        ]

    # ── transcription ─────────────────────────────────────────────────────────

    def _build_prompt(self, language: str | None) -> str | None:
        """Build initial_prompt from punctuation example + glossary terms."""
        PROMPTS = {
            "ru": "Привет! Как дела? Да, всё хорошо. Ну, в общем, вот так.",
            "en": "Hello! How are you? Yes, everything is fine. Well, that's how it is.",
        }
        parts = []
        if language and language in PROMPTS:
            parts.append(PROMPTS[language])

        # Add glossary terms (extract short names only, ~200 token budget)
        if os.path.exists(self.glossary_path):
            with open(self.glossary_path) as f:
                raw = f.read().strip()
            # Extract just the term names, skip descriptions in parentheses
            terms = []
            for entry in raw.split(","):
                entry = entry.strip()
                if "(" in entry:
                    entry = entry[:entry.index("(")].strip()
                if entry:
                    terms.append(entry)
            # Deduplicate and limit to fit ~200 tokens
            seen = set()
            unique = []
            for t in terms:
                if t not in seen:
                    seen.add(t)
                    unique.append(t)
            glossary_str = ", ".join(unique[:80])
            if glossary_str:
                parts.append(glossary_str)

        return " ".join(parts) if parts else None

    def transcribe(self, audio: np.ndarray, language: str | None = None) -> list[dict]:
        """
        Run Silero VAD to find speech chunks, then transcribe each chunk
        with MLX-Whisper. This produces shorter, more accurate segments.
        """
        base_kwargs: dict = {
            "path_or_hf_repo": self.whisper_model,
            "word_timestamps": True,
            "verbose": False,
            "condition_on_previous_text": False,
        }
        if language:
            base_kwargs["language"] = language

        prompt = self._build_prompt(language)
        if prompt:
            base_kwargs["initial_prompt"] = prompt

        # Get speech regions via VAD
        vad_regions = self._get_vad_segments(audio)

        # Merge nearby VAD regions so short chunks get enough context.
        merged_regions = []
        for r in vad_regions:
            if merged_regions and r["start"] - merged_regions[-1]["end"] < 0.3:
                merged_regions[-1]["end"] = r["end"]
            else:
                merged_regions.append(dict(r))

        all_segments: list[dict] = []
        for region in merged_regions:
            start_sample = int(region["start"] * SAMPLE_RATE)
            end_sample = int(region["end"] * SAMPLE_RATE)
            chunk = audio[start_sample:end_sample]

            if len(chunk) < SAMPLE_RATE * 0.3:
                continue

            result = mlx_whisper.transcribe(chunk, **base_kwargs)
            for seg in result.get("segments", []):
                seg["start"] += region["start"]
                seg["end"] += region["start"]
                for w in seg.get("words", []):
                    w["start"] = w.get("start", 0) + region["start"]
                    w["end"] = w.get("end", 0) + region["start"]
                all_segments.append(seg)

        return self._filter_hallucinations(all_segments)

    # ── hallucination filtering ─────────────────────────────────────────────

    @staticmethod
    def _filter_hallucinations(segments: list[dict]) -> list[dict]:
        """Remove segments that are likely hallucinated."""
        filtered = []
        for seg in segments:
            # Skip segments where Whisper thinks there's no speech
            if seg.get("no_speech_prob", 0) > 0.6:
                continue
            # Skip highly repetitive segments (hallucination loops)
            if seg.get("compression_ratio", 0) > 2.4:
                continue
            # Skip segments where the model is very uncertain
            if seg.get("avg_logprob", 0) < -1.0:
                continue
            text = seg.get("text", "").strip()
            if not text:
                continue
            filtered.append(seg)
        return filtered

    # ── diarization ───────────────────────────────────────────────────────────

    def _load_diarizer(self) -> PyannotePipeline:
        """Lazy-load and cache the pyannote diarization pipeline."""
        if self._diarizer is None:
            self._diarizer = PyannotePipeline.from_pretrained(
                self.DIARIZATION_MODEL,
                token=self.hf_token,
            )
            self._diarizer.to(_best_device())
        return self._diarizer

    def diarize(
        self,
        audio_path: str,
        segments: list[dict],
        num_speakers: int | None = None,
    ) -> list[dict]:
        """Assign speaker labels at word level, then group into segments."""
        diarizer = self._load_diarizer()

        kwargs: dict = {}
        if num_speakers:
            kwargs["num_speakers"] = num_speakers
        annotation = diarizer(audio_path, **kwargs)

        # pyannote 4.x wraps the result in DiarizeOutput
        if hasattr(annotation, "speaker_diarization"):
            diarization = annotation.speaker_diarization
        elif hasattr(annotation, "annotation"):
            diarization = annotation.annotation
        else:
            diarization = annotation

        # Flatten pyannote output to (start, end, speaker) tuples
        speaker_turns: list[tuple[float, float, str]] = [
            (turn.start, turn.end, spk)
            for turn, _, spk in diarization.itertracks(yield_label=True)
        ]

        # Assign speaker to each word individually
        labeled_words: list[dict] = []
        for seg in segments:
            for w in seg.get("words", []):
                w_start = w.get("start", seg["start"])
                w_end = w.get("end", seg["end"])
                word_text = w.get("word", "").strip()
                if not word_text:
                    continue

                # Find speaker with max overlap for this word
                best_spk = "SPEAKER_00"
                best_ov = 0.0
                for t_start, t_end, spk in speaker_turns:
                    ov = _overlap(w_start, w_end, t_start, t_end)
                    if ov > best_ov:
                        best_ov = ov
                        best_spk = spk

                labeled_words.append({
                    "start": w_start,
                    "end": w_end,
                    "text": word_text,
                    "speaker": best_spk,
                })

        # Smooth speaker flickering: if a run of 1-2 words has a different
        # speaker than both neighbors, reassign to the surrounding speaker.
        for i in range(len(labeled_words)):
            if i == 0 or i == len(labeled_words) - 1:
                continue
            cur_spk = labeled_words[i]["speaker"]
            prev_spk = labeled_words[i - 1]["speaker"]
            # Look ahead to find next different-speaker boundary
            j = i
            while j < len(labeled_words) and labeled_words[j]["speaker"] == cur_spk:
                j += 1
            run_len = j - i
            if run_len <= 2 and prev_spk != cur_spk:
                next_spk = labeled_words[j]["speaker"] if j < len(labeled_words) else prev_spk
                if prev_spk == next_spk:
                    for k in range(i, j):
                        labeled_words[k]["speaker"] = prev_spk

        # Group consecutive words by speaker into segments
        if not labeled_words:
            return []

        grouped: list[dict] = []
        cur = {
            "start": labeled_words[0]["start"],
            "end": labeled_words[0]["end"],
            "text": labeled_words[0]["text"],
            "speaker": labeled_words[0]["speaker"],
            "words": [labeled_words[0]],
        }
        for w in labeled_words[1:]:
            if w["speaker"] == cur["speaker"]:
                cur["end"] = w["end"]
                cur["text"] += " " + w["text"]
                cur["words"].append(w)
            else:
                grouped.append(cur)
                cur = {
                    "start": w["start"],
                    "end": w["end"],
                    "text": w["text"],
                    "speaker": w["speaker"],
                    "words": [w],
                }
        grouped.append(cur)

        return self._merge_consecutive(grouped)

    # ── post-processing ───────────────────────────────────────────────────────

    @staticmethod
    def _merge_consecutive(
        segments: list[dict],
        max_gap: float = 1.5,
        max_duration: float = 30.0,
        max_words: int = 50,
    ) -> list[dict]:
        """
        Merge adjacent segments from the same speaker, respecting limits
        on gap duration, total segment length, and word count.
        """
        if not segments:
            return segments

        merged = [segments[0].copy()]
        for seg in segments[1:]:
            prev = merged[-1]
            gap = seg["start"] - prev["end"]
            prev_duration = prev["end"] - prev["start"]
            prev_word_count = len(prev.get("words", []))

            can_merge = (
                seg["speaker"] == prev["speaker"]
                and gap <= max_gap
                and prev_duration < max_duration
                and prev_word_count < max_words
            )

            if can_merge:
                prev["end"] = seg["end"]
                prev["text"] += " " + seg["text"]
                prev["words"].extend(seg.get("words", []))
            else:
                merged.append(seg.copy())

        return merged

    # ── full pipeline ─────────────────────────────────────────────────────────

    def run(
        self,
        audio_path: str,
        language: str | None = None,
        num_speakers: int | None = None,
        llm_postprocess: bool = False,
        on_progress: callable | None = None,
    ) -> list[dict]:
        """Run transcription + diarization and return merged labeled segments."""
        audio_np = _load_audio_np(audio_path)

        if on_progress:
            on_progress(0.10, "Transcribing audio (MLX-Whisper)…")
        segments = self.transcribe(audio_np, language=language)

        if on_progress:
            on_progress(0.50, "Identifying speakers (pyannote)…")
        wav_path = _to_wav16k(audio_path)
        try:
            segments = self.diarize(wav_path, segments, num_speakers=num_speakers)
        finally:
            os.unlink(wav_path)

        if llm_postprocess:
            if on_progress:
                on_progress(0.75, "Post-processing with LLM…")
            from src.postprocess import postprocess_segments
            segments = postprocess_segments(
                segments,
                use_glossary=True,
                glossary_path=self.glossary_path,
                prompt_path=self.llm_prompt_path,
            )

        if on_progress:
            on_progress(0.95, "Done!")
        return segments

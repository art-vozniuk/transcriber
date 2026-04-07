from __future__ import annotations

import os
import tempfile

import numpy as np
import torch
import torchaudio
import mlx_whisper
from pyannote.audio import Pipeline as PyannotePipeline


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
    ) -> None:
        self.hf_token = hf_token
        self.whisper_model = whisper_model
        self._diarizer: PyannotePipeline | None = None

    # ── transcription ─────────────────────────────────────────────────────────

    def transcribe(self, audio: np.ndarray, language: str | None = None) -> list[dict]:
        """
        Run MLX-Whisper on a float32 numpy array (16 kHz mono).
        Passing an ndarray skips mlx_whisper's internal ffmpeg call entirely.
        """
        kwargs: dict = {
            "path_or_hf_repo": self.whisper_model,
            "word_timestamps": True,
            "verbose": False,
        }
        if language:
            kwargs["language"] = language

        result = mlx_whisper.transcribe(audio, **kwargs)
        return result.get("segments", [])

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
        """Assign speaker labels to each Whisper segment."""
        diarizer = self._load_diarizer()

        kwargs: dict = {}
        if num_speakers:
            kwargs["num_speakers"] = num_speakers
        annotation = diarizer(audio_path, **kwargs)

        # pyannote 4.x wraps the result in DiarizeOutput with field
        # 'speaker_diarization'; older versions return an Annotation directly.
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

        labeled: list[dict] = []
        for seg in segments:
            seg_start, seg_end = seg["start"], seg["end"]

            # Assign the speaker with the greatest time overlap in this segment
            speaker_scores: dict[str, float] = {}
            for t_start, t_end, spk in speaker_turns:
                ov = _overlap(seg_start, seg_end, t_start, t_end)
                if ov > 0:
                    speaker_scores[spk] = speaker_scores.get(spk, 0.0) + ov

            speaker = (
                max(speaker_scores, key=speaker_scores.get)
                if speaker_scores
                else "SPEAKER_00"
            )

            labeled.append(
                {
                    "start": seg_start,
                    "end": seg_end,
                    "text": seg["text"].strip(),
                    "speaker": speaker,
                    "words": seg.get("words", []),
                }
            )

        return self._merge_consecutive(labeled, max_gap=2.0)

    # ── post-processing ───────────────────────────────────────────────────────

    @staticmethod
    def _merge_consecutive(segments: list[dict], max_gap: float = 2.0) -> list[dict]:
        """
        Merge adjacent segments from the same speaker when the silence gap
        between them is shorter than *max_gap* seconds.
        """
        if not segments:
            return segments

        merged = [segments[0].copy()]
        for seg in segments[1:]:
            prev = merged[-1]
            gap = seg["start"] - prev["end"]
            if seg["speaker"] == prev["speaker"] and gap <= max_gap:
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
        on_progress: callable | None = None,
    ) -> list[dict]:
        """Run transcription + diarization and return merged labeled segments."""
        # Load audio once with torchaudio – no ffmpeg needed for any format.
        audio_np = _load_audio_np(audio_path)

        if on_progress:
            on_progress(0.15, "Transcribing audio (MLX-Whisper)…")
        # Pass numpy array directly; mlx_whisper skips its ffmpeg loader.
        segments = self.transcribe(audio_np, language=language)

        if on_progress:
            on_progress(0.60, "Identifying speakers (pyannote)…")
        # pyannote needs a WAV file path, so write a temp file.
        wav_path = _to_wav16k(audio_path)
        try:
            segments = self.diarize(wav_path, segments, num_speakers=num_speakers)
        finally:
            os.unlink(wav_path)

        if on_progress:
            on_progress(0.95, "Done!")
        return segments

from __future__ import annotations

import os
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

from src.model_cache import hf_repo_is_cached


LLM_MODEL = "mlx-community/Qwen2.5-7B-Instruct-4bit"

DEFAULT_SYSTEM_PROMPT = (
    "You are a transcript post-processor. "
    "Fix ONLY recognition errors in this speech transcript — do not rephrase or rewrite.\n\n"
    "RULES:\n"
    "1. Fix punctuation: add commas, periods, question marks, dashes\n"
    "2. Fix capitalization of names and proper nouns\n"
    "3. Fix broken dashes (e.g. 'что -то' → 'что-то')\n"
    "4. If a glossary is provided, map misheard words to glossary terms "
    "when they sound similar\n"
    "5. NEVER delete or skip any word from the original text\n"
    "6. NEVER rephrase — only fix individual words in place\n"
    "7. Keep filler words and hesitations exactly as they are\n"
    "8. Output ONLY the corrected text, no explanations"
)

_model_cache = {}


def _get_model():
    if "m" not in _model_cache:
        model, tokenizer = load(LLM_MODEL)
        _model_cache["m"] = model
        _model_cache["t"] = tokenizer
    return _model_cache["m"], _model_cache["t"]


def _load_file(path: str) -> str:
    """Load a text file if it exists and is non-empty."""
    if os.path.exists(path):
        with open(path, "r") as f:
            content = f.read().strip()
        if content:
            return content
    return ""


def _build_prompt(
    text: str,
    system_prompt: str,
    glossary: str = "",
    prev_context: str = "",
) -> str:
    parts = [system_prompt]
    if glossary:
        parts.append(f"\nGlossary of domain terms:\n{glossary}")
    if prev_context:
        parts.append(f"\nPrevious context:\n{prev_context}")
    parts.append(f"\nFix this transcript:\n{text}")
    return "\n".join(parts)


def postprocess_segments(
    segments: list[dict],
    use_glossary: bool = True,
    glossary_path: str = "glossary.txt",
    prompt_path: str = "llm_prompt.txt",
    on_status: callable | None = None,
) -> list[dict]:
    """Run LLM post-processing on transcript segments."""
    if on_status:
        if hf_repo_is_cached(LLM_MODEL):
            on_status(
                "llm_model",
                "LLM model",
                f"Using cached model {LLM_MODEL}",
            )
        else:
            on_status(
                "llm_model",
                "LLM model",
                f"Downloading model {LLM_MODEL}",
            )
    model, tokenizer = _get_model()

    glossary = _load_file(glossary_path) if use_glossary else ""

    # Use custom prompt file if present, otherwise fall back to default
    system_prompt = _load_file(prompt_path) or DEFAULT_SYSTEM_PROMPT

    result = []
    prev_context = ""
    total = len(segments)

    if total == 0:
        if on_status:
            on_status(
                "llm",
                "LLM post-processing",
                "No transcript segments need post-processing",
            )
        return result

    for i, seg in enumerate(segments):
        text = seg["text"]
        if on_status:
            on_status(
                "llm",
                "LLM post-processing",
                f"Post-processing segment {i+1}/{total} ({len(text)} chars)",
            )

        prompt = _build_prompt(
            text,
            system_prompt=system_prompt,
            glossary=glossary,
            prev_context=prev_context,
        )

        messages = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        fixed = generate(
            model, tokenizer,
            prompt=formatted,
            max_tokens=len(text) * 3,
            sampler=make_sampler(temp=0.1),
        )
        fixed = fixed.strip()

        # Sanity check: if LLM output is way too short or too long, keep original
        if len(fixed) < len(text) * 0.3 or len(fixed) > len(text) * 3:
            fixed = text

        new_seg = seg.copy()
        new_seg["text"] = fixed
        result.append(new_seg)

        # Keep last segment as context for next
        prev_context = fixed[-200:] if len(fixed) > 200 else fixed

    if on_status:
        on_status(
            "llm",
            "LLM post-processing",
            f"Finished AI cleanup for {total} segments",
        )
    return result

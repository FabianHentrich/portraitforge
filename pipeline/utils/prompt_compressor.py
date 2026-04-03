"""
PromptCompressor — shortens long prompts to CLIP-compatible token length.

Strategy:
  1. CLIP tokenizer counts actual tokens (identical to the text encoder
     in FLUX, SDXL, SD3.5).
  2. If the prompt is within the limit: no modification.
  3. If the prompt exceeds the limit: google/flan-t5-small compresses it via
     instruction ("compress ...") to approx. 12 words / ~15 tokens.
  4. Fallback: comma-segment-based greedy truncation if flan-t5 is not
     available or its output still exceeds the limit.

Model: google/flan-t5-small (~300 MB), CPU-only, runs in < 1s.
Download: python scripts/download_models.py --models flan_t5_small
          (or automatically on first call to compress())
"""

import logging
import pathlib

logger = logging.getLogger(__name__)

_PROJECT_ROOT = pathlib.Path(__file__).parent.parent.parent
_MODEL_DIR = _PROJECT_ROOT / "models" / "utils" / "flan-t5-small"

# CLIP tokenizer ID — identical in SDXL, FLUX, SD3.5
_CLIP_TOKENIZER_ID = "openai/clip-vit-large-patch14"

# Instruction for flan-t5: compress prompt, keep most important elements
_INSTRUCTION = (
    "Compress this image generation prompt to at most 12 words. "
    "Keep: subject, art style, key visual attributes. Remove: filler words, "
    "redundant adjectives, connector words. Prompt: "
)


class PromptCompressor:
    """
    Singleton — lazy-loaded, stays in RAM after first call (CPU, minimal).

    Usage:
        from pipeline.utils.prompt_compressor import compressor
        short = compressor.compress(long_prompt, token_limit=77)
    """

    def __init__(self) -> None:
        self._clip_tokenizer = None
        self._t5_model = None
        self._t5_tokenizer = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def count_tokens(self, text: str) -> int:
        """Counts CLIP tokens (BOS/EOS not included)."""
        tok = self._get_clip_tokenizer()
        ids = tok.encode(text, add_special_tokens=False)
        return len(ids)

    def compress(self, prompt: str, token_limit: int = 77) -> tuple[str, bool]:
        """
        Returns (compressed_prompt, was_compressed).
        If the prompt is already within the limit, it is returned unchanged.

        Strategy:
        - Up to 2x limit: Greedy truncation (comma-segment-based, low-loss, exact keywords
          are preserved). Best choice for image prompts — subject/style appear first.
        - Over 2x limit: flan-t5-small compresses the content semantically, followed
          by greedy truncation as a safety net if needed.
        """
        n = self.count_tokens(prompt)
        if n <= token_limit:
            return prompt, False

        logger.info(
            f"PromptCompressor: {n} tokens > limit {token_limit} — compressing..."
        )

        # Greedy is sufficient for slightly to moderately overlong prompts (<= 2x limit).
        # Greedy preserves exact keywords in the correct priority order; flan-t5 would
        # paraphrase to ~12 words and lose too much content.
        if n <= token_limit * 2:
            result = self._compress_greedy(prompt, token_limit)
            n2 = self.count_tokens(result)
            logger.info(
                f"PromptCompressor: Greedy -> {n2} tokens: '{result[:80]}...'"
            )
            return result, True

        # Very long prompt (> 2x limit): flan-t5-small attempts semantic condensation
        compressed = self._compress_with_t5(prompt, token_limit)
        if compressed is not None:
            n2 = self.count_tokens(compressed)
            logger.info(
                f"PromptCompressor: flan-t5 -> {n2} tokens: '{compressed[:80]}...'"
            )
            if n2 <= token_limit:
                return compressed, True
            # flan-t5 output still too long -> greedy truncation on the already shorter result
            logger.warning(
                f"PromptCompressor: flan-t5 output ({n2} tokens) still over limit "
                f"— greedy truncation"
            )
            prompt = compressed

        result = self._compress_greedy(prompt, token_limit)
        n3 = self.count_tokens(result)
        logger.info(
            f"PromptCompressor: Greedy -> {n3} tokens: '{result[:80]}...'"
        )
        return result, True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_clip_tokenizer(self):
        if self._clip_tokenizer is None:
            try:
                from transformers import CLIPTokenizer
                # Try local SDXL/FLUX model first (no HF request)
                local_candidates = [
                    _PROJECT_ROOT / "models" / "generator" / "sdxl_base" / "tokenizer",
                    _PROJECT_ROOT / "models" / "generator" / "realvisxl_v5" / "tokenizer",
                    _PROJECT_ROOT / "models" / "generator" / "flux_schnell" / "tokenizer",
                ]
                for candidate in local_candidates:
                    if candidate.is_dir():
                        self._clip_tokenizer = CLIPTokenizer.from_pretrained(str(candidate))
                        logger.debug(
                            f"PromptCompressor: CLIP tokenizer loaded from '{candidate}'"
                        )
                        return self._clip_tokenizer
                # Fallback: load from HuggingFace (requires internet on first use)
                self._clip_tokenizer = CLIPTokenizer.from_pretrained(_CLIP_TOKENIZER_ID)
                logger.debug("PromptCompressor: CLIP tokenizer loaded from HuggingFace")
            except Exception as e:
                logger.warning(
                    f"PromptCompressor: CLIP tokenizer not available ({e}) "
                    f"— counting via whitespace split (approximation)"
                )
                # Dummy tokenizer: returns words (rough approximation)
                class _FallbackTok:
                    def encode(self, text, **_):
                        return text.split()
                self._clip_tokenizer = _FallbackTok()
        return self._clip_tokenizer

    def _load_t5(self) -> bool:
        """Lazy-loads flan-t5-small. Returns True if successful."""
        if self._t5_model is not None:
            return True
        try:
            from transformers import T5ForConditionalGeneration, T5Tokenizer
            if not _MODEL_DIR.is_dir():
                logger.info(
                    f"PromptCompressor: flan-t5-small not found locally — "
                    f"downloading to {_MODEL_DIR} (one-time ~300 MB)"
                )
                from huggingface_hub import snapshot_download
                snapshot_download(
                    repo_id="google/flan-t5-small",
                    local_dir=str(_MODEL_DIR),
                    ignore_patterns=["*.msgpack", "*.h5", "flax_model*"],
                )
            self._t5_tokenizer = T5Tokenizer.from_pretrained(
                str(_MODEL_DIR), legacy=False
            )
            self._t5_model = T5ForConditionalGeneration.from_pretrained(
                str(_MODEL_DIR), low_cpu_mem_usage=True
            )
            self._t5_model.eval()
            logger.info("PromptCompressor: flan-t5-small loaded (CPU)")
            return True
        except Exception as e:
            logger.warning(f"PromptCompressor: flan-t5-small not available — {e}")
            return False

    def _compress_with_t5(self, prompt: str, token_limit: int) -> str | None:
        if not self._load_t5():
            return None
        try:
            import torch
            instruction = _INSTRUCTION + prompt
            inputs = self._t5_tokenizer(
                instruction,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            )
            with torch.no_grad():
                outputs = self._t5_model.generate(
                    **inputs,
                    max_new_tokens=40,   # ~12 words + some buffer
                    num_beams=4,
                    early_stopping=True,
                    no_repeat_ngram_size=2,
                )
            result = self._t5_tokenizer.decode(
                outputs[0], skip_special_tokens=True
            ).strip()
            return result if result else None
        except Exception as e:
            logger.warning(f"PromptCompressor: flan-t5 inference failed — {e}")
            return None

    def _compress_greedy(self, prompt: str, token_limit: int) -> str:
        """
        Splits the prompt at commas and greedily adds segments until the
        token limit is reached. First segment (subject) has highest priority.
        """
        tok = self._get_clip_tokenizer()
        segments = [s.strip() for s in prompt.split(",") if s.strip()]
        if not segments:
            return prompt

        result_parts: list[str] = []
        # Reserve 2 tokens for BOS/EOS and small buffer
        budget = token_limit - 2

        for seg in segments:
            seg_tokens = len(tok.encode(seg, add_special_tokens=False))
            current_tokens = len(
                tok.encode(", ".join(result_parts), add_special_tokens=False)
            ) if result_parts else 0
            sep_cost = 2 if result_parts else 0  # ", " costs ~2 tokens

            if current_tokens + sep_cost + seg_tokens <= budget:
                result_parts.append(seg)
            elif not result_parts:
                # First segment too long — truncate by words
                words = seg.split()
                partial = []
                for w in words:
                    test = " ".join(partial + [w])
                    if len(tok.encode(test, add_special_tokens=False)) <= budget:
                        partial.append(w)
                    else:
                        break
                result_parts.append(" ".join(partial))
                break
            # Later segments that don't fit: skip

        return ", ".join(result_parts)

    def unload(self) -> None:
        """Frees flan-t5 from RAM (CLIP tokenizer is kept)."""
        import gc
        self._t5_model = None
        self._t5_tokenizer = None
        gc.collect()
        logger.info("PromptCompressor: flan-t5 unloaded")


# Module-level singleton
compressor = PromptCompressor()

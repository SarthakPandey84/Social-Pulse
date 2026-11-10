"""Text preprocessor and validator for social media audience comments.

Provides:
- Robust input validation and empty/invalid text handling.
- Insufficient context detection (emojis only, single characters, punctuation only).
- Transparent language detection (detects non-English scripts/tokens and marks unsupported).
- Meaning-preserving normalization: preserves negations, conversational punctuation, and emojis.
"""

from dataclasses import dataclass
import re
from typing import Optional
import unicodedata
import pandas as pd

from src.data_ingestion.schemas import ClassificationStatus


@dataclass
class PreprocessedText:
    """Encapsulates the preprocessed output and validation metadata for a comment."""
    raw_text: str
    cleaned_text: str
    normalized_tokens: str
    language: str
    is_valid: bool
    status: ClassificationStatus
    rejection_reason: Optional[str] = None


class TextPreprocessor:
    """Preprocesses and validates user comments with safety guards."""

    # Explicit non-English stop patterns for popular social languages
    NON_EN_PATTERNS = [
        # Hindi / Hinglish indicators
        r"\b(hai|h|kya|bhai|yaar|nahi|acha|bohot|bahut|mera|meri|mujhe|sahi|bilkul|shukriya|aapne|kahani|sach|bola|toh)\b",
        # Spanish indicators
        r"\b(hola|gracias|muy|pero|este|esta|bueno|verdad|tambien|amigo|amigos|vida|mio|mia|describe|perfectamente)\b",
        # French indicators
        r"\b(merci|très|tres|avec|pour|c'est|tellement|vrai|aussi|mon|ma|dans|vie|moi|nous|vous)\b",
        # German indicators
        r"\b(danke|sehr|nicht|aber|genau|meine|mein|wirklich|leben|auch|ich|das|der|die|kenne|aus|eigener|erfahrung|gut)\b"
    ]

    # Non-Latin unicode blocks (Devanagari, Cyrillic, Arabic, CJK, etc.)
    NON_LATIN_SCRIPTS = re.compile(r"[\u0900-\u097F\u0400-\u04FF\u0600-\u06FF\u4E00-\u9FFF\u3040-\u30FF\uAC00-\uD7AF]")
    _NON_EN_REGEX = re.compile("|".join(NON_EN_PATTERNS), re.IGNORECASE)

    _CONTRACTION_MAP = {
        "i'm": "i am",
        "can't": "cannot",
        "don't": "do not",
        "didn't": "did not",
        "won't": "will not",
        "isn't": "is not",
        "aren't": "are not",
        "wasn't": "was not",
        "it's": "it is",
        "that's": "that is",
        "you're": "you are",
    }
    _CONTRACTION_REGEX = re.compile(
        r"\b(" + "|".join(re.escape(k) for k in _CONTRACTION_MAP) + r")\b"
    )
    _MULTI_SPACE_REGEX = re.compile(r"[ \t]+")
    _MULTI_BANG_REGEX = re.compile(r"!{3,}")
    _MULTI_QUEST_REGEX = re.compile(r"\?{3,}")
    _STRIP_PUNCT_REGEX = re.compile(r"[^\w\s\?!']")
    _WHITESPACE_REGEX = re.compile(r"\s+")

    def __init__(self):
        self.non_en_regex = self._NON_EN_REGEX

    def detect_language(self, text: str) -> str:
        """Detect language heuristically; returns 'en' or detected script/pattern ('non-en')."""
        if not text:
            return "unknown"

        # Check for non-Latin script alphabets
        if self.NON_LATIN_SCRIPTS.search(text):
            return "non-en-script"

        # Check for non-English high-frequency social stopwords
        matches = self.non_en_regex.findall(text)
        if len(matches) >= 2:
            return "non-en"

        return "en"

    def preprocess(self, raw_text: Optional[str], language_hint: Optional[str] = None) -> PreprocessedText:
        """Validate, normalize, and prepare comment text for analytical pipelines."""
        # 1. Null / Type / Empty check
        if raw_text is None or pd.isna(raw_text) or not isinstance(raw_text, str):
            return PreprocessedText(
                raw_text="" if raw_text is None else str(raw_text),
                cleaned_text="",
                normalized_tokens="",
                language=language_hint or "unknown",
                is_valid=False,
                status=ClassificationStatus.EMPTY_TEXT,
                rejection_reason="Comment text is null, missing, or non-string."
            )

        stripped = raw_text.strip()
        if not stripped:
            return PreprocessedText(
                raw_text=raw_text,
                cleaned_text="",
                normalized_tokens="",
                language=language_hint or "unknown",
                is_valid=False,
                status=ClassificationStatus.EMPTY_TEXT,
                rejection_reason="Comment text is empty or contains only whitespace."
            )

        # 2. Insufficient context check (e.g. single character, punctuation only, or pure emoji without words)
        # Count up to 3 alphanumeric characters without allocating an intermediate list
        alnum_count = 0
        for c in stripped:
            if c.isalnum():
                alnum_count += 1
                if alnum_count > 2:
                    break
        if alnum_count == 0:
            # Emojis only or symbols only like "???", "...", "❤️🔥"
            return PreprocessedText(
                raw_text=raw_text,
                cleaned_text=stripped,
                normalized_tokens=stripped,
                language=language_hint or "unknown",
                is_valid=False,
                status=ClassificationStatus.INSUFFICIENT_CONTEXT,
                rejection_reason="Comment contains only emojis or symbols without textual context."
            )

        if alnum_count <= 2 and len(stripped.split()) <= 1:
            # Single word abbreviation or letter like "k", "ok", "no", "yo" with no further context
            return PreprocessedText(
                raw_text=raw_text,
                cleaned_text=stripped,
                normalized_tokens=stripped.lower(),
                language=language_hint or "unknown",
                is_valid=False,
                status=ClassificationStatus.INSUFFICIENT_CONTEXT,
                rejection_reason="Comment too brief to reliably infer personal relatability."
            )

        # 3. Language check
        detected_lang = language_hint or self.detect_language(stripped)
        if detected_lang != "en" and language_hint != "en":
            return PreprocessedText(
                raw_text=raw_text,
                cleaned_text=stripped,
                normalized_tokens=stripped.lower(),
                language=detected_lang,
                is_valid=False,
                status=ClassificationStatus.UNSUPPORTED_LANGUAGE,
                rejection_reason=f"Unsupported language detected ({detected_lang}); English models cannot be applied."
            )

        # 4. Meaning-preserving text cleaning
        # Normalize unicode (NFKC)
        cleaned = unicodedata.normalize("NFKC", stripped)
        
        # Collapse multiple spaces while preserving newlines/casing
        cleaned = self._MULTI_SPACE_REGEX.sub(" ", cleaned)

        # Collapse excessive repetitive punctuation while preserving single/double exclamation/question marks
        cleaned = self._MULTI_BANG_REGEX.sub("!!", cleaned)
        cleaned = self._MULTI_QUEST_REGEX.sub("??", cleaned)

        # 5. Tokenized / Lowercase version for TF-IDF / vocabulary matching
        # Single-pass contraction expansion preserving negations
        tokens = self._CONTRACTION_REGEX.sub(
            lambda m: self._CONTRACTION_MAP[m.group(0)], cleaned.lower()
        )

        # Remove extraneous punctuation from normalized tokens but KEEP words and spaces
        tokens = self._STRIP_PUNCT_REGEX.sub(" ", tokens)
        tokens = self._WHITESPACE_REGEX.sub(" ", tokens).strip()

        return PreprocessedText(
            raw_text=raw_text,
            cleaned_text=cleaned,
            normalized_tokens=tokens,
            language="en",
            is_valid=True,
            status=ClassificationStatus.SUCCESS,
            rejection_reason=None
        )

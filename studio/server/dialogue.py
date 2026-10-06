"""Approved-script validation and transcript-vs-script matching.

The approved script is the only source of truth for speech. Speech recognition is used to
VERIFY audio against it; a transcript is never written back into the script.
"""

from __future__ import annotations

import difflib
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass

MAX_SCRIPT_CHARS = 400

_QUOTES = {"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "—"}


class ScriptError(ValueError):
    pass


def normalize_script(text: str) -> str:
    """Safe formatting only: Unicode NFC, typographic quotes to plain, whitespace collapsed.
    Words, punctuation and order are untouched."""
    text = unicodedata.normalize("NFC", text or "")
    for a, b in _QUOTES.items():
        if b != "—":
            text = text.replace(a, b)
    return re.sub(r"\s+", " ", text).strip()


def validate_script(text: str) -> str:
    clean = normalize_script(text)
    if not clean:
        raise ScriptError("The approved dialogue is empty.")
    if not re.search(r"[A-Za-zऀ-෿]", clean):
        raise ScriptError("The approved dialogue has no words.")
    if len(clean) > MAX_SCRIPT_CHARS:
        raise ScriptError(f"The approved dialogue is {len(clean)} characters; keep it under {MAX_SCRIPT_CHARS} "
                          "so it fits one 8-second clip.")
    if '"' in clean:
        raise ScriptError('Remove double quotes (") from the dialogue; they would break the instruction to the video model.')
    return clean


def words(text: str) -> list[str]:
    """Comparison tokens: lower case, punctuation dropped, hyphenated words joined
    ("well-being" == "wellbeing"), apostrophes kept ("family's")."""
    text = unicodedata.normalize("NFKC", text.lower()).replace("’", "'")
    text = re.sub(r"(?<=\w)-(?=\w)", "", text)
    return re.findall(r"[a-z0-9ऀ-෿']+", text)


@dataclass
class MatchRules:
    max_word_error_rate: float = 0.10   # (substitutions + deletions + insertions) / approved words
    allow_missing: int = 0              # approved words that may be absent
    allow_extra: int = 1                # extra words tolerated (e.g. a filler the recogniser hears)


def compare(approved: str, transcript: str, rules: MatchRules = MatchRules()) -> dict:
    a, b = words(approved), words(transcript)
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    missing, extra, substituted = [], [], []
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "delete":
            missing += a[i1:i2]
        elif op == "insert":
            extra += b[j1:j2]
        elif op == "replace":
            ap, hd = " ".join(a[i1:i2]), " ".join(b[j1:j2])
            sound_alike = difflib.SequenceMatcher(None, ap, hd).ratio() >= 0.6
            substituted.append({"approved": ap, "heard": hd,
                                "note": "sounds alike: possibly a recognition error, listen to check" if sound_alike else ""})
            missing += a[i1:i2]
            extra += b[j1:j2]
    # A word counted missing AND extra is either a substitution or a word that moved.
    ca, cb = Counter(a), Counter(b)
    reordered = sorted(w for w in set(missing) & set(extra) if ca[w] == cb[w])
    duplicated = sorted(w for w in cb if cb[w] > ca.get(w, 0) and w in ca)
    errors = sum(max(i2 - i1, j2 - j1) for op, i1, i2, j1, j2 in sm.get_opcodes() if op != "equal")
    wer = errors / max(1, len(a))
    char_ratio = difflib.SequenceMatcher(None, " ".join(a), " ".join(b), autojunk=False).ratio()
    true_missing = [w for w in missing if w not in reordered]
    true_extra = [w for w in extra if w not in reordered]
    reasons = []
    if wer > rules.max_word_error_rate:
        reasons.append(f"word error rate {wer:.0%} is above {rules.max_word_error_rate:.0%}")
    if len(true_missing) > rules.allow_missing:
        reasons.append(f"missing words: {', '.join(true_missing)}")
    if len(true_extra) > rules.allow_extra:
        reasons.append(f"extra words: {', '.join(true_extra)}")
    if reordered:
        reasons.append(f"words out of order: {', '.join(reordered)}")
    if duplicated:
        reasons.append(f"repeated words: {', '.join(duplicated)}")
    return {
        "approved": approved,
        "transcript": transcript,
        "word_accuracy": round(max(0.0, 1 - wer), 3),
        "word_error_rate": round(wer, 3),
        "char_similarity": round(char_ratio, 3),
        "missing": true_missing,
        "extra": true_extra,
        "substituted": substituted,
        "reordered": reordered,
        "duplicated": duplicated,
        "passed": not reasons,
        "reasons": reasons,
        "rules": rules.__dict__,
    }

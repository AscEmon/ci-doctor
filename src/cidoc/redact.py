"""Mask anything that looks like a credential before a log touches disk.

GitHub already masks registered secrets as '***', but tokens that leak via
unregistered env vars, echoed config files, or tool debug output do not get
that treatment, so this pass is a safety net.

Vendor rules (known token formats) always mask. The generic keyword rule is
narrower: masking cache keys and file names destroyed diagnostic context while
catching no secrets, so it only masks values that look random (see
_is_secret_value) and never in GitHub Actions cache-key lines.
"""

import math
import re
from collections import Counter
from dataclasses import dataclass, field

# Bump when a rule change alters what gets masked, so stored logs redacted under older
# rules can be found and refetched. 1: original rules. 2: narrowed keyword_secret, jwt.
REDACTOR_VERSION = 2

# keyword_secret thresholds. Random tokens of this length score ~4+ bits/char;
# words, versions and slugs score lower.
MIN_SECRET_LENGTH = 20
MIN_SECRET_ENTROPY = 3.5

# Order matters: whole private-key blocks first, then specific token formats,
# and the generic keyword=value rule last so it never sees an already-masked token.
_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "private_key",
        re.compile(
            r"-----BEGIN[A-Z ]*PRIVATE KEY-----"
            # Normal case: a terminated block (lines may carry GitHub's timestamp prefix).
            r"(?:[\s\S]{0,10000}?-----END[A-Z ]*PRIVATE KEY-----"
            # Truncated log: swallow the base64 body lines that follow the header.
            r"|(?:\r?\n(?:\S+Z )?[A-Za-z0-9+/=]{16,})*)"
        ),
    ),
    ("github_pat", re.compile(r"\bgithub_pat_[A-Za-z0-9_]{22,}")),
    ("github_token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}")),
    ("sk_key", re.compile(r"(?<![A-Za-z0-9])sk-[A-Za-z0-9_-]{20,}")),
    ("aws_access_key", re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b")),
    # JSON Web Token: base64url header and payload, both JSON objects ('{"' -> 'eyJ').
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_-]{5,}\.eyJ[A-Za-z0-9_-]{5,}\.[A-Za-z0-9_-]*")),
    (
        "keyword_secret",
        re.compile(
            r"(?i)((?:key|token|secret|password)[\w.-]*[\"']?\s*[:=]?\s*[\"']?)"
            # No spaces, quotes or brackets: brackets keep [REDACTED:...] masks from
            # earlier rules out. The value may not start with the separator, which the
            # prefix makes optional. Length is checked here, not in the callback, so a
            # short word after the keyword never swallows the real value that follows.
            rf"(?![:=])([^\s\"'`,;()\[\]{{}}<>]{{{MIN_SECRET_LENGTH},}})"
        ),
    ),
)

# GitHub Actions cache lines: the "key" there is a cache key (often a hash), not a secret.
_CACHE_KEY_CONTEXT = re.compile(
    r"cache restored from key|cache not found for input keys|restored from cache key"
    r"|saved cache entry with key",
    re.IGNORECASE,
)
FILE_EXTENSIONS = (
    ".jks",
    ".keystore",
    ".p12",
    ".pfx",
    ".pem",
    ".json",
    ".gradle",
    ".kts",
    ".properties",
    ".plist",
    ".xml",
    ".yaml",
    ".yml",
    ".txt",
)


RULE_NAMES: tuple[str, ...] = tuple(kind for kind, _ in _PATTERNS)
# What a masked value looks like in a stored log; group 1 is the rule name.
MASK_RE = re.compile(r"\[REDACTED:([a-z_]+)\]")


def _mask(kind: str) -> str:
    return f"[REDACTED:{kind}]"


def shannon_entropy(value: str) -> float:
    """Bits per character of `value`'s own character distribution."""
    if not value:
        return 0.0
    n = len(value)
    return sum(c / n * math.log2(n / c) for c in Counter(value).values())


def _looks_like_path(value: str) -> bool:
    return "/" in value or value.lower().endswith(FILE_EXTENSIONS)


def _is_secret_value(value: str) -> bool:
    return (
        len(value) >= MIN_SECRET_LENGTH
        and " " not in value
        and not _looks_like_path(value)
        and shannon_entropy(value) >= MIN_SECRET_ENTROPY
    )


def _in_cache_key_line(match: re.Match[str]) -> bool:
    """True if a cache phrase precedes the value on its line (the phrase ends in the key word)."""
    text = match.string
    line_start = text.rfind("\n", 0, match.start()) + 1
    return bool(_CACHE_KEY_CONTEXT.search(text, line_start, match.start(2)))


@dataclass(frozen=True)
class Redaction:
    text: str
    counts: Counter[str] = field(default_factory=Counter)

    @property
    def total(self) -> int:
        return sum(self.counts.values())


def redact(text: str) -> Redaction:
    """Return `text` with secrets masked, plus how many of each kind were found."""
    counts: Counter[str] = Counter()

    def mask_keyword_value(match: re.Match[str]) -> str:
        prefix, value = match.group(1), match.group(2)
        if _in_cache_key_line(match) or not _is_secret_value(value):
            return match.group(0)
        counts["keyword_secret"] += 1
        # Keep the key name so the log line still reads sensibly.
        return prefix + _mask("keyword_secret")

    for kind, pattern in _PATTERNS:
        if kind == "keyword_secret":
            text = pattern.sub(mask_keyword_value, text)
            continue
        text, n = pattern.subn(_mask(kind), text)
        if n:
            counts[kind] += n
    return Redaction(text, counts)

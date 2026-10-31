"""Mask anything that looks like a credential before a log touches disk.

GitHub already masks registered secrets as '***', but tokens that leak via
unregistered env vars, echoed config files, or tool debug output do not get
that treatment. This pass is deliberately aggressive: a false positive costs a
few characters of log context, a false negative leaks a credential.
"""

import re
from collections import Counter
from dataclasses import dataclass, field

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
    (
        "keyword_secret",
        re.compile(
            r"(?i)((?:key|token|secret|password)[\w.-]*[\"']?\s*[:=]?\s*[\"']?)"
            r"([A-Za-z0-9+/_=-]{41,})"
        ),
    ),
)


def _mask(kind: str) -> str:
    return f"[REDACTED:{kind}]"


def _mask_keyword_value(match: re.Match[str]) -> str:
    # Keep the key name so the log line still reads sensibly.
    return match.group(1) + _mask("keyword_secret")


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
    for kind, pattern in _PATTERNS:
        replacement = _mask_keyword_value if kind == "keyword_secret" else _mask(kind)
        text, n = pattern.subn(replacement, text)
        if n:
            counts[kind] += n
    return Redaction(text, counts)

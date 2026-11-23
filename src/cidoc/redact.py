"""Mask anything that looks like a credential before a log touches disk.

GitHub already masks registered secrets as '***', but tokens that leak via
unregistered env vars, echoed config files, or tool debug output do not get
that treatment, so this pass is a safety net.

Vendor rules (known token formats) always mask. The generic rule,
assignment_secret, only looks at assignments to a secret-sounding name
(API_TOKEN=..., client_secret: ...) and only masks values that look random.
Matching on a bare "key" word was tried and dropped: in CI logs it hits package
names, cache keys and build flags far more often than secrets.
"""

import math
import re
from collections import Counter
from dataclasses import dataclass, field

# Bump when a rule change alters what gets masked, so stored logs redacted under older
# rules can be found and refetched. 1: original rules. 2: narrowed keyword_secret, jwt.
# 3: keyword_secret replaced by assignment_secret.
REDACTOR_VERSION = 3

# assignment_secret thresholds. Random tokens this long score ~4.5+ bits/char; slugs,
# versions and hash-suffixed identifiers mostly score lower.
MIN_SECRET_LENGTH = 24
MIN_SECRET_ENTROPY = 4.0

# A whole identifier (word boundaries on both sides) containing a secret-ish part.
# Bare KEY, PUBLIC_KEY and PUBLISHABLE_KEY contain none of these and never match.
_SENSITIVE_NAME = (
    r"\b[A-Z0-9_]*(?:SECRET|TOKEN|PASSWORD|PASSWD|API_KEY|APIKEY|PRIVATE_KEY|ACCESS_KEY"
    r"|CLIENT_SECRET)[A-Z0-9_]*\b"
)
# Characters that end a value: whitespace, quotes, brackets and list separators.
_VALUE_CHARS = r"[^\s\"'`,;()\[\]{}<>]"

# Order matters: whole private-key blocks first, then specific token formats,
# and the generic assignment rule last so it never sees an already-masked token.
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
        "assignment_secret",
        re.compile(
            # Group 1: NAME, optional closing quote, '=' or ':' (required), optional
            # opening quote. Kept in the output so the line still reads sensibly.
            rf"(?i)({_SENSITIVE_NAME}[\"']?[ \t]*[:=][ \t]*[\"']?)"
            # Group 2: the value. It may not start with another separator, must be long
            # enough here (not just in the callback, so a rejected short value can't hide
            # a later one), and must end at a value boundary, never mid-token.
            rf"(?![:=])({_VALUE_CHARS}{{{MIN_SECRET_LENGTH},}})(?!{_VALUE_CHARS})"
        ),
    ),
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
# GitHub's own mask for registered secrets.
GITHUB_MASK = "***"

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


def looks_like_path(value: str) -> bool:
    """A file path, URL or file name.

    Deliberately not "contains '/'": base64 secrets often contain '/' mid-value, and
    roughly half of 40-char ones would slip through. Relative directories without a
    leading ./ or a file extension are rare as secret-named values and get masked.
    """
    return (
        value.startswith(("/", "./", "../", "~/"))
        or "://" in value
        or value.lower().endswith(FILE_EXTENSIONS)
    )


def is_secret_value(value: str) -> bool:
    return (
        len(value) >= MIN_SECRET_LENGTH
        and " " not in value
        and GITHUB_MASK not in value
        and not looks_like_path(value)
        and shannon_entropy(value) >= MIN_SECRET_ENTROPY
    )


def _is_excluded_name(prefix: str) -> bool:
    # Xcode build settings (INFOPLIST_KEY_*) hold Info.plist text, never secrets.
    return "INFOPLIST" in prefix.upper()


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

    def mask_assignment(match: re.Match[str]) -> str:
        prefix, value = match.group(1), match.group(2)
        if _is_excluded_name(prefix) or not is_secret_value(value):
            return match.group(0)
        counts["assignment_secret"] += 1
        return prefix + _mask("assignment_secret")

    for kind, pattern in _PATTERNS:
        if kind == "assignment_secret":
            text = pattern.sub(mask_assignment, text)
            continue
        text, n = pattern.subn(_mask(kind), text)
        if n:
            counts[kind] += n
    return Redaction(text, counts)

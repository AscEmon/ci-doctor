"""Read-only tools the agent can call, all bound to one failed job (RunContext).

Every tool returns plain text and never raises into the agent loop: a bad argument, a
404, a rate limit or an oversized file comes back as a short "error: ..." text the model
can read and work around.

Repository tools read at the failing commit (head_sha) through our GitHub client only,
with the existing read-only token; nothing else touches the network. Paths are checked
before any request: no absolute paths, no "..", nothing that could leave the repository.
Everything a tool returns is redacted with the miner's secret rules and capped at
`cap_tokens` (char-based estimate), ending in a "(truncated)" marker when cut.
"""

import base64
import binascii
import contextlib
import hashlib
import importlib
import re
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol
from urllib.parse import quote

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from cidoc.agent.cache import ToolCache, tool_cache_key
from cidoc.agent.context import RunContext, valid_repo, valid_sha
from cidoc.analyze.preprocess import CHARS_PER_TOKEN, MAX_LINE_CHARS
from cidoc.github_client import RetryableHTTPError
from cidoc.logview import locate_failed_steps_in_lines, split_lines
from cidoc.miner.runs import failed_step_names, runner_os_from_labels
from cidoc.redact import redact

OUTPUT_CAP_TOKENS = 2000
TRUNCATED = "... (truncated)"
ERROR_PREFIX = "error: "
ERROR_MAX_CHARS = 300

READ_FILE_MAX_LINES = 200
READ_FILE_MAX_BYTES = 16 * 1024
LIST_DIR_MAX_ENTRIES = 200
# Files are sniffed for NUL bytes in their first 8 KiB, as git does.
BINARY_SNIFF_BYTES = 8192

Scope = Literal["commit", "run", "job"]
"""What a tool's result depends on: the commit (head_sha), the run (run_id) or the job's
stored log (job_id). Each is immutable, so a result for one never changes."""


class ToolError(Exception):
    """An expected failure; its message is shown to the model after "error: "."""


class GitHubReader(Protocol):
    """The slice of GitHubClient the tools use (GET JSON only: read-only by shape)."""

    def get_json(self, url: str, params: dict[str, Any] | None = None) -> Any: ...


@dataclass(frozen=True)
class ToolEnv:
    ctx: RunContext
    github: GitHubReader | None


class NoArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]
    """JSON schema of the arguments, sent to the model."""
    args_model: type[BaseModel]
    """Validates and normalizes the model's arguments (defaults filled in)."""
    scope: Scope
    run: Callable[[ToolEnv, Any], str]
    needs_github: bool = True


TOOLS: dict[str, ToolSpec] = {}
"""Every tool, in the order they are offered to the model."""


def _register(spec: ToolSpec) -> None:
    TOOLS[spec.name] = spec


# --- shared helpers -------------------------------------------------------------------


_DRIVE = re.compile(r"^[A-Za-z]:")
_SCHEME = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")


def validate_repo_path(path: str, allow_root: bool = False) -> str:
    """A repository-relative path, normalized ("./a//b" -> "a/b"); ToolError if it is
    absolute, climbs out with "..", or is not a plain path. "" is the repository root."""
    if any(ord(c) < 32 or c == "\x7f" for c in path):
        raise ToolError("path contains control characters")
    if "\\" in path:
        raise ToolError("use forward slashes in paths")
    stripped = path.strip()
    if stripped.startswith(("/", "~")) or _DRIVE.match(stripped):
        raise ToolError(f"absolute paths are not allowed: {path!r}; use a repository path")
    if _SCHEME.match(stripped):
        raise ToolError(f"not a repository path: {path!r}")
    parts = [p for p in stripped.split("/") if p not in ("", ".")]
    if ".." in parts:
        raise ToolError(f'".." is not allowed in paths: {path!r}')
    if not parts and not allow_root:
        raise ToolError("path is empty")
    return "/".join(parts)


def _require_run(env: ToolEnv) -> tuple[GitHubReader, str, str]:
    """(github, repo, head_sha), or ToolError when this job's run is unknown."""
    repo, sha = env.ctx.repo, env.ctx.head_sha
    if env.github is None or repo is None or sha is None or env.ctx.run_id is None:
        raise ToolError("no run metadata for this job (not in the database); use the log tools")
    if not valid_repo(repo) or not valid_sha(sha):
        raise ToolError("run metadata for this job looks malformed")
    return env.github, repo, sha


def _contents_url(repo: str, path: str) -> str:
    # quote(): "?" or "#" in a path must stay part of the path, never become a query.
    return f"/repos/{repo}/contents/{quote(path, safe='/')}" if path else f"/repos/{repo}/contents"


def _clip_line(text: str) -> str:
    if len(text) > MAX_LINE_CHARS:
        return f"{text[:MAX_LINE_CHARS]}... [{len(text) - MAX_LINE_CHARS} chars truncated]"
    return text


def _decode_file(data: Any, path: str) -> list[str]:
    """The lines of a contents-API file entry; ToolError for directories, symlinks,
    submodules, files too big for the API, and binary files."""
    if isinstance(data, list):
        raise ToolError(f"{path} is a directory; use list_dir")
    if not isinstance(data, dict):
        raise ToolError(f"unexpected response for {path}")
    kind = data.get("type")
    if kind != "file":
        raise ToolError(f"{path} is a {kind}, not a file")
    size = data.get("size", 0)
    if data.get("encoding") != "base64" or not data.get("content"):
        if size:  # the contents API leaves content empty above 1 MB
            raise ToolError(f"{path} is too large to read ({size} bytes)")
        return []
    try:
        raw = base64.b64decode(data["content"])
    except (binascii.Error, ValueError):
        raise ToolError(f"cannot decode {path}") from None
    if b"\x00" in raw[:BINARY_SNIFF_BYTES]:
        raise ToolError(f"{path} is a binary file ({len(raw)} bytes); not shown")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise ToolError(f"{path} is a binary file ({len(raw)} bytes); not shown") from None
    return split_lines(text)


def number_lines(
    lines: Sequence[str], start: int, end: int, max_bytes: int | None = None
) -> tuple[list[str], int]:
    """Lines start..end (1-based, inclusive) as "<n>| <text>". Stops early once the
    output would pass `max_bytes`; returns (rendered, last line rendered)."""
    width = len(str(end))
    out: list[str] = []
    used = 0
    last = start - 1
    for n in range(start, end + 1):
        row = f"{n:>{width}}| {_clip_line(lines[n - 1])}"
        size = len(row.encode("utf-8")) + 1
        if max_bytes is not None and out and used + size > max_bytes:
            break
        out.append(row)
        used += size
        last = n
    return out, last


# --- repository tools -----------------------------------------------------------------


class ReadFileArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    start_line: int = Field(1, ge=1)
    end_line: int = Field(READ_FILE_MAX_LINES, ge=1)

    @field_validator("path")
    @classmethod
    def _repo_path(cls, value: str) -> str:
        # Normalized here, so "./a" and "a" are one call (and one cache entry).
        try:
            return validate_repo_path(value)
        except ToolError as exc:
            raise ValueError(str(exc)) from None


def read_file(env: ToolEnv, args: ReadFileArgs) -> str:
    github, repo, sha = _require_run(env)
    path = validate_repo_path(args.path)
    if args.end_line < args.start_line:
        raise ToolError("end_line is before start_line")
    lines = _decode_file(github.get_json(_contents_url(repo, path), {"ref": sha}), path)
    total = len(lines)
    if total == 0:
        return f"{path} @ {sha[:12]}: empty file"
    if args.start_line > total:
        raise ToolError(f"start_line {args.start_line} is past the end ({path} has {total} lines)")
    end = min(args.end_line, args.start_line + READ_FILE_MAX_LINES - 1, total)
    rows, last = number_lines(lines, args.start_line, end, READ_FILE_MAX_BYTES)
    header = f"{path} @ {sha[:12]}: lines {args.start_line}-{last} of {total}"
    footer = []
    if last < total:
        footer.append(f"({total - last} more lines; read_file start_line={last + 1})")
    return "\n".join([header, *rows, *footer])


_register(
    ToolSpec(
        name="read_file",
        description=(
            "Read a text file of the repository at the failing commit, with line numbers."
            f" At most {READ_FILE_MAX_LINES} lines or 16 KB per call; page with start_line."
            " Use for build config: pubspec.yaml, android/build.gradle(.kts),"
            " android/app/build.gradle, gradle-wrapper.properties, ios/Podfile, etc."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path relative to the repository root, e.g. pubspec.yaml.",
                },
                "start_line": {"type": "integer", "description": "First line (1-based)."},
                "end_line": {"type": "integer", "description": "Last line (inclusive)."},
            },
            "required": ["path"],
        },
        args_model=ReadFileArgs,
        scope="commit",
        run=read_file,
    )
)


class ListDirArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str = "."

    @field_validator("path")
    @classmethod
    def _repo_path(cls, value: str) -> str:
        try:
            return validate_repo_path(value, allow_root=True) or "."
        except ToolError as exc:
            raise ValueError(str(exc)) from None


def list_dir(env: ToolEnv, args: ListDirArgs) -> str:
    github, repo, sha = _require_run(env)
    path = validate_repo_path(args.path, allow_root=True)
    shown = path or "."
    data = github.get_json(_contents_url(repo, path), {"ref": sha})
    if isinstance(data, dict):
        raise ToolError(f"{shown} is a {data.get('type', 'file')}; use read_file")
    if not isinstance(data, list):
        raise ToolError(f"unexpected response for {shown}")
    entries = [e for e in data if isinstance(e, dict)]
    rows = []
    for entry in entries[:LIST_DIR_MAX_ENTRIES]:
        kind = str(entry.get("type", "?"))
        size = f" ({entry.get('size', 0)} bytes)" if kind == "file" else ""
        rows.append(f"{kind:<7} {entry.get('name', '?')}{size}")
    header = f"{shown} @ {sha[:12]}: {len(entries)} entries"
    footer = []
    if len(entries) > LIST_DIR_MAX_ENTRIES:
        footer.append(f"... ({len(entries) - LIST_DIR_MAX_ENTRIES} more entries not shown)")
    return "\n".join([header, *rows, *footer])


_register(
    ToolSpec(
        name="list_dir",
        description=(
            "List a repository directory at the failing commit: name, type, size."
            f" At most {LIST_DIR_MAX_ENTRIES} entries."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": 'Directory relative to the repository root; "." is the root.',
                }
            },
        },
        args_model=ListDirArgs,
        scope="commit",
        run=list_dir,
    )
)


def read_workflow(env: ToolEnv, args: NoArgs) -> str:
    github, repo, sha = _require_run(env)
    if not env.ctx.workflow_path:
        raise ToolError("the workflow path of this run is unknown")
    path = validate_repo_path(env.ctx.workflow_path)
    lines = _decode_file(github.get_json(_contents_url(repo, path), {"ref": sha}), path)
    rows, _ = number_lines(lines, 1, len(lines))
    return "\n".join([f"{path} @ {sha[:12]}: {len(lines)} lines", *rows])


_register(
    ToolSpec(
        name="read_workflow",
        description=(
            "Read the GitHub Actions workflow file of the failed run, in full, with line"
            " numbers: jobs, matrix, runs-on, setup actions and their versions."
        ),
        parameters={"type": "object", "properties": {}},
        args_model=NoArgs,
        scope="commit",
        run=read_workflow,
    )
)


# --- log tools --------------------------------------------------------------------------

READ_LOG_MAX_LINES = 150
SEARCH_DEFAULT_MATCHES = 20
SEARCH_MAX_MATCHES = 50
SEARCH_PATTERN_MAX_CHARS = 200
SEARCH_DEADLINE_S = 2.0
# Python's re cannot be interrupted mid-match, so risky patterns are refused up front.
# Measured on one 500-char line:
# - backreferences, and alternation or repetition inside an unbounded repeat ("(a|aa)*",
#   "(a+)+"): exponential, never finishes;
# - three unbounded repeats over broad items (".", negated sets, \S \W \D) such as
#   ".*a.*b.*c": ~0.9 s (four: ~13 s); two such as ".*.*z": ~27 ms;
# - three unbounded repeats in a row ("x*x*x*y"): ~3.5 s.
# Repeats over narrow, separated items ("[a-z_]+\.dart:\d+:\d+") stay well under 1 ms.
# Each line is matched on its first MAX_LINE_CHARS characters (as the excerpt shows it),
# so an accepted pattern costs at most tens of ms per line, and the scan stops at
# SEARCH_DEADLINE_S, checked between lines.
MAX_BROAD_REPEATS = 2
MAX_ADJACENT_REPEATS = 2
LARGE_REPEAT = 100
"""A {m,n} repeat with n above this counts as unbounded."""

_sre_parser: Any = importlib.import_module("re._parser")
_sre_constants: Any = importlib.import_module("re._constants")
_BACKREFERENCES = frozenset(
    {"GROUPREF", "GROUPREF_EXISTS", "GROUPREF_IGNORE", "GROUPREF_LOC_IGNORE", "GROUPREF_UNI_IGNORE"}
)
_REPEATS = frozenset({"MAX_REPEAT", "MIN_REPEAT", "POSSESSIVE_REPEAT"})
_NEGATED_CATEGORIES = frozenset({"CATEGORY_NOT_DIGIT", "CATEGORY_NOT_SPACE", "CATEGORY_NOT_WORD"})


def _is_broad(item: Any) -> bool:
    r"""Whether a repeated item matches most characters: ".", a negated set, \S, \W, \D.
    Groups count as broad (conservative)."""
    if len(item) != 1:
        return True
    op, value = item[0]
    name = str(op)
    if name == "ANY":
        return True
    if name == "IN":
        return any(
            str(o) == "NEGATE" or (str(o) == "CATEGORY" and str(v) in _NEGATED_CATEGORIES)
            for o, v in value
        )
    return name != "LITERAL"


def _check_tree(items: Any, inside_unbounded: bool, multiplier: int, broad: list[int]) -> None:
    adjacent = 0
    for op, value in items:
        name = str(op)
        if name in _BACKREFERENCES:
            raise ToolError("backreferences are not allowed in search patterns")
        is_unbounded_repeat = False
        if name in _REPEATS:
            _, high, item = value
            unbounded = high == _sre_constants.MAXREPEAT or high > LARGE_REPEAT
            if inside_unbounded and high > 1:
                raise ToolError("nested repetition (like (a+)+) is not allowed; simplify")
            if unbounded:
                is_unbounded_repeat = True
                if _is_broad(item):
                    broad[0] += multiplier
                _check_tree(item, True, multiplier, broad)
            else:
                _check_tree(item, inside_unbounded, multiplier * max(high, 1), broad)
        elif name == "BRANCH":
            if inside_unbounded:
                raise ToolError("alternation inside a repeated group is not allowed; simplify")
            for alternative in value[1]:
                _check_tree(alternative, inside_unbounded, multiplier, broad)
        elif name == "SUBPATTERN":
            _check_tree(value[3], inside_unbounded, multiplier, broad)
        elif name in ("ASSERT", "ASSERT_NOT"):
            _check_tree(value[1], inside_unbounded, multiplier, broad)
        elif name == "ATOMIC_GROUP":
            _check_tree(value, inside_unbounded, multiplier, broad)
        adjacent = adjacent + 1 if is_unbounded_repeat else 0
        if adjacent > MAX_ADJACENT_REPEATS:
            raise ToolError(
                f"more than {MAX_ADJACENT_REPEATS} unbounded repeats in a row; simplify"
            )
    if broad[0] > MAX_BROAD_REPEATS:
        raise ToolError(
            f"more than {MAX_BROAD_REPEATS} unbounded repeats of broad items"
            " (., [^...], \\S, \\W, \\D); use literal anchors or narrower classes"
        )


def compile_search_pattern(pattern: str, ignore_case: bool) -> re.Pattern[str]:
    """`pattern` compiled, or ToolError if it is invalid or could backtrack badly."""
    if not pattern:
        raise ToolError("pattern is empty")
    if len(pattern) > SEARCH_PATTERN_MAX_CHARS:
        raise ToolError(f"pattern is longer than {SEARCH_PATTERN_MAX_CHARS} characters")
    flags = re.IGNORECASE if ignore_case else 0
    try:
        compiled = re.compile(pattern, flags)
        tree = _sre_parser.parse(pattern, flags)
    except (re.error, RecursionError, OverflowError) as exc:
        raise ToolError(f"invalid regex: {exc}") from None
    _check_tree(tree, inside_unbounded=False, multiplier=1, broad=[0])
    return compiled


@dataclass(frozen=True)
class SearchResult:
    matches: list[int]
    """0-based indices of every matching line scanned, in log order."""
    stopped_at: int | None = None
    """The 0-based index the scan stopped before, when it ran out of time."""


def search_lines(
    lines: Sequence[str],
    regex: re.Pattern[str],
    candidates: Iterable[int] | None = None,
    deadline_s: float = SEARCH_DEADLINE_S,
    clock: Callable[[], float] = time.monotonic,
) -> SearchResult:
    """Indices of `candidates` (default: every line) whose first MAX_LINE_CHARS
    characters match `regex`; stops at `deadline_s`, checked between lines."""
    start = clock()
    matches: list[int] = []
    for i in candidates if candidates is not None else range(len(lines)):
        if clock() - start > deadline_s:
            return SearchResult(matches, stopped_at=i)
        if regex.search(lines[i][:MAX_LINE_CHARS]):
            matches.append(i)
    return SearchResult(matches)


# Lines that match common patterns ("fail", "error") without saying anything about the
# failure. GitHub prints a header block for each step: "##[group]Run <command>", the
# command, "shell: ... -o pipefail {0}", "env:" with indented "  NAME: value" lines, then
# "##[endgroup]". shell:/env:/assignments count as noise only inside that block, so
# test output such as "  Actual: <4>" is never dropped.
_GROUP_RUN = "##[group]Run"
_ENDGROUP = "##[endgroup]"
_COMMAND = "[command]"
_HEADER_NOISE = ("shell:", "env:")
_ENV_ASSIGNMENT = re.compile(r"^\s+[A-Za-z_][A-Za-z0-9_]*: ")


def search_noise(lines: Sequence[str]) -> set[int]:
    """0-based indices of lines search_log never returns."""
    noise: set[int] = set()
    in_header = in_env = False
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if stripped.startswith(_GROUP_RUN):
            noise.add(i)
            in_header, in_env = True, False
        elif stripped.startswith(_ENDGROUP):
            noise.add(i)
            in_header = in_env = False
        elif stripped.startswith(_COMMAND):
            noise.add(i)
        elif in_header and stripped.startswith(_HEADER_NOISE):
            noise.add(i)
            in_env = stripped.startswith("env:")
        elif in_env and _ENV_ASSIGNMENT.match(line):
            noise.add(i)
        else:
            in_env = False
    return noise


SearchScope = Literal["failed_step", "all"]


class SearchLogArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pattern: str
    max_matches: int = Field(SEARCH_DEFAULT_MATCHES, ge=1, le=SEARCH_MAX_MATCHES)
    ignore_case: bool = True
    scope: SearchScope = "failed_step"


@dataclass(frozen=True)
class Hit:
    index: int
    """0-based index of the first occurrence."""
    text: str
    count: int
    distance: int | None
    """Lines to the nearest ##[error]; None when the log has none."""


def rank_hits(lines: Sequence[str], matches: Iterable[int], errors: Sequence[int]) -> list[Hit]:
    """Identical lines merged (first occurrence, with a count), closest to an ##[error]
    first, then in log order."""
    first: dict[str, int] = {}
    counts: dict[str, int] = {}
    for i in matches:
        text = lines[i].strip()
        first.setdefault(text, i)
        counts[text] = counts.get(text, 0) + 1

    def distance(i: int) -> int | None:
        return min((abs(i - e) for e in errors), default=None)

    hits = [Hit(i, text, counts[text], distance(i)) for text, i in first.items()]
    hits.sort(key=lambda h: (h.distance is None, h.distance or 0, h.index))
    return hits


def search_log(env: ToolEnv, args: SearchLogArgs) -> str:
    regex = compile_search_pattern(args.pattern, args.ignore_case)
    lines = env.ctx.log_lines
    located = locate_failed_steps_in_lines(lines, env.ctx.failed_steps)
    scope_note = ""
    if args.scope == "failed_step" and located.failed:
        candidates = sorted(
            {
                i
                for step in located.failed
                for i in range(step.start, step.stop)
                if i not in located.hidden
            }
        )
        titles = "; ".join(step.title for step in located.failed)
        scope_note = f"in the failed step ({titles})"
    else:
        candidates = list(range(located.end))  # nothing after "Post job cleanup."
        scope_note = "in the whole log"
        if args.scope == "failed_step":
            scope_note += " (no failed step found)"
    noise = search_noise(lines)
    candidates = [i for i in candidates if i not in noise]
    found = search_lines(lines, regex, candidates)
    hits = rank_hits(lines, found.matches, located.errors)
    shown = hits[: args.max_matches]
    total = len(found.matches)
    if not hits:
        header = f"no lines match /{args.pattern}/ {scope_note}"
    else:
        header = (
            f"{total} matching lines ({len(hits)} distinct) for /{args.pattern}/ {scope_note};"
            f" showing {len(shown)} of {len(hits)}, closest to an ##[error] first"
        )
    rows = []
    for hit in shown:
        repeat = f" (x{hit.count})" if hit.count > 1 else ""
        rows.append(f"L{hit.index + 1}| {_clip_line(lines[hit.index])}{repeat}")
    footer = []
    if found.stopped_at is not None:
        footer.append(
            f"(search stopped after {SEARCH_DEADLINE_S:g}s before line {found.stopped_at + 1}"
            f" of {len(lines)}; use a narrower pattern)"
        )
    return "\n".join([header, *rows, *footer])


_register(
    ToolSpec(
        name="search_log",
        description=(
            "Search the FULL job log (not just the excerpt) with a Python regular expression;"
            ' returns matching lines as "L<n>| <line>", numbered like the excerpt, identical'
            " lines merged with a count, closest to an ##[error] first. scope"
            ' "failed_step" (default) searches the failed step only, "all" the whole job.'
            " Step headers, shell:/env: lines and [command] lines never match."
            " Case-insensitive unless ignore_case is false. Good patterns:"
            ' "Expected:|Actual:" (test assertions), "Exception|Error:" (exceptions),'
            r' "\[E\]|Some tests failed" (failed tests), "Could not|timed out|429|ECONNRESET"'
            ' (network and infrastructure). Beware: "fail" also matches "pipefail"; use'
            r" word boundaries: \bfail\b. Backreferences, nested repeats and more than two"
            " unbounded repeats are refused."
        ),
        parameters={
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "A Python regex."},
                "max_matches": {
                    "type": "integer",
                    "description": f"Distinct lines to return (1-{SEARCH_MAX_MATCHES}).",
                },
                "ignore_case": {"type": "boolean"},
                "scope": {
                    "type": "string",
                    "enum": ["failed_step", "all"],
                    "description": 'Where to search; default "failed_step".',
                },
            },
            "required": ["pattern"],
        },
        args_model=SearchLogArgs,
        scope="job",
        run=search_log,
        needs_github=False,
    )
)


class ReadLogArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)


def read_log(env: ToolEnv, args: ReadLogArgs) -> str:
    lines = env.ctx.log_lines
    total = len(lines)
    if args.end_line < args.start_line:
        raise ToolError("end_line is before start_line")
    if args.start_line > total:
        raise ToolError(f"start_line {args.start_line} is past the end (the log has {total} lines)")
    end = min(args.end_line, args.start_line + READ_LOG_MAX_LINES - 1, total)
    rows = [f"L{n}| {_clip_line(lines[n - 1])}" for n in range(args.start_line, end + 1)]
    header = f"log lines {args.start_line}-{end} of {total}"
    footer = []
    if end < args.end_line and end < total:
        footer.append(f"(at most {READ_LOG_MAX_LINES} lines per call; continue at {end + 1})")
    return "\n".join([header, *rows, *footer])


_register(
    ToolSpec(
        name="read_log",
        description=(
            'Read raw lines of the FULL job log by line number (the N in "L<N>|"), e.g. the'
            f' lines behind "... (N lines omitted)". At most {READ_LOG_MAX_LINES} per call.'
        ),
        parameters={
            "type": "object",
            "properties": {
                "start_line": {"type": "integer"},
                "end_line": {"type": "integer", "description": "Inclusive."},
            },
            "required": ["start_line", "end_line"],
        },
        args_model=ReadLogArgs,
        scope="job",
        run=read_log,
        needs_github=False,
    )
)


# --- run-context tools ------------------------------------------------------------------

JOBS_PAGE_SIZE = 100
PREVIOUS_RUNS_PAGE_SIZE = 30
COMMIT_MAX_FILES = 100
PATCH_MAX_LINES = 200
# "Flutter 3.24.0", "java: 17", "xcode_15.4", "jdk-21", "dart 3.5"
_VERSION_IN_NAME = re.compile(
    r"\b(flutter|dart|java|jdk|xcode|ndk|gradle|kotlin|node)[\s:=_-]*v?(\d+(?:\.\d+)*)",
    re.IGNORECASE,
)
_MATRIX_IN_NAME = re.compile(r"\(([^()]*)\)\s*$")


def _run_id(env: ToolEnv) -> int:
    if env.ctx.run_id is None:
        raise ToolError("no run metadata for this job (not in the database); use the log tools")
    return env.ctx.run_id


def versions_in_name(name: str) -> list[str]:
    """Toolchain versions visible in a job name: "flutter 3.24.0", and the matrix values
    GitHub appends in parentheses ("build (macos-14, 3.24.0)")."""
    found = [f"{tool.lower()} {version}" for tool, version in _VERSION_IN_NAME.findall(name)]
    matrix = _MATRIX_IN_NAME.search(name)
    if matrix and matrix.group(1).strip():
        found.append(f"matrix ({matrix.group(1).strip()})")
    return found


def _job_line(job: dict[str, Any]) -> str:
    name = str(job.get("name", "?"))
    parts = [name, str(job.get("conclusion") or job.get("status") or "?")]
    parts.append(runner_os_from_labels(job.get("labels") or []) or "unknown OS")
    failed = failed_step_names(job)
    if failed:
        parts.append("failed steps: " + "; ".join(failed))
    versions = versions_in_name(name)
    if versions:
        parts.append("versions: " + ", ".join(versions))
    return " | ".join(parts)


def sibling_jobs(env: ToolEnv, args: NoArgs) -> str:
    github, repo, _ = _require_run(env)
    run_id = _run_id(env)
    data = github.get_json(
        f"/repos/{repo}/actions/runs/{run_id}/jobs",
        {"filter": "latest", "per_page": JOBS_PAGE_SIZE},
    )
    jobs = [j for j in (data.get("jobs") or []) if isinstance(j, dict)]
    this = [j for j in jobs if j.get("id") == env.ctx.job_id]
    others = [j for j in jobs if j.get("id") != env.ctx.job_id]
    counts: dict[str, int] = {}
    for job in others:
        key = str(job.get("conclusion") or job.get("status") or "?")
        counts[key] = counts.get(key, 0) + 1
    summary = ", ".join(f"{n} {c}" for c, n in sorted(counts.items())) or "none"
    rows = [f"this job: {_job_line(this[0])}" if this else "this job: (not listed)"]
    rows.append(f"{len(others)} other jobs in this run ({summary}):")
    rows.extend(f"- {_job_line(job)}" for job in others)
    total = data.get("total_count")
    if isinstance(total, int) and total > len(jobs):
        rows.append(f"... ({total - len(jobs)} more jobs not listed)")
    return "\n".join(rows)


_register(
    ToolSpec(
        name="sibling_jobs",
        description=(
            "The other jobs of the same workflow run: name, conclusion, runner OS, failed"
            " step names, and Flutter/Java/Xcode versions or matrix values visible in the"
            " job name. No logs. Use it to see if the failure happens on only one OS,"
            " channel or version."
        ),
        parameters={"type": "object", "properties": {}},
        args_model=NoArgs,
        scope="run",
        run=sibling_jobs,
    )
)


def previous_run(env: ToolEnv, args: NoArgs) -> str:
    github, repo, sha = _require_run(env)
    run_id = _run_id(env)
    branch = env.ctx.head_branch
    if not branch:
        raise ToolError("the branch of this run is unknown")
    run = github.get_json(f"/repos/{repo}/actions/runs/{run_id}")
    workflow_id, created = run.get("workflow_id"), run.get("created_at")
    if not isinstance(workflow_id, int) or not isinstance(created, str):
        raise ToolError("cannot tell which workflow this run belongs to")
    data = github.get_json(
        f"/repos/{repo}/actions/workflows/{workflow_id}/runs",
        {
            "branch": branch,
            "status": "completed",
            "created": f"<{created}",
            "per_page": PREVIOUS_RUNS_PAGE_SIZE,
        },
    )
    # Filter here too: only runs created strictly before this one, never this run, so a
    # later run (one that may contain the fix) can never be returned.
    earlier = [
        r
        for r in (data.get("workflow_runs") or [])
        if isinstance(r, dict)
        and r.get("id") != run_id
        and isinstance(r.get("created_at"), str)
        and r["created_at"] < created
    ]
    if not earlier:
        return f"no earlier completed run of this workflow on branch {branch}"
    latest = max(earlier, key=lambda r: str(r["created_at"]))
    other_sha = str(latest.get("head_sha", ""))
    same = "same commit as the failing run" if other_sha == sha else "a different commit"
    return "\n".join(
        [
            f"latest earlier run of this workflow on branch {branch}:",
            f"conclusion: {latest.get('conclusion')}",
            f"head_sha: {other_sha[:12]} ({same})",
            f"created_at: {latest['created_at']} (failing run: {created})",
            f"event: {latest.get('event', '?')}",
        ]
    )


_register(
    ToolSpec(
        name="previous_run",
        description=(
            "The latest earlier completed run of the same workflow on the same branch:"
            ' conclusion, head_sha, created_at. Answers "was it passing before?"; with'
            " commit_changes it tells a code change from an environment change."
        ),
        parameters={"type": "object", "properties": {}},
        args_model=NoArgs,
        scope="run",
        run=previous_run,
    )
)


def _patch_excerpt(files: list[dict[str, Any]], max_lines: int) -> list[str]:
    out: list[str] = []
    remaining = sum(len(str(f.get("patch") or "").splitlines()) for f in files)
    for f in files:
        patch = str(f.get("patch") or "").splitlines()
        if not patch:
            continue
        if len(out) >= max_lines:
            break
        out.append(f"--- {f.get('filename', '?')}")
        room = max_lines - len(out)
        out.extend(_clip_line(line) for line in patch[:room])
        remaining -= min(len(patch), room)
    if remaining > 0:
        out.append(f"... (patch truncated: {remaining} more lines)")
    return out


def commit_changes(env: ToolEnv, args: NoArgs) -> str:
    github, repo, sha = _require_run(env)
    commit = github.get_json(f"/repos/{repo}/commits/{sha}")
    parents = [p.get("sha") for p in commit.get("parents") or [] if isinstance(p, dict)]
    message = str((commit.get("commit") or {}).get("message", "")).strip()
    title = message.splitlines()[0] if message else "(no message)"
    if len(parents) > 1:
        # A merge: diff against the first parent explicitly. It is an ancestor of sha, so
        # the three-dot merge base is the parent itself.
        first = str(parents[0])
        if not valid_sha(first):
            raise ToolError("the first parent of this merge commit looks malformed")
        files = github.get_json(f"/repos/{repo}/compare/{first}...{sha}").get("files") or []
        kind = f"merge commit; changes vs its first parent {first[:12]}"
    else:
        files = commit.get("files") or []
        kind = "changes vs its parent" if parents else "root commit"
    files = [f for f in files if isinstance(f, dict)]
    additions = sum(int(f.get("additions") or 0) for f in files)
    deletions = sum(int(f.get("deletions") or 0) for f in files)
    rows = [
        f"commit {sha[:12]}: {_clip_line(title)}",
        f"{kind}: {len(files)} files, +{additions} -{deletions}",
    ]
    for f in files[:COMMIT_MAX_FILES]:
        rows.append(
            f"{f.get('status', '?'):<9} {f.get('filename', '?')}"
            f" (+{f.get('additions', 0)} -{f.get('deletions', 0)})"
        )
    if len(files) > COMMIT_MAX_FILES:
        rows.append(f"... ({len(files) - COMMIT_MAX_FILES} more files not listed)")
    patch = _patch_excerpt(files, PATCH_MAX_LINES)
    if patch:
        rows.append(f"patch excerpt (at most {PATCH_MAX_LINES} lines):")
        rows.extend(patch)
    return "\n".join(rows)


_register(
    ToolSpec(
        name="commit_changes",
        description=(
            "Files changed by the failing commit vs its first parent: paths, additions,"
            f" deletions and a patch excerpt (at most {PATCH_MAX_LINES} lines). Use it to"
            " see whether the code or config changed right before the failure."
        ),
        parameters={"type": "object", "properties": {}},
        args_model=NoArgs,
        scope="commit",
        run=commit_changes,
    )
)


# --- running a tool -------------------------------------------------------------------


@dataclass(frozen=True)
class ToolOutput:
    name: str
    args: dict[str, Any]
    """Normalized arguments (defaults filled in); the raw ones if they were invalid."""
    text: str
    """What the model sees: redacted and capped."""
    ok: bool
    latency_s: float
    cached: bool = False


def cap_output(text: str, cap_tokens: int) -> str:
    """At most ~`cap_tokens` tokens of `text`, cut at a line end where possible."""
    max_chars = cap_tokens * CHARS_PER_TOKEN
    if len(text) <= max_chars:
        return text
    cut = text[: max_chars - len(TRUNCATED) - 1]
    newline = cut.rfind("\n")
    if newline > len(cut) // 2:
        cut = cut[:newline]
    return f"{cut}\n{TRUNCATED}"


def _short(message: str) -> str:
    message = " ".join(message.split())
    return message if len(message) <= ERROR_MAX_CHARS else message[:ERROR_MAX_CHARS] + "..."


def describe_error(exc: Exception) -> str:
    """A short, model-readable reason for a failed tool call."""
    if isinstance(exc, ToolError):
        return _short(str(exc))
    if isinstance(exc, RetryableHTTPError):
        code = exc.response.status_code
        if code in (403, 429):
            return "GitHub rate limit reached; use another tool or submit_diagnosis"
        return f"GitHub server error (HTTP {code}); try again later"
    if isinstance(exc, httpx.HTTPStatusError):
        code = exc.response.status_code
        if code == 404:
            return "not found (HTTP 404) at this commit"
        return f"GitHub returned HTTP {code}"
    if isinstance(exc, httpx.TransportError):
        return "network error talking to GitHub"
    return _short(f"{type(exc).__name__}: {exc}")


def _validation_message(exc: ValidationError) -> str:
    return "; ".join(
        f"{'.'.join(str(p) for p in err['loc']) or 'arguments'}: {err['msg']}"
        for err in exc.errors(include_input=False)
    )


class UnknownToolError(ValueError):
    """A tool name that does not exist (for --tools)."""


def select_tools(names: Sequence[str] | None) -> list[ToolSpec]:
    """The named tools in registry order; all of them for None."""
    if names is None:
        return list(TOOLS.values())
    unknown = sorted(set(names) - set(TOOLS))
    if unknown:
        raise UnknownToolError(
            f"unknown tools: {', '.join(unknown)} (available: {', '.join(TOOLS)})"
        )
    return [spec for name, spec in TOOLS.items() if name in names]


class ToolBox:
    """The tools enabled for one job. `call` never raises.

    With a `cache`, a successful result is stored and every later identical call (same
    job context, tool and normalized arguments) is answered from disk.
    """

    def __init__(
        self,
        env: ToolEnv,
        names: Sequence[str] | None = None,
        cap_tokens: int = OUTPUT_CAP_TOKENS,
        cache: ToolCache | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.env = env
        self.specs = select_tools(names)
        self.cap_tokens = cap_tokens
        self.cache = cache
        self._clock = clock
        self._by_name = {spec.name: spec for spec in self.specs}
        self._log_digest: str | None = None

    @property
    def names(self) -> list[str]:
        return [spec.name for spec in self.specs]

    def _ref(self, scope: Scope) -> str:
        """What a result in `scope` is pinned to (see agent.cache)."""
        ctx = self.env.ctx
        if scope == "commit":
            return f"commit:{ctx.head_sha}"
        if scope == "run":
            return f"run:{ctx.run_id}"
        if self._log_digest is None:
            joined = "\n".join(ctx.log_lines).encode("utf-8")
            self._log_digest = hashlib.sha256(joined).hexdigest()[:16]
        return f"job:{ctx.job_id}:{self._log_digest}"

    def call(self, name: str, raw_args: dict[str, Any] | None) -> ToolOutput:
        start = self._clock()
        raw = dict(raw_args or {})

        def done(text: str, ok: bool, args: dict[str, Any], cached: bool = False) -> ToolOutput:
            text = cap_output(text, self.cap_tokens)
            return ToolOutput(name, args, text, ok, self._clock() - start, cached)

        spec = self._by_name.get(name)
        if spec is None:
            return done(f"{ERROR_PREFIX}unknown tool {name!r}; available: {self.names}", False, raw)
        try:
            args = spec.args_model.model_validate(raw)
        except ValidationError as exc:
            return done(f"{ERROR_PREFIX}invalid arguments: {_validation_message(exc)}", False, raw)
        normalized = args.model_dump()
        repo, ref = self.env.ctx.repo or "", self._ref(spec.scope)
        key = tool_cache_key(repo, ref, name, normalized)
        if self.cache is not None:
            hit = self.cache.get(key)
            if hit is not None:
                return done(hit, True, normalized, cached=True)
        try:
            text = redact(spec.run(self.env, args)).text
        except Exception as exc:  # a tool must never break the agent loop
            return done(redact(ERROR_PREFIX + describe_error(exc)).text, False, normalized)
        if self.cache is not None:
            meta = {"repo": repo, "ref": ref, "tool": name, "args": normalized}
            with contextlib.suppress(OSError):  # an unwritable cache only costs a refetch
                self.cache.put(key, text, meta)
        return done(text, True, normalized)

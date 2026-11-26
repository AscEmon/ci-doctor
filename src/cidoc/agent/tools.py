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
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol
from urllib.parse import quote

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from cidoc.agent.context import RunContext, valid_repo, valid_sha
from cidoc.analyze.preprocess import CHARS_PER_TOKEN, MAX_LINE_CHARS
from cidoc.github_client import RetryableHTTPError
from cidoc.logview import split_lines
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
    """The tools enabled for one job. `call` never raises."""

    def __init__(
        self,
        env: ToolEnv,
        names: Sequence[str] | None = None,
        cap_tokens: int = OUTPUT_CAP_TOKENS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.env = env
        self.specs = select_tools(names)
        self.cap_tokens = cap_tokens
        self._clock = clock
        self._by_name = {spec.name: spec for spec in self.specs}

    @property
    def names(self) -> list[str]:
        return [spec.name for spec in self.specs]

    def call(self, name: str, raw_args: dict[str, Any] | None) -> ToolOutput:
        start = self._clock()
        raw = dict(raw_args or {})

        def done(text: str, ok: bool, args: dict[str, Any], cached: bool = False) -> ToolOutput:
            return ToolOutput(name, args, self._finish(text), ok, self._clock() - start, cached)

        spec = self._by_name.get(name)
        if spec is None:
            return done(f"{ERROR_PREFIX}unknown tool {name!r}; available: {self.names}", False, raw)
        try:
            args = spec.args_model.model_validate(raw)
        except ValidationError as exc:
            return done(f"{ERROR_PREFIX}invalid arguments: {_validation_message(exc)}", False, raw)
        normalized = args.model_dump()
        try:
            text = spec.run(self.env, args)
        except Exception as exc:  # a tool must never break the agent loop
            return done(ERROR_PREFIX + describe_error(exc), False, normalized)
        return done(text, True, normalized)

    def _finish(self, text: str) -> str:
        return cap_output(redact(text).text, self.cap_tokens)

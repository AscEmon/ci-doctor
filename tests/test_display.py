import gzip
from pathlib import Path

import pytest
from rich.console import Console

from cidoc.labeling.display import (
    build_view,
    clean_line,
    clean_log,
    is_marker,
    marker_windows,
    read_log,
    render_view,
)

TS = "2026-10-01T12:34:56.1234567Z "


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (f"{TS}Run flutter test", "Run flutter test"),
        ("2026-10-01T12:34:56Z no fraction", "no fraction"),
        (f"\ufeff{TS}first line", "first line"),
        (f"{TS}\x1b[31mFAILURE:\x1b[0m Build failed", "FAILURE: Build failed"),
        ("\x1b[1;33mwarn\x1b[K\x1b[?25l", "warn"),
        ("\x1b]8;;https://example.com\x07link\x1b]8;;\x07", "link"),
        ("mid-line 2026-10-01T12:34:56Z stays", "mid-line 2026-10-01T12:34:56Z stays"),
        ("", ""),
    ],
)
def test_clean_line(raw: str, expected: str) -> None:
    assert clean_line(raw) == expected


def test_clean_log_preserves_line_count() -> None:
    text = f"{TS}a\n{TS}\n\n{TS}\x1b[0mb\r\n"

    assert clean_log(text) == ["a", "", "", "b"]


def test_clean_log_only_splits_on_newline() -> None:
    text = "page\x0cbreak\nsep\u2028inside\nDownloading 10%\rDownloading 100%\nlast"

    assert clean_log(text) == ["page\x0cbreak", "sep\u2028inside", "Downloading 100%", "last"]


@pytest.mark.parametrize(
    "line",
    [
        "##[error]Process completed with exit code 1.",
        "FAILURE: Build failed with an exception.",
        "BUILD FAILED in 1m 2s",
        "lib/main.dart:3:1: Error: Expected ';' after this.",
        "e: file.kt:10:5 error: unresolved reference",
        "Unhandled Exception: StateError",
        "00:03 +12 -1: Some tests FAILED",
        "> Could not resolve all files for configuration ':app:debugRuntimeClasspath'.",
    ],
)
def test_markers_detected(line: str) -> None:
    assert is_marker(line)


@pytest.mark.parametrize("line", ["Resolving dependencies...", "0 errors found", "failed: 0"])
def test_non_markers(line: str) -> None:
    assert not is_marker(line)


def test_marker_windows_merge_overlaps_and_clip() -> None:
    lines = ["ok"] * 30
    for i in (1, 5, 20):
        lines[i] = "error: x"

    assert marker_windows(lines, end=30, context=3) == [(0, 9), (17, 24)]
    assert marker_windows(lines, end=21, context=3) == [(0, 9), (17, 21)]


def make_log(n: int, markers: dict[int, str]) -> str:
    """n lines numbered 1..n; `markers` maps 1-based line numbers to replacement text."""
    return "".join(f"{TS}{markers.get(i, f'line {i}')}\n" for i in range(1, n + 1))


def test_view_shows_tail_and_earlier_snippets_with_real_line_numbers() -> None:
    log = make_log(300, {50: "FAILURE: early", 290: "BUILD FAILED"})

    view = build_view(log, tail=120, context=3)

    assert view.total_lines == 300
    assert view.tail[0] == (181, "line 181")
    assert view.tail[-1] == (300, "line 300")
    assert view.snippets == [
        [(n, f"line {n}") if n != 50 else (50, "FAILURE: early") for n in range(47, 54)]
    ]
    assert view.marker_lines == {50, 290}
    assert view.omitted_snippets == 0


def test_markers_inside_tail_are_not_duplicated_as_snippets() -> None:
    view = build_view(make_log(150, {140: "error: late"}), tail=120)

    assert view.snippets == []
    assert 140 in view.marker_lines


def test_snippet_window_stops_at_tail_boundary() -> None:
    view = build_view(make_log(200, {80: "Error: edge"}), tail=120, context=3)

    assert [n for n, _ in view.snippets[0]] == [77, 78, 79, 80]


def test_short_log_is_all_tail() -> None:
    view = build_view(make_log(5, {2: "Exception"}), tail=120)

    assert [n for n, _ in view.tail] == [1, 2, 3, 4, 5]
    assert view.snippets == []


def test_snippets_capped_keeping_earliest() -> None:
    markers = {i: "error: x" for i in range(10, 400, 10)}

    view = build_view(make_log(600, markers), tail=120, context=1, max_snippets=5)

    assert len(view.snippets) == 5
    assert view.snippets[0][1] == (10, "error: x")
    assert view.omitted_snippets == len(markers) - 5


def test_read_log_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "1.log.gz"
    path.write_bytes(gzip.compress("héllo\r\n10%\r100%\nwörld\n".encode()))

    assert read_log(path) == "héllo\r\n10%\r100%\nwörld\n"


def test_render_treats_log_text_literally() -> None:
    log = make_log(200, {10: "error: [red]not markup[/red]", 199: "##[error]boom"})
    console = Console(record=True, width=120, color_system=None)

    render_view(console, build_view(log, tail=20))

    out = console.export_text()
    assert " 10 │ error: [red]not markup[/red]" in out
    assert "199 │ ##[error]boom" in out
    assert "last 20 of 200 lines" in out

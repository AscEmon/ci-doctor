import pytest

from cidoc.redact import redact

# Fake credentials are assembled at runtime so no secret-shaped literal lives in the repo.
ALNUM = "Ab1Cd2Ef3Gh4Ij5Kl6Mn7Op8Qr9St0UvWxYz"
FAKE_GHP = "ghp" + "_" + ALNUM[:36]
FAKE_GHS = "ghs" + "_" + ALNUM[:36]
FAKE_PAT = "github" + "_pat_" + ALNUM + "_" + ALNUM
FAKE_SK = "sk" + "-ant-" + ALNUM
FAKE_AKIA = "AKIA" + "ABCDEFGH23456789"
FAKE_B64 = ("QmFzZTY0U2VjcmV0VmFsdWU" * 3)[:56]
TS = "2026-10-01T12:00:00.1234567Z "


@pytest.mark.parametrize(
    ("secret", "kind"),
    [
        (FAKE_GHP, "github_token"),
        (FAKE_GHS, "github_token"),
        (FAKE_PAT, "github_pat"),
        (FAKE_SK, "sk_key"),
        (FAKE_AKIA, "aws_access_key"),
    ],
)
def test_token_formats_masked(secret: str, kind: str) -> None:
    result = redact(f"{TS}curl -H 'x: {secret}' https://example.com\n")

    assert secret not in result.text
    assert f"[REDACTED:{kind}]" in result.text
    assert result.counts == {kind: 1}
    assert result.total == 1


@pytest.mark.parametrize(
    "line",
    [
        f"API_KEY={FAKE_B64}",
        f'"client_secret": "{FAKE_B64}"',
        f"GOOGLE_SERVICES_TOKEN: {FAKE_B64}",
        f"keystore password {FAKE_B64}",
    ],
)
def test_long_value_after_keyword_masked_but_key_kept(line: str) -> None:
    result = redact(line)

    assert FAKE_B64 not in result.text
    assert result.text.endswith("[REDACTED:keyword_secret]" + ('"' if line.endswith('"') else ""))
    assert result.counts == {"keyword_secret": 1}


def test_private_key_block_masked_across_timestamped_lines() -> None:
    log = (
        f"{TS}before\n"
        f"{TS}-----BEGIN RSA PRIVATE KEY-----\n"
        f"{TS}{FAKE_B64}\n"
        f"{TS}{FAKE_B64}\n"
        f"{TS}-----END RSA PRIVATE KEY-----\n"
        f"{TS}after\n"
    )

    result = redact(log)

    assert FAKE_B64 not in result.text
    assert "before" in result.text and "after" in result.text
    assert result.counts == {"private_key": 1}


def test_truncated_private_key_masks_body_lines() -> None:
    log = f"-----BEGIN PRIVATE KEY-----\n{TS}{FAKE_B64}\n{TS}{FAKE_B64}\n{TS}next step\n"

    result = redact(log)

    assert FAKE_B64 not in result.text
    assert result.text.endswith(f"\n{TS}next step\n")


def test_counts_multiple_hits() -> None:
    result = redact(f"{FAKE_GHP} {FAKE_GHP}\nTOKEN={FAKE_B64}\n")

    assert result.counts == {"github_token": 2, "keyword_secret": 1}
    assert result.total == 3


def test_token_inside_keyword_assignment_counted_once() -> None:
    result = redact(f"GITHUB_TOKEN={FAKE_GHP}")

    assert result.text == "GITHUB_TOKEN=[REDACTED:github_token]"
    assert result.total == 1


@pytest.mark.parametrize(
    "line",
    [
        "BUILD FAILED in 2m 13s",
        "FAILURE: Build failed with an exception.",
        "Run flutter test --coverage",
        "task-runner-utils resolved",
        "Cache restored from key: flutter-linux-stable-3.24.3-x64",
        "HEAD is now at 1f2e3d4c5b6a79881f2e3d4c5b6a79881f2e3d4c",
        "token: ***",
        "Error: Could not resolve com.android.tools.build:gradle:8.5.0.",
    ],
)
def test_ordinary_log_lines_untouched(line: str) -> None:
    result = redact(line)

    assert result.text == line
    assert result.total == 0

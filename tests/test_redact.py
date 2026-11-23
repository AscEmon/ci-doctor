import pytest

from cidoc.redact import MIN_SECRET_ENTROPY, redact, shannon_entropy

# Fake credentials are assembled at runtime so no secret-shaped literal lives in the repo.
ALNUM = "Ab1Cd2Ef3Gh4Ij5Kl6Mn7Op8Qr9St0UvWxYz"
FAKE_GHP = "ghp" + "_" + ALNUM[:36]
FAKE_GHS = "ghs" + "_" + ALNUM[:36]
FAKE_PAT = "github" + "_pat_" + ALNUM + "_" + ALNUM
FAKE_SK = "sk" + "-ant-" + ALNUM
FAKE_AKIA = "AKIA" + "ABCDEFGH23456789"
FAKE_B64 = ("QmFzZTY0U2VjcmV0VmFsdWU" * 3)[:56]
TS = "2026-10-01T12:00:00.1234567Z "
FAKE_HIGH_ENTROPY = ALNUM[:32]
FAKE_SUPABASE = "sb_publishable_" + ALNUM[:24]
FAKE_JWT = (
    "eyJ"
    + "hbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
    + ".eyJ"
    + "zdWIiOiIxMjM0NTY3ODkwIiwicm9sZSI6ImFub24ifQ"
    + "."
    + "SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
)
# Cache keys as GitHub Actions prints them: long, hash-suffixed, and high-entropy, so
# only the cache-line context (not length or entropy) keeps them readable.
CACHE_KEY = "flutter-linux-stable-3.24.3-x64-8f14e45fceea167a5a36dedd4bea2543"
GRADLE_KEY = "Linux-gradle-5f3c2a1b9e8d7c6b5a4f3e2d1c0b9a8f"
PUB_KEY = "pub-cache-macos-arm64-c9f8e7d6b5a4c3b2a1f0e9d8c7b6a5f4"
PODS_KEY = "macOS-pods-0a1b2c3d4e5f60718293a4b5c6d7e8f9"


@pytest.mark.parametrize(
    ("secret", "kind"),
    [
        (FAKE_GHP, "github_token"),
        (FAKE_GHS, "github_token"),
        (FAKE_PAT, "github_pat"),
        (FAKE_SK, "sk_key"),
        (FAKE_AKIA, "aws_access_key"),
        (FAKE_JWT, "jwt"),
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


@pytest.mark.parametrize(
    "line",
    [
        f"{TS}Cache restored from key: {CACHE_KEY}",
        f"{TS}Cache not found for input keys: {GRADLE_KEY}, Linux-gradle-",
        f'{TS}Restored from cache key "{PUB_KEY}"',
        f"{TS}Saved cache entry with key {PODS_KEY}",
        f"{TS}CACHE RESTORED FROM KEY: {CACHE_KEY}",
    ],
)
def test_cache_key_lines_untouched(line: str) -> None:
    result = redact(line)

    assert result.text == line
    assert result.total == 0


@pytest.mark.parametrize("value", [CACHE_KEY, GRADLE_KEY, PUB_KEY, PODS_KEY])
def test_same_values_outside_cache_lines_are_masked(value: str) -> None:
    # Proves the cache context is what spares them: they pass length and entropy.
    assert shannon_entropy(value) >= MIN_SECRET_ENTROPY

    result = redact(f"API_KEY={value}")

    assert result.text == "API_KEY=[REDACTED:keyword_secret]"


def test_cache_context_does_not_leak_to_next_line() -> None:
    log = f"Cache restored from key: {CACHE_KEY}\nSECRET_KEY={FAKE_HIGH_ENTROPY}\n"

    result = redact(log)

    assert CACHE_KEY in result.text
    assert FAKE_HIGH_ENTROPY not in result.text
    assert result.counts == {"keyword_secret": 1}


def test_vendor_tokens_still_masked_in_cache_lines() -> None:
    result = redact(f"Cache restored from key: deps-{FAKE_GHP}")

    assert FAKE_GHP not in result.text
    assert result.counts == {"github_token": 1}


@pytest.mark.parametrize(
    "line",
    [
        "storeFile=release.jks",
        "keystore: release.jks",
        "KEYSTORE_FILE=upload-keystore-release-2026.jks",
        "key_file: firebase-adminsdk-prod-x7k2m9.json",
        "keyProperties=android/key.properties",
        "SIGNING_KEY_PATH=/home/runner/work/app/app/android/app/upload.keystore",
        "secretFile=ios/Runner/GoogleService-Info.plist",
        "token_url=https://oauth2.googleapis.com/token/abcdefghij",
        "keyAlias=distribution-cert-2026.p12",
    ],
)
def test_file_names_and_paths_untouched(line: str) -> None:
    result = redact(line)

    assert result.text == line
    assert result.total == 0


@pytest.mark.parametrize(
    "line",
    [
        "SECRET_NAME=" + "a" * 30,  # long but no entropy
        "token: abcabcabcabcabcabcabcabc",
        "API_KEY=Ab1Cd2Ef3Gh4Ij5Kl6M",  # 19 chars: below the length floor
    ],
)
def test_low_entropy_or_short_values_untouched(line: str) -> None:
    assert redact(line).total == 0


@pytest.mark.parametrize(
    ("line", "kept_prefix"),
    [
        (f"{TS}SymmetricKey: {FAKE_HIGH_ENTROPY}", f"{TS}SymmetricKey: "),
        (f"SUPABASE_PUBLISHABLE_KEY: {FAKE_SUPABASE}", "SUPABASE_PUBLISHABLE_KEY: "),
        (f'"api_key": "{FAKE_HIGH_ENTROPY}"', '"api_key": "'),
        (f"DB_PASSWORD={FAKE_HIGH_ENTROPY}", "DB_PASSWORD="),
    ],
)
def test_high_entropy_keyword_values_masked(line: str, kept_prefix: str) -> None:
    result = redact(line)

    assert FAKE_HIGH_ENTROPY not in result.text
    assert FAKE_SUPABASE not in result.text
    assert result.text.startswith(kept_prefix + "[REDACTED:keyword_secret]")
    assert result.counts == {"keyword_secret": 1}


def test_short_word_after_keyword_does_not_hide_the_value() -> None:
    result = redact(f"keystore password {FAKE_HIGH_ENTROPY}")

    assert result.text == "keystore password [REDACTED:keyword_secret]"


def test_jwt_after_keyword_counted_once_as_jwt() -> None:
    result = redact(f"SUPABASE_ANON_KEY={FAKE_JWT}")

    assert result.text == "SUPABASE_ANON_KEY=[REDACTED:jwt]"
    assert result.counts == {"jwt": 1}


def test_shannon_entropy() -> None:
    assert shannon_entropy("") == 0.0
    assert shannon_entropy("aaaa") == 0.0
    assert shannon_entropy("abcd") == 2.0

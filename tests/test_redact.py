import pytest

from cidoc.redact import (
    MIN_SECRET_ENTROPY,
    MIN_SECRET_LENGTH,
    looks_like_path,
    redact,
    shannon_entropy,
)

# Fake credentials are assembled at runtime so no secret-shaped literal lives in the repo.
ALNUM = "Ab1Cd2Ef3Gh4Ij5Kl6Mn7Op8Qr9St0UvWxYz"
FAKE_GHP = "ghp" + "_" + ALNUM[:36]
FAKE_GHS = "ghs" + "_" + ALNUM[:36]
FAKE_PAT = "github" + "_pat_" + ALNUM + "_" + ALNUM
FAKE_SK = "sk" + "-ant-" + ALNUM
FAKE_AKIA = "AKIA" + "ABCDEFGH23456789"
FAKE_B64 = ("QmFzZTY0U2VjcmV0VmFsdWU" * 3)[:56]
FAKE_JWT = (
    "eyJ"
    + "hbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
    + ".eyJ"
    + "zdWIiOiIxMjM0NTY3ODkwIiwicm9sZSI6ImFub24ifQ"
    + "."
    + "SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
)
# Random-looking values of the lengths the must-redact cases call for.
RANDOM_32 = "Xq7" + "Lm2Rp9Tv4Wz8Kd3Hn6Bc1Gs5Jf0Ya"
RANDOM_40 = "Pn4" + "Vs8Ld1Gk6Ry3Zw9Bt2Mh7Qc5Fx0Je4Ua8Ko3Wi6"
# Base64-shaped, with '/' and '+' mid-value like AWS secret access keys.
RANDOM_B64_SLASH = "wJal" + "rXUtnFEMI/K7MDENG/bPxRfiCYzEXAMPLEKEY+9q"
TS = "2026-10-01T12:00:00.1234567Z "


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
    result = redact(f"{FAKE_GHP} {FAKE_GHP}\nAPI_TOKEN={RANDOM_32}\n")

    assert result.counts == {"github_token": 2, "assignment_secret": 1}
    assert result.total == 3


@pytest.mark.parametrize(
    ("line", "masked"),
    [
        (f"GITHUB_TOKEN={FAKE_GHP}", "GITHUB_TOKEN=[REDACTED:github_token]"),
        (f"SUPABASE_ANON_TOKEN={FAKE_JWT}", "SUPABASE_ANON_TOKEN=[REDACTED:jwt]"),
    ],
)
def test_vendor_token_in_assignment_counted_once(line: str, masked: str) -> None:
    result = redact(line)

    assert result.text == masked
    assert result.total == 1


# --- assignment_secret: must redact ----------------------------------------------


@pytest.mark.parametrize(
    ("line", "masked"),
    [
        (f"API_TOKEN={RANDOM_32}", "API_TOKEN=[REDACTED:assignment_secret]"),
        (f"client_secret: {RANDOM_40}", "client_secret: [REDACTED:assignment_secret]"),
        (f'"client_secret": "{RANDOM_40}"', '"client_secret": "[REDACTED:assignment_secret]"'),
        (
            f"{TS}export DB_PASSWORD='{RANDOM_32}'",
            f"{TS}export DB_PASSWORD='[REDACTED:assignment_secret]'",
        ),
        (f"apiKey: {RANDOM_32}", "apiKey: [REDACTED:assignment_secret]"),
        (f"FIREBASE_PRIVATE_KEY={RANDOM_40}", "FIREBASE_PRIVATE_KEY=[REDACTED:assignment_secret]"),
        (
            f"aws_secret_access_key = {RANDOM_B64_SLASH}",
            "aws_secret_access_key = [REDACTED:assignment_secret]",
        ),
        (
            f"--client-secret={RANDOM_32} --verbose",
            "--client-secret=[REDACTED:assignment_secret] --verbose",
        ),
        (f"MATCH_PASSWD:{RANDOM_32}", "MATCH_PASSWD:[REDACTED:assignment_secret]"),
    ],
)
def test_assignment_secret_masked(line: str, masked: str) -> None:
    result = redact(line)

    assert result.text == masked
    assert result.counts == {"assignment_secret": 1}


def test_base64_with_slashes_is_not_treated_as_a_path() -> None:
    assert "/" in RANDOM_B64_SLASH
    assert not looks_like_path(RANDOM_B64_SLASH)


# --- assignment_secret: must NOT redact (audit regressions) ---------------------

AUDIT_FALSE_POSITIVES = [
    # Secret-ish words inside longer identifiers or paths (old rule matched mid-word).
    f"{TS}Downloading flutter_keyboard_visibility_platform_interface-2.0.0.tar.gz",
    f"{TS}+ flutter_keyboard_visibility 6.0.0 (was 5.4.1)",
    f"{TS}gnome-keyring-daemon: GNOME_KEYRING_CONTROL=/run/user/1001/keyring",
    f"{TS}00:03 +12: test/nskey_self_signed_certificate_validation_test.dart: loads",
    # Cache inputs (hash-suffixed, high entropy).
    f"{TS}  key: Linux-pub-8f14e45fceea167a5a36dedd4bea2543a1b2c3d4",
    f"{TS}Cache not found for keys: Linux-gradle-5f3c2a1b9e8d7c6b5a4f3e2d1c0b9a8f7e6d",
    f"{TS}Cache restored from key: flutter-linux-stable-3.24.3-x64-8f14e45fceea167a5a",
    # Build settings and flags that are not secrets.
    f"{TS}export INFOPLIST_KEY_NSCameraUsageDescription=Scan_QR_codes_to_pay_merchants",
    f"{TS}export INFOPLIST_KEY_API_TOKEN_USAGE={RANDOM_32}",
    f"{TS}flutter build ipa -dShorebirdPublicKey={RANDOM_40}",
    # Public / publishable / bare keys are not secrets by design.
    f"SUPABASE_PUBLISHABLE_KEY: sb_publishable_{RANDOM_32}",
    f"PUBLIC_KEY={RANDOM_40}",
    f"KEY={RANDOM_40}",
    f"SymmetricKey: {RANDOM_32}",
]


@pytest.mark.parametrize("line", AUDIT_FALSE_POSITIVES)
def test_audit_false_positives_untouched(line: str) -> None:
    result = redact(line)

    assert result.text == line
    assert result.total == 0


@pytest.mark.parametrize(
    "line",
    [
        "API_TOKEN=***",
        "client_secret: ***",
        f"API_TOKEN={RANDOM_32[: MIN_SECRET_LENGTH - 1]}",  # one char too short
        "API_TOKEN=" + "ab" * 20,  # long, low entropy
        "API_TOKEN=release-candidate-build-number",  # slug: below 4.0 bits/char
        "API_TOKEN_FILE=/home/runner/work/app/secrets/api_token.txt",
        "CLIENT_SECRET_PATH=./android/app/client_secret_1234567890.json",
        "PASSWORD_URL=https://vault.example.com/v1/secret/data/payments-app",
        "GOOGLE_SERVICES_TOKEN_JSON=firebase-adminsdk-prod-x7k2m9q4.json",
        "GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}",
        f"API_TOKEN {RANDOM_32}",  # no '=' or ':': not an assignment
    ],
)
def test_non_secret_assignments_untouched(line: str) -> None:
    result = redact(line)

    assert result.text == line
    assert result.total == 0


def test_value_is_masked_whole_never_mid_token() -> None:
    result = redact(f"API_TOKEN={RANDOM_32}.extra-suffix next")

    assert result.text == "API_TOKEN=[REDACTED:assignment_secret] next"


def test_rejected_short_value_does_not_hide_a_later_assignment() -> None:
    result = redact(f"API_TOKEN=short CLIENT_SECRET={RANDOM_40}")

    assert result.text == "API_TOKEN=short CLIENT_SECRET=[REDACTED:assignment_secret]"


@pytest.mark.parametrize(
    "line",
    [
        "BUILD FAILED in 2m 13s",
        "FAILURE: Build failed with an exception.",
        "Run flutter test --coverage",
        "task-runner-utils resolved",
        "HEAD is now at 1f2e3d4c5b6a79881f2e3d4c5b6a79881f2e3d4c",
        "token: ***",
        "Error: Could not resolve com.android.tools.build:gradle:8.5.0.",
    ],
)
def test_ordinary_log_lines_untouched(line: str) -> None:
    result = redact(line)

    assert result.text == line
    assert result.total == 0


def test_thresholds() -> None:
    assert MIN_SECRET_LENGTH == 24
    assert MIN_SECRET_ENTROPY == 4.0
    for value in (RANDOM_32, RANDOM_40, RANDOM_B64_SLASH):
        assert shannon_entropy(value) >= MIN_SECRET_ENTROPY


def test_shannon_entropy() -> None:
    assert shannon_entropy("") == 0.0
    assert shannon_entropy("aaaa") == 0.0
    assert shannon_entropy("abcd") == 2.0

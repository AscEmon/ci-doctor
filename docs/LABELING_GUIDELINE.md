# Labeling guideline

How to label a failed CI job with `cidoc label`. The goal is one category per job that
names the **root cause**, the first thing that went wrong, not the last thing that
printed an error.

Labels are written to `data/labels.jsonl`. Fix mistakes with
`cidoc label edit <job_id>` or `cidoc label edit --last`.

## How to read a log

1. **Find `##[error]`.** `cidoc label` opens on the failed step (the focus view) with
   `##[error]` lines highlighted. Press `f` for the full view (error snippets plus the
   last 120 lines).
2. **Read upward from the first `##[error]`.** The `##[error]` line itself is usually
   just `Process completed with exit code 1.`. The cause is in the lines above it:
   Gradle's `* What went wrong:` block, the first `error:` from the compiler, the first
   failing test, the pub solver message.
3. **Ignore everything after `Post job cleanup.`** Cleanup steps (cache saves, git
   config resets) run after the failure and their errors are never the cause. The
   focus view already hides them.
4. **Check whether earlier steps succeeded.** If checkout, setup or `pub get` failed,
   that is the root cause even when a later step printed more errors. Composite steps
   that finished with `outcome=success` are hidden in the focus view; a step you can
   see either failed or has no recorded outcome.
5. **Use the hint as a hint.** A `hint:` line above the prompt is a regex match, not a
   verdict. It never fills in the category. Decide from the log, then type the key.
6. **Cite evidence.** Evidence line numbers are the numbers in the left gutter: the
   lines that show the root cause, usually 1 to 3 of them.

## Categories

Key in brackets. Each has one definition and one typical error line.

### `dependency_resolution` [1]

A pub package or a Gradle/Maven artifact could not be resolved or downloaded.

```
Because app depends on intl ^0.19.0 which doesn't match any versions, version solving failed.
```

### `gradle_agp_config` [2]

An error in Gradle or Android Gradle Plugin configuration or DSL (build scripts,
plugin setup, namespace), where the required artifacts were found.

```
Namespace not specified. Specify a namespace in the module's build file.
```

### `native_compile` [3]

Kotlin, Java, C or C++ code failed to compile (app or plugin native code).

```
e: file:///home/runner/work/app/android/app/src/main/kotlin/MainActivity.kt:12:5 Unresolved reference: foo
```

### `dart_compile` [4]

Dart code failed to compile during a build or test run.

```
lib/main.dart:14:7: Error: The method 'fetchUser' isn't defined for the class 'ApiClient'.
```

### `analyzer_lint` [5]

`flutter analyze`, `dart analyze` or `dart format` reported issues (info, warning or
error) and the step failed because of them.

```
   info • Unused import: 'package:flutter/foundation.dart' • lib/main.dart:3:8 • unused_import
```

### `test_failure` [6]

A unit, widget, golden or integration test ran and failed.

```
00:42 +118 -1: Some tests failed.
```

### `signing_secrets` [7]

A signing certificate, keystore, provisioning profile or CI secret is missing, invalid
or not readable.

```
> Keystore file '/home/runner/work/app/android/app/upload.jks' not found for signing config 'release'.
```

### `toolchain_version` [8]

The Flutter, Dart, Java, Xcode, Gradle or AGP version in use is not the one the
project or a dependency requires.

```
Android Gradle plugin requires Java 17 to run. You are currently using Java 11.
```

### `ios_build` [9]

An iOS or macOS build failed in CocoaPods or Xcode for a reason other than signing or
a toolchain version.

```
[!] CocoaPods could not find compatible versions for pod "Firebase/CoreOnly"
```

### `infra_flaky` [0]

The failure came from the CI environment rather than the code: network errors, 5xx
responses, runner crashes, cache download failures, or anything a plain re-run fixed.

```
Error: read ECONNRESET
```

### `timeout_resource` [t]

The job or a step hit a time limit, or ran out of memory or disk.

```
##[error]The job running on runner GitHub Actions 12 has exceeded the maximum execution time of 60 minutes.
```

### `other` [o]

The root cause is visible in the log but fits none of the categories above. Describe it
precisely in the root cause line.

```
Error: Resource not accessible by integration
```

## Tie-break rules

When more than one category seems to fit, apply the first rule that matches.

- **Missing or invalid signing cert, keystore, provisioning profile or secret →
  `signing_secrets`**, even on iOS or Android jobs (not `ios_build`, not
  `gradle_agp_config`).
- **A pub or Gradle/Maven artifact cannot be resolved or downloaded →
  `dependency_resolution`.**
- **Gradle/AGP config or DSL error where the artifacts were found →
  `gradle_agp_config`.**
- **Flutter, Java, Xcode or AGP version mismatch → `toolchain_version`.**
- **`flutter analyze` / `dart format` failures (info, warning or error) →
  `analyzer_lint`.**
- **`pub get` succeeded with "newer versions available" → not an error; ignore it**
  and keep reading for the real failure.
- **Network, 5xx, runner crash, cache download failure, or fixed by a plain re-run →
  `infra_flaky`.**
- **The root cause is not visible in the log → skip** (press `s`). This is typical of
  reporter and aggregator steps ("Notify", "Send error", "Lint Code Base") that only
  report that another job failed.
- **Still ambiguous → check the `fix_sha` commit** (below). Use confidence 2 when
  unsure, and 3 only when the log alone is conclusive.

## Confidence

| Value | Meaning |
| ----- | ------- |
| 3 | The log alone is conclusive: the evidence lines show the root cause. |
| 2 | Most likely right, but you are unsure or relied on the fix commit. |
| 1 | A guess. Prefer skipping (`s`) if the log does not show the cause. |

## Checking the fix commit

`fix_sha` is the head commit of the next successful run of the same workflow on the
same branch. The diff between the failing commit and it often shows what was fixed.
It is not in the label header yet; look it up in the database:

```sh
sqlite3 data/cidoc.db "SELECT repo, head_sha, fix_sha FROM failures WHERE job_id = <job_id>"
```

Then open `https://github.com/<repo>/compare/<head_sha>...<fix_sha>`.

- `fix_sha` equal to `head_sha`: a re-run passed with no code change, which points to
  `infra_flaky`.
- A diff that only bumps versions in `pubspec.yaml`, `build.gradle` or `.tool-versions`
  points to `dependency_resolution` or `toolchain_version`.
- No `fix_sha`: nothing passed later on that branch yet; decide from the log alone.

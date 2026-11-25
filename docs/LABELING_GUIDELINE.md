# Labeling guideline

How to label a failed CI job with `cidoc label`. The goal is one category per job that
names the **root cause**, the first thing that went wrong, not the last thing that
printed an error.

Labels are written to `data/labels.jsonl`. Fix mistakes with
`cidoc label edit <job_id>` or `cidoc label edit --last`.

## Scope

**In scope:** jobs that build, test, analyze, package, sign or publish the app.

**Out of scope: skip them** (press `s`):

- PR hygiene bots: PR title validation, labelers, stale/first-interaction bots,
  release drafters, PR size labelers.
- Issue bots: anything triggered by issue activity or adding items to project boards.
- Aggregator jobs (`ci-success`, `all-green`, `gate`, `result`, `check`, ...) whose
  only failing step reads the results of other jobs and exits. The root cause is in
  the job that actually failed, not here.

Mining and `cidoc mine prune` already drop most of these automatically. Skip any that
slip through.

## How to read a log

1. **Find `##[error]`.** `cidoc label` opens on the failed step (the focus view) with
   error markers highlighted (`##[error]`, `FAILURE:`, `error:`, `❌`, `[!]`,
   `** BUILD FAILED **`, `fastlane finished with errors`, ...). Press `f` for the full
   view (error snippets plus the last 120 lines).
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
5. **Label the first distinct error.** Repeated identical errors (the same compile
   error in every test file, the same missing symbol in every target) are a cascade
   from one cause. Label that cause, and cite its first occurrence as evidence.
6. **Use the hint as a hint.** A `hint:` line above the prompt is a regex match, not a
   verdict. It never fills in the category. Decide from the log, then type the key.
7. **Cite evidence.** Evidence line numbers are the numbers in the left gutter: the
   lines that show the root cause, usually 1 to 3 of them.

### Never a root cause

These look like problems but never fail a job. Keep reading past them:

- `Target Integrity (Xcode)` and `Warning (Xcode)` lines.
- `pub get` reporting "N packages have newer versions incompatible with dependency
  constraints" (or any "newer versions available" note).
- Deprecation warnings.
- fastlane's "fastlane X is available" notice and its release notes ("To see all new
  releases", "Please update using"). The focus view hides it and shows the 40 lines
  above it, where fastlane's actual error is.

## Categories

Key in brackets. Each has one definition and one typical error line.

### `dependency_resolution` [1]

A pub package or Gradle/Maven artifact cannot be resolved: the requested version
truly does not exist, or version constraints conflict. Also a dependency that resolved
but has a mismatched API (a missing type or symbol) where the fix is changing its
version.

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

`flutter analyze`, `dart analyze` or `dart format` reported issues and the step failed
because of them.

```
   info • Unused import: 'package:flutter/foundation.dart' • lib/main.dart:3:8 • unused_import
```

### `test_failure` [6]

A unit, widget, golden or integration test ran and failed.

```
00:42 +118 -1: Some tests failed.
```

### `signing_secrets` [7]

A signing certificate, keystore, provisioning profile or CI secret is missing,
invalid, expired or not readable.

```
> Keystore file '/home/runner/work/app/android/app/upload.jks' not found for signing config 'release'.
```

### `toolchain_version` [8]

A tool or runtime is not the version the project, a dependency or the workflow
requires. Covers any tool: Flutter, Dart, Java, Xcode, AGP, Gradle, Python, Node,
Ruby, Go, Rust, CMake, MSVC.

```
Android Gradle plugin requires Java 17 to run. You are currently using Java 11.
```

### `ios_build` [9]

An iOS or macOS build failed in CocoaPods or Xcode (including linking) for a reason
other than signing or a toolchain version.

```
[!] CocoaPods could not find compatible versions for pod "Firebase/CoreOnly"
```

### `infra_flaky` [0]

The failure came from the CI environment rather than the code: network errors, HTTP
429/5xx, timeouts talking to a server, runner crashes, cache download failures, git
remote errors, or anything a plain re-run fixed.

```
Error: read ECONNRESET
```

### `timeout_resource` [t]

The job or a step hit a time limit, or ran out of memory or disk.

```
##[error]The job running on runner GitHub Actions 12 has exceeded the maximum execution time of 60 minutes.
```

### `ci_config` [c]

The workflow file, CI scripts, or runner setup is missing or wrong: required system
packages not installed, secrets/tokens not passed to a script, invalid CI data/config
files, project CI scripts out of sync, or the workflow calling a missing file.

```
./scripts/ci/build.sh: line 4: ./scripts/ci/prepare.sh: No such file or directory
```

### `other` [o]

The root cause is clear in the log but fits none of the categories above. Describe it
precisely in the root cause line.

```
ERROR ITMS-90189: "Redundant Binary Upload. You've already uploaded a build with build number '42'."
```

## Specific cases

These cases come up often enough to settle once. Each one overrides the category
definitions and the tie-break rules below. `cidoc analyze` and `cidoc eval` copy this
list into the model's instructions, so keep each case to one bullet.

- A compile error caused by a feature that a newer compiler or SDK removed or
  deprecated, with no code change (e.g. MSVC `STL1011` for
  `<experimental/coroutine>`) → `toolchain_version`, not `native_compile`.
- An iOS/macOS deployment target or minimum OS version below what the installed Xcode
  supports → `toolchain_version`.
- Corrupted or missing files inside a local Gradle, pub or CocoaPods cache →
  `infra_flaky`.
- HTTP 404 means the resource does not exist → never `infra_flaky`.
- A project-defined gate (binary size budget, coverage threshold) failed → `other`.
- A runtime "symbol not found" or missing type caused by a dependency version
  mismatch, where nothing failed to compile → `dependency_resolution`.

## Tie-break rules

When more than one category seems to fit, apply the first rule that matches.

1. **Out of scope** (PR/issue bots, aggregator jobs) → **skip**.
2. **Signing:** a missing, invalid or expired signing cert, keystore, provisioning
   profile or secret → `signing_secrets`, even on iOS or Android jobs (not `ios_build`,
   not `gradle_agp_config`).
   - The secret exists but the workflow does not pass it to the step (missing `env:`
     or `with:` wiring) → `ci_config`.
3. **Artifact or package download failed transiently:** HTTP 429 or 5xx, a timeout,
   `Could not GET '...'`, `Could not parse POM` → `infra_flaky`. A 404 is not
   transient (see [Specific cases](#specific-cases)).
4. **Version truly missing or constraints conflict** → `dependency_resolution`. So is
   a resolved dependency with a mismatched API (missing type or symbol) that is fixed
   by changing its version.
5. **`version solving failed` because of the SDK version:**
   - the CI's Flutter/Dart is simply too old → `toolchain_version`;
   - a deliberate minimum-version job (e.g. `pub downgrade`) and the fix was in
     `pubspec.yaml` → `dependency_resolution`.
6. **Compile error caused by a runner compiler or SDK upgrade** (the code did not
   change) → `toolchain_version`.
7. **Any tool/runtime version mismatch** → `toolchain_version`, including a workflow's
   own version guard (a step that checks a version and exits).
8. **Gradle/AGP config or DSL error where the artifacts were found** →
   `gradle_agp_config`.
9. **Xcode `Undefined symbol` / linker error from a missing system library** →
   `ios_build`.
10. **`flutter analyze` / `dart analyze` / `dart format` failures** → `analyzer_lint`.
    Check the flags (`--fatal-infos`, `--fatal-warnings`, `--no-fatal-*`) to see which
    levels fail the step, and cite as evidence only issues at a fatal level.
11. **Workflow, CI script or runner setup problem** → `ci_config` (see its
    definition).
12. **Network, 5xx, runner crash, cache download failure, `remote: error` /
    `remote: fatal` on git push or fetch, or fixed by a plain re-run** →
    `infra_flaky`.
13. **Cause clear but no category fits** → `other`. **Cause not visible in the log**
    (reporter steps, a log cut short) → **skip**.
14. **Still ambiguous** → check the `fix_sha` commit (below). Use confidence 2 when
    unsure, and 3 only when the log alone is conclusive.

### `other` candidates to watch

When labeling `other`, note in the root cause if it matches one of these. If they
keep coming up they may become categories:

- **`store_release`:** a store or registry upload rejected because the version or
  build already exists (App Store `ERROR ITMS-90189`, Play Console "Version code
  already used", pub.dev "version already exists").
- **`quality_gate`:** a size, coverage or similar threshold check failed (labeled
  `other`, see [Specific cases](#specific-cases)).

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
  points to `dependency_resolution` or `toolchain_version` (rules 4 to 7 decide which).
- A diff that only touches `.github/workflows/` or CI scripts points to `ci_config`.
- No `fix_sha`: nothing passed later on that branch yet; decide from the log alone.

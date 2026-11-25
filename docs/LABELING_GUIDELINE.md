# CI Doctor Labeling Guideline (v3)

This guideline defines how to label one failed GitHub Actions job. Human labelers and the
analyzer prompt both use it, so every rule here must be decidable from the log (plus the
fix commit when stated).

**Changes in v3**
- New category `store_release`.
- Clear boundary between `signing_secrets` and `ci_config`.
- `other` narrowed to "clear cause, truly no category".
- Categories are now chosen with an **ordered decision procedure**: the first matching
  rule wins. This removes most overlaps.

---

## 1. Before labeling: scope and visibility

**Skip (`s`) when the job is out of scope:**
- PR or issue hygiene bots: semantic PR title, labeler, stale, welcome, add-to-project.
- Aggregator jobs that only read `needs.*.result` and exit.
- Jobs that run no Flutter, Dart, Android, iOS, or macOS/Windows/Linux app tooling are in
  scope for labeling but are marked by the `stack` field and reported separately.

**Skip (`s`) when the cause is not visible:** the log does not show why the step failed,
and the fix commit does not make it clear. Do not guess. Skip is not a failure; a wrong
label is.

---

## 2. How to read a log

1. Find `##[error]`. It is usually just "Process completed with exit code 1". The cause is
   above it.
2. Ignore everything after `Post job cleanup.`
3. Check that earlier steps succeeded. A warning in a successful step is not the cause.
4. Repeated identical errors are a cascade. Label the **first distinct** error.
5. These are **never** root causes on their own:
   - "N packages have newer versions", per-package `x (y available)` lines, "discontinued"
   - "Welcome to Flutter" analytics banner
   - `Target Integrity (Xcode)` and `Warning (Xcode)` lines
   - Deprecation warnings (Node, Gradle, Kotlin), unless a lint step fails on them
   - fastlane's update changelog ("To see all new releases", "Please update using")
   - Log lines marked ERROR that say "Skipping", "optional", or "continuing"
6. `***` is GitHub's secret mask. `[REDACTED:...]` is ours. Neither is a cause by itself.
7. For retries: the same failure on every attempt means a real problem; different
   failures across attempts mean `infra_flaky`.

---

## 3. Decision procedure (first match wins)

Ask these questions **in order**. Stop at the first "yes".

| # | Question | Category |
|---|---|---|
| 1 | Did a **store or registry reject an upload or publish**? | `store_release` |
| 2 | Is a **credential's value** missing, empty, invalid, expired, or denied? | `signing_secrets` |
| 3 | Did something **outside the repo fail temporarily**? | `infra_flaky` |
| 4 | Did the job or step hit a **time, memory, or disk limit**? | `timeout_resource` |
| 5 | Is a **tool or runtime the wrong version**, or did a **tool upgrade** break it with no code change? | `toolchain_version` |
| 6 | Is a **dependency** missing, conflicting, out of sync, or API-mismatched? | `dependency_resolution` |
| 7 | Is the **workflow file, a CI script, or runner setup** itself wrong? | `ci_config` |
| 8 | Is the **Gradle / Android Gradle Plugin configuration** wrong? | `gradle_agp_config` |
| 9 | Is the **Xcode / CocoaPods project** setup or linking wrong? | `ios_build` |
| 10 | Did **Kotlin, Java, C, C++, Swift, or Objective-C code** fail to compile? | `native_compile` |
| 11 | Did **Dart code** fail to compile? | `dart_compile` |
| 12 | Did **analyze, lint, or format** fail? | `analyzer_lint` |
| 13 | Did a **test run and fail**? | `test_failure` |
| 14 | The cause is clear but none of the above fit. | `other` |

Why the order matters: a test that fails because the emulator crashed is `infra_flaky`
(rule 3), not `test_failure` (rule 13). A compile error caused by a new compiler is
`toolchain_version` (rule 5), not `native_compile` (rule 10).

---

## 4. Category definitions and boundaries

### `store_release`
A store or package registry **rejected** the upload or publish of an already built
artifact.
- Google Play: "Version code N has already been used", invalid bundle, policy rejection.
- App Store Connect / TestFlight: `ERROR ITMS-90xxx`, duplicate build number, missing
  export compliance.
- pub.dev: "Version X of package Y already exists".
- **Not this:** the store rejected the **login or API key** → `signing_secrets`.

### `signing_secrets`
The **value** of a credential is the problem: a signing certificate, keystore,
provisioning profile, store API key, or CI secret is missing, empty, invalid, expired,
or has no access.
- "Keystore file not found", "No signing certificate", "could not read Username" when
  cloning a certificates repo, "Invalid API Key", 401/403 from a store or signing service.
- An env var that should hold a secret is empty (`KEY_PASSWORD:` with no `***`).

**Boundary with `ci_config` (the most common confusion):**
- Ask: *"Would fixing this require changing a secret's value or access in the repo/account
  settings?"* → `signing_secrets`.
- Ask: *"Would fixing this require editing the workflow file or a script?"* → `ci_config`.
- If the log cannot tell you which, choose `signing_secrets` with confidence 2.

### `infra_flaky`
Something **outside the repository** failed temporarily. A re-run would most likely pass.
- HTTP 429, 5xx, "Too Many Requests", "Service Unavailable"
- Timeouts or connection errors talking to a server: `ETIMEDOUT`, `ECONNRESET`, DNS failure
- Registry, CDN, or GitHub-side errors: `remote: error`, `remote: fatal`, "Could not parse
  POM", CocoaPods CDN clone failure
- Corrupted or missing files inside a local Gradle, pub, or CocoaPods **cache**
- Runner crash, emulator failed to boot, "The runner has received a shutdown signal"
- **Not this:** HTTP **404** (the resource does not exist) → `dependency_resolution` or
  `ci_config`. Authentication errors → `signing_secrets`.

### `timeout_resource`
The job or step hit a **platform limit**: maximum execution time, out of memory, no space
left on device.
- **Not this:** one test timing out inside a passing test runner → `test_failure`, unless
  it is waiting on a network service → `infra_flaky`.

### `toolchain_version`
A tool or runtime is **not the version** the project, a dependency, or the workflow
requires, **or** a tool upgrade broke the build while the project code did not change.
Covers Flutter, Dart, Java, Kotlin plugin, Gradle, AGP, Xcode, CocoaPods, Python, Node,
Ruby, Go, Rust, CMake, MSVC.
- "requires Java 17", "Gradle version lower than Flutter's minimum supported"
- iOS deployment target below what the installed Xcode supports
- A compiler error from a feature the new compiler removed (e.g. MSVC `STL1011`
  `<experimental/coroutine>`)
- `version solving failed` because the CI's Flutter/Dart SDK is too old (normal build job)
- The workflow's own version-check script fails
- Fails only on Flutter master/beta while stable passes (when known)

### `dependency_resolution`
A pub, Gradle/Maven, or CocoaPods **dependency** is the problem.
- The requested version does not exist; version constraints conflict
- `pubspec.lock` out of sync (`--enforce-lockfile`)
- A third-party artifact download returns **404**
- A dependency resolved, but its API does not match (missing type, method, or native
  symbol), and the fix is changing its version
- A deliberate minimum-version job (`pub downgrade`) fails because `pubspec.yaml`
  constraints are wrong

### `ci_config`
The **workflow file, a CI script, or the runner setup** is wrong. The app code and the
standard tools are fine.
- The workflow calls a file or script that does not exist; missing `actions/checkout`
- A required system package is not installed (e.g. GTK dev packages for a native hook)
- A token exists but the workflow or script does not pass it on
- An invalid CI data or config file (e.g. a YAML file read by a CI script)
- Project CI scripts out of sync (overlay digest mismatch)

### `gradle_agp_config`
An error in Gradle or Android Gradle Plugin **configuration**, where the artifacts were
found: DSL errors, plugin apply errors, namespace not set, inconsistent JVM targets
between Java and Kotlin tasks.

### `ios_build`
An Xcode or CocoaPods **project** problem that is not signing and not a toolchain version:
CocoaPods version conflicts, missing pods, linker errors from a missing system library or
framework (e.g. `libresolv`), Xcode project settings.

### `native_compile`
Kotlin, Java, C, C++, Swift, or Objective-C **source code** failed to compile, and the fix
is a code change.

### `dart_compile`
Dart code failed to compile during a build or test run, and the fix is a Dart code change.

### `analyzer_lint`
`flutter analyze`, `dart analyze`, or `dart format` failed the step.
- Check the flags: with `--no-fatal-infos`, only warnings and errors are fatal.
- Deprecation lints that appear after an SDK upgrade are still `analyzer_lint`.
- Evidence = only issues at a fatal level.

### `test_failure`
A unit, widget, golden, integration, or end-to-end test **ran and failed** because of the
project's code or test code.
- Same assertion failing on every retry.
- **Not this:** the test environment failed (emulator, network, backend down) →
  `infra_flaky`. The test file did not compile → `dart_compile` or
  `dependency_resolution`.

### `other`
The cause is **clear**, but no category above fits. Examples: a project-defined quality
gate (binary size budget, coverage threshold, generated-code drift check).
- If the cause is not clear, use **skip**, not `other`.
- Always describe the cause precisely in the root cause text.

---

## 5. Root cause text

- One sentence, at most 200 characters.
- Pattern: **`<what failed>: <why it failed>`**
  - Good: `Gradle could not resolve kotlin-stdlib 2.0.21: Maven Central returned HTTP 429`
  - Bad: `package update issue generate`, `cocoapods issue`
- Name the file, package, tool, or version involved.
- Never paste instructions or notes into this field.

## 6. Evidence lines

- 1 to 4 line numbers that show the root cause directly.
- Prefer the first distinct error line over its repetitions.
- Do not include noise lines (see section 2.5).

## 7. Confidence

| Value | Use when |
|---|---|
| 3 | The log alone proves both the cause and the category. |
| 2 | The cause is clear, but the category depends on something not in the log (for example, whether a re-run passed). Check the fix commit when possible. |
| 1 | You know what failed but are guessing the category. Keep this under 10% of labels. |

## 8. Splits and integrity

- `dev`: used to tune prompts and this guideline. Labels may be re-checked.
- `test1`: the first test batch. It has been seen; it is reported as-is and never tuned on.
- `test2`: the next test batch, labeled **after** this guideline and the prompt are frozen.
  Evaluated once.
- When re-checking any label, decide from the log and this guideline only, never from the
  model's prediction. Record a reason in the root cause text.

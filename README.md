# CI Doctor for Mobile

**An LLM agent that diagnoses failed CI builds in Flutter, Android and iOS projects, evaluated on a
human-labeled dataset of real GitHub Actions failures.**

> Status: research prototype. Phase 0 (data), Phase 1 (LLM analyzer) and Phase 2 (tool-using agent)
> are complete. A paper draft and dataset release are in progress.

---

## Why this project

- CI builds fail often, and finding the real cause in a long log takes developer time.
- Mobile CI adds its own failure modes: Gradle and AGP, Xcode and CocoaPods, code signing, app
  store uploads, and many interacting tool versions.
- Existing AI tools for CI failures are general-purpose and are not evaluated on mobile projects.
- There is no public, human-labeled dataset of mobile CI failures to measure such tools against.

CI Doctor addresses this with a mobile-focused diagnosis agent, a labeled dataset, and an
evaluation protocol with frozen prompts and a held-out test set.

## Results so far

Category accuracy on the **held-out test set (test2, n = 50)**, with the prompt and labeling
guideline frozen before labeling:

| Method                                    | Accuracy  | Evidence-line hit rate | Input tokens / item |
| ----------------------------------------- | --------- | ---------------------- | ------------------- |
| Regex baseline (no LLM)                   | 38.0%     | 34.0%                  | 0                   |
| **LLM, log only** (Gemini 3.5 Flash-Lite) | **76.0%** | **88.0%**              | ~5K                 |
| LLM agent, same prompt, no tools          | 74.0%     | 86.0%                  | ~7K                 |
| LLM agent with read-only tools            | 70.0%     | 86.0%                  | ~16K                |

Main findings (preliminary, n = 50, margin of error about ±12 points):

1. The LLM doubles the accuracy of a rule-based baseline on unseen failures, and is far better at
   environment-related causes (registry rate limits, tool-version changes).
2. The model locates the right log lines 88% of the time; most errors are about choosing the
   category, not finding the error.
3. Prompt tuning and label adjudication on the dev set inflated accuracy by about 12 points
   (dev 87.7% vs. test 76.0%), which is why we report the held-out number.
4. For a small model, read-only tools gave no accuracy gain while using about 3× the tokens.
   Prompt and tool effects were separated with a no-tools ablation.

## Method in brief

1. **Mining:** discover popular open-source Flutter projects, download logs of failed GitHub
   Actions jobs (last 90 days), and pair each failure with the next successful run (fix commit).
2. **Cleaning:** remove framework repos, bots and fork PRs; redact secrets (with an audited,
   low-false-positive redaction rule).
3. **Labeling:** a human labels each failure with a category, a one-line root cause, evidence
   lines and a confidence, following the [labeling guideline](docs/LABELING_GUIDELINE.md)
   (ordered decision procedure, 14 categories including mobile-specific `store_release`).
4. **Splits:** `dev` for prompt development, `test1` (older guideline, preliminary), `test2`
   (clean, main result).
5. **Analyzer:** deterministic log preprocessing, structured JSON output, prompt-injection-aware
   system instruction, response cache, cost guard.
6. **Agent:** read-only tools bound to the failing commit (repo files, workflow, full-log search,
   sibling jobs, previous run, commit changes). The later fix commit is never shown to the agent.

## Repository layout

| Path                         | Contents                                                                 |
| ---------------------------- | ------------------------------------------------------------------------ |
| `src/cidoc/miner/`           | Repo discovery, failed-run mining, fix pairing, filters, redaction audit |
| `src/cidoc/labeling/`        | Labeling tool, rule-based hints, label storage                           |
| `src/cidoc/analyze/`         | Log preprocessing, prompt, output schema, cache, cost guard              |
| `src/cidoc/llm/`             | Provider interface, Gemini client, rate limiting, retries                |
| `src/cidoc/eval/`            | Regex baseline, metrics, eval runner, run comparison                     |
| `src/cidoc/agent/`           | Agent tools, loop, trajectory recording, tool cache                      |
| `docs/LABELING_GUIDELINE.md` | Labeling rulebook and category definitions                               |
| `tests/`                     | ~590 tests; none use the network                                         |

## Quick start

```sh
brew install uv            # or see https://docs.astral.sh/uv/
uv sync
cp .env.example .env       # set GITHUB_TOKEN (read-only) and GEMINI_API_KEY
uv run cidoc --help
```

Check the code without data or API keys:

```sh
uv run pytest
uv run ruff check .
uv run mypy
```

Reproduce the pipeline (requires tokens; `data/` is local and gitignored):

```sh
uv run cidoc mine discover && uv run cidoc mine runs   # collect failed jobs
uv run cidoc label                                     # label failures
uv run cidoc eval --model regex-baseline --split test2
uv run cidoc eval --model gemini-3.5-flash-lite --split test2
uv run cidoc eval --model gemini-3.5-flash-lite --agent --split test2
uv run cidoc analyze <job_id> --agent                  # one diagnosis with its tool calls
```

Every eval run is saved with its prompt version and guideline version, so each reported number
can be traced to the exact configuration.

## Limitations

- Test set is small (n = 50); one annotator so far (inter-annotator agreement is planned).
- One small model evaluated; stronger models are next.
- Data covers popular open-source projects and a 90-day window.
- The dataset is not yet public; a release of labels and job IDs is planned.

## Roadmap

- [x] Phase 0: dataset and labeling tool
- [x] Phase 1: LLM analyzer, baseline, held-out evaluation
- [x] Phase 2: tool-using agent and ablation
- [ ] Grow the test set, second annotator, stronger-model comparison, paper preprint
- [ ] Phase 6: security study (prompt injection through CI logs)
- [ ] Phases 3–5, 7: RAG, fix generation in a sandbox, full benchmark, GitHub App

## Author

Abu Sayed Chowdhury · [Portfolio](https://ascemon.github.io/Portfolio) ·
ORCID [0009-0004-8061-4447](https://orcid.org/0009-0004-8061-4447)

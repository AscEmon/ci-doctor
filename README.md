# ci-doctor

## Setup

1. Install [uv](https://docs.astral.sh/uv/):

   ```sh
   brew install uv
   ```

2. Install Python 3.12 and the project dependencies:

   ```sh
   uv sync
   ```

3. Create your local env file and fill in real values:

   ```sh
   cp .env.example .env
   ```

   | Variable            | Value                                                    |
   | ------------------- | -------------------------------------------------------- |
   | `GITHUB_TOKEN`      | GitHub token with read access to Actions on target repos |
   | `ANTHROPIC_API_KEY` | Anthropic API key (only needed for analyze/eval)        |

   `.env` and `data/` are gitignored. Never commit them.

4. Verify the install:

   ```sh
   uv run cidoc --help
   ```

## Labeling

Label failed jobs by hand with `uv run cidoc label`, and fix a label with
`uv run cidoc label edit <job_id>` (or `--last`). Follow the
[labeling guideline](docs/LABELING_GUIDELINE.md) for how to read a log, what each
category means, and the ordered decision procedure for choosing one.

Each label has a `split` (see the guideline's "Splits and integrity"): `dev` is
what the prompt is tuned on, and labels written before splits existed are `dev`.
`test1` is the first held-out batch, and new labels default to `test2`
(`cidoc label --split dev|test1|test2` overrides this). `uv run cidoc eval` scores
`dev` by default; use `--split test1`, `--split test2` or `--split all` for the
others.

Re-check dev labels after a guideline change with
`uv run cidoc label review --category other,ci_config,signing_secrets`. It uses the
`label edit` flow and never shows model predictions. It refuses `test1`/`test2`.

## Agent (Phase 2)

`uv run cidoc analyze <job_id> --agent` lets the model fetch context beyond the log
before it answers, through read-only tools bound to the failing commit: `read_file`,
`list_dir`, `read_workflow`, `search_log`, `read_log`, `sibling_jobs`, `previous_run`
and `commit_changes`. It prints each tool call (arguments, a short preview of the
output) and the final diagnosis, including the repository files it relied on.

- `--max-steps N` caps tool calls (default 6); after that the model must answer.
- `--tools a,b,...` offers only those tools (for ablations); default all. `--tools none`
  runs the same agent instruction and loop with only `submit_diagnosis`: the checklist
  reasoning stays, the tool references go, and it gets its own prompt version.
- `--no-call` prints the system instruction, the excerpt and the tool list, no API call.
- `CIDOC_AGENT_MAX_INPUT` caps the input tokens of all model calls of one item (default
  80,000). Tool results the model has already answered are resent compacted (first 300
  characters and their size); the latest results always go in full. The trajectory shows
  each call's input tokens.

`uv run cidoc eval --agent [--max-steps N] [--tools ...]` scores the agent on the same
labels; the run file records `agent`, the tool set and `max_steps`, and adds tool calls
per item, the share of items that used a tool, tool usage counts, model calls and tokens
per item, and items that hit `max_steps`. `--dry-run` and `--max-cost` count every
uncached item at its worst case (`max_steps` tool rounds plus the final answer), and
`CIDOC_RPM` applies to every model call, agent steps included.
`cidoc eval --show <run> --job <id>` prints an agent item's full trajectory, and
`cidoc eval --compare <phase1.json> <agent.json>` compares a Phase 1 run with an agent
run on the same split, including model calls, tokens and tool calls per item.

Tools use your `GITHUB_TOKEN` (read-only) and nothing else on the network. Their results
are cached in `data/cache/agent_tools/` and never expire: a commit, a finished run and a
stored log do not change. The commit that later fixed the failure is never shown to the
agent. Without `--agent`, `cidoc analyze` is the Phase 1 analyzer, unchanged.

## Development checks

```sh
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest
```

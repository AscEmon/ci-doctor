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

Each label has a `split`. Labels written before splits existed are `dev`, which is
the set the prompt is tuned on. New labels default to `test`, which is held out
(`cidoc label --split dev` overrides this). `uv run cidoc eval` scores `dev` by
default; use `--split test` or `--split all` for the others.

## Development checks

```sh
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest
```

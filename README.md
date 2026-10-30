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
   | `ANTHROPIC_API_KEY` | Anthropic API key                                        |

   `.env` and `data/` are gitignored. Never commit them.

4. Verify the install:

   ```sh
   uv run cidoc --help
   ```

## Development checks

```sh
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest
```

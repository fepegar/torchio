default:
    @just --list

clean:
    rm -rf .mypy_cache
    rm -rf .pytest_cache
    rm -rf .tox
    rm -rf .venv
    rm -rf dist
    rm -rf **/__pycache__
    rm -rf src/*.egg-info
    rm -f .coverage
    rm -f coverage.*

@install_uv:
	if ! command -v uv >/dev/null 2>&1; then \
		echo "uv is not installed. Installing..."; \
		curl -LsSf https://astral.sh/uv/install.sh | sh; \
	fi

setup: install_uv
    uv sync --all-extras --all-groups
    uv run pre-commit install

bump part="patch":
    uv run bump-my-version bump {{part}} --verbose

bump-dry part='patch':
    uv run bump-my-version bump {{part}} --dry-run --verbose --allow-dirty

push:
    git push && git push --tags

types:
    uv run --group quality -- tox -e types

lint:
    uv run --group quality -- ruff check

format:
    uv run --group quality -- ruff format --diff

test:
    uv run --group test -- tox -e pytest

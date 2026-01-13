# Contributing to Claude Vault

Thank you for your interest in contributing to Claude Vault! We welcome contributions from the community to help make this tool better.

## Development Setup

1.  **Fork and Clone**
    ```bash
    git clone https://github.com/MarioPadilla/claude-vault.git
    cd claude-vault
    ```

2.  **Create Virtual Environment**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Dependencies**
    Install the package in editable mode with development dependencies:
    ```bash
    pip install -e ".[dev]"
    ```
    This installs tools like `pytest`, `ruff`, `mypy`, and `pre-commit`.

4.  **Setup Pre-commit Hooks**
    ```bash
    pre-commit install
    ```
    This ensures code quality checks run automatically before every commit.

## Running Tests

We use `pytest` for testing. Run the test suite with:

```bash
pytest
```

To run with coverage report:
```bash
pytest --cov=claude_vault
```

## Code Style

We follow strict code style guidelines enforced by `ruff` and `mypy`.

- **Formatting & Linting**: Run `ruff check .` and `ruff format .`
- **Type Checking**: Run `mypy .`

## Pull Request Process

1.  Create a new branch for your feature or fix: `git checkout -b feature/amazing-feature`.
2.  Write tests for your changes.
3.  Ensure all tests and linting checks pass.
4.  Update documentation if necessary (e.g., `README.md`).
5.  Submit a Pull Request with a clear description of your changes.

## Reporting Issues

If you find a bug or have a feature request, please open an issue on GitHub. Provide as much detail as possible, including steps to reproduce the issue.

Thank you for contributing!

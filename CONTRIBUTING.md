# Contributing to Plesk Unified

Thank you for your interest in contributing to Plesk Unified! We welcome
contributions from the community. This document provides guidelines and
instructions for contributing.

## Code of conduct

Be respectful and constructive in all interactions. We maintain a welcoming and
harassment-free environment for everyone.

## Getting started

1. **Fork the repository** on GitHub.
2. **Clone your fork** locally:

   ```bash
   git clone https://github.com/barateza/mcp-plesk-unified.git
   cd mcp-plesk-unified
   ```

3. **Create a virtual environment**:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

4. **Install development dependencies**:

   ```bash
   pip install -e .
   ```

## Make changes

1. **Create a feature branch**:

   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** according to the style guide below.

3. **Test your changes**:

   ```bash
   python server.py  # Verify the server starts correctly
   ```

4. **Commit with clear messages**:

   ```bash
   git commit -m "feat: add support for X" -m "Detailed explanation of changes"
   ```

   Use conventional commits:
   - `feat:` for new features
   - `fix:` for bug fixes
   - `docs:` for documentation changes
   - `refactor:` for code refactoring
   - `test:` for test additions
   - `chore:` for maintenance tasks

## Style guide

### Python code

- Follow [PEP 8](https://pep8.org/) style guidelines.
- Use meaningful variable and function names.
- Add docstrings to functions and classes:

  ```python
  def search_knowledge_base(query: str, max_results: int = 10) -> list[dict]:
      """
      Search the Plesk knowledge base for relevant documentation.
      
      Args:
          query: The search query string
          max_results: Maximum number of results to return
          
      Returns:
          List of documentation entries matching the query
      """
  ```

- Keep lines under 100 characters when reasonable.
- Use type hints for function arguments and return values.

### Comments and documentation

- Write clear, concise comments detailing the "why" instead of the "what".
- Update `README.md` with new features.
- Document configuration changes.
- Add examples for new functionality.

## Submit changes

1. **Push to your fork**:

   ```bash
   git push origin feature/your-feature-name
   ```

2. **Create a Pull Request** on GitHub with:
   - A clear title describing the changes.
   - A description of what changed and why.
   - A reference to any related issues (e.g., "Fixes #123").
   - Screenshots or examples if applicable.

3. **Respond to review comments** promptly and professionally.

## Pull request checklist

Before submitting a PR, ensure:

- [ ] Code follows the style guide.
- [ ] Changes include proper documentation.
- [ ] You add no unnecessary dependencies.
- [ ] You regenerate the vector database if content parsing changed.
- [ ] You resolve merge conflicts with the main branch.
- [ ] Commit messages are clear and conventional.

## Report issues

### Bug reports

Include:

- A clear, descriptive title.
- Steps to reproduce.
- Expected behavior.
- Actual behavior.
- Python version and OS.
- Full error messages and traceback.
- Relevant code snippets.

### Feature requests

Include:

- A clear description of the feature.
- Use cases and benefits.
- A possible implementation approach.
- Examples of similar features in other projects.

## Questions?

- Check existing [GitHub Issues](https://github.com/barateza/mcp-plesk-unified/issues).
- Open a new Discussion or Issue.
- Be patient—maintainers are volunteers.

## Recognition

We recognize contributors in:

- Git commit history.
- Release notes for significant contributions.
- README contributors section (if implemented).

Thank you for helping make Plesk Unified better! 🙏

## Pre-commit hooks (Formatting & Linting)

We use `pre-commit` to run formatters and linters automatically on each commit.

Install and enable hooks locally:

```bash
python -m pip install --upgrade pre-commit
pre-commit install
pre-commit run --all-files  # optional: verify everything passes
```

The hooks include `ruff` (lint + auto-fix), `isort` (import sorting), and
`black` (code formatting). If a hook modifies files, re-run `git add` and
commit again.


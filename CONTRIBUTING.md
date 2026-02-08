# Contributing to Plesk Unified

Thank you for your interest in contributing to Plesk Unified! We welcome contributions from the community. This document provides guidelines and instructions for contributing.

## Code of Conduct

Be respectful and constructive in all interactions. We're committed to providing a welcoming and harassment-free environment for everyone.

## Getting Started

1. **Fork the repository** on GitHub
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

## Making Changes

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following the style guide below

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

## Style Guide

### Python Code

- Follow [PEP 8](https://pep8.org/) style guidelines
- Use meaningful variable and function names
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
- Keep lines under 100 characters when reasonable
- Use type hints for function arguments and return values

### Comments and Documentation

- Write clear, concise comments explaining the "why" not the "what"
- Keep README.md updated with new features
- Document any configuration changes
- Add examples for new functionality

## Submitting Changes

1. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

2. **Create a Pull Request** on GitHub with:
   - Clear title describing the changes
   - Description of what changed and why
   - Reference to any related issues (e.g., "Fixes #123")
   - Screenshots or examples if applicable

3. **Respond to review comments** promptly and professionally

## Pull Request Checklist

Before submitting a PR, ensure:

- [ ] Code follows the style guide
- [ ] Changes are well-documented
- [ ] No unnecessary dependencies added
- [ ] Vector database regenerated if content parsing changed
- [ ] No merge conflicts with main branch
- [ ] Commit messages are clear and conventional

## Reporting Issues

### Bug Reports

Include:
- Clear, descriptive title
- Steps to reproduce
- Expected behavior
- Actual behavior
- Python version and OS
- Full error messages/traceback
- Relevant code snippets

### Feature Requests

Include:
- Clear description of the feature
- Use cases and benefits
- Possible implementation approach (if you have ideas)
- Examples of similar features in other projects

## Questions?

- Check existing [GitHub Issues](https://github.com/barateza/mcp-plesk-unified/issues)
- Open a new Discussion or Issue
- Be patient - maintainers are volunteers

## Recognition

Contributors will be recognized in:
- Git commit history
- Release notes for significant contributions
- README contributors section (if implemented)

Thank you for helping make Plesk Unified better! 🙏

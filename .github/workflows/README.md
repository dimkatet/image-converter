# GitHub Actions CI/CD

This directory contains GitHub Actions workflows for continuous integration and deployment.

## Workflows

### CI Pipeline (`ci.yml`)

Runs on every push to `main` and on all pull requests.

**Steps:**
1. **Checkout code** - Retrieves repository code
2. **Setup Python** - Installs Python 3.12 with pip caching
3. **Install dependencies** - Installs package with dev dependencies
4. **Pyright type checking** - Static type analysis (must pass with 0 errors)
5. **Run tests** - Executes pytest test suite (all tests must pass)
6. **Generate summary** - Creates test results summary in GitHub UI

**Matrix:**
- Python versions: 3.12 (project requires >=3.12)

**Exit conditions:**
- ❌ Fails if Pyright reports any type errors
- ❌ Fails if any test fails
- ✅ Passes only if both Pyright and tests succeed

## Running CI Checks Locally

Before pushing, run the same checks locally:

```bash
# Activate virtual environment
source venv/bin/activate

# Type checking (must pass with 0 errors)
pyright

# Run all tests
pytest tests/ -v --tb=short

# Both should complete without errors
```

## Adding New Checks

To add additional CI steps (linting, formatting, coverage, etc.), edit [ci.yml](ci.yml).

Example additions:

```yaml
# Add before tests
- name: Check code formatting
  run: |
    black --check src/ tests/

# Add after tests
- name: Upload coverage
  uses: codecov/codecov-action@v3
  with:
    files: ./coverage.xml
```

## Badges

Add CI status badge to README.md:

```markdown
![CI](https://github.com/USERNAME/REPO/workflows/CI/badge.svg)
```

Replace `USERNAME` and `REPO` with your GitHub username and repository name.

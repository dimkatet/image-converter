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
5. **Run tests with coverage** - Executes pytest test suite with coverage (≥62% required)
6. **Upload coverage report** - Uploads HTML coverage report as artifact (30 day retention)
7. **Generate summary** - Creates test results summary in GitHub UI

**Matrix:**
- Python versions: 3.12 (project requires >=3.12)

**Exit conditions:**
- ❌ Fails if Pyright reports any type errors
- ❌ Fails if any test fails
- ❌ Fails if test coverage < 62%
- ✅ Passes only if all checks succeed

## Running CI Checks Locally

Before pushing, run the same checks locally:

```bash
# Activate virtual environment
source venv/bin/activate

# Type checking (must pass with 0 errors)
pyright

# Run all tests with coverage (must be ≥62%)
pytest tests/ -v --tb=short --cov=src/image_pipeline --cov-report=term --cov-report=html --cov-fail-under=62

# View detailed HTML coverage report
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
```

## Viewing Coverage Reports in CI

After each CI run, coverage reports are available as artifacts:

1. Go to the **Actions** tab in GitHub
2. Click on a workflow run
3. Scroll to **Artifacts** section at the bottom
4. Download `coverage-report.zip`
5. Extract and open `htmlcov/index.html` in a browser

Coverage reports are retained for 30 days.

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

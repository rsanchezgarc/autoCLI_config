# GitHub Actions Workflows

This directory contains automated workflows for the config_docs_system project.

## Workflows

### 1. Test Workflow (`test.yml`)

**Trigger:** Manual (workflow_dispatch)

**Purpose:** Run the test suite across multiple Python versions using conda environments.

**How to use:**
1. Go to the "Actions" tab in GitHub
2. Select "Run Tests" from the workflow list
3. Click "Run workflow" button
4. Select the branch and click "Run workflow"

**What it does:**
- Tests against Python 3.7, 3.8, 3.9, 3.10, and 3.11
- Sets up conda environment for each Python version
- Installs the package with dev dependencies
- Runs pytest with coverage reporting
- Uploads coverage results to Codecov (optional)

### 2. PyPI Publishing Workflow (`publish-to-pypi.yml`)

**Trigger:** Automatic on new release creation

**Purpose:** Automatically build and publish the package to PyPI when a new release is created.

**Setup required:**
1. Create a PyPI API token:
   - Go to https://pypi.org/manage/account/token/
   - Create a new API token with scope limited to this project
2. Add the token to GitHub secrets:
   - Go to repository Settings → Secrets and variables → Actions
   - Click "New repository secret"
   - Name: `PYPI_API_TOKEN`
   - Value: Your PyPI API token

**How to use:**
1. Create a new release on GitHub:
   - Go to "Releases" → "Create a new release"
   - Create a new tag (e.g., `v0.1.0`)
   - Fill in release title and description
   - Click "Publish release"
2. The workflow will automatically:
   - Build the package
   - Verify the distribution
   - Upload to PyPI
   - Update the release notes with installation instructions

**What it does:**
- Builds source distribution and wheel
- Validates the distribution with twine
- Publishes to PyPI using the API token
- Updates GitHub release notes with PyPI link

## Local Testing

To run tests locally with a temporary conda environment, use the provided script:

```bash
./run_tests_with_conda.sh
```

This script:
- Creates a temporary conda environment
- Installs the package with dev dependencies
- Runs pytest with coverage
- Cleans up the temporary environment

## Notes

- The test workflow runs on Ubuntu latest
- Coverage reports are automatically uploaded to Codecov if configured
- PyPI publishing requires the `PYPI_API_TOKEN` secret to be set
- Make sure to update version numbers in `setup.py` and `pyproject.toml` before creating releases

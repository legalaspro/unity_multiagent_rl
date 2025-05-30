# Running Tests

This document provides instructions on how to set up your environment for testing and how to run the various tests included in this project.

## Setup

1.  **Create and Activate a Virtual Environment (Recommended):**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
    ```

2.  **Install Dependencies:**
    Ensure all project dependencies, including testing libraries, are installed. The testing libraries (`pytest`, `pytest-mock`) are included in `requirements.txt`.
    ```bash
    pip install -r requirements.txt
    ```

## Running Tests

All tests are written using the `pytest` framework.

*   **Run All Tests:**
    To run all tests in the project, navigate to the root directory and execute:
    ```bash
    pytest
    ```
    You can add `-v` for more verbose output.

*   **Run Tests in a Specific File:**
    To run all tests within a particular file (e.g., `tests/test_maddpg.py`):
    ```bash
    pytest tests/test_maddpg.py
    ```

*   **Run a Specific Test Function:**
    To run a specific test function by name (e.g., `test_maddpg_initialization` in `tests/test_maddpg.py`):
    ```bash
    pytest tests/test_maddpg.py::test_maddpg_initialization
    ```

*   **Run Tests with a Keyword Expression:**
    To run tests whose names match a keyword expression:
    ```bash
    pytest -k "maddpg and not initialization"
    ```

## Test Coverage (Optional)

To generate a test coverage report, you can install `pytest-cov`:
```bash
pip install pytest-cov
```
Then run pytest with the coverage flag:
```bash
pytest --cov=. --cov-report html
```
This will generate an HTML report in the `htmlcov/` directory, which you can open in a web browser to see detailed coverage information.

## Continuous Integration (CI) Suggestion

To automate the running of tests, consider setting up a Continuous Integration workflow. Below is a basic example for GitHub Actions. Create a file named `.github/workflows/ci.yml`:

```yaml
name: Python CI

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  build:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.8", "3.9", "3.10"] # Specify desired Python versions

    steps:
    - uses: actions/checkout@v3
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v3
      with:
        python-version: ${{ matrix.python-version }}
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
    - name: Test with pytest
      run: |
        pytest
```
This workflow will run tests on every push to the `main` branch and on every pull request targeting `main`, across multiple Python versions.

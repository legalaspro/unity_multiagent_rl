# Running Tests

This document provides instructions on how to set up your environment for testing and how to run the various tests included in this project.

## Setup

1.  **Create and Activate a Virtual Environment (Recommended):**

    ```bash
    conda env create -f environment.yaml
    conda activate unity_multiagent_rl
    ```

    Or using pip:

    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
    pip install -r requirements.txt
    ```

2.  **Install Dependencies:**
    Ensure all project dependencies, including testing libraries, are installed. The testing libraries (`pytest`, `pytest-mock`) are included in both `requirements.txt` and `environment.yaml`.
    ```bash
    pip install -r requirements.txt
    ```

## Running Tests

All tests are written using the `pytest` framework.

### Test Categories

The project includes several categories of tests:

1. **Algorithm Tests** (`tests/algos/`): Test individual algorithm implementations (MAPPO, MADDPG, MASAC, MATD3)
2. **Network Tests** (`tests/networks/`): Test neural network modules and components
3. **Buffer Tests** (`tests/buffers/`): Test replay buffer and rollout storage functionality
4. **Evaluation Tests** (`tests/eval/`): Test evaluation and competitive evaluation systems
5. **Smoke Tests** (`tests/test_smoke_runs.py`): Quick end-to-end tests for all algorithms

### Basic Test Commands

- **Run All Tests:**
  To run all tests in the project, navigate to the root directory and execute:

  ```bash
  pytest
  ```

  You can add `-v` for more verbose output.

- **Run Tests in a Specific Directory:**
  To run all tests within a particular directory (e.g., algorithm tests):

  ```bash
  pytest tests/algos/
  ```

- **Run Tests in a Specific File:**
  To run all tests within a particular file (e.g., MADDPG tests):

  ```bash
  pytest tests/algos/test_maddpg.py
  ```

- **Run a Specific Test Function:**
  To run a specific test function by name:

  ```bash
  pytest tests/algos/test_maddpg.py::test_maddpg_initialization
  ```

- **Run Smoke Tests Only:**
  To run quick smoke tests for all algorithms:

  ```bash
  pytest tests/test_smoke_runs.py
  ```

- **Run Tests with a Keyword Expression:**
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

## Useful Test Options

- **Run tests with verbose output:**

  ```bash
  pytest -v
  ```

- **Run tests and stop on first failure:**

  ```bash
  pytest -x
  ```

- **Run tests in parallel (requires pytest-xdist):**

  ```bash
  pip install pytest-xdist
  pytest -n auto
  ```

- **Run only failed tests from last run:**
  ```bash
  pytest --lf
  ```

## Continuous Integration (CI) Suggestion

To automate the running of tests, consider setting up a Continuous Integration workflow. Below is a basic example for GitHub Actions. Create a file named `.github/workflows/ci.yml`:

```yaml
name: Python CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  build:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.11"] # Project uses Python 3.11

    steps:
      - uses: actions/checkout@v4
      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
      - name: Test with pytest
        run: |
          pytest -v
```

This workflow will run tests on every push to the `main` branch and on every pull request targeting `main`, using Python 3.11 as specified in the project requirements.

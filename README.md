# Golf performance experiments

A local Streamlit app for testing whether one golf-variable change improves a predefined performance goal.

## Setup

This project requires Python 3.11 or newer. A project-local virtual environment is recommended:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## Run the app

```bash
streamlit run app.py
```

## Run tests

```bash
pytest
```

The current application covers experiment definition, power-based protocol selection, local storage/resume, manual shot-level data entry, statistical analysis, and direction-aware practical-significance decisions. Visual reporting and historical player features are being developed incrementally.

## Development principles

- Only one experiment variable changes between A and B.
- Practical significance is defined before results are reviewed.
- Confirmatory and exploratory protocols are presented separately.
- Core domain, protocol, power, and analysis logic remains independent of Streamlit.

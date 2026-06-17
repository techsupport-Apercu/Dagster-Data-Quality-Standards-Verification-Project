# Dagster NCSI Data Cleaning Pipeline

This repository contains a Dagster project that transforms an NCSI CSV datadump through a 5-stage cleaning and enrichment pipeline.

## Overview

The project reads source data from datasets/datadump.csv and materializes staged outputs to dedicated folders under datasets/clean_stage*/.

Pipeline stages:
1. Stage 1: shorten source column names and generate a data dictionary.
2. Stage 2: clean comp_inte_with (drop nulls, uppercase values).
3. Stage 3: build companies dataset and attach company IDs to each row.
4. Stage 4: analyze sentiment from impr_on_cust using TextBlob.
5. Stage 5: derive NPS category from like_to_reco.

## Project Structure

```text
Dagster-NCSI/
├── README.md
└── Dagster/
    ├── README.md
    ├── pyproject.toml
    ├── requirements.txt
    ├── data_quality_checker/
    │   ├── __init__.py
    │   ├── assets.py
    │   ├── io_managers.py
    │   └── schedules.py
    ├── datasets/
    │   ├── datadump.csv
    │   ├── clean_stage1/
    │   │   ├── stage_1_ouput.csv
    │   │   └── stage_1_data_dictionary.csv
    │   ├── clean_stage2/
    │   │   └── stage_2_ouput.csv
    │   ├── clean_stage3/
    │   │   ├── companies_dataset.csv
    │   │   └── stage_3_ouput.csv
    │   ├── clean_stage4/
    │   │   └── stage_4_ouput.csv
    │   └── clean_stage5/
    │       └── stage_5_ouput.csv
    └── tests/
        └── test_column_shortening.py
```

## Local Setup

1. From the repository root:

```powershell
cd Dagster
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

2. Run Dagster UI:

```powershell
dagster dev
```

Open http://127.0.0.1:3000.

## Running the Pipeline

Materialize all assets (Stages 1-5):

```powershell
python -c "from data_quality_checker import defs; from dagster import materialize; materialize(defs.assets)"
```

## Running Tests

Run unit tests:

```powershell
python -m unittest tests/test_column_shortening.py
```

Current tests cover:
1. Short-column mapping behavior.
2. Stage 2 comp_inte_with cleaning.
3. Stage 3 company dataset and ID mapping.
4. Stage 4 sentiment labeling.
5. Stage 5 NPS categorization.

## Contributing

1. Create a feature branch.
2. Make changes in data_quality_checker/assets.py and related tests.
3. Run tests before opening a pull request.
4. Include sample output validation when stage logic changes.

## Notes

1. CSV IO is abstracted through a Dagster IO manager in data_quality_checker/io_managers.py.
2. The pipeline is scheduled hourly via schedules.py.

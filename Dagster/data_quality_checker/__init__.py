from pathlib import Path

from dagster import Definitions, load_assets_from_modules

from . import assets, schedules
from .io_managers import csv_io_manager

# Load all assets from assets.py module
all_assets = load_assets_from_modules([assets])

# Define the Definitions object, which Dagster uses to load your code
defs = Definitions(
    assets=all_assets,
    schedules=[schedules.hourly_verification_schedule],
    jobs=[schedules.standards_verification_job],
    resources={
        "io_manager": csv_io_manager.configured(
            {"base_dir": str(Path(__file__).resolve().parents[1] / "datasets")}
        )
    },
)

from dagster import Definitions, load_assets_from_modules, load_asset_checks_from_modules

from . import assets, schedules

# Load all assets and asset checks from assets.py module
all_assets = load_assets_from_modules([assets])
all_checks = load_asset_checks_from_modules([assets])

# Define the Definitions object, which Dagster uses to load your code
defs = Definitions(
    assets=all_assets,
    asset_checks=all_checks,
    schedules=[schedules.hourly_verification_schedule],
    jobs=[schedules.standards_verification_job],
)

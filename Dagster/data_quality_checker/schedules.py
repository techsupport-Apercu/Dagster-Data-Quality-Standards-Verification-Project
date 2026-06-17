from dagster import ScheduleDefinition, define_asset_job

# Define a job that materializes all assets in the project
standards_verification_job = define_asset_job(
    name="standards_verification_job",
    selection="*"
)

# Schedule the job to run daily at midnight (or hourly for testing, let's do hourly)
hourly_verification_schedule = ScheduleDefinition(
    job=standards_verification_job,
    cron_schedule="0 * * * *",  # Every hour
)

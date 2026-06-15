from dagster import ScheduleDefinition, define_asset_job

# Define a job that materializes both raw and cleaned user signups assets (and runs their checks)
standards_verification_job = define_asset_job(
    name="standards_verification_job",
    selection=["raw_user_signups", "cleaned_user_signups"]
)

# Schedule the job to run daily at midnight (or hourly for testing, let's do hourly)
hourly_verification_schedule = ScheduleDefinition(
    job=standards_verification_job,
    cron_schedule="0 * * * *",  # Every hour
)

import pandas as pd
from dagster import asset, asset_check, AssetCheckResult, AssetCheckSeverity

@asset
def raw_user_signups() -> pd.DataFrame:
    """Mock raw user signups data loaded from source."""
    data = {
        "user_id": [1, 2, 3, 4, None],  # One invalid user_id
        "email": ["alice@example.com", "bob@example.com", "invalid-email", "charlie@example.com", "dan@example.com"],
        "signup_date": ["2026-06-01", "2026-06-02", "2026-06-02", None, "2026-06-03"],
        "country": ["us", "ca", "US", "uk", "CA"],
        "age": [25, 30, -5, 45, 22]  # Negative age is invalid
    }
    return pd.DataFrame(data)

@asset
def cleaned_user_signups(raw_user_signups: pd.DataFrame) -> pd.DataFrame:
    """Cleaned user signups matching the organization's standards."""
    # Filter rows with null user_id
    df = raw_user_signups.dropna(subset=["user_id"]).copy()
    
    # Filter out rows with invalid emails (basic check for '@' sign)
    df = df[df["email"].str.contains("@", na=False)]
    
    # Correct invalid ages (replace negative age with None or default)
    df.loc[df["age"] < 0, "age"] = None
    
    # Standardize country codes to uppercase
    df["country"] = df["country"].str.upper()
    
    return df

@asset_check(asset=cleaned_user_signups)
def check_no_null_user_ids(cleaned_user_signups: pd.DataFrame) -> AssetCheckResult:
    """Check that user_id is never null."""
    null_count = cleaned_user_signups["user_id"].isnull().sum()
    return AssetCheckResult(
        passed=bool(null_count == 0),
        metadata={"null_records_found": int(null_count)},
        severity=AssetCheckSeverity.ERROR
    )

@asset_check(asset=cleaned_user_signups)
def check_emails_are_valid(cleaned_user_signups: pd.DataFrame) -> AssetCheckResult:
    """Check that all email strings contain an '@' sign."""
    invalid_count = (~cleaned_user_signups["email"].str.contains("@", na=False)).sum()
    return AssetCheckResult(
        passed=bool(invalid_count == 0),
        metadata={"invalid_emails_found": int(invalid_count)},
        severity=AssetCheckSeverity.ERROR
    )

@asset_check(asset=cleaned_user_signups)
def check_age_values(cleaned_user_signups: pd.DataFrame) -> AssetCheckResult:
    """Check that there are no negative ages in the output."""
    negative_count = (cleaned_user_signups["age"] < 0).sum()
    return AssetCheckResult(
        passed=bool(negative_count == 0),
        metadata={"negative_ages_found": int(negative_count)},
        severity=AssetCheckSeverity.WARN
    )

import re

import pandas as pd
from dagster import asset
from textblob import TextBlob


def build_short_column_mapping(columns: list[str], max_length: int = 18) -> dict[str, str]:
    """Build unique, shorter column names for easier reference."""
    mapping: dict[str, str] = {}
    used_names: set[str] = set()

    for index, original in enumerate(columns, start=1):
        normalized = re.sub(r"[^a-zA-Z0-9]+", "_", str(original).strip().lower()).strip("_")
        tokens = [token for token in normalized.split("_") if token]

        if tokens:
            base = "_".join(token[:4] for token in tokens[:3])
        else:
            base = f"col_{index}"

        if not base:
            base = f"col_{index}"

        target_max = min(max_length, max(1, len(str(original)) - 1))
        candidate = base[:target_max]

        # Guarantee uniqueness while preserving the shorter-than-original goal.
        suffix = 1
        while candidate in used_names:
            suffix_text = str(suffix)
            keep = max(1, target_max - len(suffix_text) - 1)
            candidate = f"{base[:keep]}_{suffix_text}"
            suffix += 1

        mapping[str(original)] = candidate
        used_names.add(candidate)

    return mapping


def clean_comp_inte_with_column(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Drop null comp_inte_with rows and normalize values to uppercase."""
    if "comp_inte_with" not in dataframe.columns:
        raise KeyError("Required column 'comp_inte_with' was not found in ncsi_datadump_short_columns.")

    cleaned = dataframe.dropna(subset=["comp_inte_with"]).copy()
    cleaned["comp_inte_with"] = cleaned["comp_inte_with"].astype(str).str.upper()
    return cleaned


def build_companies_dataset(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Create a company dimension table from stage-2 data."""
    required_columns = {"comp_inte_with", "sect"}
    missing = required_columns.difference(dataframe.columns)
    if missing:
        raise KeyError(f"Missing required columns for company dataset: {sorted(missing)}")

    companies = (
        dataframe[["comp_inte_with", "sect"]]
        .dropna(subset=["comp_inte_with"]) 
        .drop_duplicates(subset=["comp_inte_with"], keep="first")
        .sort_values(by="comp_inte_with")
        .reset_index(drop=True)
    )
    companies.insert(0, "company_id", [f"COMP_{i:05d}" for i in range(1, len(companies) + 1)])
    return companies


def attach_company_ids(stage2_dataframe: pd.DataFrame, companies_dataset: pd.DataFrame) -> pd.DataFrame:
    """Attach company_id to stage-2 rows using comp_inte_with as key."""
    if "comp_inte_with" not in stage2_dataframe.columns:
        raise KeyError("Required column 'comp_inte_with' was not found in stage-2 data.")

    if "comp_inte_with" not in companies_dataset.columns or "company_id" not in companies_dataset.columns:
        raise KeyError("companies_dataset must contain 'comp_inte_with' and 'company_id' columns.")

    merged = stage2_dataframe.merge(
        companies_dataset[["comp_inte_with", "company_id"]],
        on="comp_inte_with",
        how="left",
    )

    if merged["company_id"].isnull().any():
        raise ValueError("Some stage-2 rows could not be mapped to a company_id.")

    return merged


def classify_sentiment(text: str) -> str:
    """Classify a text value into POSITIVE, NEGATIVE, or NEUTRAL using TextBlob."""
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0:
        return "POSITIVE"
    if polarity < 0:
        return "NEGATIVE"
    return "NEUTRAL"


def add_impr_on_cust_sentiment(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Append sentiment labels derived from impr_on_cust values."""
    if "impr_on_cust" not in dataframe.columns:
        raise KeyError("Required column 'impr_on_cust' was not found in stage-3 data.")

    output = dataframe.copy()
    output["impr_on_cust_sentiment"] = (
        output["impr_on_cust"]
        .fillna("")
        .astype(str)
        .map(classify_sentiment)
    )
    return output


def classify_nps_category(value) -> str:
    """Map like_to_reco values to NPS categories."""
    text = str(value).strip().lower()

    if text in {"promoter", "passive", "detractor"}:
        return text

    number_match = re.search(r"-?\d+(?:\.\d+)?", text)
    score = float(number_match.group(0)) if number_match else 7.0

    if score >= 9:
        return "promoter"
    if score >= 7:
        return "passive"
    return "detractor"


def add_nps_category(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Append nps_category derived from like_to_reco values."""
    if "like_to_reco" not in dataframe.columns:
        raise KeyError("Required column 'like_to_reco' was not found in stage-4 data.")

    output = dataframe.copy()
    output["nps_category"] = output["like_to_reco"].map(classify_nps_category)
    return output


@asset(metadata={"filename": "datadump.csv"})
def ncsi_datadump(context) -> pd.DataFrame:
    """Load the NCSI data dump through the CSV IO manager abstraction."""
    return context.resources.io_manager.read_csv("datadump.csv")


@asset(metadata={"filename": "clean_stage1/stage_1_data_dictionary.csv"})
def ncsi_data_dictionary(ncsi_datadump: pd.DataFrame) -> pd.DataFrame:
    """Map shortened column names back to original source column names."""
    mapping = build_short_column_mapping(list(ncsi_datadump.columns))
    return pd.DataFrame(
        {
            "short_name": list(mapping.values()),
            "original_name": list(mapping.keys()),
        }
    )


@asset(metadata={"filename": "clean_stage1/stage_1_ouput.csv"})
def ncsi_datadump_short_columns(
    ncsi_datadump: pd.DataFrame,
    ncsi_data_dictionary: pd.DataFrame,
) -> pd.DataFrame:
    """Create a cleaned stage output with shortened column names."""
    reverse_lookup = dict(
        zip(
            ncsi_data_dictionary["original_name"],
            ncsi_data_dictionary["short_name"],
        )
    )
    return ncsi_datadump.rename(columns=reverse_lookup)


@asset(metadata={"filename": "clean_stage2/stage_2_ouput.csv"})
def ncsi_datadump_comp_inte_with_cleaned(
    ncsi_datadump_short_columns: pd.DataFrame,
) -> pd.DataFrame:
    """Stage-2 clean for comp_inte_with."""
    return clean_comp_inte_with_column(ncsi_datadump_short_columns)


@asset(metadata={"filename": "clean_stage3/companies_dataset.csv"})
def ncsi_companies_dataset(
    ncsi_datadump_comp_inte_with_cleaned: pd.DataFrame,
) -> pd.DataFrame:
    """Stage-3 company dimension extracted from comp_inte_with."""
    return build_companies_dataset(ncsi_datadump_comp_inte_with_cleaned)


@asset(metadata={"filename": "clean_stage3/stage_3_ouput.csv"})
def ncsi_stage_3_output(
    ncsi_datadump_comp_inte_with_cleaned: pd.DataFrame,
    ncsi_companies_dataset: pd.DataFrame,
) -> pd.DataFrame:
    """Stage-3 output with company IDs added for comp_inte_with."""
    return attach_company_ids(ncsi_datadump_comp_inte_with_cleaned, ncsi_companies_dataset)


@asset(metadata={"filename": "clean_stage4/stage_4_ouput.csv"})
def ncsi_stage_4_output(
    ncsi_stage_3_output: pd.DataFrame,
) -> pd.DataFrame:
    """Stage-4 output with sentiment column derived from impr_on_cust."""
    return add_impr_on_cust_sentiment(ncsi_stage_3_output)


@asset(metadata={"filename": "clean_stage5/stage_5_ouput.csv"})
def ncsi_stage_5_output(
    ncsi_stage_4_output: pd.DataFrame,
) -> pd.DataFrame:
    """Stage-5 output with NPS category derived from like_to_reco."""
    return add_nps_category(ncsi_stage_4_output)


def split_channels_of_interaction(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Split multi-value chan_of_inte values into separate rows, retaining i as key."""
    if "chan_of_inte" not in dataframe.columns:
        raise KeyError("Required column 'chan_of_inte' was not found in input data.")
    if "i" not in dataframe.columns:
        raise KeyError("Required column 'i' (Response ID) was not found in input data.")

    df = dataframe[["i", "chan_of_inte"]].dropna(subset=["chan_of_inte"]).copy()
    df["chan_of_inte"] = df["chan_of_inte"].astype(str).str.split(",")
    df = df.explode("chan_of_inte")
    df["chan_of_inte"] = df["chan_of_inte"].str.strip()
    df = df[df["chan_of_inte"] != ""]
    return df


@asset(metadata={"filename": "clean_stage6/stage_6_ouput.csv"})
def ncsi_stage_6_output(
    ncsi_stage_5_output: pd.DataFrame,
) -> pd.DataFrame:
    """Stage-6 output containing the split channel of interaction values mapped to response IDs."""
    return split_channels_of_interaction(ncsi_stage_5_output)

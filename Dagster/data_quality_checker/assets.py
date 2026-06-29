import re
from functools import lru_cache
from pathlib import Path

import pandas as pd
from dagster import asset
from transformers import pipeline


LOCAL_SENTIMENT_MODEL_DIR = (
    Path(__file__).resolve().parents[1]
    / "models"
    / "cardiffnlp-twitter-roberta-base-sentiment-latest"
)


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

    cleaned = dataframe.copy()
    cleaned["comp_inte_with"] = cleaned["comp_inte_with"].astype("string").str.upper()

    cleaned["comp_inte_with"] = (
        cleaned["comp_inte_with"]
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    cleaned = cleaned.dropna(subset=["comp_inte_with"]).copy()

    replacements = {
        "JUDICIARY": "NIGERIAN JUDICIARY",
        "ALIBABA POWER LIMITED [APL ELECTRIC COMPANY LIMITED]": "ABA POWER LIMITED",
        "APL ELECTRIC COMPANY LIMITED": "ABA POWER LIMITED",
        "ABA POWER LIMITED [APL ELECTRIC COMPANY LIMITED]": "ABA POWER LIMITED",
        "LAGOS TRICYCLE(KEKE NAPEP)": "KEKE NAPEP",
        "LAGOS TRICYCLE (KEKE NAPEP)": "KEKE NAPEP",
        "MAITAMA GENERAL HOSPITAL - FCT(ABUJA)": "MAITAMA GENERAL HOSPITAL",
        "UNIVERSITY OF LAGOS, HEALTH CENTER": "UNIVERSITY OF LAGOS HEALTH CENTER",
        "LAGOS YELLOW BUSES": "YELLOW BUSES",
        "LAGOS YELLOW BUSES BUS": "YELLOW BUSES",
        "COMMERCIAL BUS": "YELLOW BUSES",
        "COMMERCIAL BUS (YELLOW BUSES)": "YELLOW BUSES",
        "BRT": "BUS RAPID TRANSIT [BRT]",
        "GENERAL HOSPITAL ILORIN": "GENERAL HOSPITAL, ILORIN",
        "HOTEL DU HOLF": "HOTEL DU GOLF",
        "ARMY": "NIGERIAN ARMY",
        "GENERAL HOSPITAL, ISOLO,OSHODI,LAGOS": "ISOLO GENERAL HOSPITAL",
        "GENERAL HOSPITAL ISOLO, OSHODI, LAGOS": "ISOLO GENERAL HOSPITAL",
    }
    cleaned["comp_inte_with"] = cleaned["comp_inte_with"].replace(replacements)
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


@lru_cache(maxsize=1)
def _get_sentiment_classifier():
    """Create and cache the CardiffNLP Twitter-RoBERTa sentiment pipeline."""
    if not (LOCAL_SENTIMENT_MODEL_DIR / "config.json").exists():
        raise FileNotFoundError(
            "Local sentiment model not found. Run scripts/download_cardiffnlp_model.py first. "
            f"Expected model files under: {LOCAL_SENTIMENT_MODEL_DIR}"
        )

    return pipeline(
        task="sentiment-analysis",
        model=str(LOCAL_SENTIMENT_MODEL_DIR),
        tokenizer=str(LOCAL_SENTIMENT_MODEL_DIR),
        local_files_only=True,
    )


def classify_sentiment(text: str) -> str:
    """Classify a text value into POSITIVE, NEGATIVE, or NEUTRAL using CardiffNLP."""
    normalized_text = str(text).strip()
    if not normalized_text:
        return "NEUTRAL"

    prediction = _get_sentiment_classifier()(normalized_text, truncation=True)[0]["label"]
    label_map = {
        "LABEL_0": "NEGATIVE",
        "negative": "NEGATIVE",
        "LABEL_1": "NEUTRAL",
        "neutral": "NEUTRAL",
        "LABEL_2": "POSITIVE",
        "positive": "POSITIVE",
    }
    return label_map.get(str(prediction), "NEUTRAL")


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


def classify_age_category(age_value) -> str:
    """Map age ranges into broader age categories."""
    normalized_value = str(age_value).strip().replace("–", "-")
    normalized_value = " ".join(normalized_value.split())

    age_mapping = {
        "Below 18": "Youth",
        "18-24": "Young Adults",
        "18 - 24": "Young Adults",
        "25-29": "Early Career",
        "25 - 29": "Early Career",
        "30-34": "Mid-Level Professionals",
        "30 - 34": "Mid-Level Professionals",
        "35-39": "Mid-Career",
        "35 - 39": "Mid-Career",
        "40-49": "Experienced Professionals",
        "40 - 49": "Experienced Professionals",
        "50-59": "Senior Professionals",
        "50 - 59": "Senior Professionals",
        "60+": "Seniors",
    }
    return age_mapping.get(normalized_value)


def add_age_category(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Append age_category derived from ag values."""
    if "ag" not in dataframe.columns:
        raise KeyError("Required column 'ag' was not found in input data.")

    output = dataframe.copy()
    output["age_category"] = output["ag"].map(classify_age_category)
    return output


@asset(
    metadata={"filename": "clean_stage7/stage_7_output.csv"},
    deps=["ncsi_stage_6_output"],
)
def ncsi_stage_7_output(
    ncsi_stage_5_output: pd.DataFrame,
) -> pd.DataFrame:
    """Stage-7 output with age category derived from ag."""
    return add_age_category(ncsi_stage_5_output)


def classify_income_category(value) -> str:
    """Map income ranges to socioeconomic categories."""
    if pd.isnull(value):
        return "Unknown"

    text = " ".join(str(value).strip().split())
    text = text.replace("–", "-")

    income_mapping = {
        "Unemployed/No Earning As At Now": "No Income / Not Earning",
        "Student/Dependent": "No Income / Not Earning",
        "N20,000 - N30,000": "Low Income",
        "N31,000 - N40,000": "Low Income",
        "N41,000 - N60,000": "Low Income",
        "N61,000 - N79,000": "Lower-Middle Income",
        "N60,000 - N79,000": "Lower-Middle Income",
        "N80,000 - N99,000": "Lower-Middle Income",
        "N100,000 - N150,000": "Lower-Middle Income",
        "N151,000 - N199,000": "Middle-High Income",
        "N200,000 - N300,000": "Middle-High Income",
        "N300,000 - N500,000": "Middle-High Income",
        "N500,000 - N1m": "High Income",
        "Above N1m": "High Income",
        "Business Person/Income Not Fixed": "Income not fixed",
    }
    return income_mapping.get(text, "Unknown")


def add_income_category(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Append income_category derived from inco values."""
    if "inco" not in dataframe.columns:
        raise KeyError("Required column 'inco' was not found in input data.")

    output = dataframe.copy()
    output["income_category"] = output["inco"].map(classify_income_category)
    return output


@asset(metadata={"filename": "clean_stage8/stage_8_output.csv"})
def ncsi_stage_8_output(
    ncsi_stage_7_output: pd.DataFrame,
) -> pd.DataFrame:
    """Stage-8 output with income category derived from inco."""
    return add_income_category(ncsi_stage_7_output)


def convert_to_percentages(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Convert specific columns to percentages based on their base scale (7 or 8)."""
    output = dataframe.copy()
    base_7_cols = [
        "over_sati_with",
        "trus_exte_of",
        "prof_exte_of",
        "bran_bran_outl",
        "comp_exte_of",
        "ease_of_doin",
        "proc_and_proc",
        "cust_focu_inno",
        "enga_with_cust",
    ]
    base_8_cols = [
        "orde_of_impo",
        "orde_of_impo_1",
        "orde_of_impo_2",
        "orde_of_impo_3",
        "orde_of_impo_4",
        "orde_of_impo_5",
        "orde_of_impo_6",
        "orde_of_impo_7",
    ]
    for col in base_7_cols:
        if col in output.columns:
            output[col] = (pd.to_numeric(output[col], errors="coerce") / 7.0) * 100.0
    for col in base_8_cols:
        if col in output.columns:
            output[col] = (pd.to_numeric(output[col], errors="coerce") / 8.0) * 100.0
    return output


@asset(metadata={"filename": "clean_stage9/stage_9_output.csv"})
def ncsi_stage_9_output(
    ncsi_stage_8_output: pd.DataFrame,
) -> pd.DataFrame:
    """Stage-9 output with specific columns converted to percentages."""
    return convert_to_percentages(ncsi_stage_8_output)


def aggregate_company_importance(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Calculate the average 'order of importance' attributes per company and restructure."""
    ooi_cols = [
        "orde_of_impo",
        "orde_of_impo_1",
        "orde_of_impo_2",
        "orde_of_impo_3",
        "orde_of_impo_4",
        "orde_of_impo_5",
        "orde_of_impo_6",
        "orde_of_impo_7",
    ]
    df_subset = dataframe[["comp_inte_with"] + ooi_cols].copy()

    for col in ooi_cols:
        df_subset[col] = pd.to_numeric(df_subset[col], errors="coerce")

    melted = df_subset.melt(
        id_vars=["comp_inte_with"],
        value_vars=ooi_cols,
        var_name="Attribute",
        value_name="Score",
    )

    aggregated = (
        melted.groupby(["comp_inte_with", "Attribute"], as_index=False)["Score"]
        .mean()
        .rename(columns={"Score": "Average_score"})
    )

    return aggregated


@asset(
    metadata={"filename": "clean_stage10/stage_10_output.csv"},
    deps=["ncsi_stage_9_output"],
)
def ncsi_stage_10_output(
    ncsi_stage_8_output: pd.DataFrame,
) -> pd.DataFrame:
    """Stage-10 output with company importance aggregates."""
    return aggregate_company_importance(ncsi_stage_8_output)


def calculate_company_nps(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Calculate company-level Net Promoter Score (NPS) statistics."""
    df = dataframe.copy()

    df["is_promoter"] = (df["nps_category"] == "promoter").astype(int)
    df["is_detractor"] = (df["nps_category"] == "detractor").astype(int)
    df["is_passive"] = (df["nps_category"] == "passive").astype(int)

    agg = (
        df.groupby(["comp_inte_with", "sect"], as_index=False)
        .agg(
            count_of_responses=("nps_category", "size"),
            count_of_promoters=("is_promoter", "sum"),
            count_of_detributors=("is_detractor", "sum"),  # Wait, let's keep name consistent
            count_of_passives=("is_passive", "sum"),
        )
    )
    # Wait, the prompt says columns should be:
    # 1. company_name (mapped from comp_inte_with)
    # 2. sector (mapped from sect)
    # 3. count_of_responses
    # 4. count_of_promoters
    # 5. count_of_detractors (notice the prompt specifies 'count_of_detractors')
    # 6. count_of_passives
    # 7. nps_score
    # Let's rename comp_inte_with to company_name, sect to sector, is_detractor sum to count_of_detractors
    agg = agg.rename(columns={
        "comp_inte_with": "company_name", 
        "sect": "sector",
        "count_of_detributors": "count_of_detractors"
    })

    agg["nps_score"] = 0.0
    non_zero = agg["count_of_responses"] > 0
    agg.loc[non_zero, "nps_score"] = (
        (agg.loc[non_zero, "count_of_promoters"] - agg.loc[non_zero, "count_of_detractors"])
        / agg.loc[non_zero, "count_of_responses"]
    ) * 100.0

    return agg


@asset(
    metadata={"filename": "clean_stage11/stage_11_output.csv"},
    deps=["ncsi_stage_10_output"],
)
def ncsi_stage_11_output(
    ncsi_stage_8_output: pd.DataFrame,
) -> pd.DataFrame:
    """Stage-11 output with company-level Net Promoter Score (NPS) statistics."""
    return calculate_company_nps(ncsi_stage_8_output)


def create_states_dataset(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Extract region data and generate instance counts."""
    return (
        dataframe.groupby("regi", dropna=False)
        .size()
        .reset_index(name="Count")
        .rename(columns={"regi": "State"})
    )


def calculate_overall_cx_score(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Calculate overall CX score per company using weighted attribute scores."""
    required_columns = {
        "company_id",
        "sect",
        "prof_exte_of",
        "bran_bran_outl",
        "comp_exte_of",
        "ease_of_doin",
        "proc_and_proc",
        "cust_focu_inno",
        "enga_with_cust",
        "orde_of_impo",
        "orde_of_impo_1",
        "orde_of_impo_2",
        "orde_of_impo_3",
        "orde_of_impo_4",
        "orde_of_impo_5",
        "orde_of_impo_6",
        "orde_of_impo_7",
    }
    missing = required_columns.difference(dataframe.columns)
    if missing:
        raise KeyError(f"Missing required columns for stage-13 output: {sorted(missing)}")

    output = dataframe.copy()

    trust_column = "trust_exte_of" if "trust_exte_of" in output.columns else "trus_exte_of"
    if trust_column not in output.columns:
        raise KeyError("Missing required trust column: expected 'trust_exte_of' or 'trus_exte_of'.")

    numeric_columns = [
        trust_column,
        "prof_exte_of",
        "bran_bran_outl",
        "comp_exte_of",
        "ease_of_doin",
        "proc_and_proc",
        "cust_focu_inno",
        "enga_with_cust",
        "orde_of_impo",
        "orde_of_impo_1",
        "orde_of_impo_2",
        "orde_of_impo_3",
        "orde_of_impo_4",
        "orde_of_impo_5",
        "orde_of_impo_6",
        "orde_of_impo_7",
    ]
    for column in numeric_columns:
        output[column] = pd.to_numeric(output[column], errors="coerce")

    output["weighted_numerator"] = (
        output[trust_column] * output["orde_of_impo"]
        + output["prof_exte_of"] * output["orde_of_impo_1"]
        + output["bran_bran_outl"] * output["orde_of_impo_2"]
        + output["comp_exte_of"] * output["orde_of_impo_3"]
        + output["ease_of_doin"] * output["orde_of_impo_4"]
        + output["proc_and_proc"] * output["orde_of_impo_5"]
        + output["cust_focu_inno"] * output["orde_of_impo_6"]
        + output["enga_with_cust"] * output["orde_of_impo_7"]
    )

    output["weight_sum"] = (
        output["orde_of_impo"]
        + output["orde_of_impo_1"]
        + output["orde_of_impo_2"]
        + output["orde_of_impo_3"]
        + output["orde_of_impo_4"]
        + output["orde_of_impo_5"]
        + output["orde_of_impo_6"]
        + output["orde_of_impo_7"]
    )

    grouped = (
        output.groupby(["company_id", "sect"], as_index=False)
        .agg(
            weighted_numerator=("weighted_numerator", "sum"),
            weight_sum=("weight_sum", "sum"),
        )
        .rename(columns={"sect": "sector"})
    )

    grouped["overall_cx_score"] = pd.NA
    non_zero = grouped["weight_sum"] > 0
    grouped.loc[non_zero, "overall_cx_score"] = (
        grouped.loc[non_zero, "weighted_numerator"] / grouped.loc[non_zero, "weight_sum"]
    )

    return grouped[["company_id", "sector", "overall_cx_score"]]


def aggregate_sector_scores(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Aggregate Stage-13 overall CX scores to sector-level scores."""
    required_columns = {"sector", "overall_cx_score"}
    missing = required_columns.difference(dataframe.columns)
    if missing:
        raise KeyError(f"Missing required columns for stage-14 output: {sorted(missing)}")

    output = dataframe.copy()
    output["overall_cx_score"] = pd.to_numeric(output["overall_cx_score"], errors="coerce")

    return (
        output.groupby("sector", as_index=False)["overall_cx_score"]
        .mean()
        .rename(columns={"overall_cx_score": "sector_score"})
    )


@asset(
    metadata={"filename": "clean_stage12/stage_12_output.csv"},
    deps=["ncsi_stage_11_output"],
)
def ncsi_stage_12_output(
    ncsi_stage_8_output: pd.DataFrame,
) -> pd.DataFrame:
    """Stage-12 output with region instance counts."""
    return create_states_dataset(ncsi_stage_8_output)


@asset(
    metadata={"filename": "clean_stage13/stage_13_output.csv"},
    deps=["ncsi_stage_12_output"],
)
def ncsi_stage_13_output(
    ncsi_stage_9_output: pd.DataFrame,
) -> pd.DataFrame:
    """Stage-13 output with overall CX score per company and sector."""
    return calculate_overall_cx_score(ncsi_stage_9_output)


@asset(
    metadata={"filename": "clean_stage14/stage_14_output.csv"},
    deps=["ncsi_stage_12_output"],
)
def ncsi_stage_14_output(
    ncsi_stage_13_output: pd.DataFrame,
) -> pd.DataFrame:
    """Stage-14 output with sector-level aggregated CX scores."""
    return aggregate_sector_scores(ncsi_stage_13_output)



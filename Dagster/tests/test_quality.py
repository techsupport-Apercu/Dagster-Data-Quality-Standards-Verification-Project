from pathlib import Path
import unittest

import pandas as pd
import pandas.testing as pdt

from data_quality_checker.assets import (
    add_impr_on_cust_sentiment,
    add_nps_category,
    attach_company_ids,
    build_companies_dataset,
    build_short_column_mapping,
    classify_sentiment,
    classify_nps_category,
    clean_comp_inte_with_column,
    split_channels_of_interaction,
    classify_age_category,
    add_age_category,
    classify_income_category,
    add_income_category,
)


class TestQuality(unittest.TestCase):
    def test_shortened_names_are_shorter_than_original(self):
        datasets_dir = Path(__file__).resolve().parents[1] / "datasets"
        source_path = datasets_dir / "datadump.csv"
        source_columns = list(pd.read_csv(source_path, nrows=0).columns)

        mapping = build_short_column_mapping(source_columns)

        self.assertEqual(len(mapping), len(source_columns))
        self.assertEqual(len(set(mapping.values())), len(source_columns))

        for original_name, short_name in mapping.items():
            self.assertLess(
                len(short_name),
                len(original_name),
                msg=f"Expected '{short_name}' to be shorter than '{original_name}'",
            )


class TestCompInteWithCleaning(unittest.TestCase):
    def test_cleaning_drops_nulls_and_uppercases_values(self):
        source = pd.DataFrame(
            {
                "comp_inte_with": ["abc", None, "MiXeD", "  xYz  "],
                "other_col": [1, 2, 3, 4],
            }
        )

        cleaned = clean_comp_inte_with_column(source)

        self.assertFalse(cleaned["comp_inte_with"].isnull().any())
        self.assertTrue(cleaned["comp_inte_with"].eq(cleaned["comp_inte_with"].str.upper()).all())
        self.assertEqual(len(cleaned), 3)

    def test_cleaning_on_stage1_file_has_no_nulls_and_uppercase(self):
        datasets_dir = Path(__file__).resolve().parents[1] / "datasets"
        source_path = datasets_dir / "clean_stage1" / "stage_1_ouput.csv"
        source = pd.read_csv(source_path)

        cleaned = clean_comp_inte_with_column(source)

        self.assertIn("comp_inte_with", cleaned.columns)
        self.assertFalse(cleaned["comp_inte_with"].isnull().any())
        self.assertTrue(cleaned["comp_inte_with"].eq(cleaned["comp_inte_with"].str.upper()).all())


class TestStage3CompanyOutputs(unittest.TestCase):
    def test_companies_dataset_has_unique_companies_ids_and_sector(self):
        stage2 = pd.DataFrame(
            {
                "comp_inte_with": ["ABC", "ABC", "XYZ", "LMN"],
                "sect": [1, 1, 2, 3],
                "metric": [10, 11, 12, 13],
            }
        )

        companies = build_companies_dataset(stage2)

        self.assertEqual(companies["comp_inte_with"].nunique(), len(companies))
        self.assertEqual(companies["company_id"].nunique(), len(companies))
        self.assertTrue(set(["company_id", "comp_inte_with", "sect"]).issubset(companies.columns))
        self.assertEqual(set(companies["comp_inte_with"]), {"ABC", "XYZ", "LMN"})

    def test_stage3_output_contains_valid_company_ids_for_all_rows(self):
        datasets_dir = Path(__file__).resolve().parents[1] / "datasets"
        stage2_path = datasets_dir / "clean_stage2" / "stage_2_ouput.csv"
        stage2 = pd.read_csv(stage2_path)

        companies = build_companies_dataset(stage2)
        stage3 = attach_company_ids(stage2, companies)

        self.assertIn("company_id", stage3.columns)
        self.assertFalse(stage3["company_id"].isnull().any())

        lookup = dict(zip(companies["comp_inte_with"], companies["company_id"]))
        expected_ids = stage3["comp_inte_with"].map(lookup)
        self.assertTrue((stage3["company_id"] == expected_ids).all())


class TestStage4SentimentOutputs(unittest.TestCase):
    def test_classify_sentiment_labels(self):
        self.assertEqual(classify_sentiment("This service is excellent and amazing"), "POSITIVE")
        self.assertEqual(classify_sentiment("This service is awful and terrible"), "NEGATIVE")
        self.assertEqual(classify_sentiment(""), "NEUTRAL")

    def test_stage4_adds_sentiment_column_for_stage3_rows(self):
        datasets_dir = Path(__file__).resolve().parents[1] / "datasets"
        stage3_path = datasets_dir / "clean_stage3" / "stage_3_ouput.csv"
        stage3 = pd.read_csv(stage3_path)

        stage4 = add_impr_on_cust_sentiment(stage3)

        self.assertIn("impr_on_cust_sentiment", stage4.columns)
        self.assertEqual(len(stage4), len(stage3))
        self.assertFalse(stage4["impr_on_cust_sentiment"].isnull().any())
        allowed_labels = {"POSITIVE", "NEGATIVE", "NEUTRAL"}
        self.assertTrue(set(stage4["impr_on_cust_sentiment"].unique()).issubset(allowed_labels))


class TestStage5NpsOutputs(unittest.TestCase):
    def test_classify_nps_category(self):
        self.assertEqual(classify_nps_category(10), "promoter")
        self.assertEqual(classify_nps_category(8), "passive")
        self.assertEqual(classify_nps_category(3), "detractor")

    def test_stage5_adds_nps_category_for_stage4_rows(self):
        datasets_dir = Path(__file__).resolve().parents[1] / "datasets"
        stage4_path = datasets_dir / "clean_stage4" / "stage_4_ouput.csv"
        stage4 = pd.read_csv(stage4_path)

        stage5 = add_nps_category(stage4)

        self.assertIn("nps_category", stage5.columns)
        self.assertEqual(len(stage5), len(stage4))
        self.assertFalse(stage5["nps_category"].isnull().any())
        allowed_labels = {"promoter", "passive", "detractor"}
        self.assertTrue(set(stage5["nps_category"].unique()).issubset(allowed_labels))


class TestStage6ChannelOutputs(unittest.TestCase):
    def test_split_channels_splits_values_correctly(self):
        source = pd.DataFrame(
            {
                "i": [1, 2, 3],
                "chan_of_inte": ["In Person, Email", "Website,In Person,", None],
            }
        )
        result = split_channels_of_interaction(source)
        self.assertEqual(len(result), 4)
        self.assertEqual(list(result["i"]), [1, 1, 2, 2])
        self.assertEqual(list(result["chan_of_inte"]), ["In Person", "Email", "Website", "In Person"])

    def test_stage6_on_stage5_file_has_correct_structure(self):
        datasets_dir = Path(__file__).resolve().parents[1] / "datasets"
        stage5_path = datasets_dir / "clean_stage5" / "stage_5_ouput.csv"
        stage5 = pd.read_csv(stage5_path)

        result = split_channels_of_interaction(stage5)
        self.assertTrue(set(["i", "chan_of_inte"]).issubset(result.columns))
        self.assertEqual(len(result.columns), 2)
        self.assertFalse(result["chan_of_inte"].isnull().any())
        self.assertFalse((result["chan_of_inte"] == "").any())


class TestStage7Output(unittest.TestCase):
    @staticmethod
    def _expected_age_category(age_value):
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

    def test_stage7_output_has_valid_age_categories(self):
        datasets_dir = Path(__file__).resolve().parents[1] / "datasets"
        stage7_path = datasets_dir / "clean_stage7" / "stage_7_output.csv"

        self.assertTrue(
            stage7_path.exists(),
            msg=f"Expected stage 7 output file to exist at '{stage7_path}', but it was not found.",
        )

        stage7 = pd.read_csv(stage7_path)

        self.assertIn(
            "age_category",
            stage7.columns,
            msg="Expected 'age_category' column to exist in stage_7_output.csv.",
        )
        self.assertIn(
            "ag",
            stage7.columns,
            msg="Expected 'ag' column to exist in stage_7_output.csv for age mapping validation.",
        )

        for row_number, row in enumerate(stage7.itertuples(index=False), start=1):
            expected_age_category = self._expected_age_category(row.ag)
            self.assertIsNotNone(
                expected_age_category,
                msg=f"Row {row_number}: unsupported ag value '{row.ag}' found in stage_7_output.csv.",
            )
            self.assertEqual(
                expected_age_category,
                row.age_category,
                msg=(
                    f"Row {row_number}: ag='{row.ag}' expected age_category="
                    f"'{expected_age_category}' but found '{row.age_category}'."
                ),
            )


class TestStage8Output(unittest.TestCase):
    @staticmethod
    def _expected_income_category(income_value):
        if pd.isnull(income_value):
            return "Unknown"
        text = " ".join(str(income_value).strip().split()).replace("–", "-")
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

    def test_stage8_output_has_valid_income_categories(self):
        datasets_dir = Path(__file__).resolve().parents[1] / "datasets"
        stage8_path = datasets_dir / "clean_stage8" / "stage_8_output.csv"

        self.assertTrue(
            stage8_path.exists(),
            msg=f"Expected stage 8 output file to exist at '{stage8_path}', but it was not found.",
        )

        stage8 = pd.read_csv(stage8_path)

        self.assertIn(
            "income_category",
            stage8.columns,
            msg="Expected 'income_category' column to exist in stage_8_output.csv.",
        )
        self.assertIn(
            "inco",
            stage8.columns,
            msg="Expected 'inco' column to exist in stage_8_output.csv for income mapping validation.",
        )

        for row_number, row in enumerate(stage8.itertuples(index=False), start=1):
            expected_income_category = self._expected_income_category(row.inco)
            self.assertEqual(
                expected_income_category,
                row.income_category,
                msg=(
                    f"Row {row_number}: inco='{row.inco}' expected income_category="
                    f"'{expected_income_category}' but found '{row.income_category}'."
                ),
            )


class TestStage9Output(unittest.TestCase):
    DIVIDE_BY_7_COLUMNS = [
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

    DIVIDE_BY_8_COLUMNS = [
        "orde_of_impo",
        "orde_of_impo_1",
        "orde_of_impo_2",
        "orde_of_impo_3",
        "orde_of_impo_4",
        "orde_of_impo_5",
        "orde_of_impo_6",
        "orde_of_impo_7",
    ]

    @staticmethod
    def _to_numeric(series: pd.Series) -> pd.Series:
        return pd.to_numeric(series, errors="coerce")

    def _assert_percentage_conversion(
        self,
        stage8: pd.DataFrame,
        stage9: pd.DataFrame,
        columns: list[str],
        divisor: int,
    ) -> None:
        for column in columns:
            self.assertIn(column, stage8.columns, msg=f"Expected '{column}' to exist in stage_8_output.csv.")
            self.assertIn(column, stage9.columns, msg=f"Expected '{column}' to exist in stage_9_output.csv.")

            original = self._to_numeric(stage8[column])
            actual = self._to_numeric(stage9[column])
            expected = (original / divisor) * 100

            pdt.assert_series_equal(
                actual,
                expected,
                check_names=False,
                check_exact=False,
                rtol=1e-9,
                atol=1e-9,
                obj=f"Percentage conversion mismatch for '{column}'",
            )

    def test_stage9_output_has_expected_percentage_conversions(self):
        datasets_dir = Path(__file__).resolve().parents[1] / "datasets"
        stage8_path = datasets_dir / "clean_stage8" / "stage_8_output.csv"
        stage9_path = datasets_dir / "clean_stage9" / "stage_9_output.csv"

        self.assertTrue(
            stage8_path.exists(),
            msg=f"Expected stage 8 output file to exist at '{stage8_path}', but it was not found.",
        )
        self.assertEqual(
            stage9_path.name,
            "stage_9_output.csv",
            msg="Expected Stage 9 final output to be labeled as 'stage_9_output.csv'.",
        )
        self.assertTrue(
            stage9_path.exists(),
            msg=f"Expected stage 9 output file to exist at '{stage9_path}', but it was not found.",
        )

        stage8 = pd.read_csv(stage8_path)
        stage9 = pd.read_csv(stage9_path)

        self.assertEqual(
            len(stage9),
            len(stage8),
            msg="Expected stage_9_output.csv to preserve the same number of rows as stage_8_output.csv.",
        )

        self._assert_percentage_conversion(stage8, stage9, self.DIVIDE_BY_7_COLUMNS, divisor=7)
        self._assert_percentage_conversion(stage8, stage9, self.DIVIDE_BY_8_COLUMNS, divisor=8)


class TestStage10Output(unittest.TestCase):
    OOI_COLUMNS = [
        "orde_of_impo",
        "orde_of_impo_1",
        "orde_of_impo_2",
        "orde_of_impo_3",
        "orde_of_impo_4",
        "orde_of_impo_5",
        "orde_of_impo_6",
        "orde_of_impo_7",
    ]

    def test_stage10_output_matches_expected_company_attribute_averages(self):
        datasets_dir = Path(__file__).resolve().parents[1] / "datasets"
        stage8_path = datasets_dir / "clean_stage8" / "stage_8_output.csv"
        stage10_path = datasets_dir / "clean_stage10" / "stage_10_output.csv"

        self.assertTrue(
            stage8_path.exists(),
            msg=f"Expected stage 8 output file to exist at '{stage8_path}', but it was not found.",
        )
        self.assertEqual(
            stage10_path.name,
            "stage_10_output.csv",
            msg="Expected Stage 10 final output to be labeled as 'stage_10_output.csv'.",
        )
        self.assertTrue(
            stage10_path.exists(),
            msg=f"Expected stage 10 output file to exist at '{stage10_path}', but it was not found.",
        )

        stage8 = pd.read_csv(stage8_path)
        stage10 = pd.read_csv(stage10_path)

        self.assertListEqual(
            list(stage10.columns),
            ["comp_inte_with", "Attribute", "Average_score"],
            msg="Expected stage_10_output.csv to strictly follow columns: comp_inte_with, Attribute, Average_score.",
        )

        expected = (
            stage8[["comp_inte_with", *self.OOI_COLUMNS]]
            .groupby("comp_inte_with", as_index=False)
            .mean(numeric_only=True)
            .melt(
                id_vars=["comp_inte_with"],
                value_vars=self.OOI_COLUMNS,
                var_name="Attribute",
                value_name="Average_score",
            )
        )

        self.assertEqual(
            len(stage10),
            len(expected),
            msg="Expected one Stage 10 row per company and per order-of-importance attribute.",
        )
        self.assertSetEqual(set(stage10["Attribute"].unique()), set(self.OOI_COLUMNS))

        actual_sorted = stage10.sort_values(["comp_inte_with", "Attribute"]).reset_index(drop=True)
        expected_sorted = expected.sort_values(["comp_inte_with", "Attribute"]).reset_index(drop=True)

        actual_sorted["Average_score"] = pd.to_numeric(actual_sorted["Average_score"], errors="coerce")
        expected_sorted["Average_score"] = pd.to_numeric(expected_sorted["Average_score"], errors="coerce")

        pdt.assert_frame_equal(
            actual_sorted,
            expected_sorted,
            check_dtype=False,
            check_exact=False,
            rtol=1e-9,
            atol=1e-9,
            obj="Stage 10 output mismatch",
        )


class TestStage11Output(unittest.TestCase):
    def test_stage11_output_matches_expected_response_statistics(self):
        datasets_dir = Path(__file__).resolve().parents[1] / "datasets"
        stage8_path = datasets_dir / "clean_stage8" / "stage_8_output.csv"
        stage11_path = datasets_dir / "clean_stage11" / "stage_11_output.csv"

        self.assertTrue(
            stage8_path.exists(),
            msg=f"Expected stage 8 output file to exist at '{stage8_path}', but it was not found.",
        )
        self.assertEqual(
            stage11_path.name,
            "stage_11_output.csv",
            msg="Expected Stage 11 final output to be labeled as 'stage_11_output.csv'.",
        )
        self.assertTrue(
            stage11_path.exists(),
            msg=f"Expected stage 11 output file to exist at '{stage11_path}', but it was not found.",
        )

        stage8 = pd.read_csv(stage8_path)
        stage11 = pd.read_csv(stage11_path)

        self.assertListEqual(
            list(stage11.columns),
            [
                "company_name",
                "sector",
                "count_of_responses",
                "count_of_promoters",
                "count_of_detractors",
                "count_of_passives",
                "nps_score",
            ],
            msg=(
                "Expected stage_11_output.csv to strictly follow columns: "
                "company_name, sector, count_of_responses, count_of_promoters, "
                "count_of_detractors, count_of_passives, nps_score."
            ),
        )

        expected = (
            stage8.assign(
                is_promoter=(stage8["nps_category"] == "promoter").astype(int),
                is_detractor=(stage8["nps_category"] == "detractor").astype(int),
                is_passive=(stage8["nps_category"] == "passive").astype(int),
            )
            .groupby(["comp_inte_with", "sect"], as_index=False)
            .agg(
                count_of_responses=("nps_category", "size"),
                count_of_promoters=("is_promoter", "sum"),
                count_of_detractors=("is_detractor", "sum"),
                count_of_passives=("is_passive", "sum"),
            )
            .rename(columns={"comp_inte_with": "company_name", "sect": "sector"})
        )

        expected["nps_score"] = (
            (expected["count_of_promoters"] / expected["count_of_responses"] * 100)
            - (expected["count_of_detractors"] / expected["count_of_responses"] * 100)
        )

        self.assertEqual(
            len(stage11),
            len(expected),
            msg="Expected one Stage 11 output row per company and sector.",
        )

        actual_sorted = stage11.sort_values(["company_name", "sector"]).reset_index(drop=True)
        expected_sorted = expected.sort_values(["company_name", "sector"]).reset_index(drop=True)

        numeric_columns = [
            "count_of_responses",
            "count_of_promoters",
            "count_of_detractors",
            "count_of_passives",
            "nps_score",
        ]
        for column in numeric_columns:
            actual_sorted[column] = pd.to_numeric(actual_sorted[column], errors="coerce")
            expected_sorted[column] = pd.to_numeric(expected_sorted[column], errors="coerce")

        pdt.assert_frame_equal(
            actual_sorted,
            expected_sorted,
            check_dtype=False,
            check_exact=False,
            rtol=1e-9,
            atol=1e-9,
            obj="Stage 11 output mismatch",
        )


class TestStage12Output(unittest.TestCase):
    def test_stage12_output_matches_expected_region_counts(self):
        datasets_dir = Path(__file__).resolve().parents[1] / "datasets"
        stage8_path = datasets_dir / "clean_stage8" / "stage_8_output.csv"
        stage12_path = datasets_dir / "clean_stage12" / "stage_12_output.csv"

        self.assertTrue(
            stage8_path.exists(),
            msg=f"Expected stage 8 output file to exist at '{stage8_path}', but it was not found.",
        )
        self.assertEqual(
            stage12_path.name,
            "stage_12_output.csv",
            msg="Expected Stage 12 final output to be labeled as 'stage_12_output.csv'.",
        )
        self.assertTrue(
            stage12_path.exists(),
            msg=f"Expected stage 12 output file to exist at '{stage12_path}', but it was not found.",
        )

        stage8 = pd.read_csv(stage8_path)
        stage12 = pd.read_csv(stage12_path)

        self.assertListEqual(
            list(stage12.columns),
            ["State", "Count"],
            msg="Expected stage_12_output.csv to strictly follow columns: State, Count.",
        )

        expected = (
            stage8.groupby("regi", dropna=False)
            .size()
            .reset_index(name="Count")
            .rename(columns={"regi": "State"})
        )

        self.assertEqual(
            len(stage12),
            len(expected),
            msg="Expected one Stage 12 output row per unique regi value from Stage 8.",
        )

        actual_sorted = stage12.sort_values(["State"]).reset_index(drop=True)
        expected_sorted = expected.sort_values(["State"]).reset_index(drop=True)

        actual_sorted["Count"] = pd.to_numeric(actual_sorted["Count"], errors="coerce")
        expected_sorted["Count"] = pd.to_numeric(expected_sorted["Count"], errors="coerce")

        pdt.assert_frame_equal(
            actual_sorted,
            expected_sorted,
            check_dtype=False,
            check_exact=False,
            rtol=1e-9,
            atol=1e-9,
            obj="Stage 12 output mismatch",
        )


class TestStage13Output(unittest.TestCase):
    def test_stage13_output_matches_expected_overall_cx_score_and_format(self):
        datasets_dir = Path(__file__).resolve().parents[1] / "datasets"
        stage9_path = datasets_dir / "clean_stage9" / "stage_9_output.csv"
        stage13_path = datasets_dir / "clean_stage13" / "stage_13_output.csv"

        self.assertTrue(
            stage9_path.exists(),
            msg=f"Expected stage 9 output file to exist at '{stage9_path}', but it was not found.",
        )
        self.assertEqual(
            stage13_path.name,
            "stage_13_output.csv",
            msg="Expected Stage 13 final output to be labeled as 'stage_13_output.csv'.",
        )
        self.assertTrue(
            stage13_path.exists(),
            msg=f"Expected stage 13 output file to exist at '{stage13_path}', but it was not found.",
        )

        stage9 = pd.read_csv(stage9_path)
        stage13 = pd.read_csv(stage13_path)

        self.assertListEqual(
            list(stage13.columns),
            ["company_id", "sector", "overall_cx_score"],
            msg=(
                "Expected stage_13_output.csv to strictly follow columns: "
                "company_id, sector, overall_cx_score."
            ),
        )

        trust_column = "trust_exte_of" if "trust_exte_of" in stage9.columns else "trus_exte_of"
        self.assertIn(
            trust_column,
            stage9.columns,
            msg="Expected Stage 9 output to include 'trust_exte_of' or 'trus_exte_of'.",
        )

        weighted_columns = [
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
        for column in ["company_id", "sect", *weighted_columns]:
            self.assertIn(column, stage9.columns, msg=f"Expected '{column}' to exist in stage_9_output.csv.")

        expected = stage9.copy()
        for column in weighted_columns:
            expected[column] = pd.to_numeric(expected[column], errors="coerce")

        expected["weighted_numerator"] = (
            expected[trust_column] * expected["orde_of_impo"]
            + expected["prof_exte_of"] * expected["orde_of_impo_1"]
            + expected["bran_bran_outl"] * expected["orde_of_impo_2"]
            + expected["comp_exte_of"] * expected["orde_of_impo_3"]
            + expected["ease_of_doin"] * expected["orde_of_impo_4"]
            + expected["proc_and_proc"] * expected["orde_of_impo_5"]
            + expected["cust_focu_inno"] * expected["orde_of_impo_6"]
            + expected["enga_with_cust"] * expected["orde_of_impo_7"]
        )
        expected["weight_sum"] = (
            expected["orde_of_impo"]
            + expected["orde_of_impo_1"]
            + expected["orde_of_impo_2"]
            + expected["orde_of_impo_3"]
            + expected["orde_of_impo_4"]
            + expected["orde_of_impo_5"]
            + expected["orde_of_impo_6"]
            + expected["orde_of_impo_7"]
        )

        expected = (
            expected.groupby(["company_id", "sect"], as_index=False)
            .agg(
                weighted_numerator=("weighted_numerator", "sum"),
                weight_sum=("weight_sum", "sum"),
            )
            .rename(columns={"sect": "sector"})
        )
        expected["overall_cx_score"] = pd.NA
        non_zero = expected["weight_sum"] > 0
        expected.loc[non_zero, "overall_cx_score"] = (
            expected.loc[non_zero, "weighted_numerator"] / expected.loc[non_zero, "weight_sum"]
        )
        expected = expected[["company_id", "sector", "overall_cx_score"]]

        self.assertEqual(
            len(stage13),
            len(expected),
            msg="Expected one Stage 13 output row per company and sector.",
        )

        actual_sorted = stage13.sort_values(["company_id", "sector"]).reset_index(drop=True)
        expected_sorted = expected.sort_values(["company_id", "sector"]).reset_index(drop=True)

        actual_sorted["overall_cx_score"] = pd.to_numeric(actual_sorted["overall_cx_score"], errors="coerce")
        expected_sorted["overall_cx_score"] = pd.to_numeric(expected_sorted["overall_cx_score"], errors="coerce")

        pdt.assert_frame_equal(
            actual_sorted,
            expected_sorted,
            check_dtype=False,
            check_exact=False,
            rtol=1e-9,
            atol=1e-9,
            obj="Stage 13 output mismatch",
        )


class TestStage14Output(unittest.TestCase):
    def test_stage14_output_matches_expected_sector_scores_and_format(self):
        datasets_dir = Path(__file__).resolve().parents[1] / "datasets"
        stage13_path = datasets_dir / "clean_stage13" / "stage_13_output.csv"
        stage14_path = datasets_dir / "clean_stage14" / "stage_14_output.csv"

        self.assertTrue(
            stage13_path.exists(),
            msg=f"Expected stage 13 output file to exist at '{stage13_path}', but it was not found.",
        )
        self.assertEqual(
            stage14_path.name,
            "stage_14_output.csv",
            msg="Expected Stage 14 final output to be labeled as 'stage_14_output.csv'.",
        )
        self.assertTrue(
            stage14_path.exists(),
            msg=f"Expected stage 14 output file to exist at '{stage14_path}', but it was not found.",
        )

        stage13 = pd.read_csv(stage13_path)
        stage14 = pd.read_csv(stage14_path)

        self.assertListEqual(
            list(stage14.columns),
            ["sector", "sector_score"],
            msg="Expected stage_14_output.csv to strictly follow columns: sector, sector_score.",
        )

        for column in ["sector", "overall_cx_score"]:
            self.assertIn(column, stage13.columns, msg=f"Expected '{column}' to exist in stage_13_output.csv.")

        expected = stage13.copy()
        expected["overall_cx_score"] = pd.to_numeric(expected["overall_cx_score"], errors="coerce")

        expected = (
            expected.groupby("sector", as_index=False)["overall_cx_score"]
            .mean()
            .rename(columns={"overall_cx_score": "sector_score"})
        )

        self.assertEqual(
            len(stage14),
            len(expected),
            msg="Expected one Stage 14 output row per sector.",
        )

        actual_sorted = stage14.sort_values(["sector"]).reset_index(drop=True)
        expected_sorted = expected.sort_values(["sector"]).reset_index(drop=True)

        actual_sorted["sector_score"] = pd.to_numeric(actual_sorted["sector_score"], errors="coerce")
        expected_sorted["sector_score"] = pd.to_numeric(expected_sorted["sector_score"], errors="coerce")

        pdt.assert_frame_equal(
            actual_sorted,
            expected_sorted,
            check_dtype=False,
            check_exact=False,
            rtol=1e-9,
            atol=1e-9,
            obj="Stage 14 output mismatch",
        )


if __name__ == "__main__":
    unittest.main()

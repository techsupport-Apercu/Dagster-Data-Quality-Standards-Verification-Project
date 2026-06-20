from pathlib import Path
import unittest

import pandas as pd

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


if __name__ == "__main__":
    unittest.main()

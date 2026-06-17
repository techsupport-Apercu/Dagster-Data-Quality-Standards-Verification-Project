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
)


class TestColumnShortening(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()

from pathlib import Path

from transformers import AutoModelForSequenceClassification, AutoTokenizer


MODEL_ID = "cardiffnlp/twitter-roberta-base-sentiment-latest"
OUTPUT_DIR = (
    Path(__file__).resolve().parents[1]
    / "models"
    / "cardiffnlp-twitter-roberta-base-sentiment-latest"
)


def download_and_save_model() -> Path:
    """Download CardiffNLP sentiment model and persist it to a local directory."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)

    tokenizer.save_pretrained(OUTPUT_DIR)
    model.save_pretrained(OUTPUT_DIR)
    return OUTPUT_DIR


def main() -> None:
    model_dir = download_and_save_model()
    print(f"Model downloaded and saved to: {model_dir}")


if __name__ == "__main__":
    main()

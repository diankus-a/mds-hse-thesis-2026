import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from sentence_transformers import InputExample, SentenceTransformer
from sentence_transformers.sentence_transformer import losses
from sentence_transformers.sentence_transformer.evaluation import BinaryClassificationEvaluator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SBERT to detect duplicate events from two text columns.")
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=Path(r"C:\Users\Akdia\Downloads\pairs_full_processed.csv"),
        help="Path to CSV file with event pairs.",
    )
    parser.add_argument("--text-col-1", type=str, default="event_1", help="First text column.")
    parser.add_argument("--text-col-2", type=str, default="event_2", help="Second text column.")
    parser.add_argument("--label-col", type=str, default="label", help="Binary label column.")
    parser.add_argument(
        "--model-name",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Base SentenceTransformer model.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/sbert_duplicates"), help="Output directory.")
    parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=64, help="Training batch size.")
    parser.add_argument("--eval-batch-size", type=int, default=256, help="Evaluation batch size.")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="Optimizer learning rate.")
    parser.add_argument("--max-samples", type=int, default=0, help="Optional cap for faster experiments. 0 means full dataset.")
    parser.add_argument("--test-size", type=float, default=0.1, help="Holdout test split size.")
    parser.add_argument("--val-size", type=float, default=0.1, help="Validation split size from the remaining data.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--keep-underscores",
        action="store_true",
        help="Keep underscores instead of converting snake_case to tokenized text.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_text(value: str, keep_underscores: bool) -> str:
    text = str(value).strip().lower()
    if not keep_underscores:
        text = text.replace("_", " ")
    return " ".join(text.split())


def load_dataset(args: argparse.Namespace) -> pd.DataFrame:
    df = pd.read_csv(args.csv_path)
    required_cols = {args.text_col_1, args.text_col_2, args.label_col}
    missing_cols = required_cols.difference(df.columns)
    if missing_cols:
        raise ValueError(f"Missing columns in CSV: {sorted(missing_cols)}")

    df = df[[args.text_col_1, args.text_col_2, args.label_col]].copy()
    df = df.dropna()
    df[args.label_col] = df[args.label_col].astype(int)
    df[args.text_col_1] = df[args.text_col_1].map(lambda x: normalize_text(x, args.keep_underscores))
    df[args.text_col_2] = df[args.text_col_2].map(lambda x: normalize_text(x, args.keep_underscores))

    if args.max_samples and args.max_samples < len(df):
        df = (
            df.groupby(args.label_col, group_keys=False)
            .apply(lambda chunk: chunk.sample(n=args.max_samples // 2, random_state=args.seed))
            .sample(frac=1.0, random_state=args.seed)
            .reset_index(drop=True)
        )

    return df


def build_examples(df: pd.DataFrame, text_col_1: str, text_col_2: str, label_col: str) -> list[InputExample]:
    return [
        InputExample(texts=[row[text_col_1], row[text_col_2]], label=float(row[label_col]))
        for _, row in df.iterrows()
    ]


def evaluate_split(model: SentenceTransformer, df: pd.DataFrame, args: argparse.Namespace) -> dict:
    embeddings_1 = model.encode(
        df[args.text_col_1].tolist(),
        batch_size=args.eval_batch_size,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    embeddings_2 = model.encode(
        df[args.text_col_2].tolist(),
        batch_size=args.eval_batch_size,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    cosine_scores = torch.nn.functional.cosine_similarity(embeddings_1, embeddings_2).cpu().numpy()
    labels = df[args.label_col].to_numpy()

    thresholds = np.linspace(-1.0, 1.0, 401)
    best_threshold = 0.0
    best_f1 = -1.0
    for threshold in thresholds:
        preds = (cosine_scores >= threshold).astype(int)
        score = f1_score(labels, preds)
        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)

    preds = (cosine_scores >= best_threshold).astype(int)
    metrics = {
        "threshold": best_threshold,
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "classification_report": classification_report(labels, preds, digits=4, zero_division=0),
    }
    return metrics


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(args)
    train_df, test_df = train_test_split(
        df,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=df[args.label_col],
    )

    relative_val_size = args.val_size / (1.0 - args.test_size)
    train_df, val_df = train_test_split(
        train_df,
        test_size=relative_val_size,
        random_state=args.seed,
        stratify=train_df[args.label_col],
    )

    model = SentenceTransformer(args.model_name)
    train_examples = build_examples(train_df, args.text_col_1, args.text_col_2, args.label_col)
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=args.batch_size)
    train_loss = losses.CosineSimilarityLoss(model=model)

    evaluator = BinaryClassificationEvaluator(
        sentences1=val_df[args.text_col_1].tolist(),
        sentences2=val_df[args.text_col_2].tolist(),
        labels=val_df[args.label_col].astype(bool).tolist(),
        batch_size=args.eval_batch_size,
        show_progress_bar=True,
        write_csv=True,
        name="validation",
    )

    warmup_steps = math.ceil(len(train_dataloader) * args.epochs * 0.1)
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=args.epochs,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": args.learning_rate},
        output_path=str(args.output_dir),
        save_best_model=True,
        show_progress_bar=True,
    )

    best_model = SentenceTransformer(str(args.output_dir))
    test_metrics = evaluate_split(best_model, test_df, args)

    summary = {
        "dataset": {
            "csv_path": str(args.csv_path),
            "rows_total": int(len(df)),
            "rows_train": int(len(train_df)),
            "rows_val": int(len(val_df)),
            "rows_test": int(len(test_df)),
            "positive_rate_total": float(df[args.label_col].mean()),
        },
        "training": {
            "base_model": args.model_name,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
        },
        "test_metrics": test_metrics,
    }

    metrics_path = args.output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    print()
    print("Classification report:")
    print(test_metrics["classification_report"])
    print(f"Model saved to: {args.output_dir.resolve()}")
    print(f"Metrics saved to: {metrics_path.resolve()}")


if __name__ == "__main__":
    main()

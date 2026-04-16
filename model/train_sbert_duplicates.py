import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from sentence_transformers import InputExample, SentenceTransformer
from sentence_transformers.losses import CosineSimilarityLoss
from sentence_transformers.evaluation import BinaryClassificationEvaluator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train SBERT to detect duplicate events from two text columns."
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=Path(r"C:\Users\Akdia\Downloads\pairs_full_processed.csv"), #Заменить
        help="Path to CSV file with event pairs.",
    )
    parser.add_argument(
        "--hard-set-path",
        type=Path,
        default=None,
        help="Optional path to hard semantic test set CSV.",
    )
    parser.add_argument("--text-col-1", type=str, default="event_1")
    parser.add_argument("--text-col-2", type=str, default="event_2")
    parser.add_argument("--label-col", type=str, default="label")
    parser.add_argument(
        "--model-name",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/sbert_duplicates"),
    )
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Cap per class for fast experiments. 0 = full dataset.",
    )
    parser.add_argument("--test-size", type=float, default=0.1)
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--keep-underscores",
        action="store_true",
        help="Keep underscores instead of converting snake_case to space-separated tokens.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {args.csv_path}")

    if args.max_samples < 0:
        raise ValueError("--max-samples must be >= 0")

    if not (0.0 < args.test_size < 1.0):
        raise ValueError("--test-size must be between 0 and 1")

    if not (0.0 < args.val_size < 1.0):
        raise ValueError("--val-size must be between 0 and 1")

    if args.test_size + args.val_size >= 1.0:
        raise ValueError("--test-size + --val-size must be < 1")


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

    required = {args.text_col_1, args.text_col_2, args.label_col}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns in main dataset: {sorted(missing)}")

    df = df[[args.text_col_1, args.text_col_2, args.label_col]].copy()
    df = df.dropna().reset_index(drop=True)
    df[args.label_col] = df[args.label_col].astype(int)
    df[args.text_col_1] = df[args.text_col_1].map(
        lambda x: normalize_text(x, args.keep_underscores)
    )
    df[args.text_col_2] = df[args.text_col_2].map(
        lambda x: normalize_text(x, args.keep_underscores)
    )

    if args.max_samples > 0:
        df = (
            df.groupby(args.label_col, group_keys=False)
            .apply(
                lambda chunk: chunk.sample(
                    n=min(args.max_samples, len(chunk)),
                    random_state=args.seed,
                )
            )
            .sample(frac=1.0, random_state=args.seed)
            .reset_index(drop=True)
        )

    return df


def load_hard_set(path: Path, args: argparse.Namespace) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Hard set file not found: {path}")

    df = pd.read_csv(path)
    df = df.rename(columns={"event_1": args.text_col_1, "event_2": args.text_col_2})

    required = {args.text_col_1, args.text_col_2, args.label_col}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns in hard set: {sorted(missing)}")

    df = df[[args.text_col_1, args.text_col_2, args.label_col]].copy()
    df = df.dropna().reset_index(drop=True)
    df[args.label_col] = df[args.label_col].astype(int)
    df[args.text_col_1] = df[args.text_col_1].map(
        lambda x: normalize_text(x, args.keep_underscores)
    )
    df[args.text_col_2] = df[args.text_col_2].map(
        lambda x: normalize_text(x, args.keep_underscores)
    )
    return df


def build_examples(
    df: pd.DataFrame,
    text_col_1: str,
    text_col_2: str,
    label_col: str,
) -> list[InputExample]:
    return [
        InputExample(texts=[row[text_col_1], row[text_col_2]], label=float(row[label_col]))
        for _, row in df.iterrows()
    ]


def _encode_pairs(
    model: SentenceTransformer,
    df: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray]:
    emb1 = model.encode(
        df[args.text_col_1].tolist(),
        batch_size=args.eval_batch_size,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    emb2 = model.encode(
        df[args.text_col_2].tolist(),
        batch_size=args.eval_batch_size,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    cosine_scores = torch.nn.functional.cosine_similarity(emb1, emb2).cpu().numpy()
    labels = df[args.label_col].to_numpy()
    return cosine_scores, labels


def evaluate_split(
    model: SentenceTransformer,
    df: pd.DataFrame,
    args: argparse.Namespace,
) -> dict:
    cosine_scores, labels = _encode_pairs(model, df, args)

    best_threshold, best_f1 = 0.0, -1.0
    for threshold in np.linspace(-1.0, 1.0, 401):
        preds = (cosine_scores >= threshold).astype(int)
        score = f1_score(labels, preds, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)

    preds = (cosine_scores >= best_threshold).astype(int)
    return {
        "threshold": best_threshold,
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "classification_report": classification_report(
            labels, preds, digits=4, zero_division=0
        ),
    }


def evaluate_with_threshold(
    model: SentenceTransformer,
    df: pd.DataFrame,
    threshold: float,
    args: argparse.Namespace,
) -> dict:
    cosine_scores, labels = _encode_pairs(model, df, args)
    preds = (cosine_scores >= threshold).astype(int)

    return {
        "threshold": threshold,
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "classification_report": classification_report(
            labels, preds, digits=4, zero_division=0
        ),
    }


def main() -> None:
    args = parse_args()
    validate_args(args)
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(args)
    print(f"Dataset: {len(df)} rows, positive rate: {df[args.label_col].mean():.3f}")

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
    print(f"Split — train: {len(train_df)}  val: {len(val_df)}  test: {len(test_df)}")

    model = SentenceTransformer(args.model_name)
    train_examples = build_examples(
        train_df, args.text_col_1, args.text_col_2, args.label_col
    )
    train_loader = DataLoader(train_examples, shuffle=True, batch_size=args.batch_size)
    train_loss = CosineSimilarityLoss(model=model)

    evaluator = BinaryClassificationEvaluator(
        sentences1=val_df[args.text_col_1].tolist(),
        sentences2=val_df[args.text_col_2].tolist(),
        labels=val_df[args.label_col].astype(bool).tolist(),
        batch_size=args.eval_batch_size,
        show_progress_bar=True,
        write_csv=True,
        name="validation",
    )

    warmup_steps = math.ceil(len(train_loader) * args.epochs * 0.1)
    model.fit(
        train_objectives=[(train_loader, train_loss)],
        evaluator=evaluator,
        epochs=args.epochs,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": args.learning_rate},
        output_path=str(args.output_dir),
        save_best_model=True,
        show_progress_bar=True,
    )

    best_model = SentenceTransformer(str(args.output_dir))

    print("\nSearching optimal threshold on VAL split...")
    val_metrics = evaluate_split(best_model, val_df, args)
    optimal_threshold = val_metrics["threshold"]
    print(
        f"Optimal threshold (val): {optimal_threshold:.2f}  |  val F1: {val_metrics['f1']:.4f}"
    )

    print("\nEvaluating TEST split...")
    test_metrics = evaluate_with_threshold(best_model, test_df, optimal_threshold, args)
    print(
        f"Test  — F1: {test_metrics['f1']:.4f}  "
        f"P: {test_metrics['precision']:.4f}  "
        f"R: {test_metrics['recall']:.4f}"
    )

    hard_metrics = None
    if args.hard_set_path is not None:
        print("\nEvaluating HARD SEMANTIC TEST SET...")
        hard_df = load_hard_set(args.hard_set_path, args)
        hard_metrics = evaluate_with_threshold(
            best_model, hard_df, optimal_threshold, args
        )
        print(
            f"Hard  — F1: {hard_metrics['f1']:.4f}  "
            f"P: {hard_metrics['precision']:.4f}  "
            f"R: {hard_metrics['recall']:.4f}"
        )
    else:
        print("\nHard set path not provided — skipping.")

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
        "val_metrics": {k: v for k, v in val_metrics.items() if k != "classification_report"},
        "test_metrics": {k: v for k, v in test_metrics.items() if k != "classification_report"},
        **(
            {
                "hard_metrics": {
                    k: v for k, v in hard_metrics.items() if k != "classification_report"
                }
            }
            if hard_metrics
            else {}
        ),
    }

    metrics_path = args.output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")

    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORTS")
    print("=" * 60)

    print("\n--- VAL ---")
    print(val_metrics["classification_report"])

    print("\n--- TEST ---")
    print(test_metrics["classification_report"])

    if hard_metrics:
        print("\n--- HARD SEMANTIC TEST SET ---")
        print(hard_metrics["classification_report"])

    print(f"\nModel saved to:   {args.output_dir.resolve()}")
    print(f"Metrics saved to: {metrics_path.resolve()}")


if __name__ == "__main__":
    main()

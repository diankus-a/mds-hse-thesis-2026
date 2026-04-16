import argparse
import json
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict whether two event names are duplicates with a trained SBERT model.")
    parser.add_argument("text_1", type=str, help="First event name.")
    parser.add_argument("text_2", type=str, help="Second event name.")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("artifacts/sbert_duplicates"),
        help="Directory with the trained model and metrics.json.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Optional manual threshold. If omitted, threshold is loaded from metrics.json.",
    )
    parser.add_argument(
        "--keep-underscores",
        action="store_true",
        help="Keep underscores instead of converting snake_case to space-separated tokens.",
    )
    return parser.parse_args()


def normalize_text(value: str, keep_underscores: bool) -> str:
    text = value.strip().lower()
    if not keep_underscores:
        text = text.replace("_", " ")
    return " ".join(text.split())


def load_threshold(model_dir: Path, fallback: float) -> float:
    metrics_path = model_dir / "metrics.json"
    if not metrics_path.exists():
        return fallback
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    return float(metrics.get("test_metrics", {}).get("threshold", fallback))


def main() -> None:
    args = parse_args()
    model = SentenceTransformer(str(args.model_dir))
    threshold = args.threshold if args.threshold is not None else load_threshold(args.model_dir, fallback=0.43)

    text_1 = normalize_text(args.text_1, args.keep_underscores)
    text_2 = normalize_text(args.text_2, args.keep_underscores)

    embeddings = model.encode([text_1, text_2], convert_to_tensor=True, normalize_embeddings=True)
    score = float(torch.nn.functional.cosine_similarity(embeddings[0], embeddings[1], dim=0).item())
    is_duplicate = score >= threshold

    print(
        json.dumps(
            {
                "text_1": text_1,
                "text_2": text_2,
                "score": score,
                "threshold": threshold,
                "is_duplicate": is_duplicate,
            },
            ensure_ascii=True,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

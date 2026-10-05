from __future__ import annotations

from pathlib import Path
from typing import Any

from src.ngram_model import BayesianNGramModel


class BayesianWordPredictor:
    """Interface pública para prever a próxima palavra com confiança percentual."""

    def __init__(self, corpus_path: str | Path = "data/corpus.txt", alpha: float = 1.0):
        root = Path(__file__).resolve().parents[1]
        resolved_path = Path(corpus_path)
        if not resolved_path.is_absolute():
            resolved_path = root / resolved_path

        self.model = BayesianNGramModel(resolved_path, alpha=alpha)

    def predict_next_word(self, sentence: str) -> dict[str, Any]:
        tokens = sentence.strip().split()
        context = " ".join(tokens[-2:]) if len(tokens) >= 2 else " ".join(tokens)
        candidates = self.model.predict(context, top_k=10)

        if not candidates:
            return {"word": "", "confidence": 0.0, "candidates": []}

        best_word, best_probability = candidates[0]
        total_probability = sum(probability for _, probability in candidates[:10])
        confidence = (best_probability / total_probability) * 100 if total_probability else 0.0
        return {
            "word": best_word,
            "confidence": round(confidence, 2),
            "candidates": [
                {"word": word, "probability": round(probability * 100, 4)}
                for word, probability in candidates
            ],
        }

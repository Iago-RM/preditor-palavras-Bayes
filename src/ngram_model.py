from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Sequence

from src.preprocessor import tokenize


class BayesianNGramModel:
    """Modelo probabilístico de n-grams com suavização de Laplace."""

    def __init__(self, corpus_path: str | Path, alpha: float = 1.0):
        self.alpha = alpha
        self.corpus_path = Path(corpus_path)
        self.tokens = self._load_tokens(self.corpus_path)
        self.total_tokens = len(self.tokens)
        self.vocabulary = sorted(set(self.tokens))
        self.unigrams = Counter(self.tokens)
        self.bigrams = Counter()
        self.trigrams = Counter()

        for first, second in zip(self.tokens, self.tokens[1:]):
            self.bigrams[(first, second)] += 1

        for first, second, third in zip(self.tokens, self.tokens[1:], self.tokens[2:]):
            self.trigrams[(first, second, third)] += 1

    @staticmethod
    def _load_tokens(corpus_path: Path) -> list[str]:
        if not corpus_path.exists():
            raise FileNotFoundError(f"Arquivo do corpus não encontrado: {corpus_path}")

        text = corpus_path.read_text(encoding="utf-8")
        tokens: list[str] = []
        for chunk in text.splitlines():
            tokens.extend(tokenize(chunk))
        return tokens

    def _count_history(self, context: Sequence[str]) -> int:
        if not context:
            return self.total_tokens
        if len(context) == 1:
            return self.unigrams.get(context[0], 0)
        if len(context) >= 2:
            return self.bigrams.get((context[-2], context[-1]), 0)
        return 0

    def _count_ngram(self, history: Sequence[str], candidate: str) -> int:
        if not history:
            return self.unigrams.get(candidate, 0)
        if len(history) == 1:
            return self.bigrams.get((history[0], candidate), 0)
        if len(history) >= 2:
            return self.trigrams.get((history[-2], history[-1], candidate), 0)
        return 0

    def probability(self, context: Sequence[str], candidate: str) -> float:
        """Calcula P(candidate | context) usando suavização de Laplace."""
        history = tuple(context)
        vocab_size = len(self.vocabulary)

        if not history:
            numerator = self.unigrams.get(candidate, 0) + self.alpha
            denominator = self.total_tokens + self.alpha * vocab_size
            return numerator / denominator

        if len(history) >= 2:
            history_key = (history[-2], history[-1])
            numerator = self.trigrams.get((history_key[0], history_key[1], candidate), 0) + self.alpha
            denominator = self.bigrams.get(history_key, 0) + self.alpha * vocab_size
            if self.bigrams.get(history_key, 0) > 0:
                return numerator / denominator

        if len(history) >= 1:
            first_word = history[-1]
            numerator = self.bigrams.get((first_word, candidate), 0) + self.alpha
            denominator = self.unigrams.get(first_word, 0) + self.alpha * vocab_size
            if self.unigrams.get(first_word, 0) > 0:
                return numerator / denominator

        numerator = self.unigrams.get(candidate, 0) + self.alpha
        denominator = self.total_tokens + self.alpha * vocab_size
        return numerator / denominator

    def predict(self, context: str, top_k: int = 5) -> list[tuple[str, float]]:
        """Lista as palavras mais prováveis para um contexto informado."""
        tokens = tokenize(context)
        ranked: list[tuple[str, float]] = []

        for word in self.vocabulary:
            score = self.probability(tokens, word)
            ranked.append((word, score))

        ranked.sort(key=lambda item: item[1], reverse=True)
        return ranked[:top_k]

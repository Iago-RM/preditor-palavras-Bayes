import re
from typing import List


TOKEN_PATTERN = re.compile(r"[a-zA-ZÀ-ÖØ-öø-ÿ0-9]+(?:'[a-zA-ZÀ-ÖØ-öø-ÿ0-9]+)?")


def normalize_text(text: str) -> str:
    """Normaliza o texto para um formato uniforme e sem caracteres especiais."""
    normalized = text.lower()
    normalized = normalized.replace("—", " ").replace("-", " ")
    normalized = re.sub(r"[^a-zA-ZÀ-ÖØ-öø-ÿ0-9\s']", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def tokenize(text: str) -> List[str]:
    """Converte uma string em uma lista de tokens válidos."""
    normalized = normalize_text(text)
    if not normalized:
        return []
    return TOKEN_PATTERN.findall(normalized)

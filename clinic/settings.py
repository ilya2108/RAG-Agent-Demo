"""Настройки из .env и переменных окружения."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = ROOT / "corpus"
CASES_DIR = ROOT / "cases"
CACHE_DIR = ROOT / ".cache"


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv(ROOT / ".env")

CREDENTIALS = os.getenv("GIGACHAT_CREDENTIALS", "")
SCOPE = os.getenv("GIGACHAT_SCOPE", "GIGACHAT_API_PERS")
MODEL = os.getenv("GIGACHAT_MODEL", "GigaChat-2-Pro")
JUDGE_MODEL = os.getenv("GIGACHAT_JUDGE_MODEL", "GigaChat-2-Max")
EMBEDDINGS_MODEL = os.getenv("GIGACHAT_EMBEDDINGS_MODEL", "Embeddings")
# gigachat — эмбеддинги GigaChat API; local — локальная модель через fastembed (если на ключе нет пакета эмбеддингов)
EMBEDDINGS_BACKEND = os.getenv("EMBEDDINGS_BACKEND", "gigachat")
LOCAL_EMBEDDINGS_MODEL = os.getenv("LOCAL_EMBEDDINGS_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
EMBEDDINGS_NAME = EMBEDDINGS_MODEL if EMBEDDINGS_BACKEND == "gigachat" else "local:" + LOCAL_EMBEDDINGS_MODEL.split("/")[-1]
VERIFY_SSL = os.getenv("GIGACHAT_VERIFY_SSL_CERTS", "false").lower() in ("1", "true", "yes")
CA_BUNDLE = os.getenv("GIGACHAT_CA_BUNDLE_FILE") or None
PORT = int(os.getenv("CLINIC_PORT", "8765"))


def live_enabled() -> bool:
    return bool(CREDENTIALS)

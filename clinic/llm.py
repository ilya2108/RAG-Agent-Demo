"""Тонкая обёртка над GigaChat: чат, эмбеддинги с дисковым кэшем, разбор JSON."""
import hashlib
import json
import math
import re
import threading

from gigachat import GigaChat
from gigachat.models import Chat, Messages, MessagesRole

from . import settings

_client = None
_lock = threading.Lock()


def client() -> GigaChat:
    global _client
    if _client is None:
        if not settings.CREDENTIALS:
            raise RuntimeError("Нет ключа GigaChat: заполните GIGACHAT_CREDENTIALS в .env")
        _client = GigaChat(
            credentials=settings.CREDENTIALS,
            scope=settings.SCOPE,
            model=settings.MODEL,
            verify_ssl_certs=settings.VERIFY_SSL,
            ca_bundle_file=settings.CA_BUNDLE,
            timeout=90,
            max_retries=3,
        )
    return _client


def chat(system: str, user: str, temperature: float = 0.1, model: str | None = None) -> str:
    payload = Chat(
        model=model or settings.MODEL,
        temperature=max(temperature, 0.01),  # GigaChat не принимает 0
        messages=[
            Messages(role=MessagesRole.SYSTEM, content=system),
            Messages(role=MessagesRole.USER, content=user),
        ],
    )
    resp = client().chat(payload)
    return resp.choices[0].message.content.strip()


def judge(system: str, user: str) -> str:
    return chat(system, user, temperature=0.01, model=settings.JUDGE_MODEL)


# ---------- эмбеддинги с кэшем ----------
_cache: dict | None = None


def _cache_path():
    return settings.CACHE_DIR / f"emb_{settings.EMBEDDINGS_NAME.replace('/', '_').replace(':', '_')}.json"


def _load_cache() -> dict:
    global _cache
    if _cache is None:
        p = _cache_path()
        _cache = json.loads(p.read_text()) if p.exists() else {}
    return _cache


def _save_cache() -> None:
    settings.CACHE_DIR.mkdir(exist_ok=True)
    _cache_path().write_text(json.dumps(_cache))


_local = None


def _local_model():
    global _local
    if _local is None:
        from fastembed import TextEmbedding  # pip install fastembed
        _local = TextEmbedding(settings.LOCAL_EMBEDDINGS_MODEL, cache_dir=str(settings.CACHE_DIR / "fastembed"))
    return _local


def _prefixed(texts: list[str], kind: str) -> list[str]:
    """Модели семейства e5 обучены с префиксами «query: » и «passage: » — без них качество заметно падает."""
    if settings.EMBEDDINGS_BACKEND == "local" and "e5" in settings.LOCAL_EMBEDDINGS_MODEL.lower():
        return [f"{kind}: {t}" for t in texts]
    return texts


def embed(texts: list[str], kind: str = "passage") -> list[list[float]]:
    """kind: passage — чанки базы знаний, query — поисковые запросы и вопросы."""
    texts = _prefixed(texts, kind)
    with _lock:
        cache = _load_cache()
        keys = [hashlib.sha1(t.encode()).hexdigest() for t in texts]
        missing = [(k, t) for k, t in zip(keys, texts) if k not in cache]
        if missing and settings.EMBEDDINGS_BACKEND == "local":
            for (k, _), v in zip(missing, _local_model().embed([t for _, t in missing])):
                cache[k] = [float(x) for x in v]
        elif missing:
            for i in range(0, len(missing), 16):
                batch = missing[i:i + 16]
                resp = client().embeddings([t for _, t in batch], model=settings.EMBEDDINGS_MODEL)
                for (k, _), item in zip(batch, resp.data):
                    cache[k] = item.embedding
        if missing:
            _save_cache()
        return [cache[k] for k in keys]


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def parse_json(text: str):
    """Достаёт первый JSON-объект или массив из ответа модели; терпит висячие запятые и «ёлочки»."""
    text = re.sub(r"```(?:json)?", "", text)
    for open_, close in (("{", "}"), ("[", "]")):
        start = text.find(open_)
        end = text.rfind(close)
        if start == -1 or end <= start:
            continue
        chunk = text[start:end + 1]
        for candidate in (chunk, re.sub(r",\s*([}\]])", r"\1", chunk)):
            try:
                return json.loads(candidate, strict=False)
            except json.JSONDecodeError:
                continue
    raise ValueError(f"Модель вернула не JSON: {text[:200]}")


def judge_json(system: str, user: str):
    """Вызов судьи с одной повторной попыткой, если ответ не разобрался как JSON."""
    try:
        return parse_json(judge(system, user))
    except ValueError:
        return parse_json(judge(system + " Верни только валидный JSON, без пояснений.", user))

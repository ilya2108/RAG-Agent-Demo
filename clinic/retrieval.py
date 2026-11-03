"""Поиск: dense, BM25, hybrid (RRF), метаданные и трансформации запроса."""
import math
import re
from collections import Counter
from functools import lru_cache

from . import llm, prompts
from .corpus import build_chunks

_TOKEN = re.compile(r"[a-zа-яё0-9]+", re.I)


def _tokens(text: str) -> list[str]:
    # грубый стемминг для русского: первые 6 символов слова
    return [t.lower()[:6] for t in _TOKEN.findall(text)]


@lru_cache
def _index(strategy: str):
    chunks = build_chunks(strategy)
    return chunks, llm.embed([c.embed_text for c in chunks])


@lru_cache
def _bm25(strategy: str):
    chunks = build_chunks(strategy)
    docs = [_tokens(c.embed_text) for c in chunks]
    df = Counter(t for d in docs for t in set(d))
    avg = sum(map(len, docs)) / len(docs)
    return docs, df, avg


def _allowed(chunk, metadata_filter: bool) -> bool:
    return chunk.status == "actual" if metadata_filter else True


def dense(query_vec, strategy, metadata_filter):
    chunks, vecs = _index(strategy)
    scored = [(llm.cosine(query_vec, v), c) for c, v in zip(chunks, vecs) if _allowed(c, metadata_filter)]
    return sorted(scored, key=lambda x: -x[0])


def bm25(query: str, strategy, metadata_filter, k1=1.5, b=0.75):
    chunks = build_chunks(strategy)
    docs, df, avg = _bm25(strategy)
    q = _tokens(query)
    n = len(docs)
    scored = []
    for c, d in zip(chunks, docs):
        if not _allowed(c, metadata_filter):
            continue
        tf = Counter(d)
        s = 0.0
        for t in q:
            if t in tf:
                idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
                s += idf * tf[t] * (k1 + 1) / (tf[t] + k1 * (1 - b + b * len(d) / avg))
        scored.append((s, c))
    return sorted(scored, key=lambda x: -x[0])


def rrf(rankings: list[list[tuple]], k: int = 60):
    """Reciprocal Rank Fusion: сумма 1/(k + rank) по всем спискам."""
    score, by_id = {}, {}
    for ranking in rankings:
        for rank, (_, c) in enumerate(ranking, 1):
            score[c.chunk_id] = score.get(c.chunk_id, 0) + 1 / (k + rank)
            by_id[c.chunk_id] = c
    return sorted(((s, by_id[i]) for i, s in score.items()), key=lambda x: -x[0])


def _search(query: str, cfg: dict, vec=None):
    vec = vec or llm.embed([query], kind="query")[0]
    d = dense(vec, cfg["chunking"], cfg["metadata_filter"])
    if cfg["retrieval"] == "hybrid":
        return rrf([d, bm25(query, cfg["chunking"], cfg["metadata_filter"])])
    return d


def retrieve(question: str, cfg: dict) -> tuple[list[dict], dict]:
    """Возвращает top-k чанков и описание трансформации запроса (для трассы)."""
    t = cfg.get("query_transform", "none")
    trace = {"type": t}
    if t == "hyde":
        hypo = llm.chat(prompts.HYDE_SYSTEM, question, temperature=0.3)
        trace["hypothetical_document"] = hypo
        ranked = _search(question, cfg, vec=llm.embed([hypo], kind="query")[0])
    elif t == "fusion":
        subs = llm.parse_json(llm.chat(prompts.FUSION_SYSTEM, question, temperature=0.3))["queries"]
        trace["queries"] = subs
        ranked = rrf([_search(q, cfg) for q in [question, *subs]])
    elif t == "stepback":
        general = llm.chat(prompts.STEPBACK_SYSTEM, question, temperature=0.3)
        trace["stepback_question"] = general
        ranked = rrf([_search(question, cfg), _search(general, cfg)])
    else:
        ranked = _search(question, cfg)
    top = []
    for rank, (score, c) in enumerate(ranked[: cfg["k"]], 1):
        top.append({**c.to_dict(), "rank": rank, "score": round(float(score), 4)})
    return top, trace

"""Один прогон вопроса через пайплайн с заданной конфигурацией → полная трасса."""
from datetime import datetime

from . import llm, metrics, prompts, settings
from .retrieval import retrieve

DEFAULT = {
    "chunking": "structural",      # structural | fixed
    "retrieval": "dense",          # dense | hybrid
    "query_transform": "none",     # none | hyde | fusion | stepback
    "metadata_filter": True,       # только документы со status: actual
    "k": 3,
    "prompt": "grounded",          # grounded | loose | custom (текст — в custom_prompt)
    "temperature": 0.1,
}


def run(question: str, cfg: dict | None = None, gold=None, evaluate=True) -> dict:
    cfg = {**DEFAULT, **(cfg or {})}
    chunks, transform = retrieve(question, cfg)
    if cfg["prompt"] == "custom":
        system = (cfg.get("custom_prompt") or "").strip() or prompts.GROUNDED_SYSTEM
    else:
        system = prompts.PROMPTS[cfg["prompt"]]
    user = prompts.user_message(question, chunks)
    answer = llm.chat(system, user, temperature=cfg["temperature"])
    trace = {
        "question": question, "config": cfg, "query_transform": transform, "chunks": chunks,
        "prompt": {"system": system, "user": user}, "answer": answer,
        "models": {"generation": settings.MODEL, "judge": settings.JUDGE_MODEL, "embeddings": settings.EMBEDDINGS_NAME},
        "created": datetime.now().isoformat(timespec="seconds"), "source": "live",
    }
    if evaluate:
        trace["metrics"] = metrics.evaluate(question, answer, chunks, gold)
    return trace

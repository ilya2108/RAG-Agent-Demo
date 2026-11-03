"""Метрики в духе RAGAS: precision@k (LLM-судья без эталона), faithfulness, answer relevance.
Плюс precision@k по эталонной разметке — она в карту пациента не попадает, только в разбор."""
import json

from . import llm

RELEVANCE_SYSTEM = (
    "Ты оцениваешь поиск в RAG-системе. Дан вопрос и пронумерованные фрагменты. "
    "Для каждого фрагмента реши, полезен ли он для ответа на вопрос (1) или нет (0). "
    'Верни JSON: {"verdicts": [{"id": 1, "relevant": 1, "reason": "..."}, ...]}'
)

STATEMENTS_SYSTEM = (
    "Разбей ответ на простые самостоятельные утверждения — по одному факту в каждом. "
    "Вежливые фразы и ссылки вида [1] не включай. Если в ответе нет утверждений "
    '(только отказ), верни пустой список. Верни JSON: {"statements": ["...", ...]}'
)

VERIFY_SYSTEM = (
    "Ты проверяешь, подтверждаются ли утверждения контекстом. Для каждого утверждения "
    "ответь 1, если оно прямо следует из контекста, и 0, если нет (в том числе когда "
    'контекст о нём молчит). Верни JSON: {"verdicts": [{"statement": "...", "supported": 1, "reason": "..."}, ...]}'
)

QGEN_SYSTEM = (
    "По ответу службы поддержки восстанови 3 разных вопроса клиента, на которые этот "
    "ответ мог бы быть ответом. Также определи, уклончивый ли ответ (noncommittal = 1, "
    "если ответ в основном говорит «не знаю» / «нет информации»). "
    'Верни JSON: {"questions": ["...", "...", "..."], "noncommittal": 0}'
)


def _context(chunks):
    return "\n\n".join(f"[{i}] {c['text']}" for i, c in enumerate(chunks, 1))


def context_precision(question, chunks):
    data = llm.judge_json(RELEVANCE_SYSTEM, f"Вопрос: {question}\n\nФрагменты:\n{_context(chunks)}")
    verdicts = data["verdicts"]
    rel = [int(v.get("relevant", 0)) for v in verdicts][: len(chunks)]
    return round(sum(rel) / len(chunks), 2), verdicts


def faithfulness(answer, chunks):
    statements = llm.judge_json(STATEMENTS_SYSTEM, answer)["statements"]
    if not statements:
        return None, []
    data = llm.judge_json(
        VERIFY_SYSTEM,
        f"Контекст:\n{_context(chunks)}\n\nУтверждения:\n{json.dumps(statements, ensure_ascii=False)}",
    )
    verdicts = data["verdicts"]
    ok = sum(int(v.get("supported", 0)) for v in verdicts)
    return round(ok / len(verdicts), 2), verdicts


def answer_relevance(question, answer):
    data = llm.judge_json(QGEN_SYSTEM, answer)
    questions = data["questions"]
    vecs = llm.embed([question, *questions], kind="query")
    sims = [llm.cosine(vecs[0], v) for v in vecs[1:]]
    score = sum(sims) / len(sims) * (0 if int(data.get("noncommittal", 0)) else 1)
    return round(score, 2), {"generated_questions": questions, "noncommittal": int(data.get("noncommittal", 0))}


def gold_precision(chunks, gold):
    """Доля чанков top-k, в которых есть эталонный факт из нужного документа."""
    def hit(c):
        norm = " ".join(c["text"].split())
        return any(c["doc_id"] == g["doc_id"] and g["phrase"] in norm for g in gold)
    flags = [hit(c) for c in chunks]
    return round(sum(flags) / len(chunks), 2), flags


def evaluate(question, answer, chunks, gold=None):
    p, p_details = context_precision(question, chunks)
    f, f_details = faithfulness(answer, chunks)
    r, r_details = answer_relevance(question, answer)
    out = {
        "precision_at_k": p, "faithfulness": f, "answer_relevance": r,
        "details": {"precision": p_details, "faithfulness": f_details, "relevance": r_details},
    }
    if gold:
        out["gold_precision_at_k"], out["details"]["gold_hits"] = gold_precision(chunks, gold)
    return out

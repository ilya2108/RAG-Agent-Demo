"""Карты пациентов для команд (Markdown, без диагноза и без промпта): ../Карты_пациентов/."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from clinic.patients import PATIENTS, SYSTEM_DESCRIPTION  # noqa: E402
from clinic.settings import CASES_DIR, ROOT  # noqa: E402

OUT = ROOT.parent / "Карты_пациентов"


def f(v):
    return "—" if v is None else f"{v:.2f}"


def chunks_md(chunks):
    out = []
    for c in chunks:
        where = f"раздел «{c['section']}»" if c["section"] else f"символы {c['start']}–{c['end']}"
        dates = f"{c['valid_from']} → {c['valid_to']}" if c["valid_to"] else f"с {c['valid_from']}"
        out.append(f"**#{c['rank']}** · score {c['score']} · `{c['file']}` · {where} · v{c['version']} · {dates} · status: `{c['status']}`\n")
        out.append("```text\n" + c["text"] + "\n```\n")
    return "\n".join(out)


def metrics_md(m):
    return ("| precision@3 | faithfulness | answer relevance |\n|---|---|---|\n"
            f"| {f(m['precision_at_k'])} | {f(m['faithfulness'])} | {f(m['answer_relevance'])} |\n")


def card(p, runs):
    fr, ctl = runs["faulty"], runs["control"]
    return f"""# {p['code']} · {p['name']}

## Жалоба
{p['complaint']}

## О системе
{SYSTEM_DESCRIPTION}

Метрики посчитаны по формулам RAGAS, судья — GigaChat: precision@3 — доля найденных фрагментов, которые судья счёл полезными для ответа (без эталонной разметки); faithfulness — доля утверждений ответа, подтверждённых фрагментами; answer relevance — близость вопросов, восстановленных по ответу, к исходному вопросу.

## Случай 1 — из жалобы

**Вопрос клиента:** {fr['question']}

**Ответ ассистента:**

> {fr['answer']}

### Найденные фрагменты (в порядке попадания в промпт)

{chunks_md(fr['chunks'])}
### Метрики

{metrics_md(fr['metrics'])}
## Случай 2 — ещё один вопрос из логов (та же конфигурация)

**Вопрос клиента:** {ctl['question']}

**Ответ ассистента:**

> {ctl['answer']}

{metrics_md(ctl['metrics'])}
<details><summary>Найденные фрагменты</summary>

{chunks_md(ctl['chunks'])}
</details>

## Ваш диагноз

| | |
|---|---|
| Этап отказа (chunking / retrieval / generation) | |
| Корень проблемы — одной фразой | |
| Три улики: что в ответе, фрагментах и метриках это подтверждает | 1. <br>2. <br>3. |
| Какую гипотезу вы отбросили и почему | |
| Лечение — конкретное изменение в пайплайне | |
| Как проверим, что помогло: какой вопрос и какая метрика | |
"""


OUT.mkdir(exist_ok=True)
for p in PATIENTS:
    data = json.loads((CASES_DIR / f"{p['id']}.json").read_text())
    name = f"{p['code'].replace(' ', '_')}_{p['name'].replace(' ', '_')}.md"
    (OUT / name).write_text(card(p, data["runs"]), encoding="utf-8")
    print(name)

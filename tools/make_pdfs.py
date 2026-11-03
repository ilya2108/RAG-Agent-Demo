"""PDF-версии материалов в ../PDF/: записки, задание, карты пациентов (шаги 1–4 демо), разбор (дефект → лечение, шаги 6–7).
Нужны скриншоты: python tools/shots.py (при запущенном server.py)."""
import re
import subprocess
import sys
from pathlib import Path

import markdown

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from clinic.patients import PATIENTS, SYSTEM_DESCRIPTION  # noqa: E402
from clinic.settings import ROOT  # noqa: E402

BASE = ROOT.parent
OUT = BASE / "PDF"
SHOTS = ROOT / ".cache" / "shots"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
STAGE = {"chunking": "Chunking failure", "retrieval": "Retrieval failure", "generation": "Generation failure"}

CSS = """
@page { size: A4; margin: 16mm 15mm; }
body { font: 10.5pt/1.5 -apple-system, "Helvetica Neue", Arial, sans-serif; color: #17212B; }
h1 { font-size: 21pt; margin: 0 0 8pt; } h2 { font-size: 15pt; margin: 18pt 0 6pt; } h3 { font-size: 12pt; margin: 14pt 0 4pt; }
h2.page { page-break-before: always; }
p, li { orphans: 3; widows: 3; } code { font-family: Menlo, monospace; font-size: 9pt; background: #EEF2F4; padding: 0 3px; border-radius: 3px; }
pre { background: #1E2730; color: #E6EDF3; padding: 8pt 10pt; border-radius: 6px; font-size: 8.5pt; white-space: pre-wrap; page-break-inside: avoid; }
pre code { background: none; color: inherit; }
table { border-collapse: collapse; width: 100%; margin: 6pt 0; font-size: 9.5pt; page-break-inside: avoid; }
td, th { border: 1px solid #D9DEE4; padding: 4pt 6pt; text-align: left; vertical-align: top; } th { background: #EEF2F4; }
figure { margin: 8pt 0 12pt; page-break-inside: avoid; text-align: center; }
figure img { max-width: 100%; max-height: 228mm; width: auto; border: 1px solid #D9DEE4; border-radius: 6px; }
figcaption { font-size: 9pt; color: #5E6B78; margin-top: 3pt; text-align: left; }
.badge { display: inline-block; padding: 2pt 9pt; border-radius: 10pt; font-weight: 700; font-size: 9.5pt; color: #fff; }
.chunking { background: #B7740B; } .retrieval { background: #2459C9; } .generation { background: #6E3FB8; }
.lead { color: #5E6B78; } blockquote { border-left: 3px solid #0E7C7B; margin: 6pt 0; padding: 2pt 10pt; color: #3B4856; }
"""


def md(text: str) -> str:
    return markdown.markdown(text, extensions=["tables", "fenced_code"])


def fig(pid: str, step: int, caption: str) -> str:
    return f'<figure><img src="{(SHOTS / f"{pid}_{step}.png").as_uri()}"><figcaption>{caption}</figcaption></figure>'


def pdf(name: str, title: str, body: str) -> None:
    html = OUT / "_html" / f"{name}.html"
    html.parent.mkdir(parents=True, exist_ok=True)
    html.write_text(f'<!doctype html><html lang="ru"><head><meta charset="utf-8"><title>{title}</title><style>{CSS}</style></head><body>{body}</body></html>', encoding="utf-8")
    target = OUT / f"{name}.pdf"
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", f"--user-data-dir={OUT / '_html' / 'profile'}",
                    f"--print-to-pdf={target}", html.as_uri()], capture_output=True, timeout=120)
    print(target.name, "ok" if target.exists() else "НЕ СОБРАН")


def page_breaks(html: str) -> str:
    return re.sub(r"<h2>", '<h2 class="page">', html)


def plain(md_name: str, name: str) -> None:
    text = (BASE / md_name).read_text(encoding="utf-8")
    pdf(name, name, md(text))


def card(p: dict) -> None:
    form = (ROOT.parent / "Карты_пациентов").glob(f"{p['code'].replace(' ', '_')}_*.md")
    form_md = next(form).read_text(encoding="utf-8").split("## Ваш диагноз", 1)[1]
    body = f"""<h1>{p['code']} · {p['name']}</h1>
<h3>Жалоба</h3><p>{p['complaint']}</p>
<h3>О системе</h3><p>{SYSTEM_DESCRIPTION}</p>
<p class="lead">Метрики посчитаны по формулам RAGAS, судья — GigaChat. precision@3 — доля найденных фрагментов, которые судья счёл полезными (без эталонной разметки);
faithfulness — доля утверждений ответа, подтверждённых фрагментами; answer relevance — близость вопросов, восстановленных по ответу, к исходному вопросу.</p>
<h2 class="page">Шаг 1. Вопрос и ответ</h2>{fig(p['id'], 1, 'Демо «RAG-клиника», шаг 1')}
<h2 class="page">Шаг 2. Найденные фрагменты</h2>{fig(p['id'], 2, 'Шаг 2: фрагменты в том порядке, в каком попали в промпт')}
<h2 class="page">Шаг 3. Метрики</h2>{fig(p['id'], 3, 'Шаг 3: метрики этого прогона')}
<h2 class="page">Шаг 4. Ещё один вопрос из логов</h2>{fig(p['id'], 4, 'Шаг 4: та же система, та же конфигурация')}
<h2 class="page">Ваш диагноз</h2>{md(form_md)}"""
    pdf(f"Карта_{p['code'].replace(' ', '_')}_{p['name'].replace(' ', '_')}", p["code"], body)


def review() -> None:
    text = (BASE / "03_Разбор_пациентов.md").read_text(encoding="utf-8")
    head, rest = text.split("## Пациент 0", 1)
    rest = "## Пациент 0" + rest
    sections, tail = rest.split("## Какой паттерн", 1)
    tail = "## Какой паттерн" + tail
    parts = re.split(r"(?m)^(?=## Пациент )", sections)
    body = md(head)
    for p, part in zip(PATIENTS, [x for x in parts if x.strip()]):
        title, part_body = part.split("\n", 1)
        defect, treatment = part_body.split("**Лечение:**", 1)
        d = p["diagnosis"]
        body += f'<h2 class="page">{title[3:]}</h2><p><span class="badge {d["stage"]}">{STAGE[d["stage"]]}</span> <b>{d["title"]}</b></p>'
        body += "<h3>Дефект</h3>" + md(defect) + fig(p["id"], 6, "Демо, шаг 6 «Диагноз»: улики и эталонная разметка чанков")
        body += '<h3>Лечение</h3>' + md("**Лечение:**" + treatment) + fig(p["id"], 7, "Демо, шаг 7 «Лечение»: конфигурация, ответ и метрики после исправления")
    body += page_breaks(md(tail))
    pdf("03_Разбор_пациентов", "Разбор пациентов", body)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    plain("01_Записки_спикера.md", "01_Записки_спикера")
    plain("02_Задание_командам.md", "02_Задание_командам")
    for p in PATIENTS:
        card(p)
    review()

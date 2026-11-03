"""Живые прогоны всех пациентов через GigaChat.

  python bake.py               — прогнать всех, записать в cases/live/, показать, воспроизвёлся ли симптом
  python bake.py zero A        — только выбранных
  python bake.py --promote     — после прогона заменить эталонные cases/<id>.json живыми,
                                 если симптом воспроизвёлся и лечение помогло
"""
import json
import sys

from clinic import pipeline
from clinic.patients import BY_ID, PATIENTS, public
from clinic.settings import CASES_DIR


def bake(p: dict) -> dict:
    runs = {
        "faulty": pipeline.run(p["question"], p["faulty"], gold=p["gold"]),
        "control": pipeline.run(p["control_question"], p["faulty"], gold=p["gold_control"]),
        "fixed": pipeline.run(p["question"], p["fixed"], gold=p["gold"]),
    }
    return {"patient": public(p), "runs": runs}


def fmt(m):
    f = "—" if m["faithfulness"] is None else f"{m['faithfulness']:.2f}"
    return f"P@k {m['precision_at_k']:.2f}  F {f}  AR {m['answer_relevance']:.2f}  gold P@k {m['gold_precision_at_k']:.2f}"


def main(argv):
    promote = "--promote" in argv
    ids = [a for a in argv if not a.startswith("--")] or [p["id"] for p in PATIENTS]
    live_dir = CASES_DIR / "live"
    live_dir.mkdir(parents=True, exist_ok=True)
    ok_all = True
    for pid in ids:
        p = BY_ID[pid]
        print(f"\n=== {p['code']} · {p['name']} ===")
        data = bake(p)
        (live_dir / f"{pid}.json").write_text(json.dumps(data, ensure_ascii=False, indent=1))
        faulty, fixed = data["runs"]["faulty"], data["runs"]["fixed"]
        sym, cure = p["symptom"](faulty["answer"]), p["cure"](fixed["answer"])
        print("с дефектом :", fmt(faulty["metrics"]))
        print("  ответ    :", faulty["answer"][:300].replace("\n", " "))
        print("после      :", fmt(fixed["metrics"]))
        print("  ответ    :", fixed["answer"][:300].replace("\n", " "))
        print(f"симптом воспроизвёлся: {'да' if sym else 'НЕТ'} · лечение помогло: {'да' if cure else 'НЕТ'}")
        ok_all &= sym and cure
        if promote and sym and cure:
            (CASES_DIR / f"{pid}.json").write_text(json.dumps(data, ensure_ascii=False, indent=1))
            print("→ эталонный прогон заменён живым")
    print("\nВсе симптомы воспроизвелись." if ok_all else
          "\nЕсть расхождения: для этих пациентов оставьте эталонные прогоны (cases/<id>.json).")


if __name__ == "__main__":
    main(sys.argv[1:])

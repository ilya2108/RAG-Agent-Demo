"""Локальный веб-интерфейс RAG-клиники: python server.py → http://localhost:8765"""
import json
import os
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from clinic import pipeline, prompts, settings
from clinic.corpus import build_chunks, load_docs
from clinic.patients import BY_ID, PATIENTS
from clinic.settings import CASES_DIR, ROOT

WEB = ROOT / "web"


def load_case(pid: str) -> dict:
    data = json.loads((CASES_DIR / f"{pid}.json").read_text())
    live = CASES_DIR / "live" / f"{pid}.json"
    data["live"] = json.loads(live.read_text())["runs"] if live.exists() else {}
    return data


def save_live(pid: str, variant: str, trace: dict) -> None:
    path = CASES_DIR / "live" / f"{pid}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(path.read_text()) if path.exists() else {"patient": {"id": pid}, "runs": {}}
    data["runs"][variant] = trace
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1))


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        raw = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, fmt, *args):
        print("  ", self.command, self.path, args[1] if len(args) > 1 else "")

    def do_GET(self):
        self.path = self.path.split("?", 1)[0]
        if self.path in ("/", "/index.html"):
            return self._send(200, (WEB / "index.html").read_bytes(), "text/html; charset=utf-8")
        if self.path == "/api/state":
            return self._send(200, {
                "live": settings.live_enabled(),
                "models": {"generation": settings.MODEL, "judge": settings.JUDGE_MODEL, "embeddings": settings.EMBEDDINGS_NAME},
                "defaults": pipeline.DEFAULT,
                "prompts": prompts.PROMPTS,
                "cases": [load_case(p["id"]) for p in PATIENTS],
            })
        if self.path == "/api/corpus":
            return self._send(200, {
                "docs": [{k: getattr(d, k) for k in ("doc_id", "title", "version", "status", "valid_from", "valid_to", "file", "body")}
                         for d in load_docs()],
                "chunks": {s: [c.to_dict() for c in build_chunks(s)] for s in ("structural", "fixed")},
            })
        self._send(404, {"error": "not found"})

    def do_POST(self):
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            if not settings.live_enabled():
                return self._send(400, {"error": "Нет ключа GigaChat: заполните .env и перезапустите сервер"})
            if self.path == "/api/run":
                p, variant = BY_ID[body["patient"]], body["variant"]
                question = p["control_question"] if variant == "control" else p["question"]
                cfg = p["fixed"] if variant == "fixed" else p["faulty"]
                gold = p["gold_control"] if variant == "control" else p["gold"]
                trace = pipeline.run(question, cfg, gold=gold)
                save_live(p["id"], variant, trace)
                return self._send(200, trace)
            if self.path == "/api/lab":
                return self._send(200, pipeline.run(body["question"], body.get("config")))
            self._send(404, {"error": "not found"})
        except Exception as e:  # noqa: BLE001 — ошибку показываем в интерфейсе
            traceback.print_exc()
            self._send(500, {"error": f"{type(e).__name__}: {e}"})


if __name__ == "__main__":
    url = f"http://localhost:{settings.PORT}"
    mode = "живой режим (GigaChat)" if settings.live_enabled() else "офлайн: только сохранённые прогоны"
    print(f"RAG-клиника: {url}  ·  {mode}")
    if not os.getenv("CLINIC_NO_BROWSER"):
        webbrowser.open(url)
    ThreadingHTTPServer(("127.0.0.1", settings.PORT), Handler).serve_forever()

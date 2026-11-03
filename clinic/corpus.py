"""Загрузка базы знаний и две стратегии чанкинга."""
import re
from dataclasses import dataclass, asdict
from functools import lru_cache

from .settings import CORPUS_DIR


@dataclass
class Doc:
    doc_id: str
    title: str
    version: str
    status: str
    valid_from: str
    valid_to: str
    file: str
    body: str


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    section: str | None
    text: str          # то, что видит модель
    embed_text: str    # то, что уходит в эмбеддинг
    start: int
    end: int
    version: str
    status: str
    valid_from: str
    valid_to: str
    file: str

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("embed_text")
        return d


def _parse(path) -> Doc:
    raw = path.read_text(encoding="utf-8")
    _, meta_block, body = raw.split("---", 2)
    meta = dict(line.split(":", 1) for line in meta_block.strip().splitlines())
    meta = {k.strip(): v.strip() for k, v in meta.items()}
    return Doc(
        doc_id=meta["doc_id"], title=meta["title"], version=meta.get("version", ""),
        status=meta.get("status", "actual"), valid_from=meta.get("valid_from", ""),
        valid_to=meta.get("valid_to", ""), file=path.name, body=body.strip(),
    )


@lru_cache
def load_docs() -> tuple[Doc, ...]:
    return tuple(_parse(p) for p in sorted(CORPUS_DIR.glob("*.md")))


def _chunk(doc: Doc, n: int, section, text, embed_text, start, end) -> Chunk:
    return Chunk(
        chunk_id=f"{doc.doc_id}#{n}", doc_id=doc.doc_id, title=doc.title, section=section,
        text=text, embed_text=embed_text, start=start, end=end, version=doc.version,
        status=doc.status, valid_from=doc.valid_from, valid_to=doc.valid_to, file=doc.file,
    )


def fixed_chunks(doc: Doc, size: int = 520, overlap: int = 0) -> list[Chunk]:
    """Окно фиксированной длины в символах, граница сдвигается к ближайшему пробелу.
    Структуру документа не учитывает — как наивный сплиттер по умолчанию."""
    body, out, pos, n = doc.body, [], 0, 0
    while pos < len(body):
        end = min(pos + size, len(body))
        if end < len(body):
            space = body.rfind(" ", pos + size // 2, end)
            nl = body.rfind("\n", pos + size // 2, end)
            end = max(space, nl) if max(space, nl) > pos else end
        text = body[pos:end].strip()
        if text:
            out.append(_chunk(doc, n, None, text, text, pos, end))
            n += 1
        if end >= len(body):
            break
        pos = max(end - overlap, pos + 1)
    return out


def structural_chunks(doc: Doc, max_chars: int = 1500) -> list[Chunk]:
    """Чанк = раздел второго уровня целиком (таблицы не режутся).
    В начало добавляется заголовок «Документ › Раздел» — контекст для модели и для эмбеддинга."""
    body = doc.body
    heads = [(m.start(), m.group(1).strip()) for m in re.finditer(r"^##\s+(.+)$", body, re.M)]
    if not heads:
        heads = [(0, None)]
    bounds = [(0, heads[0][0], None)] if heads[0][0] > 0 else []
    for i, (start, name) in enumerate(heads):
        stop = heads[i + 1][0] if i + 1 < len(heads) else len(body)
        bounds.append((start, stop, name))
    out, n = [], 0
    for start, stop, name in bounds:
        part = body[start:stop].strip()
        part = re.sub(r"^#{1,2}\s+.+\n?", "", part).strip()  # заголовки уже в шапке чанка
        if not part:
            continue
        header = f"{doc.title} › {name}" if name else doc.title
        text = f"[{header}]\n{part}"
        out.append(_chunk(doc, n, name, text, text, start, stop))
        n += 1
    return out


@lru_cache
def build_chunks(strategy: str, size: int = 520, overlap: int = 0) -> tuple[Chunk, ...]:
    chunks = []
    for doc in load_docs():
        chunks += fixed_chunks(doc, size, overlap) if strategy == "fixed" else structural_chunks(doc)
    return tuple(chunks)

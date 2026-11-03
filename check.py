"""Проверка ключа и доступа к GigaChat: токен, список моделей, чат, эмбеддинги."""
import sys

from clinic import llm, settings


def main() -> int:
    if not settings.live_enabled():
        print("✗ GIGACHAT_CREDENTIALS пуст. Скопируйте .env.example в .env и вставьте ключ авторизации.")
        return 1
    print(f"scope={settings.SCOPE}  model={settings.MODEL}  judge={settings.JUDGE_MODEL}  emb={settings.EMBEDDINGS_NAME}")
    try:
        llm.client().get_token()
        print("✓ токен получен")
        names = [m.id_ for m in llm.client().get_models().data]
        print("✓ доступные модели:", ", ".join(names))
        wanted = {settings.MODEL, settings.JUDGE_MODEL} | ({settings.EMBEDDINGS_MODEL} if settings.EMBEDDINGS_BACKEND == "gigachat" else set())
        for want in wanted:
            if want not in names:
                print(f"! модели {want} нет в списке — поправьте имя в .env")
        print("✓ чат:", llm.chat("Отвечай одним словом.", "Столица Татарстана?")[:60])
        vec = llm.embed(["проверка эмбеддингов"])[0]
        print(f"✓ эмбеддинги: размерность {len(vec)}")
    except Exception as e:  # noqa: BLE001 — показываем любую ошибку как есть
        print("✗ ошибка:", type(e).__name__, e)
        if "402" in str(e):
            print("  402 — на ключе нет пакета для этой модели. Для эмбеддингов: EMBEDDINGS_BACKEND=local в .env")
        print("  401 — неверный ключ или scope; SSL — поставьте GIGACHAT_VERIFY_SSL_CERTS=false "
              "или укажите сертификат Минцифры в GIGACHAT_CA_BUNDLE_FILE")
        return 1
    print("Готово. Дальше: python bake.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())

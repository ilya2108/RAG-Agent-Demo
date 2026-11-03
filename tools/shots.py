"""Скриншоты шагов демо для PDF: нужен запущенный server.py. python tools/shots.py → .cache/shots/<id>_<шаг>.png"""
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image, ImageChops

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from clinic.settings import PORT, ROOT  # noqa: E402

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
OUT = ROOT / ".cache" / "shots"
IDS = ["zero", "A", "B", "C", "D"]


def shot(pid: str, stage: int) -> Path:
    path = OUT / f"{pid}_{stage}.png"
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--force-device-scale-factor=2",
                    f"--user-data-dir={OUT / ('profile_' + pid + str(stage))}", "--window-size=1200,4200",
                    "--virtual-time-budget=4000", f"--screenshot={path}", f"http://localhost:{PORT}/?shot#{pid}/{stage}"],
                   capture_output=True, timeout=90)  # Chrome иногда выходит с кодом 2, но файл пишет
    if not path.exists():
        return shot(pid, stage)
    img = Image.open(path).convert("RGB")
    bg = Image.new("RGB", img.size, img.getpixel((img.width - 2, img.height - 2)))
    box = ImageChops.difference(img, bg).getbbox()
    if box:
        img = img.crop((0, 0, img.width, min(img.height, box[3] + 40)))
    img.save(path)
    return path


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [(p, s) for p in IDS for s in range(1, 8)]
    with ThreadPoolExecutor(2) as ex:
        for path in ex.map(lambda a: shot(*a), jobs):
            print(path.name, Image.open(path).size)

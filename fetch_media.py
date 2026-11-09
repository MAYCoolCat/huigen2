#!/usr/bin/env python3
"""
Докачивает свежие фото с Pinterest по темам деньги / роскошь.
Работает без API-ключей.

Фото складываются в указанную папку и попадают в монтаж photos_to_video.py.

Примеры:
    python3 fetch_media.py /Users/yaroslav/Desktop/huigen
    python3 fetch_media.py ./huigen --photos 80 --categories money,luxury
"""
import argparse
import hashlib
import json
import random
import re
import sys
import time
import urllib.parse
from pathlib import Path

import requests

CATEGORIES = {
    "money": ["money aesthetic", "cash money", "gold bars", "counting money",
              "stack of money", "stock market", "bitcoin", "luxury wallet cash"],
    "luxury": ["luxury lifestyle", "luxury villa", "yacht life", "private jet",
               "penthouse view", "luxury watch", "luxury hotel", "rich lifestyle",
               "mansion pool", "champagne luxury"],
}

PIN_ID_RE = re.compile(r"/pin/(\d+)")
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")


# ───────────────────────── Pinterest ─────────────────────────

def _pinterest_client():
    try:
        from pinterest_downloader import Pinterest
    except ImportError:
        sys.exit("Нужна библиотека pinterest-downloader. Установи:\n"
                 "    pip install pinterest-downloader")
    return Pinterest(timeout=40)   # таймаут побольше — Pinterest любит тормозить


def pinterest_photos(client, query, max_pages, debug=False):
    """
    Возвращает список пинов-изображений по запросу, проходя до max_pages страниц.
    Использует search_all, чтобы не возиться с bookmark вручную.
    """
    try:
        res = client.search_all(query, max_pages=max_pages)
    except Exception as e:
        print(f"  Pinterest: исключение при поиске «{query}»: {e}", file=sys.stderr)
        return []

    if not res.get("ok"):
        msg = (res.get("error") or {}).get("message", "unknown error")
        print(f"  Pinterest: поиск «{query}» -> ok=False, ошибка: {msg}", file=sys.stderr)
        if debug:
            print(f"  DEBUG raw: {res}", file=sys.stderr)
        return []

    pins = res.get("pins", [])
    if debug:
        print(f"  DEBUG «{query}»: получено пинов {len(pins)}", file=sys.stderr)

    # Оставляем только изображения
    return [pin for pin in pins if pin.get("media_type") == "image"]


def image_url_from_pin(pin):
    """Достаёт прямой URL оригинала изображения из структуры пина."""
    images = pin.get("images") or {}
    for key in ("orig", "originals", "736x", "564x"):
        node = images.get(key)
        if isinstance(node, dict) and node.get("url"):
            return node["url"]
    return None


def stable_key(pin, img_url):
    """
    Стабильный ключ для .downloaded.json:
    числовой ID из поля url, если он есть, иначе — хэш URL изображения.
    """
    url = pin.get("url") or ""
    m = PIN_ID_RE.search(url)
    if m:
        return m.group(1)
    return "h_" + hashlib.sha1(img_url.encode()).hexdigest()[:16]


def download_image(url, folder, key):
    """Скачивает изображение по прямому URL и сохраняет в folder с именем <key><ext>."""
    r = requests.get(url, headers={"User-Agent": UA}, timeout=60, stream=True)
    r.raise_for_status()

    # Расширение: из пути URL, иначе из Content-Type
    path_ext = Path(urllib.parse.urlparse(url).path).suffix.lower()
    if path_ext in (".jpg", ".jpeg", ".png", ".webp"):
        ext = path_ext
    else:
        ctype = (r.headers.get("Content-Type") or "").lower()
        if "png" in ctype:
            ext = ".png"
        elif "webp" in ctype:
            ext = ".webp"
        else:
            ext = ".jpg"

    dest = folder / f"{key}{ext}"
    tmp = dest.with_name(dest.name + ".part")
    with open(tmp, "wb") as f:
        for chunk in r.iter_content(chunk_size=65536):
            if chunk:
                f.write(chunk)
    tmp.rename(dest)
    return dest


# ───────────────────────── main ─────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Докачка фото с Pinterest по категориям")
    ap.add_argument("folder", type=Path, help="папка, куда складывать")
    ap.add_argument("--photos", type=int, default=40, help="сколько новых фото (40)")
    ap.add_argument("--categories", default=",".join(CATEGORIES),
                    help=f"через запятую: {', '.join(CATEGORIES)}")
    ap.add_argument("--max-page", type=int, default=5, help="макс. страниц на запрос (5)")
    ap.add_argument("--debug", action="store_true", help="печатать сырые ответы Pinterest")
    ap.add_argument("--seed", type=int)
    args = ap.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
    cats = [c.strip() for c in args.categories.split(",") if c.strip()]
    bad = [c for c in cats if c not in CATEGORIES]
    if bad:
        sys.exit(f"Неизвестные категории: {bad}. Доступны: {list(CATEGORIES)}")

    client = _pinterest_client()

    folder = args.folder
    folder.mkdir(parents=True, exist_ok=True)
    state_path = folder / ".downloaded.json"
    seen_ids = set(json.loads(state_path.read_text())) if state_path.exists() else set()
    cache = {}

    def save_state():
        state_path.write_text(json.dumps(sorted(seen_ids)))

    def fetch_photo():
        """Одна попытка достать новое фото."""
        for _ in range(40):
            q = random.choice(CATEGORIES[random.choice(cats)])
            ck = q
            try:
                if ck not in cache:
                    cache[ck] = pinterest_photos(client, q, args.max_page, args.debug)
                    time.sleep(1.5)   # пауза между запросами к Pinterest
            except Exception as e:
                print(f"  Pinterest: ошибка запроса «{q}» ({e})", file=sys.stderr)
                continue

            # Отбираем пины с валидным URL изображения и ещё не скачанные
            candidates = []
            for pin in cache[ck]:
                img_url = image_url_from_pin(pin)
                if not img_url:
                    continue
                key = stable_key(pin, img_url)
                if key in seen_ids:
                    continue
                candidates.append((pin, img_url, key))

            if not candidates:
                continue

            pin, img_url, key = random.choice(candidates)
            try:
                dest = download_image(img_url, folder, key)
            except Exception as e:
                print(f"  не скачалось {key} ({img_url}) -> {e}", file=sys.stderr)
                seen_ids.add(key)   # чтобы не зацикливаться на битом URL
                save_state()
                continue

            seen_ids.add(key)
            save_state()
            print(f"  + photo: {key}  [{q}]  -> {dest.name}")
            return True

        return False

    print(f"Источник: pinterest | категории: {', '.join(cats)} | max_pages: {args.max_page}")
    got = 0
    for _ in range(args.photos):
        got += fetch_photo()

    print(f"Готово: фото {got}/{args.photos} -> {folder.resolve()}")


if __name__ == "__main__":
    main()

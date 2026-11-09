#!/usr/bin/env python3
"""
Делает N горизонтальных видео 16:9 из фото одной папки.
В каждом видео: свой случайный порядок фото (0.09 сек на фото) и своя
мотивационная цитата, сгенерированная через Ollama.

Требования:
    pip install pillow
    ffmpeg в PATH, запущенный Ollama (ollama serve) со скачанной моделью

Примеры:
    python photos_to_video.py ./photos
    python photos_to_video.py ./photos --count 10 --model qwen2.5:7b
    python photos_to_video.py ./photos --topic "спорт и упорство" --height 2160
    python photos_to_video.py ./photos --quote "Своя цитата"   # одна и та же во всех
    python photos_to_video.py ./photos --quotes-file quotes100.txt   # цитаты из файла
"""
import argparse
import functools
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor
import json
import random
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

from PIL import Image, ImageColor, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

# запасные цитаты, если Ollama недоступна
FALLBACK_QUOTES = [
    "Дисциплина — это мост между целями и результатами.",
    "Не жди идеального момента. Бери момент и делай его идеальным.",
    "Маленькие шаги каждый день приводят к большим переменам.",
    "Ты сильнее, чем думаешь, и способнее, чем кажется.",
    "Путь в тысячу миль начинается с одного шага.",
    "Успех — это сумма небольших усилий, повторяемых изо дня в день.",
    "Не бойся идти медленно. Бойся стоять на месте.",
    "Мечты не работают, пока не работаешь ты.",
    "Делай сегодня то, за что завтра скажешь себе спасибо.",
    "Сложно — значит, ты растёшь.",
]

FONT_CANDIDATES = [
    # положи Montserrat-Light.ttf / Inter-Light.ttf и т.п. в папку fonts/ рядом со скриптом —
    # он подхватится первым. Ниже — системные запасные (с кириллицей).
    "/System/Library/Fonts/SFNS.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "C:/Windows/Fonts/segoeuil.ttf",
    "C:/Windows/Fonts/arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
]


# ───────────────────────── Ollama ─────────────────────────

def clean_quote(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        return ""
    q = lines[0]
    q = re.sub(r"^\s*(\d+[.)]|[-•*])\s*", "", q)          # нумерация/маркеры
    q = q.strip().strip('"«»“”„\'').strip()
    return q


def ollama_quote(url, model, topic, language, max_words, timeout):
    prompt = (
        f"Придумай одну короткую оригинальную мотивационную цитату на {language} языке "
        f"(не больше {max_words} слов). Тема: {topic}. "
        "Ответь только текстом цитаты: без кавычек, без автора, без пояснений."
    )
    body = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 1.0, "seed": random.randint(0, 2**31 - 1)},
    }).encode()
    req = urllib.request.Request(
        url.rstrip("/") + "/api/generate", data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.load(r)
    return clean_quote(data.get("response", ""))


def generate_quotes(count, args):
    quotes, attempts = [], 0
    fallback = FALLBACK_QUOTES[:]
    random.shuffle(fallback)
    use_ollama = True

    while len(quotes) < count:
        q = ""
        if use_ollama and attempts < count * 5:
            attempts += 1
            try:
                q = ollama_quote(args.ollama_url, args.model, args.topic,
                                 args.language, args.max_words, args.ollama_timeout)
            except urllib.error.HTTPError as e:
                print(f"Ollama вернула ошибку {e.code} (модель '{args.model}' скачана? "
                      f"`ollama pull {args.model}`). Беру запасные цитаты.", file=sys.stderr)
                use_ollama = False
            except Exception as e:
                print(f"Ollama недоступна ({e}). Беру запасные цитаты.", file=sys.stderr)
                use_ollama = False
            else:
                if not q or len(q.split()) > args.max_words * 2:
                    q = ""
                elif q.lower() in (x.lower() for x in quotes):
                    q = ""
                else:
                    print(f"  цитата {len(quotes) + 1}/{count}: {q}")
        else:
            use_ollama = False
            if not fallback:
                fallback = FALLBACK_QUOTES[:]
                random.shuffle(fallback)
            q = fallback.pop()
            if q in quotes and len(quotes) < len(FALLBACK_QUOTES):
                continue
        if q:
            quotes.append(q)
    return quotes


def load_quotes(path):
    """Цитаты из текстового файла: по одной на строку, нумерация '1.' и пустые строки игнорируются."""
    if not path.is_file():
        sys.exit(f"Файл с цитатами не найден: {path}")
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = re.sub(r"^\s*\d+[.)]\s*", "", line).strip()
        if line and not line.startswith("#"):
            out.append(line)
    if not out:
        sys.exit(f"В файле {path} нет цитат.")
    return out


def pick_from_pool(pool, n):
    """n цитат из списка: без повторов, пока список не кончился (потом перемешиваем заново)."""
    out = []
    while len(out) < n:
        batch = pool[:]
        random.shuffle(batch)
        out += batch[: n - len(out)]
    return out


AUDIO_EXTS = {".mp3", ".m4a", ".wav", ".flac", ".aac", ".ogg", ".opus"}


def collect_audio(path):
    """Файл -> [файл]; папка -> все аудиофайлы в ней."""
    if path is None:
        return []
    if not path.exists():
        sys.exit(f"Музыка не найдена: {path}")
    if path.is_file():
        return [path]
    tracks = sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in AUDIO_EXTS)
    if not tracks:
        sys.exit(f"В папке {path} нет аудиофайлов ({', '.join(sorted(AUDIO_EXTS))}).")
    return tracks


# ───────────────────────── Рендер ─────────────────────────

def find_font(user_font):
    if user_font:
        if Path(user_font).exists():
            return user_font
        sys.exit(f"Шрифт не найден: {user_font}")
    local = sorted(
        p for p in (Path(__file__).resolve().parent / "fonts").glob("*")
        if p.suffix.lower() in (".ttf", ".otf")
    ) if (Path(__file__).resolve().parent / "fonts").is_dir() else []
    for p in [str(x) for x in local] + FONT_CANDIDATES:
        if Path(p).exists():
            try:
                ImageFont.truetype(p, 20)
                return p
            except Exception:
                continue
    sys.exit("Не найден шрифт с кириллицей. Положи .ttf в папку fonts/ или укажи --font")


def text_w(s, font, track):
    return sum(font.getlength(c) for c in s) + track * max(len(s) - 1, 0)


def wrap_tracked(text, font, track, max_w):
    lines, line = [], ""
    for word in text.split():
        trial = f"{line} {word}".strip()
        if text_w(trial, font, track) <= max_w or not line:
            line = trial
        else:
            lines.append(line)
            line = word
    if line:
        lines.append(line)
    return lines


def build_overlay(text, W, H, font_path, font_size, position,
                  uppercase=False, tracking=0.04, glow=0.5, glow_color="#ffffff",
                  boldness=0.025, scrim=0.5):
    """Читаемый минималистичный текст: утолщение тонкого шрифта, тёмная мягкая подложка,
    чёткая тень и лёгкое свечение."""
    if uppercase:
        text = text.upper()
    max_w, max_h = int(W * 0.72), int(H * 0.4)
    size = font_size or int(H * 0.068)
    while True:
        font = ImageFont.truetype(font_path, size)
        track = size * tracking
        lines = wrap_tracked(text, font, track, max_w)
        line_h = int(size * 1.4)
        total_h = line_h * len(lines)
        if total_h <= max_h or size <= 20:
            break
        size -= 2
    stroke = max(0, round(size * boldness))     # «утолщает» тонкий шрифт

    if position == "top":
        y0 = int(H * 0.08)
    elif position == "center":
        y0 = (H - total_h) // 2
    else:
        y0 = int(H * 0.92) - total_h

    mask = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(mask)
    x_min, x_max = W, 0
    for i, line in enumerate(lines):
        lw = text_w(line, font, track)
        x = (W - lw) / 2
        x_min, x_max = min(x_min, x), max(x_max, x + lw)
        y = y0 + i * line_h
        for ch in line:
            d.text((x, y), ch, font=font, fill=255, stroke_width=stroke, stroke_fill=255)
            x += font.getlength(ch) + track

    def layer(rgb, alpha):
        l = Image.new("RGBA", (W, H), tuple(rgb) + (0,))
        l.putalpha(alpha)
        return l

    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))

    # 1) мягкая тёмная подложка под всем блоком текста
    if scrim > 0:
        pad = size * 0.9
        box = Image.new("L", (W, H), 0)
        ImageDraw.Draw(box).rounded_rectangle(
            (x_min - pad, y0 - pad * 0.4, x_max + pad, y0 + total_h + pad * 0.1),
            radius=int(size * 0.8), fill=255)
        box = box.filter(ImageFilter.GaussianBlur(size * 0.7)).point(
            lambda v: int(v * min(1.0, scrim) * 0.8))
        overlay = Image.alpha_composite(overlay, layer((0, 0, 0), box))

    # 2) чёткая тень у самих букв
    off = max(2, size // 22)
    sh = Image.new("L", (W, H), 0)
    sh.paste(mask, (off, off))
    sh = sh.filter(ImageFilter.GaussianBlur(max(1.5, size * 0.04))).point(
        lambda v: min(230, int(v * 1.6)))
    overlay = Image.alpha_composite(overlay, layer((0, 0, 0), sh))

    # 3) лёгкое свечение
    if glow > 0:
        gc = ImageColor.getrgb(glow_color)
        wide = mask.filter(ImageFilter.GaussianBlur(size * 0.30)).point(lambda v: min(255, int(v * 2.2 * glow)))
        near = mask.filter(ImageFilter.GaussianBlur(size * 0.08)).point(lambda v: min(255, int(v * 1.6 * glow)))
        overlay = Image.alpha_composite(overlay, layer(gc, wide))
        overlay = Image.alpha_composite(overlay, layer(gc, near))

    # 4) сам текст
    return Image.alpha_composite(overlay, layer((255, 255, 255), mask))


@functools.lru_cache(maxsize=4)
def build_shade(W, H, darken, vignette):
    """Чёрный слой: ровное затемнение + виньетка, темнеющая к углам (считается один раз)."""
    base = int(max(0.0, min(1.0, darken)) * 255)
    strength = max(0.0, min(1.0, vignette))
    # радиальный градиент: 0 в центре, 255 в углах (середины краёв ~180)
    g = Image.radial_gradient("L").resize((W, H), Image.BICUBIC)

    def vig(v):
        t = max(0.0, (v / 255 - 0.4) / 0.6)
        return int(255 * strength * t ** 2)

    v_alpha = g.point([vig(v) for v in range(256)])
    # складываем затемнение и виньетку: 1-(1-d)*(1-v)
    total = v_alpha.point([int(255 - (255 - base) * (255 - a) / 255) for a in range(256)])
    shade = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    shade.putalpha(total)
    return shade


def render_frame(path, W, H, overlay, fit="stretch", shade=None):
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    if fit == "stretch":
        # растягиваем фото точно на весь кадр (пропорции искажаются)
        bg = img.resize((W, H), Image.LANCZOS)
    elif fit == "cover":
        # заполняем кадр целиком, лишнее обрезается
        bg = ImageOps.fit(img, (W, H), Image.LANCZOS)
    else:
        # contain: фото целиком по центру, фон — размытая копия
        small = ImageOps.fit(img, (max(W // 8, 2), max(H // 8, 2)), Image.BILINEAR)
        bg = small.filter(ImageFilter.GaussianBlur(8)).resize((W, H), Image.BICUBIC)
        bg = ImageEnhance.Brightness(bg).enhance(0.45)
        fg = ImageOps.contain(img, (W, H), Image.LANCZOS)
        bg.paste(fg, ((W - fg.width) // 2, (H - fg.height) // 2))
    if shade is not None:
        bg.paste(shade, (0, 0), shade)   # затемнение + виньетка под текстом
    if overlay is not None:
        bg.paste(overlay, (0, 0), overlay)
    return bg.tobytes()


def collect_images(folder, recursive):
    it = folder.rglob("*") if recursive else folder.iterdir()
    good = []
    for p in sorted(it):
        if p.is_file() and p.suffix.lower() in EXTS:
            try:
                with Image.open(p) as im:
                    im.verify()
                good.append(p)
            except Exception:
                print(f"Пропускаю битый файл: {p.name}", file=sys.stderr)
    return good


def make_video(images, out_path, quote, args, W, H, font_path, audio=None):
    overlay = build_overlay(quote, W, H, font_path, args.font_size, args.position,
                            args.uppercase, args.tracking, args.glow, args.glow_color,
                            args.boldness, args.scrim)
    shade = build_shade(W, H, args.darken, args.vignette) if (args.darken > 0 or args.vignette > 0) else None
    n = len(images)

    # Текст накладывается уже ПОСЛЕ обработки (размытие / сглаживание), чтобы он не «плыл»
    png = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    png.close()
    overlay.save(png.name)

    in_fps = str(args.fps)
    total_frames = round(n * args.duration * args.fps)
    vid_dur = total_frames / args.fps

    chain = []
    if args.blur > 0:
        chain.append(f"gblur=sigma={args.blur}")
    if args.smooth >= 2:
        chain.append(f"tmix=frames={args.smooth}")
    head = "[0:v]" + ",".join(chain) + "[bg];" if chain else ""
    src_lbl = "[bg]" if chain else "[0:v]"
    fc = f"{head}{src_lbl}[1:v]overlay=shortest=1:format=auto,format=yuv420p[v]"
    if audio:
        af = [f"volume={args.audio_volume}"]
        if args.audio_fade > 0:
            af.append(f"afade=t=out:st={max(0.0, vid_dur - args.audio_fade):.3f}:d={args.audio_fade}")
        fc += f";[2:a]{','.join(af)}[a]"

    cmd = ["ffmpeg", "-y", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", in_fps, "-i", "-",
           "-loop", "1", "-framerate", str(args.fps), "-i", png.name]
    if audio:
        cmd += ["-stream_loop", "-1"]                    # короткий трек зацикливается
        if args.audio_start > 0:
            cmd += ["-ss", str(args.audio_start)]
        cmd += ["-i", str(audio)]
    cmd += ["-filter_complex", fc, "-map", "[v]"]
    if audio:
        cmd += ["-map", "[a]", "-c:a", "aac", "-b:a", "256k"]
    cmd += ["-t", f"{vid_dur:.3f}", "-r", str(args.fps), "-c:v", "libx264", "-preset", args.preset, "-crf", str(args.crf),
            "-pix_fmt", "yuv420p", "-profile:v", "high",
            "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
            "-movflags", "+faststart", str(out_path)]

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    try:
        cur_idx, buf = -1, b""
        for i in range(total_frames):
            idx = min(int(i / args.fps / args.duration + 1e-9), n - 1)
            if idx != cur_idx:
                buf = render_frame(images[idx], W, H, None, args.fit, shade)
                cur_idx = idx
            proc.stdin.write(buf)
    except BrokenPipeError:
        pass
    finally:
        if proc.stdin:
            proc.stdin.close()
        proc.wait()
        os.unlink(png.name)
    return proc.returncode == 0


def run_task(t):
    order, out, quote, args, W, H, font_path, label, audio = t
    music = f" | музыка: {audio.name}" if audio else ""
    print(f"{label} {out.name}: «{quote}»{music}", flush=True)
    ok = make_video(order, out, quote, args, W, H, font_path, audio)
    print(f"  {'готово' if ok else 'ОШИБКА ffmpeg'}: {out.name}", flush=True)
    return ok


# ───────────────────────── main ─────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Фото -> N видео 16:9 с цитатами от Ollama")
    ap.add_argument("folder", type=Path, help="папка с фотографиями")
    ap.add_argument("-o", "--outdir", type=Path, default=Path("videos"), help="куда сохранять")
    ap.add_argument("--count", type=int, default=10, help="сколько видео сделать (10)")

    g = ap.add_argument_group("Ollama / цитаты")
    g.add_argument("--model", default="llama3.1", help="модель Ollama (llama3.1)")
    g.add_argument("--ollama-url", default="http://localhost:11434")
    g.add_argument("--ollama-timeout", type=int, default=120)
    g.add_argument("--topic", default="деньги, богатство, успех, дисциплина")
    g.add_argument("--language", default="русском", help="язык цитат: «русском», «английском»…")
    g.add_argument("--max-words", type=int, default=14)
    g.add_argument("--quote", help="фиксированная цитата для всех видео (без Ollama)")
    g.add_argument("--quotes-file", type=Path,
                   help="текстовый файл с цитатами (по одной на строку) вместо Ollama")
    g.add_argument("--mix", action="store_true",
                   help="с --quotes-file: половина цитат из файла, половина от Ollama")

    v = ap.add_argument_group("видео")
    v.add_argument("--duration", type=float, default=0.09, help="сек на одно фото (0.09)")
    v.add_argument("--fps", type=int, default=30)
    v.add_argument("--height", type=int, default=1080, choices=[720, 1080, 1440, 2160])
    v.add_argument("--fit", choices=["stretch", "cover", "contain"], default="cover",
                   help="cover: заполнить кадр без искажений, лишнее обрезается (по умолчанию); "
                        "contain: фото целиком + размытый фон; stretch: растянуть (искажает)")
    v.add_argument("--darken", type=float, default=0.3, help="затемнение кадров 0..1 (0.3)")
    v.add_argument("--vignette", type=float, default=0.7, help="сила виньетки по углам 0..1 (0.7)")
    v.add_argument("--smooth", type=int, default=0,
                   help="плавность: смешивать N соседних кадров, 0 = выкл (рекомендую 2)")
    v.add_argument("--blur", type=float, default=0.0,
                   help="лёгкое размытие кадров (sigma, например 1.5), 0 = выкл")
    v.add_argument("--preset", default="slow", help="пресет x264: ultrafast…veryslow (slow)")
    v.add_argument("--jobs", type=int, default=1, help="сколько видео собирать параллельно (1)")
    v.add_argument("--crf", type=int, default=14, help="меньше = лучше качество (14)")
    v.add_argument("--position", choices=["top", "center", "bottom"], default="center")
    v.add_argument("--font", help="путь к .ttf шрифту с кириллицей")
    v.add_argument("--font-size", type=int, help="размер шрифта в px (авто)")
    v.add_argument("--uppercase", action="store_true", help="ЗАГЛАВНЫМИ БУКВАМИ")
    v.add_argument("--tracking", type=float, default=0.04, help="разрядка букв в em (0.04)")
    v.add_argument("--glow", type=float, default=0.5, help="сила свечения, 0 = выкл (0.5)")
    v.add_argument("--boldness", type=float, default=0.025,
                   help="утолщение букв, доля от размера шрифта, 0 = как в шрифте (0.025)")
    v.add_argument("--scrim", type=float, default=0.5,
                   help="тёмная мягкая подложка под текстом 0..1, 0 = выкл (0.5)")
    v.add_argument("--glow-color", default="#ffffff", help="цвет свечения (#ffffff)")
    v.add_argument("--audio", type=Path,
                   help="музыка: файл или ПАПКА с треками (на каждое видео берётся свой трек)")
    v.add_argument("--audio-volume", type=float, default=1.0, help="громкость музыки (1.0)")
    v.add_argument("--audio-fade", type=float, default=1.0, help="затухание в конце, сек (1.0), 0 = выкл")
    v.add_argument("--audio-start", type=float, default=0.0, help="с какой секунды трека начинать (0)")
    v.add_argument("--seed", type=int, help="базовый seed порядка фото (видео i получает seed+i)")
    v.add_argument("--recursive", action="store_true", help="искать фото и во вложенных папках")
    args = ap.parse_args()

    if not shutil.which("ffmpeg"):
        sys.exit("ffmpeg не найден. Установи его (https://ffmpeg.org) и добавь в PATH.")
    if not args.folder.is_dir():
        sys.exit(f"Папка не найдена: {args.folder}")

    images = collect_images(args.folder, args.recursive)
    if not images:
        sys.exit("В папке нет подходящих изображений.")

    H = args.height
    W = H * 16 // 9
    W -= W % 2
    font_path = find_font(args.font)
    args.outdir.mkdir(parents=True, exist_ok=True)

    print(f"Фото: {len(images)} | Видео: {args.count} шт., {W}x{H} @ {args.fps} fps, "
          f"~{len(images) * args.duration:.1f} сек каждое")

    # 1. сначала все цитаты (чтобы Ollama не мешала рендеру и ошибки были видны сразу)
    if args.quote:
        quotes = [args.quote] * args.count
    elif args.quotes_file:
        pool = load_quotes(args.quotes_file)
        n_file = args.count // 2 if args.mix else args.count
        quotes = pick_from_pool(pool, n_file)
        print(f"Цитаты из файла: {n_file} (в файле {len(pool)})")
        if args.mix:
            print(f"Генерирую ещё {args.count - n_file} цитат через Ollama ({args.model})...")
            quotes += generate_quotes(args.count - n_file, args)
            random.shuffle(quotes)
    else:
        print(f"Генерирую {args.count} цитат через Ollama ({args.model})...")
        quotes = generate_quotes(args.count, args)

    # 2. потом видео
    audio_pool = collect_audio(args.audio)
    audio_for = pick_from_pool(audio_pool, len(quotes)) if audio_pool else [None] * len(quotes)
    if audio_pool:
        print(f"Музыка: {len(audio_pool)} трек(ов) в {args.audio}")

    tasks = []
    for i, quote in enumerate(quotes, start=1):
        rng = random.Random(args.seed + i if args.seed is not None else None)
        order = images[:]
        rng.shuffle(order)
        out = args.outdir / f"video_{i:02d}.mp4"
        tasks.append((order, out, quote, args, W, H, font_path, f"[{i}/{args.count}]", audio_for[i - 1]))

    if args.jobs > 1:
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            results = list(ex.map(run_task, tasks))
    else:
        results = [run_task(t) for t in tasks]
    done = sum(results)

    (args.outdir / "quotes.txt").write_text(
        "\n".join(f"video_{i:02d}.mp4\t{q}" for i, q in enumerate(quotes, 1)), encoding="utf-8")
    print(f"Готово: {done}/{args.count} видео в {args.outdir.resolve()}")


if __name__ == "__main__":
    main()

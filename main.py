RUN pip install opencv-python-headless

# bot.py
import os
import io
import math
import asyncio
from collections import defaultdict

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from aiogram import Bot, Dispatcher, types, executor
from aiogram.types import InputFile

# -----------------------
# Настройка токена
# -----------------------
BOT_TOKEN = "8005621509:AAEVOnQZz9qrQ6deUqC-4-P9olPghdvH0DE"

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher(bot)

# получим username бота (будет заполнен в main)
BOT_USERNAME = None

# -----------------------
# Пути к ассетам
# -----------------------
ASSETS_DIR = "assets"
TEMPLATES_DIR = os.path.join(ASSETS_DIR, "templates")
FONTS_DIR = os.path.join(ASSETS_DIR, "fonts")

# -----------------------
# Временное хранилище альбомов
# ключ теперь включает chat_id чтобы избежать коллизий разных чатов
# -----------------------
albums = defaultdict(lambda: {"photos": [], "caption": None, "task": None, "chat_id": None})

# -----------------------
# Константы шаблонов (из ТЗ)
# -----------------------
KABAN = {
    "template_path": os.path.join(TEMPLATES_DIR, "kaban.png"),
    "photo_size": (187, 187),
    "photo_radius": 38,
    "headline": {"font": os.path.join(FONTS_DIR, "Soyuz_Grotesk_Bold_0.otf"), "size": 149, "color": "#CAFFC4", "y": 81},
    "card_title_font": os.path.join(FONTS_DIR, "SF-Pro-Display-Bold.otf"),
    "card_title_size": 59,
    "card_title_color": "#FFFFFF",
    "card_sub_font": os.path.join(FONTS_DIR, "SF-Pro-Display-Regular.otf"),
    "card_sub_size": 40,
    "card_sub_color": "#DBDBDB",
    "cards": [
        {"photo": (144, 279), "title": (370, 305), "sub": (370, 385)},
        {"photo": (1003, 279), "title": (1230, 305), "sub": (1230, 385)},
        {"photo": (144, 527), "title": (370, 553), "sub": (370, 633)},
        {"photo": (1003, 527), "title": (1230, 553), "sub": (1230, 633)},
    ],
}

MAKAS = {
    "template_path": os.path.join(TEMPLATES_DIR, "makas.png"),
    "canvas_size": (1280, 720),
    "photo_size": (251, 200),
    "photo_radius": 30,
    "text_color": "#373A36",
    "main_title": {"font": os.path.join(FONTS_DIR, "helvetica_bold.otf"), "size": 64, "xy": (107, 68)},
    "main_sub": {"font": os.path.join(FONTS_DIR, "helvetica_regular.otf"), "size": 40, "xy": (107, 153)},
    "card_title_font": os.path.join(FONTS_DIR, "helvetica_bold.otf"),
    "card_title_size": 30,
    "card_sub_font": os.path.join(FONTS_DIR, "helvetica_regular.otf"),
    "card_sub_size": 18,
    "cards": [
        {"photo": (107, 241), "title": (135, 455), "sub": (135, 496)},
        {"photo": (379, 241), "title": (407, 455), "sub": (407, 496)},
        {"photo": (651, 241), "title": (679, 455), "sub": (679, 496)},
        {"photo": (923, 241), "title": (951, 455), "sub": (951, 496)},
    ],
}

# Новый шаблон фото: makas_channel.png (размер 1280x720)
MAKAS_CHANNEL = {
    "template_path": os.path.join(TEMPLATES_DIR, "makas_channel.png"),
    # в нашем алгоритме фото будут размещаться в квадратах 330x330, с перспективой
    "photo_size": (330, 330),
    "radius": 40,
    # позиции и конфиг (как в примере)
    # config: (rx, ry, ox, oy)
    # ox/oy - top-left координата вставки в шаблон
    "configs": [
        (20, -20, 0, 0),               # левый верх (ox,oy)
        (20, 20, 1280 - 330, 0),       # правый верх
        (-20, -20, 0, 720 - 330),      # левый низ
        (-20, 20, 1280 - 330, 720 - 330),  # правый низ
    ],
    # подписи заголовков (два централизованных заголовка на Y220 и Y308)
    "title_font": os.path.join(FONTS_DIR, "helvetica_bold.otf"),
    "title_size": 86,
    "title_ys": [220, 308],
    "canvas_size": (1280, 720),
    "text_color": (55, 58, 54),  # '#373A36' ~ (55,58,54)
}

# -----------------------------------------------------------
# --- Utility image helpers (Pillow)
# -----------------------------------------------------------
def center_crop_and_resize(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
    src_w, src_h = img.size
    target_ratio = target_w / target_h
    src_ratio = src_w / src_h
    if src_ratio > target_ratio:
        new_w = int(src_h * target_ratio)
        left = (src_w - new_w) // 2
        img = img.crop((left, 0, left + new_w, src_h))
    else:
        new_h = int(src_w / target_ratio)
        top = (src_h - new_h) // 2
        img = img.crop((0, top, src_w, top + new_h))
    img = img.resize((target_w, target_h), Image.LANCZOS)
    return img

def round_corners(im: Image.Image, radius: int):
    circle = Image.new('L', (radius * 2, radius * 2), 0)
    draw = ImageDraw.Draw(circle)
    draw.ellipse((0, 0, radius * 2, radius * 2), fill=255)
    alpha = Image.new('L', im.size, 255)
    w, h = im.size
    alpha.paste(circle.crop((0, 0, radius, radius)), (0, 0))
    alpha.paste(circle.crop((0, radius, radius, radius * 2)), (0, h - radius))
    alpha.paste(circle.crop((radius, 0, radius * 2, radius)), (w - radius, 0))
    alpha.paste(circle.crop((radius, radius, radius * 2, radius * 2)), (w - radius, h - radius))
    im.putalpha(alpha)
    return im

# -----------------------------------------------------------
# --- Parsing input messages according to ТЗ
# -----------------------------------------------------------
def is_makas_by_second_line(text: str) -> bool:
    lines = text.splitlines()
    return len(lines) >= 2 and lines[1].strip() != ""

def parse_photo_input_kaban(lines):
    items = [line for line in lines]
    while items and items[0].strip() == "":
        items.pop(0)
    if not items:
        return None
    header = items[0].strip()
    rest = items[1:]
    grouped = []
    i = 0
    while i < len(rest):
        if rest[i].strip() == "":
            i += 1
            continue
        t = rest[i].strip()
        s = ""
        if i + 1 < len(rest) and rest[i+1].strip() != "":
            s = rest[i+1].strip()
            i += 2
        else:
            i += 1
        grouped.append((t, s))
    while len(grouped) < 4:
        grouped.append(("", ""))
    grouped = grouped[:4]
    return {"header": header, "cards": grouped}

def parse_photo_input_makas(lines):
    lines = [l for l in lines]
    while lines and lines[0].strip() == "":
        lines.pop(0)
    if len(lines) < 2:
        return parse_photo_input_kaban(lines)
    main_title = lines[0].strip()
    main_sub = lines[1].strip()
    rest = lines[2:]
    grouped = []
    i = 0
    while i < len(rest):
        if rest[i].strip() == "":
            i += 1
            continue
        t = rest[i].strip()
        s = ""
        if i + 1 < len(rest) and rest[i+1].strip() != "":
            s = rest[i+1].strip()
            i += 2
        else:
            i += 1
        grouped.append((t, s))
    while len(grouped) < 4:
        grouped.append(("", ""))
    grouped = grouped[:4]
    return {"main_title": main_title, "main_sub": main_sub, "cards": grouped}

def parse_text_input_kaban(lines):
    items = [l for l in lines]
    while items and items[0].strip() == "":
        items.pop(0)
    header = items[0].strip() if items else ""
    rest = items[1:]
    cards = []
    cur = []
    for line in rest:
        if line.strip() == "":
            if cur:
                cards.append(cur)
                cur = []
            continue
        cur.append(line.strip())
    if cur:
        cards.append(cur)
    norm = []
    for c in cards[:4]:
        title = c[0] if len(c) > 0 else ""
        sub = c[1] if len(c) > 1 else ""
        desc = c[2] if len(c) > 2 else ""
        norm.append((title, sub, desc))
    while len(norm) < 4:
        norm.append(("", "", ""))
    return {"header": header, "cards": norm}

def parse_text_input_makas(lines):
    items = [l for l in lines]
    while items and items[0].strip() == "":
        items.pop(0)
    if len(items) < 2:
        return parse_text_input_kaban(items)
    header = items[0].strip()
    subheader = items[1].strip()
    rest = items[2:]
    cards = []
    cur = []
    for line in rest:
        if line.strip() == "":
            if cur:
                cards.append(cur)
                cur = []
            continue
        cur.append(line.strip())
    if cur:
        cards.append(cur)
    norm = []
    for c in cards[:4]:
        title = c[0] if len(c) > 0 else ""
        sub = c[1] if len(c) > 1 else ""
        desc = c[2] if len(c) > 2 else ""
        norm.append((title, sub, desc))
    while len(norm) < 4:
        norm.append(("", "", ""))
    return {"header": header, "subheader": subheader, "cards": norm}

# Парсер для нового текстового шаблона "канал"
def parse_text_input_channel(lines):
    # формат:
    # 'канал'
    # {общий заголовок/подзаголовок}
    # <пустая строка>
    # затем блоки товаров: title, subtitle, description
    items = [l for l in lines]
    # убираем пустые в начале
    while items and items[0].strip() == "":
        items.pop(0)
    # ожидание, что первый непустой == 'канал' (может быть с регистром)
    if not items:
        return None
    if items[0].strip().lower() != "канал":
        # не канал — fallback: вернуть None
        return None
    # удаляем 'канал'
    items = items[1:]
    # удаляем пустые в начале
    while items and items[0].strip() == "":
        items.pop(0)
    main_sub = items[0].strip() if items else ""
    rest = items[1:]
    # парсим карточки как в других шаблонах
    cards = []
    cur = []
    for line in rest:
        if line.strip() == "":
            if cur:
                cards.append(cur)
                cur = []
            continue
        cur.append(line.strip())
    if cur:
        cards.append(cur)
    norm = []
    for c in cards[:4]:
        title = c[0] if len(c) > 0 else ""
        sub = c[1] if len(c) > 1 else ""
        desc = c[2] if len(c) > 2 else ""
        norm.append((title, sub, desc))
    while len(norm) < 4:
        norm.append(("", "", ""))
    return {"subheader": main_sub, "cards": norm}

# -----------------------------------------------------------
# --- Compose images (Pillow) для KABAN / MAKAS
# -----------------------------------------------------------
def compose_kaban(photos: list, parsed):
    tpl = Image.open(KABAN["template_path"]).convert("RGBA")
    draw = ImageDraw.Draw(tpl)
    headline_conf = KABAN["headline"]
    try:
        font_head = ImageFont.truetype(headline_conf["font"], headline_conf["size"])
    except Exception:
        font_head = ImageFont.load_default()
    w, h = tpl.size
    headline_text = parsed.get("header", "")
    bbox = draw.textbbox((0,0), headline_text, font=font_head)
    text_w = bbox[2] - bbox[0]
    x = (w - text_w) // 2
    y = headline_conf["y"]
    draw.text((x, y), headline_text, font=font_head, fill=headline_conf["color"])

    for idx in range(4):
        card_conf = KABAN["cards"][idx]
        photo_pos = card_conf["photo"]
        target_w, target_h = KABAN["photo_size"]
        if idx < len(photos) and photos[idx] is not None:
            img = Image.open(photos[idx]).convert("RGBA")
            img = center_crop_and_resize(img, target_w, target_h)
            img = round_corners(img, KABAN["photo_radius"])
            tpl.paste(img, photo_pos, img)
        title_text, sub_text = parsed["cards"][idx]
        try:
            font_title = ImageFont.truetype(KABAN["card_title_font"], KABAN["card_title_size"])
            font_sub = ImageFont.truetype(KABAN["card_sub_font"], KABAN["card_sub_size"])
        except Exception:
            font_title = ImageFont.load_default()
            font_sub = ImageFont.load_default()
        draw.text(card_conf["title"], title_text, font=font_title, fill=KABAN["card_title_color"])
        draw.text(card_conf["sub"], sub_text, font=font_sub, fill=KABAN["card_sub_color"])

    bio = io.BytesIO()
    bio.name = "kaban_result.png"
    tpl.convert("RGB").save(bio, "PNG")
    bio.seek(0)
    return bio

def compose_makas(photos: list, parsed):
    tpl = Image.open(MAKAS["template_path"]).convert("RGBA")
    draw = ImageDraw.Draw(tpl)
    try:
        font_main = ImageFont.truetype(MAKAS["main_title"]["font"], MAKAS["main_title"]["size"])
        font_sub = ImageFont.truetype(MAKAS["main_sub"]["font"], MAKAS["main_sub"]["size"])
    except Exception:
        font_main = ImageFont.load_default()
        font_sub = ImageFont.load_default()
    draw.text(MAKAS["main_title"]["xy"], parsed.get("main_title", parsed.get("header", "")), font=font_main, fill=MAKAS["text_color"])
    draw.text(MAKAS["main_sub"]["xy"], parsed.get("main_sub", ""), font=font_sub, fill=MAKAS["text_color"])

    for idx in range(4):
        card_conf = MAKAS["cards"][idx]
        photo_pos = card_conf["photo"]
        target_w, target_h = MAKAS["photo_size"]
        if idx < len(photos) and photos[idx] is not None:
            img = Image.open(photos[idx]).convert("RGBA")
            img = center_crop_and_resize(img, target_w, target_h)
            img = round_corners(img, MAKAS["photo_radius"])
            tpl.paste(img, photo_pos, img)
        title_text, sub_text = parsed["cards"][idx]
        try:
            font_title = ImageFont.truetype(MAKAS["card_title_font"], MAKAS["card_title_size"])
            font_sub = ImageFont.truetype(MAKAS["card_sub_font"], MAKAS["card_sub_size"])
        except Exception:
            font_title = ImageFont.load_default()
            font_sub = ImageFont.load_default()
        draw.text(card_conf["title"], title_text, font=font_title, fill=MAKAS["text_color"])
        draw.text(card_conf["sub"], sub_text, font=font_sub, fill=MAKAS["text_color"])

    bio = io.BytesIO()
    bio.name = "makas_result.png"
    tpl.convert("RGB").save(bio, "PNG")
    bio.seek(0)
    return bio

# -----------------------------------------------------------
# --- Новый фото шаблон: makas_channel (OpenCV)
# -----------------------------------------------------------
def rounded_mask(size, radius):
    w, h = size
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(mask, (radius, 0), (w - radius, h), 255, -1)
    cv2.rectangle(mask, (0, radius), (w, h - radius), 255, -1)
    cv2.circle(mask, (radius, radius), radius, 255, -1)
    cv2.circle(mask, (w - radius, radius), radius, 255, -1)
    cv2.circle(mask, (radius, h - radius), radius, 255, -1)
    cv2.circle(mask, (w - radius, h - radius), radius, 255, -1)
    return mask

def perspective_points(size, rx, ry, fov):
    w, h = size
    f = w / (2 * math.tan(math.radians(fov / 2)))

    def rot_x(a):
        a = math.radians(a)
        return np.array([[1, 0, 0],
                         [0, math.cos(a), -math.sin(a)],
                         [0, math.sin(a), math.cos(a)]])

    def rot_y(a):
        a = math.radians(a)
        return np.array([[math.cos(a), 0, math.sin(a)],
                         [0, 1, 0],
                         [-math.sin(a), 0, math.cos(a)]])

    R = rot_y(ry) @ rot_x(rx)

    pts = np.array([
        [-w / 2, -h / 2, 0],
        [w / 2, -h / 2, 0],
        [w / 2, h / 2, 0],
        [-w / 2, h / 2, 0]
    ])

    proj = []
    for p in pts:
        x, y, z = R @ p
        z += f
        proj.append([f * x / z + w / 2, f * y / z + h / 2])

    return np.array(proj, dtype=np.float32)

def paste_cv(img, overlay, mask):
    inv = cv2.bitwise_not(mask)
    bg = cv2.bitwise_and(img, img, mask=inv)
    fg = cv2.bitwise_and(overlay, overlay, mask=mask)
    return cv2.add(bg, fg)

def compose_makas_channel(photo_bytes_list, caption_lines):
    """
    photo_bytes_list: list of BytesIO images (4 items)
    caption_lines: list of lines from caption - for makas_channel we expect either titles in first two (or empty)
    returns BytesIO PNG
    """
    try:
        template_path = MAKAS_CHANNEL["template_path"]
        template = cv2.imread(template_path)
        if template is None:
            raise FileNotFoundError(f"Template not found: {template_path}")
        H, W = template.shape[:2]
        photo_size = MAKAS_CHANNEL["photo_size"]
        radius = MAKAS_CHANNEL["radius"]

        # for each of 4 photos
        for idx in range(4):
            rx, ry, ox, oy = MAKAS_CHANNEL["configs"][idx]
            # get image from bytes
            bio = photo_bytes_list[idx]
            bio.seek(0)
            arr = np.frombuffer(bio.read(), dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                # create placeholder gray
                img = np.full((photo_size[1], photo_size[0], 3), 200, dtype=np.uint8)
            img = cv2.resize(img, photo_size)
            mask = rounded_mask(photo_size, radius)
            img_masked = cv2.bitwise_and(img, img, mask=mask)

            src = np.array([[0, 0], [photo_size[0], 0], [photo_size[0], photo_size[1]], [0, photo_size[1]]], dtype=np.float32)
            dst = perspective_points(photo_size, rx, ry, 70)
            M = cv2.getPerspectiveTransform(src, dst)
            warped = cv2.warpPerspective(img_masked, M, photo_size)
            warped_mask = cv2.warpPerspective(mask, M, photo_size)

            # roi in template: ensure boundaries
            x0 = int(ox)
            y0 = int(oy)
            x1 = x0 + photo_size[0]
            y1 = y0 + photo_size[1]
            if x0 < 0 or y0 < 0 or x1 > W or y1 > H:
                # skip or clamp - we'll clamp coordinates
                x0 = max(0, min(x0, W - photo_size[0]))
                y0 = max(0, min(y0, H - photo_size[1]))
                x1 = x0 + photo_size[0]
                y1 = y0 + photo_size[1]

            roi = template[y0:y1, x0:x1]
            merged = paste_cv(roi, warped, warped_mask)
            template[y0:y1, x0:x1] = merged

        # Добавим заголовки (центрально) — ожидается 2 строки заголовков в caption_lines (или пустые)
        # используем PIL для рисования текста, потому что OpenCV рисование шрифтов менее удобное
        pil_img = Image.fromarray(cv2.cvtColor(template, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        try:
            font = ImageFont.truetype(MAKAS_CHANNEL["title_font"], MAKAS_CHANNEL["title_size"])
        except Exception:
            font = ImageFont.load_default()

        # берем первые две непустые строки caption_lines (или пустые)
        titles = [l.strip() for l in caption_lines if l.strip() != ""]
        # If caption was "канал" then caption_lines may include that first; we'll skip that if present
        if titles and titles[0].lower() == "канал":
            titles = titles[1:]
        # ensure at least two elements
        while len(titles) < 2:
            titles.append("")

        for i in range(2):
            text = titles[i]
            y = MAKAS_CHANNEL["title_ys"][i]
            # центрирование
            w_text, h_text = draw.textbbox((0, 0), text, font=font)[2:]
            x = (MAKAS_CHANNEL["canvas_size"][0] - w_text) // 2
            draw.text((x, y), text, font=font, fill=MAKAS_CHANNEL["text_color"])

        out_buf = io.BytesIO()
        pil_img.convert("RGB").save(out_buf, "PNG")
        out_buf.seek(0)
        return out_buf
    except Exception as e:
        # поднимем исключение дальше, обработчик вызовет fallback
        raise

# -----------------------------------------------------------
# --- Compose text (HTML) according to templates
# -----------------------------------------------------------
def format_text_kaban(parsed):
    header = parsed["header"]
    parts = [f"<b>{escape_html(header)}</b>", ""]
    for t, s, d in parsed["cards"]:
        parts.append(f"<blockquote>{escape_html(t)}</blockquote>")
        parts.append(f"<b>● {escape_html(s)}</b>")
        parts.append(f" · {escape_html(d)}")
        parts.append("")
    parts.append("<b>Приятных покупок! ☺️</b>")
    return "\n".join(parts)

def format_text_makas(parsed):
    header = parsed["header"]
    sub = parsed["subheader"]
    parts = [f"<b>{escape_html(header)}</b>", escape_html(sub), ""]
    for t, s, d in parsed["cards"]:
        parts.append(f"<b>▎{escape_html(t)}</b>")
        parts.append(f"    {escape_html(s)}")
        parts.append(f"     - {escape_html(d)}")
        parts.append("")
    parts.append("<b>Приятных покупок! 😊</b>")
    return "\n".join(parts)

def format_text_channel(parsed, bot_username_display):
    # parsed contains 'subheader' and 'cards' (title, sub, desc)
    parts = [f"<b>{escape_html(parsed.get('subheader',''))}</b>", ""]
    for t, s, d in parsed["cards"]:
        parts.append(f"<b>▎{escape_html(t)}</b> - {escape_html(s)}")
        parts.append(f"{escape_html(d)}")
        parts.append("")
    parts.append(f"🤖 <b>Приобрести</b> — {{bot_username}}")
    return "\n".join(parts)

def escape_html(s: str) -> str:
    if s is None:
        return ""
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))

# -----------------------------------------------------------
# --- Telegram handlers
# -----------------------------------------------------------

@dp.message_handler(content_types=types.ContentType.PHOTO)
async def handle_album_photos(message: types.Message):
    """
    Сборка альбомов по media_group_id. Ждём короткую паузу, затем обрабатываем
    """
    try:
        if not message.media_group_id:
            # не альбом — игнорируем обычные одиночные фото
            return

        key = f"{message.chat.id}:{message.media_group_id}"
        album = albums[key]
        album["chat_id"] = message.chat.id

        if message.caption:
            # сохраняем полную подпись (берём последнюю, т.к. caption может быть только в одном из сообщений)
            album["caption"] = message.caption

        # сохраняем photo object (последний вариант размера)
        album["photos"].append(message.photo[-1])

        # если уже стоит задача на обработку - просто вернёмся (она ждёт)
        if album["task"]:
            return

        # отложенная обработка: ждём, пока придут все части (телеграм шлёт несколько сообщений подряд)
        async def process_album_task(chat_message: types.Message, key_local: str):
            await asyncio.sleep(0.9)  # время ожидания поступления всех частей

            try:
                al = albums.get(key_local)
                if not al:
                    return
                photos_objs = al["photos"]
                caption = al["caption"] or ""
                # ожидаем ровно 4 фото
                if len(photos_objs) != 4:
                    await bot.send_message(chat_message.chat.id, "❌ Нужно отправить ровно 4 фото в одном сообщении (альбом).")
                    albums.pop(key_local, None)
                    return

                # скачиваем фото в BytesIO
                photo_files = []
                for p in photos_objs:
                    file = await bot.get_file(p.file_id)
                    bio = io.BytesIO()
                    await bot.download_file(file.file_path, bio)
                    bio.seek(0)
                    photo_files.append(bio)

                # логика выбора шаблона:
                # если caption начинается с 'канал' (первый непустой элемент) -> использовать makas_channel (и текстовый каналным шаблон)
                caption_lines = [l for l in (caption.splitlines() if caption else [])]
                first_nonempty = ""
                for l in caption_lines:
                    if l.strip():
                        first_nonempty = l.strip().lower()
                        break

                # выбираем шаблон фото
                if first_nonempty == "канал":
                    # используем новый шаблон makas_channel
                    try:
                        result = compose_makas_channel(photo_files, caption_lines)
                    except Exception as e:
                        # fallback: попробуем обычный makas
                        try:
                            parsed = parse_photo_input_makas(caption_lines)
                            result = compose_makas(photo_files, parsed)
                        except Exception as e2:
                            await bot.send_message(chat_message.chat.id, "❌ Ошибка при создании изображения (channel).")
                            albums.pop(key_local, None)
                            return
                else:
                    # старый подход: если вторая строка caption непустая => makas, иначе kaban
                    makas = is_makas_by_second_line(caption)
                    lines = caption.splitlines()
                    try:
                        if makas:
                            parsed = parse_photo_input_makas(lines)
                            result = compose_makas(photo_files, parsed)
                        else:
                            parsed = parse_photo_input_kaban(lines)
                            result = compose_kaban(photo_files, parsed)
                    except Exception as e:
                        await bot.send_message(chat_message.chat.id, "❌ Ошибка при создании изображения.")
                        albums.pop(key_local, None)
                        return

                result.seek(0)
                await bot.send_photo(chat_message.chat.id, photo=InputFile(result))

            except Exception as ex:
                try:
                    await bot.send_message(message.chat.id, "❌ Произошла внутренняя ошибка при обработке альбома.")
                except Exception:
                    pass
            finally:
                # очистка
                albums.pop(key_local, None)

        # создаём таск
        album["task"] = asyncio.create_task(process_album_task(message, key))

    except Exception as e:
        # никогда не падаем
        try:
            await message.answer("❌ Внутренняя ошибка при обработке фото.")
        except Exception:
            pass

@dp.message_handler(content_types=types.ContentType.TEXT)
async def handle_text(message: types.Message):
    """
    Обработка текстов в трех вариантах:
    - если текст начинается с 'канал' -> новый текстовый шаблон channel
    - иначе: если вторая строка непустая -> makas
    - иначе -> kaban
    """
    try:
        text = (message.text or "").strip()
        if not text:
            return

        lines = text.splitlines()
        # detect 'канал' template
        first_nonempty = ""
        for l in lines:
            if l.strip():
                first_nonempty = l.strip().lower()
                break

        if first_nonempty == "канал":
            parsed = parse_text_input_channel(lines)
            if not parsed:
                await message.answer("❌ Неверный формат для шаблона 'канал'.")
                return
            bot_display = f"@{BOT_USERNAME}" if BOT_USERNAME else "@bot"
            out = format_text_channel(parsed, bot_display)
            await message.answer(out, parse_mode="HTML")
            return

        # обычные шаблоны
        makas = is_makas_by_second_line(text)
        if makas:
            parsed = parse_text_input_makas(lines)
            out = format_text_makas(parsed)
        else:
            parsed = parse_text_input_kaban(lines)
            out = format_text_kaban(parsed)

        await message.answer(out, parse_mode="HTML")

    except Exception as e:
        # логируем и сообщаем аккуратно
        print("TEXT HANDLER ERROR:", e)
        try:
            await message.answer("❌ Ошибка при обработке текста. Проверь структуру сообщения.")
        except Exception:
            pass

# -----------------------------------------------------------
# --- Глобальный обработчик ошибок (чтобы бот не падал)
# -----------------------------------------------------------
@dp.errors_handler()
async def global_error_handler(update, exception):
    print("GLOBAL ERROR:", exception)
    # возвращаем True чтобы aiogram не логировал лишнее и не падал
    return True

# -----------------------------------------------------------
# --- Вспомогательные функции загрузки (если нужно отдельно)
# -----------------------------------------------------------
async def download_file_from_photo(photo: types.PhotoSize):
    file = await bot.get_file(photo.file_id)
    b = await file.download(io.BytesIO())
    b.seek(0)
    tmp = io.BytesIO(b.read())
    tmp.seek(0)
    return tmp

async def download_file_from_document(document: types.Document):
    file = await bot.get_file(document.file_id)
    b = await file.download(io.BytesIO())
    b.seek(0)
    tmp = io.BytesIO(b.read())
    tmp.seek(0)
    return tmp

# -----------------------------------------------------------
# --- Запуск
# -----------------------------------------------------------
if __name__ == "__main__":
    # получим username бота заранее
    loop = asyncio.get_event_loop()
    try:
        me = loop.run_until_complete(bot.get_me())
        BOT_USERNAME = me.username or None
    except Exception as e:
        print("Не удалось получить username бота:", e)
        BOT_USERNAME = None

    print("Bot starting...")
    executor.start_polling(dp, skip_updates=True)


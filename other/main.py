import logging
from PIL import Image, ImageDraw, ImageFont
from PIL import ImageFilter
from io import BytesIO
from telegram import Update, InputFile
from telegram.ext import ApplicationBuilder, MessageHandler, ContextTypes, filters
import asyncio
import random
import traceback

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TEMPLATE_PATH = 'shablon.png'
FONT_BOLD_PATH = 'helvetica_bold.otf'
FONT_REGULAR_PATH = 'helvetica_regular.otf'

PHOTO_WIDTH, PHOTO_HEIGHT = 251, 200
RADIUS = 30
COLOR = '#373A36'

POSITIONS = {
    'main_title': (107, 68),
    'main_subtitle': (107, 153),
    'photos': [(107, 241), (379, 241), (651, 241), (923, 241)],
    'titles': [(135, 455), (407, 455), (679, 455), (951, 455)],
    'subtitles': [(135, 496), (407, 496), (679, 496), (951, 496)],
}

media_groups = {}
GROUP_TIMEOUT = 2

generated_images = {}

async def handle_media_group(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        msg = update.message
        media_group_id = msg.media_group_id

        if not media_group_id:
            await msg.reply_text("Пожалуйста, отправляйте фото как альбом (медиа-группу).")
            return

        if media_group_id not in media_groups:
            media_groups[media_group_id] = {
                'messages': [],
                'caption': msg.caption or "",
            }
            asyncio.create_task(process_media_group_after_delay(media_group_id, context))

        media_groups[media_group_id]['messages'].append(msg)
    except Exception as e:
        logger.error(f"Ошибка в handle_media_group: {e}\n{traceback.format_exc()}")


async def process_media_group_after_delay(media_group_id, context: ContextTypes.DEFAULT_TYPE):
    try:
        await asyncio.sleep(GROUP_TIMEOUT)
        group = media_groups.pop(media_group_id, None)
        if not group:
            return

        msgs = group['messages']
        caption = group['caption']

        if len(msgs) != 4:
            for msg in msgs:
                await msg.reply_text("Пожалуйста, отправляйте ровно 4 фото в одном альбоме.")
            return

        images = []
        for msg in msgs:
            photo = msg.photo[-1]
            file = await context.bot.get_file(photo.file_id)
            bio = BytesIO()
            await file.download_to_memory(out=bio)
            bio.seek(0)
            images.append(Image.open(bio).convert("RGB"))

        lines = [line.strip() for line in caption.split('\n') if line.strip() != '']
        if len(lines) != 10:
            await msgs[0].reply_text("Неверный формат текста. Должно быть ровно 10 непустых строк.")
            return

        font_main_title = ImageFont.truetype(FONT_BOLD_PATH, 64)
        font_main_subtitle = ImageFont.truetype(FONT_REGULAR_PATH, 40)
        font_titles = ImageFont.truetype(FONT_BOLD_PATH, 30)
        font_subtitles = ImageFont.truetype(FONT_REGULAR_PATH, 18)

        template = Image.open(TEMPLATE_PATH).convert("RGBA")
        draw = ImageDraw.Draw(template)

        draw.text(POSITIONS['main_title'], lines[0], font=font_main_title, fill=COLOR)
        draw.text(POSITIONS['main_subtitle'], lines[1], font=font_main_subtitle, fill=COLOR)

        for i in range(4):
            photo = process_image(images[i])
            template.paste(photo, POSITIONS['photos'][i], photo)
            draw.text(POSITIONS['titles'][i], lines[2 + i * 2], font=font_titles, fill=COLOR)
            draw.text(POSITIONS['subtitles'][i], lines[2 + i * 2 + 1], font=font_subtitles, fill=COLOR)

        output = BytesIO()
        output.name = "result.png"
        template.save(output, format='PNG')
        output.seek(0)

        sent_msg = await msgs[0].reply_photo(photo=output)
        generated_images[sent_msg.message_id] = output.getvalue()

    except Exception as e:
        logger.error(f"Ошибка в process_media_group_after_delay: {e}\n{traceback.format_exc()}")
        for msg in media_groups.get(media_group_id, {}).get('messages', []):
            await msg.reply_text("Произошла ошибка при обработке. Попробуйте ещё раз.")


def process_image(img: Image.Image) -> Image.Image:
    try:
        w, h = img.size
        new_height = PHOTO_HEIGHT
        new_width = int(w * (new_height / h))
        img = img.resize((new_width, new_height), Image.LANCZOS)

        if new_width > PHOTO_WIDTH:
            left = (new_width - PHOTO_WIDTH) // 2
            img = img.crop((left, 0, left + PHOTO_WIDTH, new_height))
        else:
            img = img.resize((PHOTO_WIDTH, new_height), Image.LANCZOS)

        scale = 3
        large_mask = Image.new('L', (PHOTO_WIDTH * scale, PHOTO_HEIGHT * scale), 0)
        draw = ImageDraw.Draw(large_mask)
        draw.rounded_rectangle(
            [0, 0, PHOTO_WIDTH * scale, PHOTO_HEIGHT * scale],
            radius=RADIUS * scale,
            fill=255
        )

        mask = large_mask.resize((PHOTO_WIDTH, PHOTO_HEIGHT), Image.LANCZOS)
        img.putalpha(mask)
        return img
    except Exception as e:
        logger.error(f"Ошибка в process_image: {e}\n{traceback.format_exc()}")
        return img


async def handle_caption_reply(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        msg = update.message
        if not msg.reply_to_message or not msg.reply_to_message.photo:
            return

        replied_msg_id = msg.reply_to_message.message_id
        if replied_msg_id not in generated_images:
            return

        lines = [line.strip() for line in msg.text.strip().split('\n') if line.strip() != '']
        if len(lines) != 14:
            await msg.reply_text("Неверный формат текста. Нужно 14 непустых строк: 2 для заголовка и по 3 на каждый из 4 блоков.")
            return

        main_title = lines[0]
        main_subtitle = lines[1]
        blocks = [lines[i:i+3] for i in range(2, 14, 3)]
        emoji = random.choice(["😄", "🥹", "☺️", "😊", "🙂", "😉", "😌", "🥰", "😋", "😎", "🤭", "🤠", "😸", "🫶", "✌️", "💪"])

        caption = f"<b>{main_title}</b>\n{main_subtitle}\n\n"
        for block in blocks:
            caption += f"▎<b>{block[0]}</b>\n    {block[1]}\n     - {block[2]}\n\n"
        caption += f"<b>Приятных покупок! </b>{emoji}"

        photo_bytes = BytesIO(generated_images[replied_msg_id])
        photo_bytes.name = "final.png"
        await msg.reply_photo(photo=InputFile(photo_bytes), caption=caption, parse_mode='HTML')

    except Exception as e:
        logger.error(f"Ошибка в handle_caption_reply: {e}\n{traceback.format_exc()}")
        await update.message.reply_text("Произошла ошибка при создании подписи. Попробуйте снова.")


def main():
    try:
        app = ApplicationBuilder().token("8005621509:AAEVOnQZz9qrQ6deUqC-4-P9olPghdvH0DE").build()

        app.add_handler(MessageHandler(filters.PHOTO, handle_media_group))
        app.add_handler(MessageHandler(filters.TEXT & filters.REPLY, handle_caption_reply))

        app.run_polling()
    except Exception as e:
        logger.critical(f"Ошибка в main: {e}\n{traceback.format_exc()}")


if __name__ == '__main__':
    main()

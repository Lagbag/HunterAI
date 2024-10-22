import os
import shutil
import subprocess
from pathlib import Path

import cv2
import yt_dlp
from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from ultralytics import YOLO

app = FastAPI()

# Подключение Jinja2 шаблонов
templates = Jinja2Templates(directory="templates")

# Путь для сохранения загруженных файлов
UPLOAD_DIR = Path("uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Путь для статических файлов (загруженные и обработанные видео/изображения)
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

# Загружаем YOLOv8x модель
model = YOLO('yolov8x.pt')


# Функция для обработки видео

def process_video(video_path, output_path):
    # Открываем видеофайл
    video = cv2.VideoCapture(video_path)

    # Проверка, что файл открылся успешно
    if not video.isOpened():
        raise Exception(f"Не удалось открыть видео: {video_path}")

    # Получаем параметры видео (ширина, высота, fps)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*"XVID")

    # Используем новый выходной файл для ffmpeg
    temp_output_path = output_path.replace(".mp4", "_temp.mp4")

    out = cv2.VideoWriter(temp_output_path, fourcc, fps, (width, height))

    # Применение YOLOv8 к каждому кадру
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break

        # Детекция объектов на каждом кадре с помощью YOLOv8
        results = model(frame)  # Модель YOLOv8 применена к кадру
        annotated_frame = results[0].plot()  # Аннотирование кадра

        # Сохранение аннотированного кадра в выходное видео
        out.write(annotated_frame)

    video.release()
    out.release()

    # Конвертация с помощью ffmpeg, избегаем перезаписи
    command = ['ffmpeg', '-y', '-i', temp_output_path, '-vcodec', 'libx264', '-acodec', 'aac', output_path]
    subprocess.run(command, check=True)

    # Удаление временного файла
    os.remove(temp_output_path)


# Функция для обработки изображения
def process_image(image_path, output_path):
    image = cv2.imread(image_path)
    results = model(image)
    annotated_image = results[0].plot()  # Рисуем детекцию на изображении
    cv2.imwrite(output_path, annotated_image)


# Главная страница с формой для загрузки видео и изображений
@app.get("/", response_class=HTMLResponse)
async def index(request: Request, video_name: str = None, image_name: str = None):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "video_name": video_name,
        "image_name": image_name
    })


# Маршрут для обработки видео с ПК
@app.post("/upload_video/")
async def upload_video(request: Request, file: UploadFile = File(...)):
    video_path = UPLOAD_DIR / file.filename
    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    output_path = UPLOAD_DIR / f"processed_{file.filename}"
    process_video(str(video_path), str(output_path))

    return templates.TemplateResponse("index.html", {
        "request": request,
        "video_name": output_path.name,
        "image_name": None
    })


# Маршрут для обработки изображений с ПК
@app.post("/upload_image/")
async def upload_image(request: Request, file: UploadFile = File(...)):
    image_path = UPLOAD_DIR / file.filename
    with open(image_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    output_path = UPLOAD_DIR / f"processed_{file.filename}"
    process_image(str(image_path), str(output_path))

    return templates.TemplateResponse("index.html", {
        "request": request,
        "video_name": None,
        "image_name": output_path.name
    })


# Маршрут для скачивания и обработки видео с YouTube через yt-dlp
@app.post("/youtube/")
async def youtube_video(request: Request, url: str = Form(...)):
    try:
        ydl_opts = {
            'format': 'best',  # Выбираем лучшее качество
            'outtmpl': str(UPLOAD_DIR / '%(title)s.%(ext)s'),
            'progress_hooks': [lambda d: print(f"Status: {d['status']}")]  # Простой хук для мониторинга прогресса
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(url, download=True)
            video_title = info_dict.get('title', None)
            video_ext = info_dict.get('ext', 'mp4')

        video_path = UPLOAD_DIR / f"{video_title}.{video_ext}"
        output_path = UPLOAD_DIR / f"processed_{video_title}.{video_ext}"

        # Обработка видео
        process_video(str(video_path), str(output_path))

        return templates.TemplateResponse("index.html", {
            "request": request,
            "video_name": output_path.name,
            "image_name": None
        })
    except Exception as e:
        return {"error": str(e)}


# Запуск сервера
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

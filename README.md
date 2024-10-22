# 🎥 Приложение для обработки видео и изображений / Video and Image Processing Web Application

Приветствуем в **Приложении для обработки видео и изображений**! Это мощный веб-инструмент, который позволяет загружать видео и изображения, обрабатывать их с использованием нейросети YOLOv8 для обнаружения объектов и загружать обработанные результаты.

Welcome to the **Video and Image Processing Web Application**! This is a powerful web tool that allows users to upload videos or images, process them using the YOLOv8 neural network for object detection, and download the processed results.

## 🌟 Особенности / Features

- **Загрузка и обработка видео**: Загрузите видео с вашего устройства и обработайте его с помощью YOLOv8 для обнаружения объектов.
- **Обработка изображений**: Загрузите изображение и обнаружьте объекты с использованием модели YOLOv8.
- **Обработка видео с YouTube**: Вставьте ссылку на YouTube, и приложение загрузит, обработает и обнаружит объекты на видео.
- **Загрузка обработанных файлов**: После обработки вы сможете скачать видео или изображение с аннотациями.

- **Upload and Process Videos**: Upload your videos from your computer and process them using YOLOv8 for object detection.
- **Image Processing**: Upload an image and detect objects using the same YOLOv8 model.
- **YouTube Video Processing**: Provide a YouTube URL, and the app will download, process, and detect objects in the video for you.
- **Download Processed Files**: After processing, you can download the resulting video or image with annotations.

## 🛠️ Установка / Installation

### Необходимые компоненты / Prerequisites

- **Python 3.8+**
- **pip** (менеджер пакетов Python)
- **ffmpeg** (для обработки видео)

- **Python 3.8+**
- **pip** (Python package manager)
- **ffmpeg** (for video processing)

### Клонирование репозитория / Clone the Repository

```bash
git clone https://github.com/Lagbag/HunterAI.git
cd video-image-processing-app
```

Создание и активация виртуального окружения / Create and Activate Virtual Environment
bash
```
python -m venv venv
source venv/bin/activate   # На Windows используйте `venv\Scripts\activate`
```
bash
```
python -m venv venv
source venv/bin/activate   # On Windows use `venv\Scripts\activate`
```
Установка зависимостей / Install Dependencies
bash
```
pip install -r requirements.txt
```
Запуск приложения / Run the Application
bash
```
uvicorn main:app --reload
```

Приложение будет доступно по адресу http://127.0.0.1:8000/ в вашем браузере.

The app will be available at http://127.0.0.1:8000/ in your browser.

👨‍💻 Технологии, используемые в проекте / Technologies Used
FastAPI: Современный, быстрый веб-фреймворк для создания API.

YOLOv8: Современная модель для обнаружения объектов на изображениях и видео.

Jinja2: Шаблонизатор для рендеринга HTML.

ffmpeg: Инструмент для обработки и конвертации видео.

FastAPI: A modern, fast (high-performance) web framework for building APIs.

YOLOv8: State-of-the-art object detection model for video and image processing.

Jinja2: Template engine for rendering HTML pages.

ffmpeg: Tool for video processing and conversion.

📜 Лицензия / License
Этот проект лицензирован по лицензии MIT — подробности см. в файле [Лицензия MIT](LICENSE)

This project is licensed under the MIT License - see the [MIT LICENSE](LICENSE) file for details.

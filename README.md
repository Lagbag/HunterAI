```markdown
# 🎥 Приложение для обработки видео и изображений / Video and Image Processing Web Application

Приветствуем в **Приложении для обработки видео и изображений**! Это интуитивный и мощный веб-инструмент, который позволяет загружать видео и изображения, обрабатывать их с использованием передовой нейросети **YOLOv8** для обнаружения объектов, и загружать обработанные результаты с аннотациями.

Welcome to the **Video and Image Processing Web Application**! This is an intuitive and powerful web tool that allows users to upload videos or images, process them using the cutting-edge **YOLOv8** neural network for object detection, and download the processed results with annotations.

## 🌟 Основные возможности / Key Features

- **Загрузка и обработка видео**: Загрузите видео с вашего устройства, и оно будет обработано с помощью YOLOv8 для обнаружения объектов.
- **Обработка изображений**: Загрузите изображение и получите аннотированную версию с выделенными объектами.
- **Загрузка и обработка видео с YouTube**: Вставьте ссылку на видео с YouTube, и приложение загрузит его, обработает и выделит объекты.
- **Скачивание обработанных файлов**: После обработки вы сможете скачать обработанное видео или изображение с аннотациями.

- **Upload and Process Videos**: Upload a video from your device and have it processed using YOLOv8 for object detection.
- **Image Processing**: Upload an image and receive an annotated version highlighting detected objects.
- **YouTube Video Processing**: Provide a YouTube URL, and the app will download, process, and annotate detected objects in the video.
- **Download Processed Files**: Download the processed video or image with annotations after the detection.

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

### Создание и активация виртуального окружения / Create and Activate Virtual Environment

```bash
python -m venv venv
source venv/bin/activate   # На Windows используйте `venv\Scripts\activate`
```

```bash
python -m venv venv
source venv/bin/activate   # On Windows use `venv\Scripts\activate`
```

### Установка зависимостей / Install Dependencies

```bash
pip install -r requirements.txt
```

### Запуск приложения / Run the Application

```bash
uvicorn main:app --reload
```

Приложение будет доступно по адресу `http://127.0.0.1:8000/` в вашем браузере.

The app will be available at `http://127.0.0.1:8000/` in your browser.

## 👨‍💻 Технологии, используемые в проекте / Technologies Used

- **FastAPI**: Современный и высокопроизводительный веб-фреймворк для создания API.
- **YOLOv8**: Современная модель для обнаружения объектов на изображениях и видео.
- **Jinja2**: Мощный шаблонизатор для рендеринга HTML.
- **ffmpeg**: Инструмент для обработки и конвертации видео.

- **FastAPI**: A modern and fast web framework for building APIs.
- **YOLOv8**: A state-of-the-art object detection model for video and image processing.
- **Jinja2**: A powerful template engine for rendering HTML pages.
- **ffmpeg**: A tool for video processing and conversion.

## 📜 Лицензия / License

Этот проект лицензирован по лицензии MIT — подробности см. в файле [Лицензия MIT](LICENSE).

This project is licensed under the MIT License - see the [MIT LICENSE](LICENSE) file for details.

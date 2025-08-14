# Используем легковесный Python
FROM python:3.12-slim

# Указываем порт, который будет слушать FastAPI
EXPOSE 8000

# Не создаём .pyc файлы
ENV PYTHONDONTWRITEBYTECODE=1
# Выводим логи сразу, без буферизации
ENV PYTHONUNBUFFERED=1

# Устанавливаем зависимости
WORKDIR /app
COPY requirements.txt .
RUN python -m pip install --upgrade pip
RUN python -m pip install -r requirements.txt

# Копируем проект в контейнер
COPY . /app

# Создаём не-root пользователя для безопасности
RUN adduser --disabled-password --gecos "" appuser && chown -R appuser /app
USER appuser

# Запуск FastAPI через Gunicorn + Uvicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "-k", "uvicorn.workers.UvicornWorker", "main:app"]

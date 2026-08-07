# KazMorph — production-образ для развёртывания.
# Python 3.12 + CPU-only torch (инференс на CPU, без CUDA → образ в разы меньше).

FROM python:3.12-slim

# gcc/python3-dev — для сборки C-расширения pycrfsuite (sklearn-crfsuite).
# libpq-dev — для сборки psycopg2 (драйвер PostgreSQL).
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc python3-dev libpq-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1) Зависимости отдельным слоем — требования меняются реже кода (кэш билда).
COPY requirements.txt .
# torch ставим CPU-only; затем остальное из requirements.txt. Так как
# torch>=2.0.0 уже удовлетворён, тяжёлый CUDA-wheel не подтягивается.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt

# 2) Код приложения. .dockerignore исключает models/ data/ files/ reports/ .venv и т.д.
COPY . .

# Непривилегированный пользователь (см. docs/security.md — отдельный системный user).
RUN useradd --create-home --uid 1000 appuser && chown -R appuser /app
USER appuser

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

EXPOSE 8080

# Несколько воркеров для стабильности под нагрузкой (см. docs/pilot_launch.md §4).
# Сессионные cookie подписаны (SessionMiddleware), поэтому multi-worker безопасен.
CMD ["sh", "-c", "alembic upgrade head && uvicorn main:app --host 0.0.0.0 --port 8080 --workers ${WORKERS:-4}"]

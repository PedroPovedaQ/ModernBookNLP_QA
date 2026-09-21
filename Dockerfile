FROM python:3.11-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 tini && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY service/requirements.lock /app/service/requirements.lock
RUN pip install --no-cache-dir torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r service/requirements.lock && \
    pip install --no-cache-dir https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl
COPY ModernBookNLP /app/ModernBookNLP
COPY service /app/service
RUN useradd --uid 10001 --create-home booknlp && mkdir /data && chown booknlp:booknlp /data
USER booknlp
ENV BOOKNLP_DATA_DIR=/data/jobs BOOKNLP_MODEL_DIR=/data/models HF_HOME=/data/huggingface HF_HUB_DISABLE_XET=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:'+os.environ.get('PORT','8000')+'/healthz',timeout=3)"
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["python", "-m", "service.serve"]

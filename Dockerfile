# Weights are baked into the image (verified at build time) so a cold boot reads them
# from the local rootfs instead of a network download or a slow cold volume read.
FROM python:3.11-slim-bookworm AS weights
RUN pip install --no-cache-dir huggingface-hub==0.36.2
ENV BOOKNLP_MODEL_DIR=/opt/booknlp/models HF_HOME=/opt/booknlp/huggingface HF_HUB_DISABLE_XET=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /build
COPY service/__init__.py service/config.py service/model.py service/bake_weights.py /build/service/
RUN python -m service.bake_weights && rm -rf /opt/booknlp/huggingface/hub/.locks /opt/booknlp/huggingface/xet

FROM python:3.11-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 tini && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY service/requirements.lock /app/service/requirements.lock
RUN pip install --no-cache-dir torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r service/requirements.lock && \
    pip install --no-cache-dir https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl
# Separate layers per large artifact keep image layers under ~1.7 GB.
COPY --from=weights --chown=10001:10001 /opt/booknlp/models/ModernBERT_T2000.safetensors /opt/booknlp/models/
COPY --from=weights --chown=10001:10001 /opt/booknlp/models/coref_google_bert_uncased_L-12_H-768_A-12-v1.0.model /opt/booknlp/models/entities_google_bert_uncased_L-6_H-768_A-12-v1.0.model /opt/booknlp/models/
COPY --from=weights --chown=10001:10001 /opt/booknlp/huggingface /opt/booknlp/huggingface
COPY ModernBookNLP /app/ModernBookNLP
COPY service /app/service
RUN useradd --uid 10001 --create-home booknlp && mkdir /data && chown booknlp:booknlp /data
USER booknlp
ENV BOOKNLP_DATA_DIR=/data/jobs BOOKNLP_MODEL_DIR=/opt/booknlp/models HF_HOME=/opt/booknlp/huggingface BOOKNLP_WEIGHTS_BAKED=1 BOOKNLP_SKIP_BASE_WEIGHTS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_XET=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:'+os.environ.get('PORT','8000')+'/healthz',timeout=3)"
ENTRYPOINT ["/usr/bin/tini", "-s", "--"]
CMD ["python", "-m", "service.serve"]

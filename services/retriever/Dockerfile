FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/data/huggingface

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt requirements-prod.txt ./
RUN python3 -m pip install --no-cache-dir -r requirements.txt -r requirements-prod.txt \
    && python3 -m pip install --no-cache-dir huggingface-hub==0.28.1

COPY production.py production_core.py ./

EXPOSE 7860
CMD ["python3", "-m", "uvicorn", "production:app", "--host", "0.0.0.0", "--port", "7860"]

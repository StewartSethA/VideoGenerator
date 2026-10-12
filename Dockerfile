FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg espeak-ng \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir uv==0.13.0
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --locked --no-dev && useradd --create-home --uid 1000 editor \
    && mkdir /data && chown editor:editor /data
USER editor
WORKDIR /data
ENV HF_HOME=/data/.cache/huggingface
CMD ["/app/.venv/bin/videogenerator", "serve", "--workspace", "/data", "--no-browser"]

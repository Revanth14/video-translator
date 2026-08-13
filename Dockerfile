FROM python:3.10-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DUB_MVP_RUNS_DIRECTORY=/runs \
    DUB_MVP_POLL_SECONDS=1

RUN apt-get update \
    && apt-get install --yes --no-install-recommends ffmpeg ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN python -m pip install --no-cache-dir .

COPY docker/worker-entrypoint.sh /usr/local/bin/dub-mvp-worker
RUN chmod 0555 /usr/local/bin/dub-mvp-worker \
    && mkdir /runs \
    && chown 10001:10001 /runs

USER 10001:10001
VOLUME ["/runs"]
ENTRYPOINT ["/usr/local/bin/dub-mvp-worker"]

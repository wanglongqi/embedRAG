FROM python:3.11-slim AS base

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libgomp1 && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

WORKDIR /app
COPY pyproject.toml ./
RUN uv pip install --system ".[dev]"
COPY src/ src/
COPY config/ config/
RUN uv pip install --system -e .

# ── Writer node ──
FROM base AS writer
ENV EMBEDRAG_ROLE=writer
EXPOSE 8001 9090
CMD ["uvicorn", "embedrag.writer.app:create_writer_app", "--factory", \
     "--host", "0.0.0.0", "--port", "8001"]

# ── Query node ──
FROM base AS query
ENV EMBEDRAG_ROLE=query
EXPOSE 8000 9090
CMD ["uvicorn", "embedrag.query.app:create_query_app", "--factory", \
     "--host", "0.0.0.0", "--port", "8000"]

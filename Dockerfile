FROM ghcr.io/astral-sh/uv:python3.12-trixie-slim AS base

# Install minimal runtime deps, avoid recommended packages, clean up caches.
RUN apt-get update \
    && apt-get install -y --no-install-recommends wget ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN wget -q https://raw.githubusercontent.com/dapr/cli/master/install/install.sh -O - | bash && \
    dapr init --slim && \
    mkdir -p /app

WORKDIR /app

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1

# Copy from the cache instead of linking since it's a mounted volume
ENV UV_LINK_MODE=copy

# Use BuildKit cache mounts as before for faster dependency installs
# (these mounts require Docker BuildKit)
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project --no-dev --no-editable


COPY . /app
# RUN cp /app/.dapr/config.yaml.docker /app/.dapr/config.yaml && \
#     cp /app/.dapr/components/statestore.yaml.docker /app/.dapr/components/statestore.yaml && \
#     cp /app/.dapr/components/pubsub.yaml.docker /app/.dapr/components/pubsub.yaml && \
#     cp /app/.dapr/components/secrets.yaml.docker /app/.dapr/components/secrets.yaml
RUN cp /app/conf/llm.json.docker /app/conf/llm.json

# Install application (this will reuse cache above)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable

# Copy Dapr config & components instead of running `dapr init` in the image.
# Prefer running `dapr init` in host/dev or a separate init container.
RUN cp -a /app/.dapr/config.yaml /root/.dapr/config.yaml \
    && cp -a /app/.dapr/components/*.yaml /root/.dapr/components/ || true

# Remove build-only tools to shrink final image
RUN apt-get purge -y --auto-remove wget \
    && rm -rf /var/lib/apt/lists/* /root/.cache/uv

CMD ["/app/unified_start_with_placement.sh"]

FROM python:3.12-slim AS build

# The following does not work in Podman unless you build in Docker
# compatibility mode: <https://github.com/containers/podman/issues/8477>
# You can manually prepend every RUN script with `set -ex` too.
SHELL ["sh", "-exc"]

ENV DEBIAN_FRONTEND=noninteractive


RUN <<EOT
apt-get update -qy
apt-get install -qyy \
    -o APT::Install-Recommends=false \
    -o APT::Install-Suggests=false \
    build-essential \
    ca-certificates
EOT

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# - Silence uv complaining about not being able to use hard links,
# - tell uv to byte-compile packages for faster application startups,
# - prevent uv from accidentally downloading isolated Python builds,
# - use the existing Python installation,
# - and finally declare `/app` as the target for `uv sync`.
# Source: https://hynek.me/articles/docker-uv/
ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never \
    UV_PYTHON=python3.12 \
    UV_PROJECT_ENVIRONMENT=/app

WORKDIR /src

# Synchronize DEPENDENCIES without the application itself.
# This layer is cached until uv.lock or pyproject.toml change, which are
# only temporarily mounted into the build container.
RUN --mount=type=cache,target=/root/.cache \
    --mount=type=bind,source=hyperFakeWorker/uv.lock,target=uv.lock \
    --mount=type=bind,source=hyperFakeWorker/pyproject.toml,target=pyproject.toml \
    uv sync \
        --locked \
        --no-dev \
        --no-install-project

# Now install the rest from `/src`: The APPLICATION w/o dependencies.
# `/src` will NOT be copied into the runtime container.
COPY hyperFakeWorker/ /src
RUN --mount=type=cache,target=/root/.cache \
    uv sync \
        --locked \
        --no-dev \
        --no-editable

##########################################################################

FROM python:3.12-slim
SHELL ["sh", "-exc"]

ENV PATH=/app/bin:$PATH

RUN <<EOT
groupadd -r app
useradd -r -d /app -g app -N app
EOT

RUN <<EOT
apt-get update -qy
apt-get install -qyy \
    -o APT::Install-Recommends=false \
    -o APT::Install-Suggests=false \
    ca-certificates

apt-get clean
rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
EOT

# Copy the pre-built `/app` directory to the runtime container
# and change the ownership to user app and group app.
COPY --from=build --chown=app:app /app /app

# Copy the ONNX models to the container
COPY --chown=app:app hyperFakeWorker/models/*.onnx /app/models/

USER app
WORKDIR /app


# smoke test that the application can, in fact, be imported.
RUN <<EOT
python -V
python -Im site
python -c "import worker.main"
hyperfakeworker --help
EOT

# See <https://hynek.me/articles/docker-signals/>.
STOPSIGNAL SIGINT


# Auto-discover models script
RUN echo '#!/bin/bash\n\
args=()\n\
for model in /app/models/*.onnx; do\n\
  if [ -f "$model" ]; then\n\
    basename=$(basename "$model" .onnx)\n\
    args+=("-m" "$model" "$basename")\n\
  fi\n\
done\n\
exec hyperfakeworker "${args[@]}" \\\n\
     "--address" "localhost:50051" \\\n\
     "--database-type" "http" \\\n\
     "--runtime" "docker" \\\n\
     "--timeout" "20" \\\n\
     "--auto-remove" \\\n\
     "--log-level" "debug" \\\n\
     "--log-format" "text" \\\n\
     "--log-file" "stdout" \\\n\
     "server"' > /app/start.sh && chmod +x /app/start.sh

CMD ["/app/start.sh"]

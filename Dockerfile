FROM python:3.13-slim@sha256:e48f59cdba69a3cbd9ed2cf0f348d90d45a8d350d98134cf1070bc106ee9ed90 AS python-base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app
RUN rm -f /etc/apt/apt.conf.d/docker-clean && printf '%s\n' 'Acquire::Retries "5";' 'Acquire::http::Timeout "30";' 'Acquire::https::Timeout "30";' > /etc/apt/apt.conf.d/80-balootx-retries
# The digest-pinned Debian base records the archive timestamp used to build it.
RUN sources=/etc/apt/sources.list.d/debian.sources \
    && snapshot="$(sed -nE 's|^# http://snapshot\.debian\.org/archive/debian/([0-9]{8}T[0-9]{6}Z)$|\1|p' "$sources")" \
    && test -n "$snapshot" \
    && sed -Ei "s|^URIs: http://deb\.debian\.org/(debian(-security)?)$|URIs: https://snapshot.debian.org/archive/\1/$snapshot/\nCheck-Valid-Until: no|" "$sources" \
    && test "$(grep -c '^URIs:' "$sources")" -eq 2 \
    && test "$(grep -Fc "/$snapshot/" "$sources")" -eq 2
COPY requirements-docker.txt /opt/baloot-build/requirements-docker.txt
ENV PIP_CONSTRAINT=/opt/baloot-build/requirements-docker.txt PIP_ONLY_BINARY=:all:
ARG VERIFIER_EXTRAS=dev,grpc-dev,postgres,s3,redis,security,optimization

FROM python-base AS docker-constraints
COPY pyproject.toml ./
RUN <<'SH'
set -eu
python - "$VERIFIER_EXTRAS" >/tmp/verifier-requirements.txt <<'PY'
import sys
import tomllib

with open("pyproject.toml", "rb") as stream:
    project = tomllib.load(stream)["project"]
print(*project["dependencies"], sep="\n")
for extra in sys.argv[1].split(","):
    print(*project["optional-dependencies"][extra], sep="\n")
PY
python -m pip install --dry-run --ignore-installed --retries 0 -r /tmp/verifier-requirements.txt
SH

FROM node:26-bookworm-slim@sha256:5df79989ed685e09f0a98019db4cebfb22fb9df1a53c0161664a011f7ff6b0f1 AS remotion-build
WORKDIR /remotion
COPY src/baloot_platform/video/remotion/package.json src/baloot_platform/video/remotion/package-lock.json ./
RUN --mount=type=cache,id=balootx-remotion-npm,target=/root/.npm,sharing=locked sha256sum package-lock.json > .lock-sha256 && npm ci --ignore-scripts
RUN node --input-type=module -e "import {ensureBrowser} from '@remotion/renderer'; await ensureBrowser({logLevel: 'warn'});"

FROM remotion-build AS foundry-build
WORKDIR /foundry
COPY src/baloot_platform/service/foundry_dashboard/package.json src/baloot_platform/service/foundry_dashboard/package-lock.json ./
RUN --mount=type=cache,id=balootx-foundry-npm,target=/root/.npm,sharing=locked npm ci --ignore-scripts

FROM remotion-build AS remotion-runtime
RUN --mount=type=cache,id=balootx-remotion-npm,target=/root/.npm,sharing=locked npm prune --omit=dev --ignore-scripts

FROM python-base AS build
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked apt-get update && apt-get install -y --no-install-recommends build-essential cmake ninja-build git ffmpeg fonts-dejavu-core fonts-noto-core fonts-noto-extra && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml LICENSE THIRD-PARTY-NOTICES ./
COPY src ./src
COPY native ./native
RUN --mount=type=cache,target=/root/.cache/pip python -m pip install build setuptools wheel && python -m build --no-isolation --wheel
RUN cmake -S native -B /tmp/native -G Ninja -DCMAKE_BUILD_TYPE=Release && cmake --build /tmp/native && ctest --test-dir /tmp/native --output-on-failure

FROM python-base AS runtime
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked apt-get update && apt-get install -y --no-install-recommends ffmpeg fonts-noto-core fonts-noto-extra ca-certificates curl && rm -rf /var/lib/apt/lists/* && useradd --uid 10001 --create-home baloot
COPY --from=build /app/dist/*.whl /tmp/
RUN --mount=type=cache,target=/root/.cache/pip pip install /tmp/*.whl && rm -f /tmp/*.whl
COPY --from=build /tmp/native/libbaloot_native.so /usr/local/lib/libbaloot_native.so
COPY configs /app/configs
COPY schemas /app/schemas
COPY generated /app/generated
COPY release/saudi-baloot-official-source-verification-20260831.json release/saudi-baloot-official-source-verification-20260831-v2.json /app/release/
COPY proto /app/proto
COPY migrations /app/migrations
COPY scripts /app/scripts
COPY assets /app/assets
RUN mkdir -p /app/outputs /app/.runtime && chown -R baloot:baloot /app/outputs /app/.runtime
USER 10001
ENV BALOOT_ROOT=/app BALOOT_NATIVE_LIBRARY=/usr/local/lib/libbaloot_native.so BALOOT_ENV=production
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/healthz',timeout=3)"
CMD ["baloot-api"]

FROM runtime AS production-runtime
USER root
RUN --mount=type=cache,target=/root/.cache/pip pip install "psycopg[binary,pool]>=3.2" "boto3>=1.37"
USER 10001

FROM production-runtime AS video-worker
USER root
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked apt-get update && apt-get install -y --no-install-recommends libnss3 libdbus-1-3 libatk1.0-0 libgbm1 libasound2t64 libxrandr2 libxkbcommon0 libxfixes3 libxcomposite1 libxdamage1 libatk-bridge2.0-0 libpango-1.0-0 libcairo2 libcups2 && rm -rf /var/lib/apt/lists/*
COPY --from=remotion-runtime /usr/local/bin/node /usr/local/bin/node
COPY --from=remotion-runtime /remotion/node_modules /opt/baloot-remotion/node_modules
COPY --from=remotion-runtime /remotion/.lock-sha256 /opt/baloot-remotion/.lock-sha256
RUN renderer_dir="$(python -c 'from pathlib import Path; import baloot_platform.video.remotion_renderer as module; print(Path(module.__file__).parent / "remotion")')" && ln -s /opt/baloot-remotion/node_modules "$renderer_dir/node_modules" && chown -h baloot:baloot "$renderer_dir/node_modules"
USER 10001
HEALTHCHECK NONE
CMD ["baloot-worker"]

FROM production-runtime AS grpc
USER root
RUN --mount=type=cache,target=/root/.cache/pip pip install "grpcio>=1.74" "protobuf>=5.29"
USER 10001
EXPOSE 50051
HEALTHCHECK NONE
CMD ["baloot-grpc"]

FROM production-runtime AS mcp
EXPOSE 8000
HEALTHCHECK NONE
CMD ["baloot-mcp"]

FROM build AS test
COPY configs ./configs
COPY schemas ./schemas
COPY generated ./generated
COPY release/saudi-baloot-official-source-verification-20260831.json release/saudi-baloot-official-source-verification-20260831-v2.json /app/release/
COPY proto ./proto
COPY tests ./tests
RUN --mount=type=cache,target=/root/.cache/pip wheel_path="$(echo /app/dist/baloot_platform-*.whl)" && python -m pip install --no-cache-dir "${wheel_path}[${VERIFIER_EXTRAS}]" && PYTHONPATH=src pytest -q && ctest --test-dir /tmp/native --output-on-failure

FROM python-base AS verifier
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
COPY --from=build /app/dist/*.whl /tmp/
COPY --from=build /tmp/native/libbaloot_native.so /usr/local/lib/libbaloot_native.so
COPY --from=build /tmp/native/libbaloot_native.so /app/build/native-release/libbaloot_native.so
COPY --from=build /tmp/native/baloot_native_tests /usr/local/bin/baloot_native_tests
COPY --from=remotion-build /usr/local/bin/node /usr/local/bin/node
COPY --from=remotion-build /remotion/node_modules ./src/baloot_platform/video/remotion/node_modules
COPY --from=remotion-build /remotion/.lock-sha256 ./src/baloot_platform/video/remotion/.lock-sha256
COPY --from=foundry-build /foundry/node_modules ./src/baloot_platform/service/foundry_dashboard/node_modules
RUN --mount=type=cache,target=/root/.cache/pip wheel_path="$(echo /tmp/baloot_platform-*.whl)" && python -m pip install --no-cache-dir "${wheel_path}[${VERIFIER_EXTRAS}]" && rm -f /tmp/*.whl
COPY . .
ENV BALOOT_ROOT=/app BALOOT_NATIVE_LIBRARY=/usr/local/lib/libbaloot_native.so LD_LIBRARY_PATH=/usr/local/lib BALOOT_VERIFIER_TEST_TIMEOUT_SECONDS=1800 SOURCE_DATE_EPOCH=315532800
ENV PATH=/app/src/baloot_platform/video/remotion/node_modules/.bin:$PATH
CMD ["python", "scripts/verifier_preflight.py"]

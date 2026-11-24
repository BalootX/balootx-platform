from __future__ import annotations

import contextlib
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from threading import RLock
from typing import Any, Callable, Iterator, Mapping, MutableMapping

from baloot_platform.version import __version__


SEMANTIC_ATTRIBUTES = {
    "game_id": "baloot.game.id", "round_id": "baloot.round.id", "decision_id": "baloot.decision.id",
    "bundle_hash": "baloot.bundle.hash", "model_hash": "baloot.model.hash", "observer_seat": "baloot.observer.seat",
    "plugin_id": "baloot.plugin.id", "job_id": "baloot.job.id", "shared_board_id": "baloot.shared.board.id",
    "tenant_id": "baloot.tenant.id", "configuration_family": "baloot.configuration.family",
    "configuration_profile_id": "baloot.configuration.profile.id", "configuration_sha256": "baloot.configuration.sha256",
    "configuration_bundle_sha256": "baloot.configuration.bundle.sha256", "plugin_isolation": "baloot.plugin.isolation",
    "search_budget": "baloot.search.budget", "analysis_method": "baloot.analysis.method", "request_id": "baloot.request.id",
    "room_id": "baloot.live.room.id", "operation_id": "baloot.operation.id", "cache_result": "baloot.cache.result",
}

_FORBIDDEN_ATTRIBUTE_PARTS = {"hand", "hands", "cards", "hidden", "private", "password", "token", "secret", "voice"}
EVENT_LOGGER = logging.getLogger("baloot_platform.events")


def configure_event_logging() -> None:
    if not EVENT_LOGGER.handlers:
        EVENT_LOGGER.addHandler(logging.StreamHandler())
    EVENT_LOGGER.setLevel(logging.INFO)
    EVENT_LOGGER.propagate = False


def log_event(event: str, *, level: int = logging.INFO, **fields: str | int | None) -> None:
    EVENT_LOGGER.log(level, json.dumps({"event": event, **fields}))


def log_worker_result(worker: str, job_id: str | None, status: str) -> None:
    log_event(
        "worker.job_result",
        level=logging.ERROR if status == "failed" else logging.INFO,
        trace_id=uuid.uuid4().hex,
        worker=worker,
        job_id=job_id,
        status=status,
    )


def run_worker(worker: str, run: Callable[[], None]) -> None:
    configure_event_logging()
    try:
        run()
    except Exception as exc:
        log_event(
            "worker.failed",
            level=logging.ERROR,
            trace_id=uuid.uuid4().hex,
            worker=worker,
            error_type=type(exc).__name__,
        )
        raise SystemExit(1) from None


class ServerDiagnosticFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        # Transport diagnostics may contain request values in both arguments and exceptions.
        fields = {
            "event": "server.diagnostic",
            "trace_id": uuid.uuid4().hex,
            "level": record.levelname,
            "source": f"{record.module}.{record.funcName}:{record.lineno}",
        }
        if record.exc_info and record.exc_info[0]:
            fields["error_type"] = record.exc_info[0].__name__
        return json.dumps(fields)


@dataclass(slots=True)
class TelemetrySnapshot:
    counters: dict[str, int] = field(default_factory=dict)
    histograms: dict[str, list[float]] = field(default_factory=dict)


class Telemetry:
    """OpenTelemetry SDK/OTLP integration with a deterministic fallback.

    Hidden cards, credentials and personal payloads are rejected from semantic
    attributes. The fallback remains testable when exporters are disabled.
    """

    def __init__(self, service_name: str = "baloot-platform"):
        self.service_name = service_name; self._lock = RLock(); self._snapshot = TelemetrySnapshot()
        self._tracer = None; self._meter = None; self._counter_instruments: dict[str, Any] = {}; self._histogram_instruments: dict[str, Any] = {}
        self._propagate = None
        self._configure_otel()

    def _configure_otel(self) -> None:
        try:
            from opentelemetry import metrics, propagate, trace
            from opentelemetry.sdk.metrics import MeterProvider
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor
            resource = Resource.create({"service.name": self.service_name, "service.version": os.environ.get("BALOOT_VERSION", __version__)})
            if not isinstance(trace.get_tracer_provider(), TracerProvider):
                provider = TracerProvider(resource=resource)
                endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "").strip()
                if endpoint:
                    try:
                        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
                        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint.rstrip("/") + "/v1/traces")))
                    except Exception:
                        pass
                trace.set_tracer_provider(provider)
            if not isinstance(metrics.get_meter_provider(), MeterProvider):
                metrics.set_meter_provider(MeterProvider(resource=resource))
            self._tracer = trace.get_tracer(self.service_name)
            self._meter = metrics.get_meter(self.service_name)
            self._propagate = propagate
        except Exception:
            self._tracer = None; self._meter = None; self._propagate = None

    @staticmethod
    def attributes(values: Mapping[str, Any] | None = None, **kwargs: Any) -> dict[str, Any]:
        source = dict(values or {}); source.update(kwargs); result: dict[str, Any] = {}
        for key, value in source.items():
            normalized = str(key).lower().replace("-", "_")
            if value is None or any(part in normalized for part in _FORBIDDEN_ATTRIBUTE_PARTS):
                continue
            mapped = SEMANTIC_ATTRIBUTES.get(key, key)
            if isinstance(value, (str, bool, int, float)):
                result[mapped] = value
            elif isinstance(value, (tuple, list)) and all(isinstance(item, (str, bool, int, float)) for item in value):
                result[mapped] = list(value)
        return result

    def extract_context(self, carrier: Mapping[str, str]) -> Any:
        return self._propagate.extract(dict(carrier)) if self._propagate is not None else None

    def inject_context(self, carrier: MutableMapping[str, str], context: Any | None = None) -> MutableMapping[str, str]:
        if self._propagate is not None:
            self._propagate.inject(carrier, context=context)
        return carrier

    @contextlib.contextmanager
    def span(self, name: str, attributes: Mapping[str, Any] | None = None, context: Any | None = None, **kwargs: Any) -> Iterator[None]:
        attrs = self.attributes(attributes, **kwargs); start = time.perf_counter()
        if self._tracer is None:
            try: yield
            finally: self.observe(f"span.{name}.duration_ms", (time.perf_counter() - start) * 1000.0, attributes=attrs)
            return
        with self._tracer.start_as_current_span(name, context=context, attributes=attrs):
            try: yield
            finally: self.observe(f"span.{name}.duration_ms", (time.perf_counter() - start) * 1000.0, attributes=attrs)

    def increment(self, name: str, amount: int = 1, *, attributes: Mapping[str, Any] | None = None) -> None:
        with self._lock: self._snapshot.counters[name] = self._snapshot.counters.get(name, 0) + int(amount)
        if self._meter is not None:
            instrument = self._counter_instruments.get(name)
            if instrument is None:
                instrument = self._meter.create_counter(name); self._counter_instruments[name] = instrument
            instrument.add(int(amount), self.attributes(attributes))

    def observe(self, name: str, value: float, *, attributes: Mapping[str, Any] | None = None) -> None:
        with self._lock:
            values = self._snapshot.histograms.setdefault(name, []); values.append(float(value))
            if len(values) > 4096: del values[: len(values) - 4096]
        if self._meter is not None:
            instrument = self._histogram_instruments.get(name)
            if instrument is None:
                instrument = self._meter.create_histogram(name); self._histogram_instruments[name] = instrument
            instrument.record(float(value), self.attributes(attributes))

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {"counters": dict(self._snapshot.counters), "histograms": {key: list(values) for key, values in self._snapshot.histograms.items()},
                    "semanticAttributes": dict(SEMANTIC_ATTRIBUTES), "otlpConfigured": bool(os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")),
                    "sensitiveAttributePolicy": "deny"}


TELEMETRY = Telemetry()

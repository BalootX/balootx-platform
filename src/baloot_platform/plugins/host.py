from __future__ import annotations

# ruff: noqa: E701, E702

import json
import os
import selectors
import signal
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class PluginHostPolicy:
    timeout_seconds: float
    maximum_response_bytes: int
    maximum_memory_mb: int
    maximum_cpu_seconds: int
    allow_network: bool
    allow_filesystem: bool
    maximum_stderr_bytes: int


def _resource_limiter(policy: PluginHostPolicy):
    if os.name != "posix": return None
    def apply() -> None:
        try:
            import resource
            memory = max(64, int(policy.maximum_memory_mb)) * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
            cpu = max(1, int(policy.maximum_cpu_seconds)); resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu + 1))
            resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
            os.umask(0o077)
        except Exception:
            pass
    return apply


class SubprocessPluginHost:
    """Bounded JSON-RPC sidecar host with continuous stderr drainage."""

    def __init__(self, command: list[str], *, policy: PluginHostPolicy,
                 environment: dict[str, str] | None = None):
        if not command: raise ValueError("plugin command is required")
        self.command = list(command); self.policy = policy
        clean_env = {"PATH": os.environ.get("PATH", ""), "LANG": "C.UTF-8", "PYTHONUNBUFFERED": "1",
                     "BALOOT_PLUGIN_NETWORK": "1" if self.policy.allow_network else "0",
                     "BALOOT_PLUGIN_FILESYSTEM": "1" if self.policy.allow_filesystem else "0"}
        clean_env.update(environment or {})
        self.process = subprocess.Popen(self.command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                        bufsize=0, env=clean_env, start_new_session=True, preexec_fn=_resource_limiter(self.policy))
        self._lock = threading.RLock(); self._sequence = 0; self._stderr = deque(); self._stderr_size = 0
        self._stderr_thread = threading.Thread(target=self._drain_stderr, name=f"plugin-stderr-{self.process.pid}", daemon=True)
        self._stderr_thread.start()

    def _drain_stderr(self) -> None:
        if self.process.stderr is None: return
        while True:
            chunk = self.process.stderr.read(4096)
            if not chunk: return
            self._stderr.append(bytes(chunk)); self._stderr_size += len(chunk)
            while self._stderr_size > self.policy.maximum_stderr_bytes and self._stderr:
                self._stderr_size -= len(self._stderr.popleft())

    def stderr_tail(self) -> str:
        return b"".join(self._stderr).decode("utf-8", errors="replace")

    def _terminate(self) -> None:
        if self.process.poll() is not None: return
        try:
            (os.killpg(self.process.pid, signal.SIGTERM) if os.name == "posix" else self.process.terminate())
            self.process.wait(timeout=1.5)
        except Exception:
            try: os.killpg(self.process.pid, signal.SIGKILL) if os.name == "posix" else self.process.kill()
            except Exception: pass

    def _read_line(self) -> bytes:
        assert self.process.stdout is not None
        if os.name == "nt":
            return self._read_line_windows()
        selector = selectors.DefaultSelector(); selector.register(self.process.stdout, selectors.EVENT_READ)
        deadline = time.monotonic() + max(0.05, float(self.policy.timeout_seconds)); value = bytearray()
        try:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0: self._terminate(); raise TimeoutError("plugin response timed out")
                ready = selector.select(remaining)
                if not ready: self._terminate(); raise TimeoutError("plugin response timed out")
                chunk = os.read(self.process.stdout.fileno(), min(65536, self.policy.maximum_response_bytes + 1 - len(value)))
                if not chunk: raise RuntimeError(f"plugin process closed its response stream: {self.stderr_tail()[-1000:]}")
                value.extend(chunk)
                if len(value) > self.policy.maximum_response_bytes: self._terminate(); raise RuntimeError("plugin response exceeds configured limit")
                newline = value.find(b"\n")
                if newline >= 0:
                    if value[newline + 1:].strip(): self._terminate(); raise RuntimeError("plugin emitted more than one response frame")
                    return bytes(value[:newline])
        finally: selector.close()

    def _read_line_windows(self) -> bytes:
        """Read a bounded response from a Windows pipe without socket selectors."""

        import ctypes
        import msvcrt

        assert self.process.stdout is not None
        handle = msvcrt.get_osfhandle(self.process.stdout.fileno())
        deadline = time.monotonic() + max(0.05, float(self.policy.timeout_seconds))
        value = bytearray()
        available = ctypes.c_ulong(0)
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self._terminate()
                raise TimeoutError("plugin response timed out")
            ok = ctypes.windll.kernel32.PeekNamedPipe(
                handle, None, 0, None, ctypes.byref(available), None
            )
            if not ok:
                if self.process.poll() is not None:
                    raise RuntimeError(
                        f"plugin process closed its response stream: {self.stderr_tail()[-1000:]}"
                    )
                raise OSError(ctypes.get_last_error(), "unable to inspect plugin response pipe")
            if available.value == 0:
                if self.process.poll() is not None:
                    raise RuntimeError(
                        f"plugin process closed its response stream: {self.stderr_tail()[-1000:]}"
                    )
                time.sleep(min(0.01, remaining))
                continue
            capacity = self.policy.maximum_response_bytes + 1 - len(value)
            chunk = os.read(self.process.stdout.fileno(), min(65_536, available.value, capacity))
            if not chunk:
                raise RuntimeError(
                    f"plugin process closed its response stream: {self.stderr_tail()[-1000:]}"
                )
            value.extend(chunk)
            if len(value) > self.policy.maximum_response_bytes:
                self._terminate()
                raise RuntimeError("plugin response exceeds configured limit")
            newline = value.find(b"\n")
            if newline >= 0:
                if value[newline + 1 :].strip():
                    self._terminate()
                    raise RuntimeError("plugin emitted more than one response frame")
                return bytes(value[:newline])

    def call(self, method: str, params: dict[str, Any] | None = None) -> Any:
        with self._lock:
            if self.process.poll() is not None: raise RuntimeError("plugin process is not running")
            self._sequence += 1; request = {"jsonrpc": "2.0", "id": self._sequence, "method": str(method), "params": params or {}}
            assert self.process.stdin is not None
            self.process.stdin.write(json.dumps(request, ensure_ascii=False, separators=(",", ":")).encode() + b"\n"); self.process.stdin.flush()
            response = json.loads(self._read_line().decode("utf-8"))
            if response.get("jsonrpc") != "2.0" or response.get("id") != self._sequence: raise RuntimeError("plugin response identity mismatch")
            if "error" in response: raise RuntimeError(str(response["error"]))
            return response.get("result")

    def health(self) -> dict[str, Any]:
        return dict(self.call("health", {}))

    def close(self) -> None:
        self._terminate(); self._stderr_thread.join(timeout=0.5)
    def __enter__(self): return self
    def __exit__(self, *_args): self.close()


class WasmtimeComponentHost:
    """Deny-by-default WebAssembly Component host.

    The optional Wasmtime dependency is required only for external components.
    No WASI filesystem or network capabilities are linked by default.
    """
    def __init__(self, component: str | Path, *, policy: PluginHostPolicy):
        self.component = Path(component); self.policy = policy; self._closed = False
        try:
            import wasmtime
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Wasmtime component hosting requires the optional wasmtime package") from exc
        if not self.component.is_file(): raise FileNotFoundError(self.component)
        config = wasmtime.Config(); config.consume_fuel = True; config.epoch_interruption = True
        self.engine = wasmtime.Engine(config); self.store = wasmtime.Store(self.engine)
        try: self.store.set_fuel(max(100_000, int(self.policy.maximum_cpu_seconds) * 1_000_000))
        except Exception: pass
        self._wasmtime = wasmtime
        # Component API support varies by package release. Instantiate only
        # through a statically typed adapter; no WASI capabilities are granted.
        if not hasattr(wasmtime, "Component"):
            raise RuntimeError("installed Wasmtime package lacks Component Model support")
        self.instance = None

    def call(self, method: str, params: dict[str, Any] | None = None) -> Any:
        raise RuntimeError("component must be bound to the versioned Baloot WIT world before invocation")
    def close(self) -> None: self._closed = True
    def __enter__(self): return self
    def __exit__(self, *_args): self.close()

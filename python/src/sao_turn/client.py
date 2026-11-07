"""Serialized PCM16/16 kHz JSONL client; failures close the worker before reuse."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import asdict, dataclass, fields
import json
import math
from pathlib import Path
import struct
from typing import Any


class TurnClientError(RuntimeError):
    """Native startup, transport, or lifecycle failure."""


class TurnClientTimeout(TurnClientError):
    """The subprocess failed to answer within its configured deadline."""


class TurnProtocolError(TurnClientError):
    """The subprocess did not produce the expected bounded JSONL schema."""


def _integer(name: str, value: Any, minimum: int, maximum: int) -> None:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")


def _probability(name: str, value: Any) -> None:
    if type(value) not in (float, int) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be a finite probability in [0, 1]")


@dataclass(frozen=True, slots=True)
class TurnContext:
    """Replacement context for the current turn; transcripts are not appended."""

    agent_speaking: bool = False
    mode: str = "conversation"
    expected_digits: int | None = None
    text: str = ""
    text_final: bool = False

    def __post_init__(self) -> None:
        if type(self.agent_speaking) is not bool or type(self.text_final) is not bool:
            raise ValueError("agent_speaking and text_final must be booleans")
        if self.mode not in ("conversation", "digits"):
            raise ValueError("mode must be 'conversation' or 'digits'")
        if self.expected_digits is not None:
            _integer("expected_digits", self.expected_digits, 1, 128)
        if not isinstance(self.text, str) or len(self.text.encode("utf-8")) > 65_536:
            raise ValueError("text must be a string of at most 65536 UTF-8 bytes")

    @classmethod
    def from_value(cls, value: TurnContext | Mapping[str, Any] | None) -> TurnContext:
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise ValueError("context must be TurnContext or a mapping")
        unknown = set(value) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f"unknown context fields: {sorted(unknown, key=str)}")
        return cls(**dict(value))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> TurnContext:
        """Strict mapping constructor used by HTTP/WebSocket adapters."""
        return cls.from_value(value)


@dataclass(frozen=True, slots=True)
class TurnConfig:
    """Rust policy configuration, validated with the same bounds as Config."""

    speech_start_threshold: float = 0.65
    speech_end_threshold: float = 0.35
    completion_threshold: float = 0.50
    min_voiced_ms: int = 200
    min_silence_ms: int = 200
    max_silence_ms: int = 8_000
    max_backchannel_ms: int = 1_200
    early_silence_ms: int | None = None
    early_completion_threshold: float | None = None

    def __post_init__(self) -> None:
        for name in ("speech_start_threshold", "speech_end_threshold", "completion_threshold"):
            _probability(name, getattr(self, name))
        for name in ("min_voiced_ms", "min_silence_ms", "max_silence_ms", "max_backchannel_ms"):
            _integer(name, getattr(self, name), 1, 2**64 - 1)
        if self.speech_end_threshold >= self.speech_start_threshold:
            raise ValueError("speech_end_threshold must be below speech_start_threshold")
        if self.max_silence_ms < self.min_silence_ms:
            raise ValueError("max_silence_ms must be at least min_silence_ms")
        if self.max_backchannel_ms < self.min_voiced_ms:
            raise ValueError("max_backchannel_ms must be at least min_voiced_ms")
        gate, threshold = self.early_silence_ms, self.early_completion_threshold
        if (gate is None) != (threshold is None):
            raise ValueError("early_silence_ms and early_completion_threshold must be set together")
        if gate is not None and threshold is not None:
            _integer("early_silence_ms", gate, 1, self.min_silence_ms - 1)
            _probability("early_completion_threshold", threshold)
            if threshold < self.completion_threshold:
                raise ValueError("early_completion_threshold must be at least completion_threshold")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> TurnConfig:
        if not isinstance(value, Mapping):
            raise ValueError("config must be a mapping")
        unknown = set(value) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f"unknown config fields: {sorted(unknown, key=str)}")
        return cls(**dict(value))

    def to_dict(self) -> dict[str, Any]:
        return {name: value for name, value in asdict(self).items() if value is not None}


def _reject_constant(value: str) -> None:
    raise ValueError(f"nonfinite JSON number: {value}")


def _validate_response(response: Any) -> dict[str, Any]:
    if not isinstance(response, dict) or set(response) != {"frames", "buffered_samples"}:
        raise ValueError("response must contain frames and buffered_samples")
    _integer("buffered_samples", response["buffered_samples"], 0, 511)
    if not isinstance(response["frames"], list):
        raise ValueError("frames must be a list")
    previous = None
    for frame in response["frames"]:
        if not isinstance(frame, dict):
            raise ValueError("frame must be an object")
        _integer("at_ms", frame.get("at_ms"), 0, 2**64 - 1)
        if previous is not None and frame["at_ms"] <= previous:
            raise ValueError("native frame timestamps must increase")
        previous = frame["at_ms"]
        _probability("speech_probability", frame.get("speech_probability"))
        if frame.get("completion_probability") is not None:
            _probability("completion_probability", frame["completion_probability"])
        for name in ("decision", "baseline_decision"):
            if name not in frame:
                raise ValueError(f"frame is missing {name}")
            decision = frame[name]
            if decision is None:
                continue
            if not isinstance(decision, dict) or decision.get("kind") not in {
                "continue", "end", "wait", "backchannel", "interrupt"
            }:
                raise ValueError(f"invalid {name}")
            if type(decision.get("forced")) is not bool or not isinstance(decision.get("reason"), str):
                raise ValueError(f"invalid {name} reason or forced flag")
            for key in ("at_ms", "turn_start_ms", "last_speech_ms", "voiced_ms"):
                _integer(key, decision.get(key), 0, 2**64 - 1)
            if decision["at_ms"] != frame["at_ms"]:
                raise ValueError("decision and frame timestamps disagree")
            if decision.get("score") is not None:
                _probability("score", decision["score"])
    return response


class SaoTurnClient:
    """Async client for ``sao-turn ... stream``; never invokes a shell.

    Paths are converted to absolute paths at construction. ``config`` is an
    optional JSON file path; use ``TurnConfig.to_dict()`` when authoring it.
    Pass ``joint_model`` for the joint backend, or ``model`` and ``vad_model``
    for the legacy pair.
    Native readiness is confirmed by an empty reset packet after model loading.
    """

    def __init__(
        self,
        *,
        executable: str | Path,
        model: str | Path | None = None,
        vad_model: str | Path | None = None,
        joint_model: str | Path | None = None,
        runtime: str | Path | None = None,
        config: str | Path | None = None,
        threads: int = 1,
        request_timeout: float = 10.0,
        startup_timeout: float = 60.0,
        shutdown_timeout: float = 3.0,
        max_response_bytes: int = 4 * 1024 * 1024,
        stderr_limit: int = 65_536,
    ) -> None:
        _integer("threads", threads, 1, 128)
        _integer("max_response_bytes", max_response_bytes, 256, 64 * 1024 * 1024)
        _integer("stderr_limit", stderr_limit, 0, 4 * 1024 * 1024)
        for name, value in (("request_timeout", request_timeout), ("startup_timeout", startup_timeout), ("shutdown_timeout", shutdown_timeout)):
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a positive finite duration")
        if joint_model is None and (model is None or vad_model is None):
            raise ValueError("provide model and vad_model, or joint_model")
        if joint_model is not None and (model is not None or vad_model is not None):
            raise ValueError("joint_model cannot be combined with model or vad_model")
        self.executable = Path(executable).expanduser().resolve()
        self.model = Path(model).expanduser().resolve() if model is not None else None
        self.vad_model = Path(vad_model).expanduser().resolve() if vad_model is not None else None
        self.joint_model = Path(joint_model).expanduser().resolve() if joint_model is not None else None
        self.runtime = Path(runtime).expanduser().resolve() if runtime is not None else None
        self.config = Path(config).expanduser().resolve() if config is not None else None
        self.threads = threads
        self.request_timeout = float(request_timeout)
        self.startup_timeout = float(startup_timeout)
        self.shutdown_timeout = float(shutdown_timeout)
        self.max_response_bytes = max_response_bytes
        self.stderr_limit = stderr_limit
        self._lock = asyncio.Lock()
        self._process: asyncio.subprocess.Process | None = None
        self._stderr_task: asyncio.Task[None] | None = None
        self._stderr = bytearray()
        self._context = TurnContext()
        self._last_at_ms: int | None = None

    @property
    def running(self) -> bool:
        return self._process is not None and self._process.returncode is None

    @property
    def stderr_tail(self) -> str:
        return self._stderr.decode("utf-8", errors="replace")

    @property
    def context(self) -> TurnContext:
        return self._context

    def _command(self) -> list[str]:
        for name in ("executable", "model", "vad_model", "joint_model", "runtime", "config"):
            path = getattr(self, name)
            if path is not None and not path.is_file():
                raise ValueError(f"{name} is not a file: {path}")
        if self.config is not None:
            if self.config.stat().st_size > 65_536:
                raise ValueError("config JSON exceeds 65536 bytes")
            TurnConfig.from_dict(json.loads(self.config.read_text(encoding="utf-8"), parse_constant=_reject_constant))
        command = [str(self.executable), "--threads", str(self.threads)]
        if self.joint_model is not None:
            command.extend(["--joint-model", str(self.joint_model)])
        else:
            command.extend(["--model", str(self.model), "--vad-model", str(self.vad_model)])
        if self.runtime is not None:
            command.extend(["--runtime", str(self.runtime)])
        if self.config is not None:
            command.extend(["--config", str(self.config)])
        return command + ["stream"]

    async def start(self) -> None:
        async with self._lock:
            if self.running:
                return
            await self._close_unlocked()
            command = self._command()
            self._stderr.clear()
            self._context = TurnContext()
            self._last_at_ms = None
            try:
                self._process = await asyncio.wait_for(
                    asyncio.create_subprocess_exec(
                        *command, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE, limit=self.max_response_bytes + 1,
                    ), self.startup_timeout,
                )
                self._stderr_task = asyncio.create_task(self._drain_stderr(self._process), name="sao-turn-stderr")
                response = await asyncio.wait_for(
                    self._exchange({"samples": [], "context": self._context.to_dict(), "reset": True}),
                    self.startup_timeout,
                )
                if response["frames"] or response["buffered_samples"] != 0:
                    raise TurnProtocolError("startup reset did not produce an empty stream")
            except BaseException as error:
                await asyncio.shield(self._close_unlocked())
                if isinstance(error, asyncio.CancelledError):
                    raise
                self._raise_failure("startup", error)

    async def request(
        self,
        samples: Sequence[int] = (),
        *,
        context: TurnContext | Mapping[str, Any] | None = None,
        reset: bool = False,
    ) -> dict[str, Any]:
        if type(reset) is not bool:
            raise ValueError("reset must be boolean")
        if len(samples) > 160_000:
            raise ValueError("one request may contain at most ten seconds of audio")
        values = list(samples)
        for sample in values:
            _integer("PCM sample", sample, -32_768, 32_767)
        new_context = TurnContext.from_value(context) if context is not None else (TurnContext() if reset else None)
        packet: dict[str, Any] = {"samples": values, "reset": reset}
        if new_context is not None:
            packet["context"] = new_context.to_dict()
        async with self._lock:
            if not self.running:
                await self._close_unlocked()
                raise TurnClientError("native client is not running; call start() first")
            try:
                response = await asyncio.wait_for(self._exchange(packet), self.request_timeout)
                if reset:
                    self._last_at_ms = None
                frames = response["frames"]
                if frames:
                    if self._last_at_ms is not None and frames[0]["at_ms"] <= self._last_at_ms:
                        raise TurnProtocolError("native clock moved backwards without reset")
                    self._last_at_ms = frames[-1]["at_ms"]
                if new_context is not None:
                    self._context = new_context
                return response
            except BaseException as error:
                await asyncio.shield(self._close_unlocked())
                if isinstance(error, asyncio.CancelledError):
                    raise
                self._raise_failure("request", error)

    async def process_audio(
        self, pcm16: bytes | bytearray | memoryview, *, context: TurnContext | Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not isinstance(pcm16, (bytes, bytearray, memoryview)):
            raise ValueError("audio must contain little-endian PCM16 bytes")
        audio = bytes(pcm16)
        if len(audio) % 2:
            raise ValueError("PCM16 byte length must be even")
        if len(audio) > 320_000:
            raise ValueError("one request may contain at most ten seconds of audio")
        samples = list(struct.unpack(f"<{len(audio) // 2}h", audio))
        return await self.request(samples, context=context)

    async def update_context(self, context: TurnContext | Mapping[str, Any]) -> dict[str, Any]:
        return await self.request(context=context)

    async def reset(self, context: TurnContext | Mapping[str, Any] | None = None) -> dict[str, Any]:
        return await self.request(context=TurnContext.from_value(context), reset=True)

    async def close(self) -> None:
        async with self._lock:
            await self._close_unlocked()

    async def __aenter__(self) -> SaoTurnClient:
        await self.start()
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.close()

    async def _exchange(self, packet: dict[str, Any]) -> dict[str, Any]:
        process = self._process
        if process is None or process.stdin is None or process.stdout is None:
            raise TurnClientError("native pipes are unavailable")
        message = json.dumps(packet, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("utf-8") + b"\n"
        if len(message) > 2_000_000:
            raise ValueError("encoded request exceeds native 2 MB packet limit")
        process.stdin.write(message)
        await process.stdin.drain()
        try:
            line = await process.stdout.readline()
        except ValueError as error:
            # asyncio StreamReader reports its own line-length limit as a
            # ValueError before the explicit bounded-line check below.
            raise TurnProtocolError("native response exceeded the stream reader limit") from error
        if not line:
            raise TurnProtocolError("native stdout closed before a response")
        if len(line) > self.max_response_bytes or not line.endswith(b"\n"):
            raise TurnProtocolError("native response exceeded limit or ended without a newline")
        try:
            return _validate_response(json.loads(line, parse_constant=_reject_constant))
        except (ValueError, TypeError, KeyError, UnicodeDecodeError) as error:
            raise TurnProtocolError(f"invalid native response: {error}") from error

    async def _drain_stderr(self, process: asyncio.subprocess.Process) -> None:
        if process.stderr is None:
            return
        try:
            while chunk := await process.stderr.read(4096):
                self._stderr.extend(chunk)
                if len(self._stderr) > self.stderr_limit:
                    del self._stderr[:len(self._stderr) - self.stderr_limit]
        except (OSError, RuntimeError):
            return

    async def _close_unlocked(self) -> None:
        process, self._process = self._process, None
        stderr_task, self._stderr_task = self._stderr_task, None
        try:
            if process is not None:
                if process.stdin is not None:
                    with suppress(BrokenPipeError, ConnectionResetError, RuntimeError):
                        process.stdin.close()
                try:
                    await asyncio.wait_for(process.wait(), self.shutdown_timeout)
                except asyncio.TimeoutError:
                    with suppress(ProcessLookupError):
                        process.terminate()
                    try:
                        await asyncio.wait_for(process.wait(), self.shutdown_timeout)
                    except asyncio.TimeoutError:
                        with suppress(ProcessLookupError):
                            process.kill()
                        await asyncio.wait_for(process.wait(), self.shutdown_timeout)
        except asyncio.CancelledError:
            # Teardown itself may be cancelled by a pipeline shutdown. The
            # detached process must still be reaped before this task exits.
            if process is not None and process.returncode is None:
                with suppress(ProcessLookupError):
                    process.kill()
                await asyncio.shield(asyncio.wait_for(process.wait(), self.shutdown_timeout))
            raise
        finally:
            if stderr_task is not None:
                try:
                    await asyncio.wait_for(asyncio.shield(stderr_task), self.shutdown_timeout)
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    stderr_task.cancel()
                    with suppress(asyncio.CancelledError):
                        await stderr_task

    def _raise_failure(self, operation: str, error: BaseException) -> None:
        detail = f"; native stderr: {self.stderr_tail}" if self.stderr_tail else ""
        kind = TurnClientTimeout if isinstance(error, asyncio.TimeoutError) else TurnClientError
        if isinstance(error, TurnProtocolError):
            kind = TurnProtocolError
        raise kind(f"native {operation} failed: {error}{detail}") from error

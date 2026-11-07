from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

from sao_turn.client import SaoTurnClient, TurnClientError, TurnClientTimeout, TurnConfig, TurnContext, TurnProtocolError


EMPTY = {"frames": [], "buffered_samples": 0}


def native_response(at_ms=32):
    return {"frames": [{"at_ms": at_ms, "speech_probability": 0.1,
                        "completion_probability": None, "decision": None,
                        "baseline_decision": None}], "buffered_samples": 0}


class FakeWriter:
    def __init__(self, process):
        self.process = process
        self.closed = False

    def write(self, message):
        if self.closed:
            raise BrokenPipeError("closed")
        self.process.packets.append(json.loads(message))
        if self.process.responses:
            response = self.process.responses.popleft()
            if response is not None:
                self.process.stdout.feed_data(response if isinstance(response, bytes) else json.dumps(response).encode() + b"\n")

    async def drain(self):
        await asyncio.sleep(0)
        if self.process.broken_pipe:
            raise BrokenPipeError("broken test pipe")

    def close(self):
        self.closed = True
        if not self.process.ignore_close:
            self.process.exit()


class FakeProcess:
    def __init__(self, responses=(), stderr=b"", ignore_close=False):
        self.responses = deque(responses)
        self.stdout = asyncio.StreamReader()
        self.stderr = asyncio.StreamReader()
        self.stderr.feed_data(stderr)
        self.stdin = FakeWriter(self)
        self.packets = []
        self.returncode = None
        self.ended = asyncio.Event()
        self.ignore_close = ignore_close
        self.terminated = False
        self.broken_pipe = False

    def exit(self):
        if self.returncode is None:
            self.returncode = 0
            self.stdout.feed_eof()
            self.stderr.feed_eof()
            self.ended.set()

    def terminate(self):
        self.terminated = True
        self.exit()

    def kill(self):
        self.exit()

    async def wait(self):
        await self.ended.wait()
        return self.returncode


class ContextTests(unittest.TestCase):
    def test_context_and_config_strict_validation(self):
        context = TurnContext.from_mapping({"mode": "digits", "expected_digits": 4, "text": "one 23"})
        self.assertEqual(context.to_dict()["expected_digits"], 4)
        for invalid in [{"text_fnal": True}, {"agent_speaking": 1}, {"mode": "credit_card"},
                        {"expected_digits": True}, {"expected_digits": 0}, {"expected_digits": 129},
                        {"text": "x" * 65_537}, {"text_final": "false"}]:
            with self.subTest(invalid=str(invalid)[:80]), self.assertRaises(ValueError):
                TurnContext.from_mapping(invalid)
        for invalid in [{"min_silence_ms": 0}, {"completion_threshold": math.nan},
                        {"speech_end_threshold": 0.9}, {"max_backchannel_ms": 1},
                        {"max_silence_ms": 100}, {"extra": True}]:
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                TurnConfig.from_dict(invalid)
        self.assertEqual(TurnConfig().completion_threshold, 0.5)

    def test_early_completion_config_round_trip(self):
        self.assertNotIn("early_silence_ms", TurnConfig().to_dict())
        config = TurnConfig(early_silence_ms=32, early_completion_threshold=0.99)
        self.assertEqual(TurnConfig.from_dict(config.to_dict()), config)
        for invalid in [
            {"early_silence_ms": 32},
            {"early_completion_threshold": 0.99},
            {"early_silence_ms": 200, "early_completion_threshold": 0.99},
            {"early_silence_ms": 32, "early_completion_threshold": 0.4},
        ]:
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                TurnConfig.from_dict(invalid)


class ClientTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.directory = tempfile.TemporaryDirectory()
        root = Path(self.directory.name)
        for name in ("turn.exe", "model.onnx", "vad.onnx", "runtime.dll"):
            (root / name).touch()
        self.options = dict(executable=root / "turn.exe", model=root / "model.onnx",
                            vad_model=root / "vad.onnx", runtime=root / "runtime.dll",
                            request_timeout=0.05, startup_timeout=0.05, shutdown_timeout=0.01)
        self.clients = []

    async def asyncTearDown(self):
        for client in self.clients:
            await client.close()
        self.directory.cleanup()

    def client(self, **kwargs):
        client = SaoTurnClient(**(self.options | kwargs))
        self.clients.append(client)
        return client

    async def test_startup_handshake_arguments_pcm_and_context_reset(self):
        process = FakeProcess([EMPTY, native_response(), EMPTY, EMPTY])
        client = self.client()
        with patch("sao_turn.client.asyncio.create_subprocess_exec", new=AsyncMock(return_value=process)) as spawn:
            await client.start()
            await client.start()
            self.assertEqual(spawn.await_count, 1)
            command = spawn.call_args.args
            self.assertEqual(command[-1], "stream")
            self.assertIn("--model", command)
            self.assertTrue(Path(command[0]).is_absolute())
            self.assertEqual(process.packets[0]["reset"], True)
            self.assertEqual(process.packets[0]["samples"], [])
            response = await client.process_audio(struct.pack("<3h", -32768, 0, 32767), context={"text": "hello"})
            self.assertEqual(response["frames"][0]["at_ms"], 32)
            self.assertEqual(process.packets[1]["samples"], [-32768, 0, 32767])
            await client.update_context(replace(client.context, text="", text_final=False))
            self.assertFalse(process.packets[2]["reset"])
            await client.reset()
            self.assertEqual(client.context, TurnContext())
            self.assertTrue(process.packets[3]["reset"])
            await client.close()
            self.assertTrue(process.stdin.closed)
            self.assertFalse(client.running)

    async def test_joint_model_starts_and_streams_without_legacy_paths(self):
        joint = Path(self.directory.name) / "joint.onnx"
        joint.touch()
        response = native_response()
        response["frames"][0].update(completion_probability=0.8, joint_inference_ms=0.12)
        process = FakeProcess([EMPTY, response])
        client = self.client(model=None, vad_model=None, joint_model=joint)
        with patch("sao_turn.client.asyncio.create_subprocess_exec", new=AsyncMock(return_value=process)) as spawn:
            await client.start()
            command = spawn.call_args.args
            self.assertEqual(command[command.index("--joint-model") + 1], str(joint.resolve()))
            self.assertNotIn("--model", command)
            self.assertNotIn("--vad-model", command)
            frames = (await client.process_audio(bytes(1024)))["frames"]
            self.assertEqual(frames[0]["joint_inference_ms"], 0.12)
            self.assertEqual(frames[0]["completion_probability"], 0.8)

    async def test_timeouts_close_and_explicit_restart_is_supported(self):
        first = FakeProcess([EMPTY, None])
        second = FakeProcess([EMPTY, EMPTY])
        client = self.client()
        with patch("sao_turn.client.asyncio.create_subprocess_exec", new=AsyncMock(side_effect=[first, second])):
            await client.start()
            with self.assertRaises(TurnClientTimeout):
                await client.process_audio(b"\0\0")
            self.assertFalse(client.running)
            self.assertTrue(first.stdin.closed)
            with self.assertRaises(TurnClientError):
                await client.process_audio(b"\0\0")
            await client.start()
            await client.process_audio(b"\0\0")
            self.assertTrue(client.running)

    async def test_startup_timeout_and_os_rejection_are_visible(self):
        process = FakeProcess([None], stderr=b"loading failed")
        client = self.client()
        with patch("sao_turn.client.asyncio.create_subprocess_exec", new=AsyncMock(return_value=process)):
            with self.assertRaisesRegex(TurnClientTimeout, "loading failed"):
                await client.start()
            self.assertFalse(client.running)
        with patch("sao_turn.client.asyncio.create_subprocess_exec", new=AsyncMock(side_effect=OSError("blocked by application control"))) as spawn:
            with self.assertRaisesRegex(TurnClientError, "application control"):
                await client.start()
            self.assertEqual(spawn.await_count, 1)

    async def test_malformed_eof_and_oversize_responses_close(self):
        bad_responses = [b"not JSON\n", b'{"frames":[],"buffered_samples":NaN}\n',
                         {"frames": [], "buffered_samples": 512}, b" " * 1_024 + b"\n",
                         {"frames": [{"at_ms": 32}], "buffered_samples": 0}]
        for bad in bad_responses:
            with self.subTest(bad=str(bad)[:50]):
                process = FakeProcess([EMPTY, bad])
                client = self.client(max_response_bytes=512)
                with patch("sao_turn.client.asyncio.create_subprocess_exec", new=AsyncMock(return_value=process)):
                    await client.start()
                    with self.assertRaises(TurnProtocolError):
                        await client.process_audio(b"\0\0")
                    self.assertFalse(client.running)
        process = FakeProcess([EMPTY])
        client = self.client()
        with patch("sao_turn.client.asyncio.create_subprocess_exec", new=AsyncMock(return_value=process)):
            await client.start()
            process.stdout.feed_eof()
            with self.assertRaises(TurnProtocolError):
                await client.process_audio(b"\0\0")

    async def test_cancellation_closes_inflight_protocol(self):
        process = FakeProcess([EMPTY, None])
        client = self.client()
        with patch("sao_turn.client.asyncio.create_subprocess_exec", new=AsyncMock(return_value=process)):
            await client.start()
            task = asyncio.create_task(client.process_audio(b"\0\0"))
            for _ in range(5):
                await asyncio.sleep(0)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.assertFalse(client.running)
            self.assertTrue(process.stdin.closed)

    async def test_lock_serializes_requests_and_stderr_is_bounded(self):
        process = FakeProcess([EMPTY, None, native_response(64)], stderr=b"a" * 300 + b"TAIL")
        client = self.client(stderr_limit=8)
        with patch("sao_turn.client.asyncio.create_subprocess_exec", new=AsyncMock(return_value=process)):
            await client.start()
            first = asyncio.create_task(client.process_audio(b"\0\0"))
            second = asyncio.create_task(client.process_audio(b"\0\0"))
            for _ in range(5):
                await asyncio.sleep(0)
            self.assertEqual(len(process.packets), 2)
            process.stdout.feed_data(json.dumps(native_response(32)).encode() + b"\n")
            await asyncio.gather(first, second)
            self.assertEqual(len(process.packets), 3)
            self.assertEqual(client.stderr_tail, "aaaaTAIL")

    async def test_bad_inputs_do_not_touch_live_protocol(self):
        process = FakeProcess([EMPTY])
        client = self.client()
        with patch("sao_turn.client.asyncio.create_subprocess_exec", new=AsyncMock(return_value=process)):
            await client.start()
            for pcm in [b"x", b"\0" * 320_002]:
                with self.assertRaises(ValueError):
                    await client.process_audio(pcm)
            for samples in [[True], [32768], [1.5]]:
                with self.assertRaises(ValueError):
                    await client.request(samples)
            with self.assertRaises(ValueError):
                await client.update_context({"expected_digits": 0})
            self.assertEqual(len(process.packets), 1)
            self.assertTrue(client.running)

    async def test_broken_pipe_backwards_clock_and_hung_shutdown(self):
        process = FakeProcess([EMPTY, native_response(64), native_response(32)], ignore_close=True)
        client = self.client()
        with patch("sao_turn.client.asyncio.create_subprocess_exec", new=AsyncMock(return_value=process)):
            await client.start()
            await client.process_audio(b"\0\0")
            with self.assertRaises(TurnProtocolError):
                await client.process_audio(b"\0\0")
            self.assertTrue(process.terminated)
        process = FakeProcess([EMPTY, EMPTY])
        client = self.client()
        with patch("sao_turn.client.asyncio.create_subprocess_exec", new=AsyncMock(return_value=process)):
            await client.start()
            process.broken_pipe = True
            with self.assertRaises(TurnClientError):
                await client.process_audio(b"\0\0")
            self.assertFalse(client.running)

    async def test_config_file_passes_through_after_validation(self):
        path = Path(self.directory.name) / "policy.json"
        path.write_text(json.dumps(TurnConfig().to_dict()), encoding="utf-8")
        process = FakeProcess([EMPTY])
        client = self.client(config=path)
        with patch("sao_turn.client.asyncio.create_subprocess_exec", new=AsyncMock(return_value=process)) as spawn:
            await client.start()
            self.assertIn("--config", spawn.call_args.args)
        await client.close()
        path.write_text('{"completion_threshhold":0.5}', encoding="utf-8")
        with self.assertRaises(ValueError):
            await client.start()

    async def test_cancelling_close_still_reaps_child(self):
        process = FakeProcess([EMPTY], ignore_close=True)
        client = self.client(shutdown_timeout=1.0)
        with patch("sao_turn.client.asyncio.create_subprocess_exec", new=AsyncMock(return_value=process)):
            await client.start()
            closing = asyncio.create_task(client.close())
            for _ in range(3):
                await asyncio.sleep(0)
            closing.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await closing
            self.assertIsNotNone(process.returncode)
            self.assertFalse(client.running)

    async def test_stream_reader_limit_is_a_protocol_error(self):
        process = FakeProcess([EMPTY, b"x" * 2_000 + b"\n"])
        process.stdout = asyncio.StreamReader(limit=512)
        client = self.client(max_response_bytes=512)
        with patch("sao_turn.client.asyncio.create_subprocess_exec", new=AsyncMock(return_value=process)):
            await client.start()
            with self.assertRaisesRegex(TurnProtocolError, "reader limit"):
                await client.process_audio(b"\0\0")


@unittest.skipUnless(os.getenv("SAO_TURN_RUN_NATIVE") == "1", "opt-in real native integration")
class NativeTests(unittest.IsolatedAsyncioTestCase):
    async def test_persistent_native_silence_chunking_and_reset(self):
        client = SaoTurnClient(
            executable=os.environ["SAO_TURN_NATIVE_EXE"], model=os.environ["SAO_TURN_MODEL"],
            vad_model=os.environ["SAO_TURN_VAD"], runtime=os.environ.get("SAO_TURN_RUNTIME"),
        )
        async with client:
            first = await client.process_audio(bytes(26))
            self.assertEqual(first["buffered_samples"], 13)
            second = await client.process_audio(bytes(998))
            self.assertEqual(second["frames"][0]["at_ms"], 32)
            self.assertIsNone(second["frames"][0]["decision"])
            await client.update_context({"mode": "digits", "expected_digits": 4})
            await client.reset()
            third = await client.process_audio(bytes(1_024))
            self.assertEqual(third["frames"][0]["at_ms"], 32)


if __name__ == "__main__":
    unittest.main()

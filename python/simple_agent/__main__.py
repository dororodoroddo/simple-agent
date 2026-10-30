"""Run the agent core with deterministic fixture output for a local smoke check."""
import asyncio
import json
import os
import sys
import uuid

from .config import Config
from .engine import execute_run
from .fixtures import FixtureTransport
from .protocol import fixture_event
from .storage import MemoryStore
from .native_io import NativeTransport


async def main():
    request = json.load(sys.stdin)
    message = request.get("message")
    if not isinstance(message, str) or not message.strip():
        raise SystemExit("input.message must be a non-empty string")
    outputs = request.get("fixtureOutputs")
    config = Config.from_environment()
    if outputs is not None and (not isinstance(outputs, list) or not outputs):
        raise SystemExit("input.fixtureOutputs must be a non-empty list")
    run = {
        "id": "local-" + str(uuid.uuid4()),
        "traceId": "local-trace-" + str(uuid.uuid4()),
        "mode": "fixture",
        "source": "synthetic_fixture",
        "createdAt": "local",
        "input": {"eventRaw": fixture_event(message, "1"), "artifactIds": []},
    }
    transport = FixtureTransport(outputs=outputs) if outputs is not None else NativeTransport(os.environ, config)
    result = await execute_run(run, MemoryStore(), transport, config)
    print(json.dumps({key: result.get(key) for key in ("status", "output", "reply", "toolCalls", "error")}, ensure_ascii=False))
    if result.get("status") != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())

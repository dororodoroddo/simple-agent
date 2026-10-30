"""Streaming file references and JSON payloads for provider requests."""

import base64
import json
from .errors import RunError

CHUNK = 1024 * 1024
PART_SIZE = 8 * CHUNK  # Transfer granularity, never an input file limit.


class StoredFile:
    def __init__(self, store, meta):
        self.store, self.meta = store, meta
        self.size = meta["bytes"]

    async def chunks(self, size=CHUNK):
        for offset in range(0, self.size, size):
            expected = min(size, self.size - offset)
            data = await self.store.read_range(self.meta["id"], offset, expected)
            if len(data) != expected:
                raise RunError("file_truncated", "원본 파일을 끝까지 읽을 수 없습니다.")
            yield data

    async def prefix(self, size=4096):
        return await self.store.read_range(self.meta["id"], 0, min(size, self.size))


class FileSlice:
    """Resumable transfer from the offset confirmed by the provider."""
    def __init__(self,file,offset):
        self.file,self.offset=file,offset
        self.size=file.size-offset
    async def chunks(self):
        for offset in range(self.offset,self.file.size,CHUNK):
            expected=min(CHUNK,self.file.size-offset)
            data=await self.file.store.read_range(self.file.meta['id'],offset,expected)
            if len(data)!=expected:raise RunError('file_truncated','원본 파일이 중단되었습니다.')
            yield data


async def byte_chunks(data):
    for start in range(0, len(data), CHUNK):
        yield data[start : start + CHUNK]


class BinaryReference(dict):
    """Internal stream value; ordinary caller JSON never acquires file semantics.

    A dict keeps evidence JSON-serializable. Current request construction restores
    the type during replay; archived requests are only compared, never transmitted.
    """


def binary_reference(meta, prefix=""):
    return BinaryReference({
        "$binary": meta["id"],
        "sha256": meta["sha256"],
        "bytes": meta["bytes"],
        "encoding": "base64",
        "prefix": prefix,
    })


def json_wire_size(value):
    """Exact streamed JSON size, including base64, without reading file bytes."""
    if isinstance(value, BinaryReference):
        return len(json.dumps(value.get("prefix", ""), ensure_ascii=False).encode()) + 4 * ((value["bytes"] + 2) // 3)
    if isinstance(value, dict):
        return 2 + max(0, len(value) - 1) + sum(len(json.dumps(k, ensure_ascii=False).encode()) + 1 + json_wire_size(v) for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return 2 + max(0, len(value) - 1) + sum(json_wire_size(v) for v in value)
    return len(json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode())


async def json_chunks(value, store):
    if isinstance(value, BinaryReference):
        meta = await store.get_meta(value["$binary"])
        if meta["sha256"] != value["sha256"] or meta["bytes"] != value["bytes"]:
            raise RunError(
                "file_changed", "원본 파일의 크기 또는 해시가 변경되었습니다."
            )
        yield json.dumps(value.get("prefix", ""), ensure_ascii=False)[:-1].encode()
        import hashlib

        hasher = hashlib.sha256()
        carry = b""
        async for data in StoredFile(store, meta).chunks():
            hasher.update(data)
            data = carry + data
            complete = len(data) // 3 * 3
            if complete:
                yield base64.b64encode(data[:complete])
            carry = data[complete:]
        if hasher.hexdigest() != meta["sha256"]:
            raise RunError("file_changed", "원본 바이트 해시가 변경되었습니다.")
        if carry:
            yield base64.b64encode(carry)
        yield b'"'
    elif isinstance(value, dict):
        yield b"{"
        for i, (key, item) in enumerate(value.items()):
            if i:
                yield b","
            yield json.dumps(key, ensure_ascii=False).encode() + b":"
            async for chunk in json_chunks(item, store):
                yield chunk
        yield b"}"
    elif isinstance(value, (list, tuple)):
        yield b"["
        for i, item in enumerate(value):
            if i:
                yield b","
            async for chunk in json_chunks(item, store):
                yield chunk
        yield b"]"
    else:
        yield json.dumps(
            value, ensure_ascii=False, allow_nan=False, separators=(",", ":")
        ).encode()


class JsonBody:
    """Stream a request directly without duplicating the full input buffer."""
    def __init__(self, value, store):
        self.value, self.store = value, store
        self.size = json_wire_size(value)

    def chunks(self):
        return json_chunks(self.value, self.store)

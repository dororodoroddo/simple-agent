import asyncio
import copy
import inspect
import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from .errors import RunError
from .protocol import digest, json_text, redact, SECRET_ENV_KEYS
from .tools import default_registry
from .usage import aggregate_usage


def error_record(exc, *, secrets=()):
    def diagnostic(error, seen):
        seen.add(id(error))
        frames = []
        trace = error.__traceback__
        while trace is not None:
            frames.append({"file": trace.tb_frame.f_code.co_filename, "line": trace.tb_lineno,
                           "function": trace.tb_frame.f_code.co_name})
            trace = trace.tb_next
        result = {"exception": type(error).__name__, "message": str(error), "traceback": frames}
        cause = error.__cause__ or (None if error.__suppress_context__ else error.__context__)
        if cause is not None and id(cause) not in seen:
            result["cause"] = diagnostic(cause, seen)
        return result

    code = (
        "cancelled"
        if isinstance(exc, asyncio.CancelledError)
        else "timeout"
        if isinstance(exc, TimeoutError)
        else "execution_error"
    )
    record = exc.record() if isinstance(exc, RunError) else {
        "code": code,
        "message": "실행 시간 초과 또는 처리 오류입니다.",
        "details": {"exception": type(exc).__name__},
    }
    if isinstance(exc, MemoryError) or isinstance(exc, OSError) and exc.errno in {12, 28, 122}:
        record.update(code="runtime_resource_exhausted", message="실행환경의 메모리 또는 저장 공간이 부족합니다.")
        record["details"]["errno"] = getattr(exc, "errno", None)
    details = record.setdefault("details", {})
    if "diagnostic" not in details:
        details["diagnostic"] = diagnostic(exc, set())
    def clean(value):
        if isinstance(value, dict): return {key: clean(item) for key, item in value.items()}
        if isinstance(value, list): return [clean(item) for item in value]
        if isinstance(value, str):
            for secret in secrets:
                value = value.replace(secret, "<redacted>")
        return value
    return clean(redact(record))


def raise_record(record):
    raise RunError(record["code"], record["message"], details=record.get("details"))


@dataclass
class Dependencies:
    clock: object = time.time
    monotonic: object = time.monotonic
    new_id: object = lambda: str(uuid.uuid4())
    registry: object = field(default_factory=default_registry)
    permissions: frozenset | None = None


class RunContext:
    def __init__(self, run, store, transport, config, deps=None):
        self.run, self.store, self.transport, self.config = (
            run,
            store,
            transport,
            config,
        )
        self.deps = deps or Dependencies()
        self.registry = self.deps.registry
        self.permissions = self.deps.permissions if self.deps.permissions is not None else frozenset(s.permission for s in self.registry.specs.values())
        self.replay = getattr(transport, "replay", None)
        self.cursor = 0
        self.replay_ids = {None: None}
        self.replay_children = {}
        self.replay_positions = {}
        for operation in (self.replay or {}).get("operations", []):
            self.replay_children.setdefault(operation.get("parentCallId"), []).append(operation)
        env = getattr(transport, "env", None)
        self.secrets = [value for key in SECRET_ENV_KEYS
                        if isinstance(value := (env.get(key) if hasattr(env, "get") else getattr(env, key, None)), str) and value]
        self.storage_failed = False
        self.closed = False
        self.restored_artifact_ids = set()

    def assert_open(self):
        if self.closed:
            raise RunError("execution_closed", "종료된 실행에서는 새 도구나 외부 작업을 시작할 수 없습니다.")

    def error(self, exc):
        return error_record(exc, secrets=self.secrets)

    def now(self):
        return datetime.fromtimestamp(self.deps.clock(), timezone.utc).isoformat()

    async def save(self):
        self.run["usageSummary"] = aggregate_usage(
            self.run.get("usage", []),
            "replay" if self.replay is not None else self.run["mode"],
        )
        try:
            await self.store.save_run(self.run)
        except Exception:
            self.storage_failed = True
            raise

    async def stage(
        self, name, data, status="completed", *, call_id=None, parent_id=None,
        checkpoint=False,
    ):
        event = {
            "id": self.deps.new_id(),
            "runId": self.run["id"],
            "index": self.run.get("stageOffset",0) + len(self.run["stages"]),
            "name": name,
            "status": status,
            "at": self.now(),
            "callId": call_id,
            "parentCallId": parent_id,
            "data": copy.deepcopy(data),
        }
        self.run["stages"].append(event)
        return event

    async def pack(self, value):
        if isinstance(value, bytes):
            meta = await self.store.put_artifact(
                value, "operation.bin", "application/octet-stream", self.run["id"]
            )
            return {"__simple_agent_bytes__": meta["id"], "sha256": digest(value)}
        if isinstance(value, (list, tuple)):
            return [await self.pack(v) for v in value]
        if isinstance(value, dict):
            return {k: await self.pack(v) for k, v in value.items()}
        return value

    async def unpack(self, value):
        if isinstance(value, dict) and set(value) == {"__simple_agent_bytes__", "sha256"}:
            _, data = await self.store.get_artifact(value["__simple_agent_bytes__"])
            if digest(data) != value["sha256"]:
                raise RunError("replay_corrupt", "저장된 파일 해시가 다릅니다.")
            return data
        if isinstance(value, list):
            return [await self.unpack(v) for v in value]
        if isinstance(value, dict):
            return {k: await self.unpack(v) for k, v in value.items()}
        return value

    async def effect(
        self,
        kind,
        request,
        execute,
        *,
        timeout=None,
        parent_id=None,
        changing=False,
        call_id=None,
    ):
        self.assert_open()
        if getattr(self.store, "owner", None) and not getattr(self.store, "response_first", False):
            await self.store.assert_owner(self.run["id"])
        record = {
            "id": call_id or self.deps.new_id(),
            "index": len(self.run["operations"]),
            "kind": kind,
            "request": copy.deepcopy(request),
            "parentCallId": parent_id,
            "status": "started",
            "changing": changing,
        }
        if changing and getattr(self, 'services', None) is not None:
            lease = self.services._node_lease.get()
            while lease is not None:
                lease['changing'] = True
                lease = lease.get('parent')
        self.run["operations"].append(record)
        await self.stage(
            kind + "_started",
            {"operationIndex": record["index"]},
            "started",
            call_id=record["id"],
            parent_id=parent_id,
        )
        guarded = (getattr(self.store, "response_first", False) and self.replay is None
                   and (kind in {"model", "media_upload"} or changing))
        if guarded:
            await self.store.begin_effect(self.run["id"], record)
        start = self.deps.monotonic()
        try:
            if self.replay is not None:
                if parent_id not in self.replay_ids:
                    raise RunError("replay_mismatch", "원래 기록과 호출의 부모가 다릅니다.")
                original_parent = self.replay_ids[parent_id]
                operations = self.replay_children.get(original_parent, [])
                position = self.replay_positions.get(original_parent, 0)
                if position >= len(operations):
                    raise RunError(
                        "replay_incomplete", "재생에 필요한 전체 작업 기록이 없습니다."
                    )
                source = operations[position]
                if source["kind"] != kind or json_text(source["request"]) != json_text(
                    request
                ):
                    raise RunError(
                        "replay_mismatch",
                        "요청 내용 또는 작업 순서가 원래 기록과 다릅니다.",
                        details={"operationIndex": source["index"]},
                    )
                # Independent branches may interleave differently without I/O waits.
                # Calls within each parent still consume the recorded sequence exactly.
                self.replay_positions[original_parent] = position + 1
                self.replay_ids[record["id"]] = source["id"]
                record["originalCallId"] = source["id"]
                self.cursor += 1
                if source["status"] not in {"completed", "failed", "indeterminate"}:
                    raise RunError(
                        "replay_incomplete", "완료 여부가 기록되지 않은 작업입니다."
                    )
                if "error" in source:
                    raise_record(source["error"])
                if "result" not in source:
                    raise RunError("replay_incomplete", "작업 결과 기록이 없습니다.")
                packed = copy.deepcopy(source["result"])
                result = await self.unpack(packed)
            else:

                async def invoke():
                    result = execute()
                    return await result if inspect.isawaitable(result) else result

                result = await invoke() if timeout is None else await asyncio.wait_for(invoke(), timeout)
                packed = await self.pack(result)
            record.update(
                {
                    "result": packed,
                    "status": "completed",
                    "durationMs": round((self.deps.monotonic() - start) * 1000),
                }
            )
        except (Exception, asyncio.CancelledError) as exc:
            err = self.error(exc)
            uncertain = err["code"] == "indeterminate" or (changing or kind == "model") and err["code"] in {
                "timeout",
                "cancelled",
                "network_error",
                "execution_error",
                "storage_bridge_failed",
                "invalid_tool_output",
                "indeterminate",
            }
            if uncertain and err["code"] != "indeterminate":
                err = {
                    "code": "indeterminate",
                    "message": "외부 작업의 완료 여부가 불명확하여 재실행하지 않았습니다.",
                    "details": {"cause": err},
                }
            record.update(
                {
                    "status": "indeterminate" if uncertain else "failed",
                    "error": err,
                    "durationMs": round((self.deps.monotonic() - start) * 1000),
                }
            )
            if guarded:
                try:
                    await self.store.end_effect(self.run["id"], record)
                except Exception as persist_exc:
                    err = {"code": "indeterminate", "message": "외부 작업 결과의 제어 상태를 확정하지 못했습니다.",
                           "details": {"cause": err, "controlError": self.error(persist_exc)}}
                    record.update(status="indeterminate", error=err)
            if err['code'] == 'indeterminate' and getattr(self, 'services', None) is not None:
                lease = self.services._node_lease.get()
                while lease is not None:
                    lease['indeterminate'] = True
                    lease = lease.get('parent')
            await self.stage(
                kind + "_result",
                {"operationIndex": record["index"], "error": err},
                record["status"],
                call_id=record["id"],
                parent_id=parent_id,
            )
            raise_record(err)
        if guarded:
            try:
                await self.store.end_effect(self.run["id"], record)
            except Exception as exc:
                err = self.error(exc)
                record.update(status="indeterminate", error=err)
                if getattr(self, 'services', None) is not None:
                    lease = self.services._node_lease.get()
                    while lease is not None:
                        lease['indeterminate'] = True
                        lease = lease.get('parent')
                await self.stage(kind + "_result", {"operationIndex": record["index"], "error": err},
                                 "indeterminate", call_id=record["id"], parent_id=parent_id)
                raise
        await self.stage(
            kind + "_result",
            {"operationIndex": record["index"]},
            call_id=record["id"],
            parent_id=parent_id,
        )
        return result

    async def artifact(self, data, filename, mime):
        request = {"sha256": digest(data), "filename": filename, "mime": mime}
        meta = await self.effect(
            "artifact_store",
            request,
            lambda: self.store.put_artifact(data, filename, mime, self.run["id"]),
        )
        # Replay grants only artifacts proven by this execution's recorded sequence.
        if meta["id"] not in self.run["artifactIds"]:
            self.run["artifactIds"].append(meta["id"])
        return meta

    def verify_replay_end(self):
        if self.replay is not None and self.cursor != len(
            self.replay.get("operations", [])
        ):
            raise RunError(
                "replay_mismatch", "원래 기록에 아직 소비하지 않은 작업이 있습니다."
            )

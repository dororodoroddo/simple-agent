"""In-memory stores for local agent runs and isolated E2E checks."""
import copy
import json
import uuid

from .errors import RunError
from .protocol import digest, json_text


class MemoryStore:
    def __init__(self):
        self.runs = {}
        self.events = {}
        self.artifacts = {}
        self.data = {}
        self.rag = {}
        self.rag_documents = {}
        self.pending_work = {}

    async def load_pending(self, conversation_id):
        return copy.deepcopy(self.pending_work.get(conversation_id))

    async def save_pending(self, conversation_id, value):
        if value is None:
            self.pending_work.pop(conversation_id, None)
        else:
            self.pending_work[conversation_id] = copy.deepcopy(value)
        return {"saved": True}

    async def save_data(self, scope, key, value):
        self.data[(scope, key)] = json.loads(json_text(value))
        return {"scope": scope, "key": key, "saved": True}

    async def load_data(self, scope, key):
        identity = (scope, key)
        return {"scope": scope, "key": key, "found": identity in self.data,
                "value": copy.deepcopy(self.data.get(identity))}

    async def rag_put(self, scope, key, item):
        self.rag[(scope, digest(key))] = json.loads(json_text(item))
        return {"scope": scope, "key": key, "saved": True}

    async def rag_list(self, scope, cursor=None):
        return {"keys": sorted(key for item_scope, key in self.rag if item_scope == scope), "cursor": None}

    async def rag_read(self, scope, key):
        try:
            return copy.deepcopy(self.rag[(scope, key)])
        except KeyError as exc:
            raise RunError("rag_entry_missing", "검색 중 자료가 없어졌습니다.") from exc

    async def rag_index_put(self, scope, document_id, item):
        self.rag_documents[(scope, digest(document_id))] = json.loads(json_text(item))
        return {"scope": scope, "document_id": document_id, "saved": True}

    async def rag_index_list(self, scope, cursor=None):
        return {"keys": sorted(key for item_scope, key in self.rag_documents if item_scope == scope), "cursor": None}

    async def rag_index_read(self, scope, key):
        try:
            return copy.deepcopy(self.rag_documents[(scope, key)])
        except KeyError as exc:
            raise RunError("rag_entry_missing", "검색 중 인덱스 문서가 없습니다.") from exc

    async def rag_index_delete(self, scope, document_id):
        self.rag_documents.pop((scope, digest(document_id)), None)
        return {"scope": scope, "document_id": document_id, "deleted": True}

    async def save_run(self, run):
        previous = self.events.get(run["id"], [])
        incoming = run.get("stages", [])
        if incoming[:len(previous)] != previous:
            raise RunError("event_mutation", "이미 기록한 이벤트를 변경할 수 없습니다.")
        self.events[run["id"]] = copy.deepcopy(incoming)
        self.runs[run["id"]] = copy.deepcopy(run)

    async def get_run(self, run_id, hydrate=True):
        try:
            return copy.deepcopy(self.runs[run_id])
        except KeyError as exc:
            raise RunError("not_found", "실행 기록이 없습니다.") from exc

    async def put_artifact(self, data, filename, mime, run_id=None):
        artifact_id = str(uuid.uuid4())
        meta = {"id": artifact_id, "filename": filename, "mime": mime, "bytes": len(data),
                "sha256": digest(data), "runId": run_id}
        self.artifacts[artifact_id] = (meta, data)
        return meta

    async def get_artifact(self, artifact_id):
        try:
            return self.artifacts[artifact_id]
        except KeyError as exc:
            raise RunError("not_found", "원본 파일이 없습니다.") from exc

    async def get_meta(self, artifact_id):
        return (await self.get_artifact(artifact_id))[0]

    async def read_range(self, artifact_id, offset, size):
        return (await self.get_artifact(artifact_id))[1][offset:offset + size]

    async def put_chunks(self, chunks, filename, mime, run_id=None):
        return await self.put_artifact(b"".join([part async for part in chunks]), filename, mime, run_id)

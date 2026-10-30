"""Direct provider HTTP transport for the platform-neutral Python agent."""
import httpx

from .errors import RunError
from .files import CHUNK
from .protocol import validate_download_url
from .transport import ProviderTransport


class NativeTransport(ProviderTransport):
    async def http(self, method, url, headers=None, body=None, limit=None, timeout=None):
        if hasattr(body, "chunks"):
            if headers is None:
                headers = {}
            headers = {**headers, "Content-Length": str(body.size)}
            body = body.chunks()
        async with httpx.AsyncClient(timeout=timeout, trust_env=False, follow_redirects=False) as client:
            async with client.stream(method, url, headers=headers, content=body) as response:
                data = bytearray()
                async for part in response.aiter_bytes():
                    data.extend(part)
                    if limit is not None and len(data) > limit:
                        raise RunError("body_too_large", "API 응답 처리 한도를 초과했습니다.")
                return response.status_code, dict(response.headers), bytes(data)

    async def download_to_store(self, url, store, run_id):
        validate_download_url(url)
        async with httpx.AsyncClient(timeout=None, trust_env=False, follow_redirects=False) as client:
            async with client.stream("GET", url) as response:
                if not 200 <= response.status_code < 300:
                    raise RunError("file_download_failed", "파일 다운로드에 실패했습니다.",
                                   details={"httpStatus": response.status_code})
                filename = url.rsplit("/", 1)[-1].split("?", 1)[0] or "download.bin"
                mime = response.headers.get("content-type", "application/octet-stream").split(";", 1)[0]
                async def chunks():
                    async for part in response.aiter_bytes(CHUNK):
                        yield part
                return await store.put_chunks(chunks(), filename, mime, run_id)

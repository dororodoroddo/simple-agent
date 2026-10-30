"""Provider protocol shared by native Python transports."""

import asyncio
import json
import time
from urllib.parse import urlsplit
from .protocol import json_text
from .errors import RunError
from .providers import adapter


class ProviderTransport:
    def __init__(self, env, config):
        self.env, self.config = env, config

    def key(self, provider):
        if not self.config.live_api_enabled:
            raise RunError(
                "live_api_disabled", "실제 API 호출이 비활성화되어 있습니다."
            )
        name = adapter(provider).key_name
        value = self.env.get(name) if hasattr(self.env, "get") else getattr(self.env, name, None)
        if not isinstance(value, str) or not value.strip():
            raise RunError("missing_api_key", f"{name}가 등록되지 않았습니다.")
        return value.strip()

    async def request_json(self, request):
        from .files import JsonBody
        provider=request['provider']
        headers = adapter(provider).headers(self.key(provider))
        body = JsonBody(request['body'], self.store)
        try:
            status,headers,data=await self.http('POST',request['url'],headers,body)
        except RunError:raise
        except Exception as exc:
            raise RunError('network_error','제공자 응답을 받지 못했습니다.',details={'exception':type(exc).__name__}) from exc
        from .io_retry import retry_after
        details = {'httpStatus': status, 'requestId': next((v for k,v in headers.items() if k.lower() == 'x-request-id'), None),
                   'retryAfterSeconds': retry_after(headers)}
        try: payload=json.loads(data)
        except ValueError as exc:
            if not 200 <= status < 300:
                raise RunError('model_http_error', f'제공자 API HTTP {status}',
                               details={**details, 'body': data.decode(errors='replace')}) from exc
            raise RunError('indeterminate' if request.get('role') in {'llm','embedding'} else 'invalid_api_json',
                           '제공자의 응답을 해석할 수 없어 실행 결과를 확인하지 못했습니다.',
                           details={**details,'body':data.decode(errors='replace')}) from exc
        if not 200 <= status < 300:
            raise RunError('model_http_error',f'제공자 API HTTP {status}',details={**details,'response':payload})
        return payload

    async def model(self, request, run):
        from .limits import validate_request
        from .files import json_wire_size
        if request.get('role') == 'embedding':
            valid = request['provider'] == 'openai' and request['url'] == 'https://api.openai.com/v1/embeddings'
            if not valid:
                raise RunError('invalid_provider_url', '허용되지 않은 임베딩 API 주소입니다.')
            return await self.request_json(request)
        allowed={'gemini':{'https://generativelanguage.googleapis.com/v1beta/interactions'},
                 'openai':{'https://api.openai.com/v1/responses','https://api.openai.com/v1/chat/completions'}}
        if request['url'] not in allowed.get(request['provider'],set()):
            raise RunError('invalid_provider_url','허용되지 않은 모델 API 주소입니다.')
        validate_request(request,json_wire_size(request['body']))
        return await self.request_json(request)  # Exactly one attempt; no retry helper.

    async def count_tokens(self, request, run):
        expected='https://api.openai.com/v1/responses/input_tokens'
        if request['provider']=='gemini':
            # URL is built from the caller's model by token_request, never user-supplied.
            import re
            valid=re.fullmatch(r'https://generativelanguage\.googleapis\.com/v1beta/models/[^/]+:countTokens',request['url'])
        else: valid=request['url']==expected
        if not valid: raise RunError('invalid_provider_url','허용되지 않은 토큰 계수 주소입니다.')
        return await self.request_json(request)

    async def upload_file(self, data, artifact, run):
        from .limits import validate_media

        validate_media(artifact, "gemini")
        key = self.key("gemini")
        status, headers, start = await self.http(
            "POST",
            "https://generativelanguage.googleapis.com/upload/v1beta/files",
            {
                "x-goog-api-key": key,
                "X-Goog-Upload-Protocol": "resumable",
                "X-Goog-Upload-Command": "start",
                "X-Goog-Upload-Header-Content-Length": str(
                    data.size if hasattr(data, "size") else len(data)
                ),
                "X-Goog-Upload-Header-Content-Type": artifact["mime"],
                "Content-Type": "application/json",
            },
            json_text({"file": {"display_name": artifact["filename"]}}),
        )
        if not 200 <= status < 300:
            raise RunError(
                "video_upload_failed",
                "Gemini 파일 업로드를 시작하지 못했습니다.",
                details={"httpStatus": status},
            )
        upload_url = headers.get("x-goog-upload-url", "")
        p = urlsplit(upload_url)
        if (
            p.scheme != "https"
            or p.hostname != "generativelanguage.googleapis.com"
            or p.port not in (None, 443)
            or p.username
        ):
            raise RunError(
                "invalid_upload_url", "Gemini 업로드 주소 검증에 실패했습니다."
            )
        from .io_retry import transfer
        offset=0
        async def upload():
            from .files import FileSlice
            content=FileSlice(data,offset) if hasattr(data,'chunks') else data[offset:]
            return await self.http('POST',upload_url,{'X-Goog-Upload-Offset':str(offset),'X-Goog-Upload-Command':'upload, finalize','Content-Type':artifact['mime']},content)
        try:
            status,upload_headers,body=await upload()
            from .io_retry import retry_after
            delay=retry_after(upload_headers)
            if status in {429,503} and delay is not None:
                run.setdefault('transfers',[]).append({'operation':'gemini_upload_retry_after','httpStatus':status,'waitSeconds':delay})
                await asyncio.sleep(delay)
                raise RunError('network_error','서버가 명시한 대기 후 업로드 진행 상태를 확인합니다.')
        except Exception as exc:
            if isinstance(exc,RunError) and exc.code!='network_error':raise
            # Query server-confirmed progress once; never create a new upload silently.
            qstatus,qheaders,qbody=await self.http('POST',upload_url,{'X-Goog-Upload-Command':'query'})
            state=qheaders.get('x-goog-upload-status')
            received=qheaders.get('x-goog-upload-size-received','')
            if qstatus==200 and state=='active' and received.isdigit() and int(received)<=artifact['bytes']:
                offset=int(received)
                run.setdefault('transfers',[]).append({'operation':'gemini_upload_resume','offset':offset,'source':'https://ai.google.dev/api/files'})
                status,_,body=await upload()
            else:
                raise RunError('indeterminate','업로드 완료 여부를 확인하지 못했습니다. 새 업로드를 자동 생성하지 않습니다.',details={'httpStatus':qstatus,'receivedBytes':received,'uploadStatus':state}) from exc
        if not 200 <= status < 300:
            raise RunError(
                "video_upload_failed",
                "Gemini 파일 업로드에 실패했습니다.",
                details={"httpStatus": status},
            )
        file = json.loads(body).get("file", {})
        name = file.get("name", "")
        import re

        if not re.fullmatch(r"files/[a-zA-Z0-9_-]+", name):
            raise RunError("invalid_upload_response", "Gemini 파일 식별자가 없습니다.")
        run.setdefault("providerFiles", []).append(
            {"provider": "gemini", "name": name, "artifactId": artifact["id"]}
        )
        while file.get("state") == "PROCESSING":
            await asyncio.sleep(2)
            status, _, body = await transfer(lambda: self.http("GET", "https://generativelanguage.googleapis.com/v1beta/"+name, {"x-goog-api-key":key}),
                safe=True, events=run.setdefault('transfers',[]), operation='gemini_file_status')
            if status != 200:
                raise RunError(
                    "video_processing_failed", "영상 처리 상태 조회에 실패했습니다."
                )
            file = json.loads(body)
        if file.get("state") != "ACTIVE" or not isinstance(file.get("uri"), str):
            raise RunError(
                "video_processing_failed",
                "영상 모델이 파일을 준비하지 못했습니다.",
                details={"state": file.get("state")},
            )
        return file["uri"], {
            "name": name,
            "state": file["state"],
            "mimeType": file.get("mimeType"),
            "source": "real_provider",
        }

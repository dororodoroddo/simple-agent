import hashlib
import json
from datetime import datetime, timezone
from urllib.parse import urlsplit
from .errors import RunError


def json_text(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(value):
    return hashlib.sha256(value if isinstance(value, bytes) else value.encode()).hexdigest()


def input_reply(message, files, nodes):
    return ('수신 메시지:\n' + message + '\n\n수신 파일:\n' + ('\n'.join(files) if files else '없음')
            + '\n\n통과 노드:\n' + (' → '.join(nodes) if nodes else '없음'))


def normalize_event(raw):
    try:
        frame = json.loads(raw, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
    except (ValueError, TypeError) as exc:
        raise RunError("invalid_json", "올바른 JSON 이벤트가 아닙니다.") from exc
    if not isinstance(frame, dict):
        raise RunError("invalid_event", "입력 이벤트는 JSON 객체여야 합니다.")
    message = frame.get("message")
    conversation = frame.get("conversationId")
    sender = frame.get("senderId")
    message_id = frame.get("messageId")
    if not isinstance(message, str) or not message.strip():
        raise RunError("invalid_message", "빈 메시지 또는 잘못된 message입니다.")
    if not all(isinstance(value, str) and value for value in (conversation, sender)):
        raise RunError("invalid_event", "발신자 또는 대화 식별자가 없습니다.")
    if isinstance(message_id, bool) or not isinstance(message_id, (str, int)):
        raise RunError("invalid_event", "메시지 ID가 없습니다.")
    timestamp = frame.get("timestamp")
    if not isinstance(timestamp, str):
        raise RunError("invalid_event", "이벤트 시각이 없습니다.")
    try:
        stamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        if stamp.tzinfo is None: raise ValueError("timezone required")
    except ValueError as exc:
        raise RunError("invalid_event", "이벤트 시각은 시간대를 포함해야 합니다.") from exc
    return {"messageId": str(message_id), "message": message, "timestamp": timestamp,
            "sender": sender, "conversation": conversation, "event": frame}


def fixture_event(message, message_id="9007199254740993123"):
    return json_text({"messageId": int(message_id), "message": message,
        "timestamp": now(), "conversationId": "fixture-room", "senderId": "fixture-user"})


def handler_event_raw(raw):
    """Return the provider-neutral input event unchanged."""
    return raw


def media_url(content, *, fixture=False):
    value = content.strip()
    try:
        validate_download_url(value, fixture=fixture)
        return value
    except (RunError, ValueError):
        return None


def validate_download_url(url, *, fixture=False):
    try:
        p = urlsplit(url)
        valid = p.scheme == "https" and not p.username and not p.password and p.port in (None, 443)
        valid = valid and (p.hostname == "files.simple-agent.invalid" or (fixture and p.hostname == "fixtures.simple-agent.invalid"))
    except ValueError:
        valid = False
    if not valid:
        raise RunError("blocked_download", "허용되지 않은 파일 다운로드 주소입니다.")


def reply_text(text):
    if not isinstance(text, str) or not text.strip():
        raise RunError("empty_output", "모델이 답변을 반환하지 않았습니다.")
    return text



SECRET_ENV_KEYS = ("OPENAI_API_KEY", "GEMINI_API_KEY")
SECRET_KEYS = {"authorization", "x-goog-api-key", "secret", "apikey", "api_key", *(key.lower() for key in SECRET_ENV_KEYS)}


def is_secret_key(key):
    import re
    return key.lower() in SECRET_KEYS or bool(re.search(r"(?:^|_)(?:api_key|token|secret|password)$", key, re.I))


def redact(value):
    if isinstance(value, dict):
        return {k: ("<redacted>" if is_secret_key(k) else redact(v)) for k,v in value.items()}
    if isinstance(value, list): return [redact(v) for v in value]
    if isinstance(value, str):
        import re
        # JSON string tokens only: preserve whitespace, escaped user content and large IDs.
        token = re.compile(r'"(?:[^"\\]|\\.)*"')
        spans=[]
        for match in token.finditer(value):
            try: key=json.loads(match.group())
            except ValueError: continue
            if not is_secret_key(key): continue
            tail=match.end()
            colon=re.match(r'\s*:\s*',value[tail:])
            if colon:
                start=tail+colon.end()
                try: _,length=json.JSONDecoder().raw_decode(value[start:])
                except ValueError: continue
                spans.append((start,start+length))
        for start,end in reversed(spans): value=value[:start]+'"<redacted>"'+value[end:]
        return value
    return value

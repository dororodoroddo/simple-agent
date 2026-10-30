"""Provider limits apply only at their own adapter boundary (2026-10-26)."""

from .errors import RunError

FILE_BYTES = 50_000_000
IMAGE_PAYLOAD_BYTES = 512_000_000
GEMINI_FILE_BYTES = 2_000_000_000
GEMINI_INLINE_BYTES = 100_000_000
GEMINI_PDF_BYTES = 50_000_000

LIMITS = {
    "openai_file": {
        "eachBytesExclusive": FILE_BYTES,
        "requestTotalBytes": FILE_BYTES,
        "source": "https://developers.openai.com/api/docs/guides/file-inputs",
    },
    "openai_image": {
        "requestPayloadBytes": IMAGE_PAYLOAD_BYTES,
        "imagesPerRequest": 1500,
        "source": "https://developers.openai.com/api/docs/guides/images-vision",
    },
    "openai_audio": {
        "formats": ["wav", "mp3"],
        "fileBytes": None,
        "source": "https://developers.openai.com/api/docs/guides/audio-chat-completions",
    },
    "gemini_file": {
        "eachBytes": GEMINI_FILE_BYTES,
        "source": "https://ai.google.dev/gemini-api/docs/files",
    },
    "gemini_inline": {"requestPayloadBytes": GEMINI_INLINE_BYTES, "source": "https://ai.google.dev/gemini-api/docs/file-input-methods"},
    "gemini_pdf": {"eachBytes": GEMINI_PDF_BYTES, "pages": 1000, "source": "https://ai.google.dev/gemini-api/docs/document-processing"},
}


def require_limit(ok, boundary, actual, limit):
    if not ok:
        raise RunError(
            "provider_input_limit",
            "제공자 입력 규격을 초과했습니다.",
            details={
                "boundary": boundary,
                "actual": actual,
                "limit": limit,
                "source": LIMITS[boundary]["source"],
            },
        )


def validate_files(files):
    for item in files:
        require_limit(
            item["bytes"] < FILE_BYTES, "openai_file", item["bytes"], FILE_BYTES
        )
    require_limit(
        sum(x["bytes"] for x in files) <= FILE_BYTES,
        "openai_file",
        sum(x["bytes"] for x in files),
        FILE_BYTES,
    )


def validate_media(meta, provider):
    if provider == "gemini":
        require_limit(meta["bytes"] <= GEMINI_FILE_BYTES, "gemini_file", meta["bytes"], GEMINI_FILE_BYTES)
        if meta["mime"] == "application/pdf":
            require_limit(meta["bytes"] <= GEMINI_PDF_BYTES, "gemini_pdf", meta["bytes"], GEMINI_PDF_BYTES)
            if "pages" in meta:
                require_limit(meta["pages"] <= 1000, "gemini_pdf", meta["pages"], 1000)
        return
    if meta["kind"] == "document":
        validate_files([meta])
    # Image payload (including encoding and prompt) is checked after serialization.


def validate_request(request, payload_bytes):
    """Check exactly the provider request, not a global upload gate."""
    if request["provider"] == "gemini":
        parts = request["body"].get("input", [])
        pdf = any(p.get("type") == "document" and p.get("mime_type") == "application/pdf" and "data" in p for p in parts if isinstance(p, dict))
        limit = GEMINI_PDF_BYTES if pdf else GEMINI_INLINE_BYTES
        require_limit(payload_bytes <= limit, "gemini_inline", payload_bytes, limit)
        return
    if request["provider"] != "openai":
        return
    files = []
    images = 0

    def walk(value):
        nonlocal images
        if isinstance(value, dict):
            if value.get("type") == "input_file" and isinstance(
                value.get("file_data"), dict
            ):
                files.append(value["file_data"])
            if value.get("type") in {"input_image", "image_url"}:
                images += 1
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    # Schemas and tool definitions are caller JSON, not media inputs.
    walk(request["body"].get("input", request["body"].get("messages", [])))
    if files:
        validate_files(files)
    if images:
        require_limit(images <= 1500, "openai_image", images, 1500)
        require_limit(
            payload_bytes <= IMAGE_PAYLOAD_BYTES,
            "openai_image",
            payload_bytes,
            IMAGE_PAYLOAD_BYTES,
        )

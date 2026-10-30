"""File metadata and bounded header inspection; no content parsing or extraction."""
from email.message import Message
from email.utils import collapse_rfc2231_value
from pathlib import PurePosixPath
from urllib.parse import unquote, urlsplit

from .errors import RunError

# Responses input_file types: extensions and MIME-only inputs share this table.
# https://developers.openai.com/api/docs/guides/file-inputs#full-list-of-accepted-file-types
OPENAI_DOCUMENT_TYPES = (
    ('pdf', 'application/pdf'),
    ('xla xlb xlc xlm xls xlt xlw', 'application/vnd.ms-excel'),
    ('xlsx', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'),
    ('csv', 'text/csv application/csv application/vnd.google-apps.spreadsheet'),
    ('tsv', 'text/tab-separated-values text/tsv'),
    ('iif', 'text/x-iif application/x-iif'),
    ('doc dot', 'application/msword'),
    ('docx', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'),
    ('odt', 'application/vnd.oasis.opendocument.text'),
    ('rtf', 'application/rtf text/rtf'),
    ('pages', 'application/vnd.apple.pages application/vnd.google-apps.document application/vnd.apple.iwork'),
    ('pot ppa pps ppt pwz wiz', 'application/vnd.ms-powerpoint'),
    ('pptx', 'application/vnd.openxmlformats-officedocument.presentationml.presentation'),
    ('key', 'application/vnd.apple.keynote application/vnd.google-apps.presentation'),
    ('asm bat c cc conf cpp css cxx def dic eml h hh htm html ics ifb in js json ksh list log '
     'markdown md mht mhtml mime mjs nws pl py rst s sql srt text txt vcf vtt xml',
     'text/plain application/javascript application/typescript text/xml text/x-shellscript text/x-rst '
     'text/x-makefile text/x-lisp text/x-asm text/vbscript text/css message/rfc822 application/x-sql '
     'application/x-scala application/x-rust application/x-powershell text/x-diff text/x-patch '
     'application/x-patch text/markdown text/x-java text/x-script.python text/x-python text/x-c '
     'text/x-c++ text/x-golang text/html text/x-php application/x-php application/x-httpd-php '
     'application/x-httpd-php-source text/x-ruby text/x-sh text/x-bash application/x-bash text/x-zsh '
     'text/x-tex text/x-csharp application/json text/x-typescript text/javascript text/x-go text/x-rust '
     'text/x-scala text/x-kotlin text/x-swift text/x-lua text/x-r text/x-julia text/x-perl '
     'text/x-objectivec text/x-objectivec++ text/x-erlang text/x-elixir text/x-haskell text/x-clojure '
     'text/x-groovy text/x-dart text/x-awk application/x-awk text/jsx text/tsx text/x-handlebars '
     'text/x-mustache text/x-ejs text/x-jinja2 text/x-liquid text/x-erb text/x-twig text/x-pug '
     'text/x-jade text/x-tmpl text/x-cmake text/x-dockerfile text/x-gradle text/x-ini text/x-properties '
     'text/x-protobuf application/x-protobuf text/x-sql text/x-sass text/x-scss text/x-less text/x-hcl '
     'text/x-terraform application/x-terraform text/x-toml application/x-toml application/graphql '
     'application/x-graphql text/x-graphql application/x-ndjson application/json5 application/x-json5 '
     'text/x-yaml application/toml application/x-yaml application/yaml text/x-astro text/srt '
     'application/x-subrip text/x-subrip text/vtt text/x-vcard text/calendar'),
)
OPENAI_DOCUMENT_EXTENSIONS = {ext for exts, _ in OPENAI_DOCUMENT_TYPES for ext in exts.split()}
OPENAI_DOCUMENT_MIMES = {mime for _, mimes in OPENAI_DOCUMENT_TYPES for mime in mimes.split()}
IMAGE = {"jpg", "jpeg", "png", "gif", "webp"}
AUDIO = {"mp3", "wav", "m4a", "aac", "ogg", "flac", "wma", "tta", "tak"}
VIDEO = {"mp4", "m4v", "mov", "mpeg", "mpg", "avi", "flv", "wmv", "webm",
         "3gp", "mkv", "ts", "asf", "ogv"}
DOCUMENT = OPENAI_DOCUMENT_EXTENSIONS | {"hwp", "hwpx"}
MIMES = {
    "txt": "text/plain", "md": "text/markdown", "csv": "text/csv", "tsv": "text/tsv",
    "pdf": "application/pdf", "json": "application/json", "hwp": "application/x-hwp",
    "hwpx": "application/hwp+zip", "xls": "application/vnd.ms-excel",
    "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "zip": "application/zip",
    "jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png", "gif": "image/gif", "webp": "image/webp",
    "mp3": "audio/mpeg", "wav": "audio/wav", "m4a": "audio/mp4", "aac": "audio/aac",
    "ogg": "audio/ogg", "flac": "audio/flac", "wma": "audio/x-ms-wma", "tta": "audio/x-tta", "tak": "audio/x-tak",
    "mp4": "video/mp4", "m4v": "video/mp4", "mov": "video/quicktime", "mpeg": "video/mpeg",
    "mpg": "video/mpeg", "avi": "video/x-msvideo", "flv": "video/x-flv", "wmv": "video/x-ms-wmv",
    "webm": "video/webm", "3gp": "video/3gpp", "mkv": "video/x-matroska", "ts": "video/mp2t",
    "asf": "video/x-ms-asf", "ogv": "video/ogg",
}
for _extensions, _mimes in OPENAI_DOCUMENT_TYPES:
    for _ext in _extensions.split():
        MIMES.setdefault(_ext, _mimes.split()[0])
MIME_ALIASES = {
    "image/jpg": "image/jpeg", "audio/mp3": "audio/mpeg", "audio/x-wav": "audio/wav",
    "audio/m4a": "audio/mp4", "audio/x-m4a": "audio/mp4", "audio/x-flac": "audio/flac",
    "video/mov": "video/quicktime", "video/avi": "video/x-msvideo", "video/wmv": "video/x-ms-wmv",
    "video/mpg": "video/mpeg", "video/x-m4v": "video/mp4", "application/x-zip-compressed": "application/zip",
    "application/haansofthwp": "application/x-hwp", "application/vnd.hancom.hwp": "application/x-hwp",
    "application/vnd.hancom.hwpx": "application/hwp+zip", "text/json": "application/json",
}


def normalize_mime(value):
    mime = value.split(";", 1)[0].strip().lower()
    return MIME_ALIASES.get(mime, mime)


def download_filename(url, disposition=""):
    """RFC 6266 filename* takes precedence; the URL path is the fallback."""
    if any(ord(c) < 32 or ord(c) == 127 for c in disposition):
        raise RunError("invalid_filename", "잘못된 Content-Disposition입니다.")
    message = Message()
    message["Content-Disposition"] = disposition
    params = message.get_params(header="content-disposition", unquote=True) or []
    names = [value for key, value in params[1:] if key.lower() == "filename"]
    extended = [value for value in names if isinstance(value, tuple)]
    plain = [value for value in names if isinstance(value, str)]
    if len(extended) > 1 or len(plain) > 1:
        raise RunError("invalid_filename", "중복된 파일명 매개변수입니다.")
    filename = plain[0] if plain else unquote(urlsplit(url).path.rsplit("/", 1)[-1])
    if extended:
        try:
            filename = collapse_rfc2231_value(extended[0], errors="strict")
        except (LookupError, UnicodeError):
            pass  # Invalid extended encoding uses the regular filename or URL.
    if not filename or filename in {".", ".."} or any(c in "/\\" or ord(c) < 32 or ord(c) == 127 for c in filename):
        raise RunError("invalid_filename", "잘못된 파일명입니다.")
    return filename


def _signature(prefix):
    """Return a container label, compatible extensions, and an unambiguous default."""
    for magic, extensions, default in (
        (b"%PDF-", {"pdf"}, "pdf"), (b"\x89PNG\r\n\x1a\n", {"png"}, "png"),
        (b"\xff\xd8\xff", {"jpg", "jpeg"}, "jpg"), (b"GIF87a", {"gif"}, "gif"), (b"GIF89a", {"gif"}, "gif"),
        (b"fLaC", {"flac"}, "flac"), (b"ID3", {"mp3"}, "mp3"),
        (b"OggS", {"ogg", "ogv"}, ""),
        (b"PK\x03\x04", {"zip", "hwpx", "xlsx", "docx", "pptx", "odt", "pages", "key"}, "zip"),
        (b"PK\x05\x06", {"zip", "hwpx", "xlsx", "docx", "pptx", "odt", "pages", "key"}, "zip"),
        (b"PK\x07\x08", {"zip", "hwpx", "xlsx", "docx", "pptx", "odt", "pages", "key"}, "zip"),
        (b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1", {"hwp", "xls", "xla", "xlb", "xlc", "xlm", "xlt", "xlw", "doc", "dot", "ppt", "pot", "ppa", "pps", "pwz", "wiz"}, ""),
        (b"\x1a\x45\xdf\xa3", {"mkv", "webm"}, ""),
        (b"\x30\x26\xb2\x75\x8e\x66\xcf\x11\xa6\xd9\x00\xaa\x00\x62\xce\x6c", {"asf", "wmv", "wma"}, ""),
        (b"FLV", {"flv"}, "flv"), (b"TTA1", {"tta"}, "tta"), (b"tBaK", {"tak"}, "tak"),
    ):
        if prefix.startswith(magic):
            label = "zip" if "zip" in extensions else "ole" if "hwp" in extensions else "ebml" if "mkv" in extensions else "asf" if "asf" in extensions else default or "ogg"
            return label, extensions, default
    if prefix[:4] == b"RIFF":
        for tag, ext in ((b"WAVE", "wav"), (b"WEBP", "webp"), (b"AVI ", "avi")):
            if prefix[8:12] == tag:
                return ext, {ext}, ext
    if len(prefix) >= 12 and prefix[4:8] in {b"ftyp", b"wide", b"moov", b"mdat"}:
        brand = prefix[8:12] if prefix[4:8] == b"ftyp" else b""
        if brand in {b"M4A ", b"M4B "}:
            return "iso-bmff", {"m4a"}, "m4a"
        if brand == b"qt  ":
            return "quicktime", {"mov"}, "mov"
        if brand.startswith(b"3g"):
            return "iso-bmff", {"3gp"}, "3gp"
        return "iso-bmff", {"mp4", "m4v", "m4a", "mov", "3gp"}, ""
    if len(prefix) >= 2 and prefix[0] == 255:
        if prefix[1] & 0xF6 == 0xF0:
            return "aac", {"aac"}, "aac"
        if prefix[1] & 0xE0 == 0xE0 and prefix[1] & 0x18 != 0x08 and prefix[1] & 0x06:
            return "mp3", {"mp3"}, "mp3"
    return None, set(), ""


def identify_file(prefix, filename, declared="application/octet-stream"):
    extension = PurePosixPath(filename).suffix.lower().lstrip(".")
    mime = normalize_mime(declared)
    candidates = {ext for ext, value in MIMES.items() if value == mime}
    exact_mime = candidates.copy()
    for exts, mimes in OPENAI_DOCUMENT_TYPES:
        if mime in mimes.split():
            candidates.update(exts.split())
    if mime == "text/plain":
        candidates.update({"csv", "tsv"})
    elif mime == "application/zip":
        candidates = {"zip", "hwpx", "xlsx", "docx", "pptx", "odt", "pages", "key"}
    signature, compatible, default = _signature(prefix)
    selected = extension if extension in MIMES else ""
    if not selected and candidates:
        # MIME takes precedence over the header, including when they conflict.
        choices = exact_mime or candidates
        selected = next((ext for ext in ("txt", "jpg", "mp4", "mpeg", "zip", "xls", "doc", "ppt") if ext in choices), sorted(choices)[0])
    selected_mime = MIMES[selected] if extension in MIMES else mime if candidates else MIMES.get(default, 'application/octet-stream')
    selected = selected or default
    issues = []
    if selected and signature and selected not in compatible:
        issues.append("header_mismatch")
    required_header = IMAGE | {"pdf", "mp3", "wav", "flac", "ogg", "m4a", "mp4", "m4v", "mov", "3gp", "zip", "xlsx", "hwpx", "docx", "pptx", "odt"}
    if selected in required_header and not signature:
        issues.append("header_missing")
    if candidates and selected and selected not in candidates:
        issues.append("mime_mismatch")
    kind = next((kind for kind, extensions in (("document", DOCUMENT), ("image", IMAGE),
                ("audio", AUDIO), ("video", VIDEO), ("archive", {"zip"})) if selected in extensions), "unknown")
    if selected and mime.split("/")[0] in {"image", "audio", "video"} and mime.split("/")[0] != kind and "mime_mismatch" not in issues:
        issues.append("mime_mismatch")
    if issues:
        status = "conflict"
    elif not selected:
        status = "unknown"
    elif default and selected in compatible and (len(compatible) == 1 or selected in {"jpg", "jpeg", "zip"}):
        status = "identified"
    else:
        status = "metadata_only"
    return {"kind": kind, "extension": selected, "mime": selected_mime,
            "status": status, "evidence": {"filenameExtension": extension, "declaredMime": declared, "signature": signature}, "issues": issues}

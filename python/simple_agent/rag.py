"""명시적 임베딩과 검색을 위한 공통 요청·계산 코드."""
from collections import Counter
import math
import re
import unicodedata
from .errors import RunError

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_MAX_TOKENS = 8192
EMBEDDING_DIMENSIONS = 1536
EMBEDDING_SPEC = "https://developers.openai.com/api/docs/models/text-embedding-3-small"


def valid_vector(vector):
    if not isinstance(vector, list) or not vector:
        raise RunError("invalid_vector", "비어 있지 않은 숫자 벡터가 필요합니다.")
    if any(type(x) not in (int, float) or not math.isfinite(x) for x in vector):
        raise RunError("invalid_vector", "벡터에는 유한한 숫자만 사용할 수 있습니다.")
    norm = math.hypot(*vector)
    if norm == 0 or not math.isfinite(norm):
        raise RunError("invalid_vector", "길이가 0이거나 계산할 수 없는 벡터입니다.")
    return norm


def valid_space_vector(space, vector):
    norm = valid_vector(vector)
    parts = space.rsplit(":", 2) if isinstance(space, str) else []
    if (len(parts) != 3 or not all(parts) or not parts[2].isascii()
            or not parts[2].isdigit() or int(parts[2]) < 1):
        raise RunError("invalid_embedding_space", "space에는 제공자:모델:차원 형식의 임베딩 공간이 필요합니다.")
    if len(vector) != int(parts[2]):
        raise RunError("embedding_dimension_mismatch", "벡터 차원이 지정한 임베딩 공간과 다릅니다.",
                       details={"space": space, "actual": len(vector), "expected": int(parts[2])})
    return norm


def embedding_request(text, dimensions=None):
    if not isinstance(text, str) or not text:
        raise RunError("invalid_embedding_input", "임베딩 입력은 비어 있지 않은 텍스트여야 합니다.")
    if dimensions is not None and (type(dimensions) is not int or not 1 <= dimensions <= EMBEDDING_DIMENSIONS):
        raise RunError("invalid_embedding_dimensions", "dimensions는 1부터 1536까지의 정수여야 합니다.", details={"source": EMBEDDING_SPEC})
    # Offline cl100k_base vocabulary, identical to the one published by OpenAI.
    from puretiktoken import get_encoding
    token_count = len(get_encoding("cl100k_base").encode_ordinary(text))
    if token_count > EMBEDDING_MAX_TOKENS:
        raise RunError("embedding_input_too_long", "OpenAI 임베딩 입력은 8192토큰 이하여야 합니다.",
                       details={"tokens": token_count, "limit": EMBEDDING_MAX_TOKENS, "source": EMBEDDING_SPEC})
    body = {"model": EMBEDDING_MODEL, "input": text, "encoding_format": "float"}
    if dimensions is not None:
        body["dimensions"] = dimensions
    return {"provider": "openai", "role": "embedding", "method": "POST",
            "url": "https://api.openai.com/v1/embeddings", "body": body}


def embedding_result(payload, dimensions=None):
    try:
        vector = payload["data"][0]["embedding"]
        valid_vector(vector)
        if len(vector) != (dimensions if dimensions is not None else EMBEDDING_DIMENSIONS):
            raise ValueError("unexpected embedding dimension")
    except (KeyError, IndexError, TypeError, ValueError, RunError) as exc:
        raise RunError("invalid_embedding_response", "임베딩 벡터를 응답에서 읽을 수 없습니다.", details={"response": payload}) from exc
    return {"vector": vector, "space": f"openai:{EMBEDDING_MODEL}:{len(vector)}", "response": payload}


def keyword_tokens(text):
    """Words plus Hangul bigrams for matching inflected Korean text."""
    tokens = []
    for word in re.findall(r"[^\W_]+", unicodedata.normalize("NFKC", text).casefold()):
        tokens.append(word)
        for part in re.findall(r"[가-힣]+", word):
            if len(part) > 2:
                tokens.extend(part[i:i + 2] for i in range(len(part) - 1))
    return tokens


async def search(store, scope, space, query, vector, limit):
    valid_space_vector(space, vector)
    documents = []
    cursor = None
    while True:
        page = await store.rag_list(scope, cursor)
        for key in page["keys"]:
            item = await store.rag_read(scope, key)
            if item.get("space") != space:
                continue
            documents.append({"key": item["key"], "text": item["text"], "metadata": item["metadata"],
                              "vector": item["vector"]})
        cursor = page.get("cursor")
        if cursor is None:
            break
    return {"scope": scope, "space": space, "matches": rank_candidates(documents, query, vector)[:limit]}


def rank_candidates(candidates, query, vector):
    """Shared flat/child ranking. Candidate keys must be unique and comparable."""
    query_norm = valid_vector(vector)
    terms = set(keyword_tokens(query))
    documents, document_frequency, total_length = [], Counter(), 0
    for candidate in candidates:
        doc = dict(candidate)
        other = doc.pop("vector")
        norm = valid_vector(other)
        if len(other) != len(vector):
            raise RunError("embedding_dimension_mismatch", "같은 임베딩 공간에 서로 다른 차원의 벡터가 있습니다.")
        words = keyword_tokens(doc["text"])
        frequencies = Counter(words)
        document_frequency.update(terms & frequencies.keys())
        total_length += len(words)
        doc.update(vector_score=math.fsum((a / query_norm) * (b / norm) for a, b in zip(vector, other, strict=True)),
                   frequencies=frequencies, length=len(words))
        documents.append(doc)
    # BM25 uses the standard k1=1.2 and b=0.75; rank fusion keeps the two
    # unrelated score scales separate. Ties are deterministic by key.
    count = len(documents)
    average_length = total_length / count if count else 0
    for doc in documents:
        frequencies = doc.pop("frequencies")
        length = doc.pop("length")
        doc["keyword_score"] = math.fsum(
            math.log1p((count - document_frequency[term] + 0.5) / (document_frequency[term] + 0.5))
            * frequencies[term] * 2.2 / (frequencies[term] + 1.2 * (0.25 + 0.75 * length / average_length))
            for term in terms if frequencies[term] and average_length
        )
        doc["vector_rank"] = None
        doc["keyword_rank"] = None
        doc["score"] = 0.0
    for rank, doc in enumerate(sorted(documents, key=lambda d: (-d["vector_score"], d["key"])), 1):
        doc["vector_rank"] = rank
        doc["score"] += 1 / rank
    keyword_docs = (doc for doc in documents if doc["keyword_score"] > 0)
    for rank, doc in enumerate(sorted(keyword_docs, key=lambda d: (-d["keyword_score"], d["key"])), 1):
        doc["keyword_rank"] = rank
        doc["score"] += 1 / rank
    return sorted(documents, key=lambda d: (-d["score"], d["key"]))

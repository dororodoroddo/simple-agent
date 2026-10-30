"""Parent Document Retrieval: rank short children and return their full parents."""
from .chunking import CHUNK_INPUT, CHUNKS, SPAN, TEXT, chunk_text, validate_chunks
from .errors import RunError
from .protocol import digest, json_text
from .rag import rank_candidates, valid_space_vector
from .tools import ToolSpec, object_schema, validate_schema

VECTOR = {'type': 'array', 'minItems': 1, 'items': {'type': 'number'}}
DOCUMENT = object_schema({
    'document_id': TEXT, 'source': {'type': 'object', 'minProperties': 1},
    'chunks': CHUNKS, 'space': TEXT,
    'vectors': {'type': 'object', 'additionalProperties': VECTOR},
})
INDEX_INPUT = object_schema({'scope': TEXT, **DOCUMENT['properties']})
RETRIEVE_INPUT = object_schema({'scope': TEXT, 'space': TEXT, 'query': TEXT, 'vector': VECTOR,
    'child_limit': {'type': 'integer', 'minimum': 1}, 'parent_limit': {'type': 'integer', 'minimum': 1}})
DELETE_INPUT = object_schema({'scope': TEXT, 'document_id': TEXT})
CHILD_MATCH = object_schema({
    **CHUNKS['properties']['children']['items']['properties'],
    'score': {'type': 'number'}, 'vector_score': {'type': 'number'}, 'keyword_score': {'type': 'number'},
    'vector_rank': {'type': 'integer', 'minimum': 1},
    'keyword_rank': {'anyOf': [{'type': 'integer', 'minimum': 1}, {'type': 'null'}]},
})
RETRIEVE_OUTPUT = object_schema({'scope': TEXT, 'space': TEXT, 'matches': {'type': 'array', 'items': object_schema({
    'document_id': TEXT, 'parent_id': TEXT, 'text': TEXT, 'source': {'type': 'object'},
    'metadata': {'type': 'object'}, 'span': {'anyOf': [SPAN, {'type': 'null'}]},
    'score': {'type': 'number'}, 'children': {'type': 'array', 'minItems': 1, 'items': CHILD_MATCH},
})}})


def validate_document(item):
    validate_schema(item, DOCUMENT, 'invalid_rag_index')
    validate_chunks(item['chunks'])
    ids = {child['id'] for child in item['chunks']['children']}
    if ids != item['vectors'].keys():
        raise RunError('invalid_rag_index', '모든 하위 청크 ID에 정확히 하나의 벡터가 필요합니다.')
    for vector in item['vectors'].values():
        valid_space_vector(item['space'], vector)
    try:
        json_text(item)
    except (TypeError, ValueError) as exc:
        raise RunError('invalid_rag_index', '출처·청크·벡터는 유한한 JSON 값이어야 합니다.') from exc


def index_document(args):
    return {key: args[key] for key in DOCUMENT['properties']}


async def retrieve(store, args):
    parents, candidates, cursor = {}, [], None
    while True:
        page = await store.rag_index_list(args['scope'], cursor)
        for key in page['keys']:
            item = await store.rag_index_read(args['scope'], key)
            validate_document(item)
            if digest(item['document_id']) != key:
                raise RunError('invalid_rag_index', '저장된 문서 ID가 검색 키와 일치하지 않습니다.')
            if item['space'] != args['space']:
                continue
            for parent in item['chunks']['parents']:
                identity = (item['document_id'], parent['id'])
                parents[identity] = {'document_id': item['document_id'], 'parent_id': parent['id'],
                    'text': parent['text'], 'span': parent['span'], 'metadata': parent['metadata'],
                    'source': item['source']}
            for child in item['chunks']['children']:
                candidates.append({**child, 'key': (item['document_id'], child['id']),
                    'vector': item['vectors'][child['id']]})
        cursor = page.get('cursor')
        if cursor is None:
            break
    # A parent is scored by its strongest child, not by the number of overlapping hits.
    selected = {}
    for child in rank_candidates(candidates, args['query'], args['vector'])[:int(args['child_limit'])]:
        document_id, _ = child.pop('key')
        identity = (document_id, child['parent_id'])
        if identity not in selected:
            if len(selected) == args['parent_limit']:
                continue
            selected[identity] = {**parents[identity], 'score': child['score'], 'children': []}
        selected[identity]['children'].append(child)
    return {'scope': args['scope'], 'space': args['space'], 'matches': list(selected.values())}


def register_tools(registry):
    async def chunk(args, ctx):
        return chunk_text(args)

    async def index(args, ctx):
        return await ctx.store.rag_index_put(args['scope'], args['document_id'], index_document(args))

    async def search(args, ctx):
        return await retrieve(ctx.store, args)

    async def delete(args, ctx):
        return await ctx.store.rag_index_delete(args['scope'], args['document_id'])

    registry.register(ToolSpec('rag_chunk', '텍스트를 수동 지정하거나 줄·문장 경계와 겹침을 맞춰 나눕니다. 마크다운 표와 앞뒤 설명은 길이를 초과하더라도 별도 상위 문맥으로 보존합니다.',
        CHUNK_INPUT, CHUNKS, chunk, 'rag_chunk'))
    registry.register(ToolSpec('rag_index', '출처·상위 문맥·하위 청크·명시적 벡터를 문서 단위로 저장하거나 교체합니다.',
        INDEX_INPUT, object_schema({'scope': TEXT, 'document_id': TEXT, 'saved': {'const': True}}),
        index, 'rag_write', read_only=False, external=True,
        validate_input=lambda args, ctx: validate_document(index_document(args))))
    registry.register(ToolSpec('rag_retrieve', '하위 청크의 키워드·벡터 순위로 검색하고 중복 없는 상위 문맥과 출처를 반환합니다.',
        RETRIEVE_INPUT, RETRIEVE_OUTPUT, search, 'rag_read', external=True,
        validate_input=lambda args, ctx: valid_space_vector(args['space'], args['vector'])))
    registry.register(ToolSpec('rag_delete', '지정한 문서의 상위 문맥·하위 청크·벡터를 함께 삭제합니다.',
        DELETE_INPUT, object_schema({'scope': TEXT, 'document_id': TEXT, 'deleted': {'const': True}}),
        delete, 'rag_write', read_only=False, external=True))

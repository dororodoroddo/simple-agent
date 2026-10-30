"""Explicit text chunking. No file parsing, model calls or storage side effects."""
import copy
import re
from bisect import bisect_left, bisect_right
from .errors import RunError
from .tools import object_schema, validate_schema

TEXT = {'type': 'string', 'minLength': 1}
OFFSET = {'type': 'integer', 'minimum': 0}
SPAN = object_schema({'start': OFFSET, 'end': OFFSET})
MANUAL_PARENT = {'type': 'object', 'properties': {
    'text': TEXT, 'metadata': {'type': 'object'},
    'children': {'type': 'array', 'minItems': 1, 'items': {'anyOf': [TEXT,
        object_schema({'text': TEXT, 'start': OFFSET})]}},
}, 'required': ['text', 'children'], 'additionalProperties': False}
CHUNK_INPUT = {'type': 'object', 'oneOf': [
    object_schema({'mode': {'const': 'manual'}, 'parents': {'type': 'array', 'minItems': 1, 'items': MANUAL_PARENT}}),
    object_schema({'mode': {'const': 'length'}, 'text': TEXT,
        'parent_chars': {'type': 'integer', 'minimum': 1}, 'child_chars': {'type': 'integer', 'minimum': 1},
        'parent_overlap': OFFSET, 'child_overlap': OFFSET}),
]}
CHUNKS = object_schema({
    'parents': {'type': 'array', 'minItems': 1, 'items': object_schema({
        'id': TEXT, 'text': TEXT, 'span': {'anyOf': [SPAN, {'type': 'null'}]}, 'metadata': {'type': 'object'}})},
    'children': {'type': 'array', 'minItems': 1, 'items': object_schema({
        'id': TEXT, 'parent_id': TEXT, 'text': TEXT, 'start': OFFSET, 'end': OFFSET})},
})


def _nonblank(text):
    if not text.strip():
        raise RunError('invalid_chunks', '공백만 있는 문맥·문장은 청크로 사용할 수 없습니다.')


def _boundaries(text):
    return [[match.end() for match in re.finditer(pattern, text)] for pattern in (
        r'\r\n|\r|\n',
        r'(?:[!?。！？]+|\.(?![A-Za-z0-9]))[\"\u201d\u2019\)\]]*[ \t]*',
        r'[ \t]+',
    )]


def split_spans(text, size, overlap):
    """Align both ends, including overlap, to lines, sentences, then spaces."""
    boundaries = _boundaries(text)
    start, previous_end = 0, 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            for ends in boundaries:
                index = bisect_right(ends, end) - 1
                if index >= 0 and ends[index] > previous_end and text[start:ends[index]].strip():
                    end = ends[index]
                    break
                # Prefer a whole next unit to overlap that would force a finer split.
                previous = bisect_left(ends, previous_end)
                if start < previous_end and previous < len(ends) and ends[previous] == previous_end:
                    fresh_end = min(previous_end + size, len(text))
                    index = bisect_right(ends, fresh_end) - 1
                    if fresh_end != len(text):
                        if index < 0 or ends[index] <= previous_end:
                            continue
                        fresh_end = ends[index]
                    start = previous_end
                    end = fresh_end
                    break
        if text[start:end].strip():
            yield start, end
        if end == len(text):
            break
        target = max(start + 1, end - overlap)
        start = target
        for ends in boundaries:
            index = bisect_left(ends, target)
            if index < len(ends) and ends[index] <= end:
                start = ends[index]
                break
        if not text[start:end].strip():
            start = end
        previous_end = end


def _table_cells(line):
    """Split unescaped pipes; optional outside pipes do not create cells."""
    cells, current, escaped = [], [], False
    for char in line.strip():
        if char == '|' and not escaped:
            cells.append(''.join(current).strip())
            current = []
        else:
            current.append(char)
        escaped = char == '\\' and not escaped
    if not cells:
        return None
    cells.append(''.join(current).strip())
    if not cells[0]:
        cells.pop(0)
    if cells and not cells[-1]:
        cells.pop()
    return cells


def table_contexts(text):
    """Top-level pipe tables with their adjacent paragraphs and preceding heading."""
    lines = [m.group() for m in re.finditer(r'[^\r\n]*(?:\r\n|\r|\n|$)', text) if m.group()]
    offsets = [0]
    blocked, fence = set(), None
    heading = re.compile(r' {0,3}#{1,6}(?:\s|$)')
    block_start = re.compile(r'(?: {4}|\t| {0,3}(?:#{1,6}(?:\s|$)|>|[-+*]\s|\d+[.)]\s|(?:[-*_]\s*){3,}$|<))')
    for index, line in enumerate(lines):
        offsets.append(offsets[-1] + len(line))
        marker = re.match(r' {0,3}(`{3,}|~{3,})(.*)', line)
        if fence:
            blocked.add(index)
            if marker and marker[1][0] == fence[0] and len(marker[1]) >= len(fence) and not marker[2].strip():
                fence = None
        elif marker:
            fence = marker[1]
            blocked.add(index)
        elif block_start.match(line):
            blocked.add(index)

    tables, index = [], 0
    while index + 1 < len(lines):
        header, delimiter = _table_cells(lines[index]), _table_cells(lines[index + 1])
        single_column = re.fullmatch(r' {0,3}:?-+:?[ \t]*(?:\r\n|\r|\n)?', lines[index + 1])
        if delimiter is None and single_column:
            delimiter = [lines[index + 1].strip()]
        if (index in blocked or (index + 1 in blocked and not single_column) or not header or not delimiter
                or len(header) != len(delimiter)
                or not all(re.fullmatch(r':?-+:?', cell) for cell in delimiter)):
            index += 1
            continue
        end = index + 2
        while end < len(lines) and lines[end].strip() and end not in blocked:
            end += 1
        tables.append((index, end))
        index = end

    table_lines = {i for start, end in tables for i in range(start, end)}

    def paragraph_line(index):
        return (0 <= index < len(lines) and index not in blocked
                and index not in table_lines and bool(lines[index].strip()))

    for table_start, table_end in tables:
        start, end = table_start, table_end
        before = start - 1
        while before >= 0 and not lines[before].strip():
            before -= 1
        if paragraph_line(before):
            start = before
            while paragraph_line(start - 1):
                start -= 1
            before = start - 1
            while before >= 0 and not lines[before].strip():
                before -= 1
        if before >= 0 and heading.match(lines[before]):
            start = before
        after = end
        while after < len(lines) and not lines[after].strip():
            after += 1
        if paragraph_line(after):
            end = after + 1
            while paragraph_line(end):
                end += 1
        # Retain separator whitespace with the preceding region.
        while end < len(lines) and not lines[end].strip():
            end += 1
        yield offsets[start], offsets[end], {
            'kind': 'markdown_table',
            'table_span': {'start': offsets[table_start], 'end': offsets[table_end]},
        }


def parent_spans(text, size, overlap):
    """Table contexts are indivisible parents; ordinary text keeps size limits."""
    cursor = 0
    for start, end, metadata in table_contexts(text):
        for left, right in split_spans(text[cursor:start], size, overlap):
            yield cursor + left, cursor + right, {}
        yield start, end, metadata
        cursor = end
    for left, right in split_spans(text[cursor:], size, overlap):
        yield cursor + left, cursor + right, {}


def chunk_text(arguments):
    validate_schema(arguments, CHUNK_INPUT)
    parents, children = [], []

    def append_parent(text, span, metadata, pieces):
        _nonblank(text)
        pid = 'p' + str(len(parents))
        parents.append({'id': pid, 'text': text, 'span': span, 'metadata': copy.deepcopy(metadata)})
        for number, (start, end) in enumerate(pieces):
            value = text[start:end]
            _nonblank(value)
            children.append({'id': f'{pid}:c{number}', 'parent_id': pid,
                             'text': value, 'start': start, 'end': end})

    if arguments['mode'] == 'length':
        text = arguments['text']
        _nonblank(text)
        # JSON Schema integer also accepts JSON numbers such as 12.0.
        parent_size, child_size = int(arguments['parent_chars']), int(arguments['child_chars'])
        parent_overlap, child_overlap = int(arguments['parent_overlap']), int(arguments['child_overlap'])
        if parent_overlap >= parent_size or child_overlap >= child_size or child_size > parent_size:
            raise RunError('invalid_chunk_config', '겹침은 해당 청크 길이보다 작고 하위 길이는 상위 길이 이하여야 합니다.')
        for start, end, metadata in parent_spans(text, parent_size, parent_overlap):
            parent = text[start:end]
            append_parent(parent, {'start': start, 'end': end}, metadata, split_spans(parent, child_size, child_overlap))
    else:
        for parent in arguments['parents']:
            text, spans = parent['text'], []
            for child in parent['children']:
                if isinstance(child, str):
                    start, value = text.find(child), child
                    if start >= 0 and text.find(child, start + 1) >= 0:
                        raise RunError('ambiguous_child_span', '같은 문장이 반복되면 text와 start로 위치를 지정하세요.')
                else:
                    start, value = int(child['start']), child['text']
                if start < 0 or text[start:start + len(value)] != value:
                    raise RunError('invalid_child_span', '하위 문장은 지정한 상위 문맥의 원문과 일치해야 합니다.')
                span = (start, start + len(value))
                if span in spans:
                    raise RunError('duplicate_child_span', '동일 위치의 하위 청크가 중복되었습니다.')
                spans.append(span)
            append_parent(text, None, parent.get('metadata', {}), spans)
    result = {'parents': parents, 'children': children}
    validate_chunks(result)
    return result


def validate_chunks(chunks):
    validate_schema(chunks, CHUNKS, 'invalid_chunks')
    parents = {parent['id']: parent for parent in chunks['parents']}
    if len(parents) != len(chunks['parents']):
        raise RunError('invalid_chunks', '상위 청크 ID가 중복되었습니다.')
    for parent in parents.values():
        _nonblank(parent['text'])
        span = parent['span']
        if span is not None and span['end'] - span['start'] != len(parent['text']):
            raise RunError('invalid_chunks', '상위 원문 위치와 글자 수가 일치하지 않습니다.')
    ids, covered, positions = set(), set(), set()
    for child in chunks['children']:
        parent = parents.get(child['parent_id'])
        start, end = int(child['start']), int(child['end'])
        position = (child['parent_id'], start, end)
        if child['id'] in ids or position in positions or parent is None:
            raise RunError('invalid_chunks', '중복 청크 또는 없는 상위 청크 참조입니다.')
        _nonblank(child['text'])
        if end <= start or end > len(parent['text']) or parent['text'][start:end] != child['text']:
            raise RunError('invalid_chunks', '하위 청크의 위치·원문이 상위 문맥과 일치하지 않습니다.')
        ids.add(child['id']); positions.add(position); covered.add(child['parent_id'])
    if covered != parents.keys():
        raise RunError('invalid_chunks', '모든 상위 문맥에는 검색할 하위 청크가 필요합니다.')

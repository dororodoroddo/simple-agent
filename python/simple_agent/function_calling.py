"""OpenAI function contracts. Validation never executes a function or repairs a schema."""
import json
import re
from .errors import RunError
from .tools import compile_schema, validate_schema


FUNCTION_TOOL_SCHEMA = {
    'type': 'object', 'additionalProperties': False,
    'required': ['type', 'name', 'description', 'parameters', 'strict'],
    'properties': {
        'type': {'const': 'function'},
        'name': {'type': 'string', 'pattern': '^[a-zA-Z0-9_-]{1,64}$'},
        'description': {'type': 'string', 'minLength': 1},
        'parameters': {'type': 'object'}, 'strict': {'type': 'boolean'},
    },
}


def json_object(text):
    def reject(value):
        raise ValueError('Non-finite JSON number: ' + value)
    def pairs(items):
        value = {}
        for key, item in items:
            if key in value: raise ValueError('Duplicate JSON key: ' + key)
            value[key] = item
        return value
    value = json.loads(text, parse_constant=reject, object_pairs_hook=pairs)
    if not isinstance(value, dict):
        raise ValueError('Function arguments must be a JSON object')
    return value


def validate_strict_schema(schema):
    """Check the strict subset's object rules without adding required fields/defaults."""
    if schema.get('type') != 'object' or 'anyOf' in schema:
        raise RunError('invalid_strict_schema', 'strict 함수의 루트는 object여야 합니다.')

    def visit(value):
        if not isinstance(value, dict):
            raise RunError('invalid_strict_schema', '스키마는 객체여야 합니다.')
        unsupported = {'allOf', 'not', 'if', 'then', 'else', 'dependentRequired', 'dependentSchemas', 'patternProperties'}
        if unsupported.intersection(value):
            raise RunError('invalid_strict_schema', '지원하지 않는 strict 스키마 구문입니다.')
        if not any(key in value for key in ('type', '$ref', 'anyOf', 'enum', 'const')):
            raise RunError('invalid_strict_schema', 'strict 스키마에는 명시적인 자료형이 필요합니다.')
        types = value.get('type', [])
        types = [types] if isinstance(types, str) else types
        if 'object' in types:
            properties = value.get('properties', {})
            required = value.get('required', [])
            if value.get('additionalProperties') is not False or set(required) != set(properties):
                raise RunError('invalid_strict_schema', '모든 object 속성을 required에 선언하고 additionalProperties=false로 지정하세요.')
        for child in value.get('properties', {}).values(): visit(child)
        for child in value.get('$defs', {}).values(): visit(child)
        for child in value.get('definitions', {}).values(): visit(child)
        for child in value.get('anyOf', []): visit(child)
        if 'items' in value: visit(value['items'])
    try:
        compile_schema(schema)
        visit(schema)
    except RunError:
        raise
    except Exception as exc:
        raise RunError('invalid_strict_schema', '유효하지 않은 strict 스키마입니다.') from exc


def validate_history(history):
    """History follows the original user prompt; no new media bypasses file adapters."""
    pending, seen = set(), set()
    for item in history:
        kind = item.get('type')
        if kind == 'function_call':
            cid = item.get('call_id')
            if not isinstance(cid, str) or not cid or cid in seen or not isinstance(item.get('name'), str):
                raise RunError('invalid_llm_history', '이력의 함수 호출 ID가 없거나 중복되었습니다.')
            try: json_object(item['arguments'])
            except (ValueError, TypeError, KeyError) as exc:
                raise RunError('invalid_llm_history', '이력의 함수 인자가 JSON 객체가 아닙니다.') from exc
            seen.add(cid); pending.add(cid)
        elif kind == 'function_call_output':
            cid = item.get('call_id')
            if not isinstance(cid, str) or cid not in pending or not isinstance(item.get('output'), str):
                raise RunError('invalid_llm_history', '함수 결과에 대응하는 미완료 호출이 없습니다.')
            pending.remove(cid)
        elif kind == 'reasoning':
            pass  # Preserve all returned reasoning fields, including encrypted_content.
        elif kind == 'message' or kind is None and 'role' in item:
            role, content = item.get('role'), item.get('content')
            expected = {'output_text', 'refusal'} if role == 'assistant' else {'input_text'} if role == 'user' else set()
            if not isinstance(content, list) or not content or any(not isinstance(p, dict) or p.get('type') not in expected for p in content):
                raise RunError('invalid_llm_history', '이력에는 사용자 텍스트·모델 출력·추론·함수 호출과 결과만 전달하세요.')
        else:
            raise RunError('invalid_llm_history', '지원하지 않는 호출 이력 항목입니다.')
    if pending:
        raise RunError('invalid_llm_history', '결과가 없는 함수 호출이 이력에 남아 있습니다.')


def allowed_names(config):
    names = {tool['name'] for tool in config.get('tools', [])}
    choice = config.get('tool_choice', 'auto')
    if choice == 'none': return set()
    if isinstance(choice, dict):
        return {choice['name']} if choice['type'] == 'function' else {tool['name'] for tool in choice['tools']}
    return names


def requires_call(config):
    choice = config.get('tool_choice', 'auto')
    return choice == 'required' or isinstance(choice, dict) and (choice['type'] == 'function' or choice.get('mode') == 'required')


def validate_function_config(config, provider, api):
    keys = {'tools', 'tool_choice', 'parallel_tool_calls', 'history'}
    if keys.intersection(config) and (provider, api) != ('openai', 'responses'):
        raise RunError('unsupported_llm_config', '함수 도구와 호출 이력은 OpenAI Responses에서 지원합니다.')
    names = set()
    for tool in config.get('tools', []):
        validate_schema(tool, FUNCTION_TOOL_SCHEMA, 'invalid_llm_config')
        if tool['name'] in names:
            raise RunError('invalid_llm_config', '도구 이름이 중복되었습니다.')
        names.add(tool['name'])
        if tool['parameters'].get('type') != 'object':
            raise RunError('invalid_llm_config', '함수 입력 스키마의 루트는 object여야 합니다.')
        try: compile_schema(tool['parameters'])
        except Exception as exc:
            raise RunError('invalid_llm_config', '함수 입력 스키마가 유효하지 않습니다.') from exc
        if tool['strict']: validate_strict_schema(tool['parameters'])
    choice = config.get('tool_choice', 'auto')
    if isinstance(choice, dict):
        if choice.get('type') == 'function' and set(choice) == {'type', 'name'} and isinstance(choice['name'], str):
            selected = {choice['name']}
        elif choice.get('type') == 'allowed_tools' and set(choice) == {'type', 'mode', 'tools'} and choice['mode'] in {'auto', 'required'}:
            refs = choice['tools']
            if not isinstance(refs, list) or any(not isinstance(t, dict) or set(t) != {'type', 'name'} or t['type'] != 'function' or not isinstance(t['name'], str) for t in refs):
                raise RunError('invalid_llm_config', 'allowed_tools에는 함수 이름 목록이 필요합니다.')
            selected = {t['name'] for t in refs}
            if len(selected) != len(refs):
                raise RunError('invalid_llm_config', 'allowed_tools의 이름이 중복되었습니다.')
        else:
            raise RunError('invalid_llm_config', 'tool_choice 구조가 유효하지 않습니다.')
        if not selected.issubset(names):
            raise RunError('invalid_llm_config', 'tool_choice가 등록되지 않은 함수를 지정했습니다.')
    elif choice not in {'auto', 'none', 'required'}:
        raise RunError('invalid_llm_config', '지원하지 않는 tool_choice입니다.')
    if requires_call(config) and not allowed_names(config):
        raise RunError('invalid_llm_config', '필수 호출에는 허용된 함수가 필요합니다.')
    validate_history(config.get('history', []))


def function_calls(payload, config):
    calls, ids = [], {i['call_id'] for i in config.get('history', []) if i.get('type') == 'function_call'}
    specs = {tool['name']: tool for tool in config.get('tools', [])}
    output = payload.get('output', [])
    if not isinstance(output, list) or any(not isinstance(item, dict) for item in output):
        raise RunError('invalid_function_call', '모델 출력 항목은 객체 목록이어야 합니다.')
    for item in output:
        if item.get('type') != 'function_call': continue
        cid, name = item.get('call_id'), item.get('name')
        if not isinstance(cid, str) or not cid or cid in ids or not isinstance(name, str) or name not in allowed_names(config) or item.get('status', 'completed') != 'completed':
            raise RunError('invalid_function_call', '모델의 함수 이름·호출 ID·완료 상태가 유효하지 않습니다.')
        try: args = json_object(item['arguments'])
        except (ValueError, TypeError, KeyError) as exc:
            raise RunError('invalid_function_call', '함수 인자가 JSON 객체가 아닙니다.') from exc
        validate_schema(args, specs[name]['parameters'], 'invalid_function_call')
        ids.add(cid)
        calls.append({'name': name, 'arguments': args, 'call_id': cid})
    if config.get('parallel_tool_calls') is False and len(calls) > 1:
        raise RunError('invalid_function_call', '순차 호출 설정에서 복수 함수가 반환되었습니다.')
    choice = config.get('tool_choice')
    if isinstance(choice, dict) and choice['type'] == 'function' and len(calls) != 1:
        raise RunError('invalid_function_call', '특정 함수 지정은 정확히 한 호출을 반환해야 합니다.')
    if requires_call(config) and not calls:
        raise RunError('invalid_function_call', '필수 함수 호출이 반환되지 않았습니다.')
    return calls

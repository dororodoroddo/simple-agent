"""Provider wire adapters. Configuration is supplied by the caller, never inferred from env."""
import base64
import copy
import json
from .errors import RunError
from .files import StoredFile, binary_reference

from .file_formats import IMAGE, OPENAI_DOCUMENT_EXTENSIONS, OPENAI_DOCUMENT_MIMES

# Current adapter formats, not a guarantee that every model accepts them.
# https://ai.google.dev/gemini-api/docs/{image-understanding,audio,video-understanding}
GEMINI_TEXT = {"txt", "md", "csv", "tsv"}
GEMINI_FORMATS = GEMINI_TEXT | {"pdf", "jpg", "jpeg", "png", "webp", "mp3", "wav", "m4a", "aac", "ogg", "flac",
                         "mp4", "m4v", "mov", "mpeg", "mpg", "avi", "flv", "wmv", "webm", "3gp"}
GEMINI_MIMES = {"audio/mp4": "audio/m4a", "video/quicktime": "video/mov",
               "video/x-msvideo": "video/avi", "video/x-ms-wmv": "video/wmv"}


def model_input_support(format_info):
    usable = format_info['status'] != 'unknown'
    ext = format_info['extension']
    return {"openai/responses": usable and (ext in IMAGE | OPENAI_DOCUMENT_EXTENSIONS or format_info['mime'] in OPENAI_DOCUMENT_MIMES),
            "openai/chat/completions": usable and ext in IMAGE | {"mp3", "wav"},
            "gemini/interactions": usable and ext in GEMINI_FORMATS and (ext not in GEMINI_TEXT or format_info['mime'] in {
                'text/plain', 'text/markdown', 'text/csv', 'application/csv', 'text/tsv', 'text/tab-separated-values'})}


def model_file_metadata(format_info, provider):
    if format_info['status'] == 'unknown':
        raise RunError('unsupported_format', '파일 형식을 식별할 수 없습니다.', details=format_info)
    meta = {key: format_info[key] for key in ('kind', 'extension', 'mime')}
    if provider == 'gemini':
        # Use the same provider MIME for both Files upload and Interactions input.
        meta['mime'] = 'text/plain' if meta['extension'] in GEMINI_TEXT else GEMINI_MIMES.get(meta['mime'], meta['mime'])
    return meta


# callLLM 공통 설정. 세부 범위·모델별 제약은 아래 검증과 README 참고.
# 공식 규격: https://developers.openai.com/api/reference/resources/responses/methods/create
#            https://ai.google.dev/api/interactions-api
LLM_CONFIG_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["model", "max_input_tokens", "max_output_tokens", "thinking"],
    "properties": {
        "provider": {"enum": ["gemini", "openai"]},
        "model": {"type": "string", "minLength": 1},
        "max_input_tokens": {"type": ["integer", "null"], "minimum": 1,
                             "description": "정수: 공식 계수 후 검사. 명시적 null: 제공자 자체 한도 사용."},
        "max_output_tokens": {"type": "integer", "minimum": 1},
        # OpenAI: none/minimal/low/medium/high/xhigh/max; Gemini: minimal/low/medium/high.
        # 모델별 지원 수준은 다르며 null은 제공자 설정 생략이다.
        "thinking": {"type": ["string", "null"], "minLength": 1,
                     "description": "공통 사고 수준. 예: low, medium, high. null은 설정 생략."},
        "thinking_summary": {"type": ["string", "null"], "minLength": 1,
                             "description": "auto 또는 none. OpenAI는 concise/detailed도 지원."},
        "api": {"enum": ["interactions", "responses", "chat/completions"]},
        # sampling: OpenAI 0~2 및 0~1. Gemini Interactions는 현재 공식 규격에 없음.
        "temperature": {"type": ["number", "null"], "minimum": 0, "maximum": 2},
        "top_p": {"type": ["number", "null"], "minimum": 0, "maximum": 1},
        # 텍스트 지시 및 구조화 출력. JSON Schema 지원 범위는 모델별로 다름.
        "system_instruction": {"type": ["string", "null"]},
        "response_schema": {"type": ["object", "null"], "properties": {
            "name": {"type": "string", "pattern": "^[a-zA-Z0-9_-]{1,64}$"},
            "schema": {"type": "object"}, "strict": {"type": "boolean"}},
            "required": ["name", "schema"], "additionalProperties": False},
        # seed/stop: Gemini Interactions와 OpenAI Chat Completions만 지원.
        # Chat seed는 signed int64, stop은 최대 4개. 모델별 지원 여부는 다름.
        "seed": {"type": ["integer", "null"]},
        "stop_sequences": {"type": ["array", "null"], "items": {"type": "string"}},
        "tools": {"type": "array", "items": {"type": "object"}},
        "tool_choice": {"type": ["string", "object"]},
        "parallel_tool_calls": {"type": "boolean"},
        "history": {"type": "array", "items": {"type": "object"}},
    },
}


def validate_llm_config(config):
    from .tools import validate_schema
    validate_schema(config, LLM_CONFIG_SCHEMA, "invalid_llm_config")
    provider, thinking = config.get('provider', 'openai'), config['thinking']
    api = config.get('api', 'interactions' if provider == 'gemini' else 'responses')
    from .function_calling import validate_function_config
    validate_function_config(config, provider, api)
    if api not in ({'interactions'} if provider == 'gemini' else {'responses','chat/completions'}):
        raise RunError('invalid_llm_config', 'config.api가 config.provider와 일치하지 않습니다.')
    if not config['model'].strip() or any(c.isspace() for c in config['model']):
        raise RunError('invalid_llm_config', 'config.model에는 실제 모델 ID를 입력해야 합니다.')
    summary = config.get('thinking_summary')
    if (thinking is not None and not thinking.strip()) or (summary is not None and not summary.strip()):
        raise RunError('invalid_llm_config', 'thinking과 thinking_summary는 빈 문자열이 아닌 값 또는 null이어야 합니다.')
    levels = {'minimal', 'low', 'medium', 'high'} if provider == 'gemini' else {'none', 'minimal', 'low', 'medium', 'high', 'xhigh', 'max'}
    if thinking is not None and thinking not in levels:
        raise RunError('invalid_llm_config', '제공자가 지원하지 않는 thinking 값입니다.', details={'provider': provider, 'allowed': sorted(levels)})
    summaries = {'auto', 'none'} if provider == 'gemini' else {'auto', 'none', 'concise', 'detailed'}
    if summary is not None and summary not in summaries:
        raise RunError('invalid_llm_config', '제공자가 지원하지 않는 thinking_summary 값입니다.', details={'provider': provider, 'allowed': sorted(summaries)})
    if api == 'chat/completions' and (thinking is not None or summary is not None):
        raise RunError('invalid_llm_config', '이 Chat Completions 어댑터는 thinking 설정을 지원하지 않습니다. null을 명시하세요.')
    if api == 'responses' and config['max_output_tokens'] < 16:
        raise RunError('invalid_llm_config', 'OpenAI Responses max_output_tokens의 공식 최솟값은 16입니다.')
    if api == 'chat/completions' and config['max_input_tokens'] is not None:
        raise RunError('unsupported_token_count', 'Chat Completions 음성 입력의 공식 사전 계수 경로가 없습니다. config.max_input_tokens=null로 제공자 한도 적용을 명시해야 합니다.')
    if provider == 'gemini' and any(config.get(k) is not None for k in ('temperature', 'top_p')):
        raise RunError('unsupported_llm_config', '현재 Gemini Interactions 경로는 temperature와 top_p를 지원하지 않습니다.', details={'provider': provider})
    if api == 'responses' and any(config.get(k) is not None for k in ('seed', 'stop_sequences')):
        raise RunError('unsupported_llm_config', 'OpenAI Responses 경로는 seed와 stop_sequences를 지원하지 않습니다.', details={'api': api})
    if api == 'chat/completions' and config.get('stop_sequences') is not None and len(config['stop_sequences']) > 4:
        raise RunError('invalid_llm_config', 'OpenAI Chat Completions의 stop_sequences는 최대 4개입니다.')
    if api == 'chat/completions' and config.get('seed') is not None and not -(2**63) <= config['seed'] < 2**63:
        raise RunError('invalid_llm_config', 'OpenAI Chat Completions의 seed는 signed int64 범위여야 합니다.')
    schema = config.get('response_schema')
    if schema and schema.get('strict') is True:
        from .function_calling import validate_strict_schema
        validate_strict_schema(schema['schema'])
    elif schema:
        from .tools import compile_schema
        try:
            compile_schema(schema['schema'])
        except Exception as exc:
            raise RunError('invalid_llm_config', '출력 JSON 스키마가 유효하지 않습니다.') from exc
    if schema and provider == 'gemini' and 'strict' in schema:
        raise RunError('unsupported_llm_config', 'Gemini JSON Schema에는 OpenAI strict 설정을 적용할 수 없습니다.')
    if schema and provider == 'gemini' and config['max_input_tokens'] is not None:
        raise RunError('unsupported_token_count', 'Gemini의 구조화 출력 스키마를 포함한 입력 토큰 사전 계수는 현재 경로에서 지원하지 않습니다. max_input_tokens=null을 명시하세요.')
    return {**config, 'provider': provider, 'api': api}


class OpenAIAdapter:
    key_name = 'OPENAI_API_KEY'
    def headers(self, key):
        return {'Content-Type':'application/json', 'Authorization':'Bearer '+key}

class GeminiAdapter:
    key_name = 'GEMINI_API_KEY'
    def headers(self, key):
        return {'Content-Type':'application/json', 'x-goog-api-key':key}

ADAPTERS = {'openai':OpenAIAdapter(), 'gemini':GeminiAdapter()}

def adapter(provider):
    if provider not in ADAPTERS:
        raise RunError('unsupported_provider', '지원하지 않는 모델 제공자입니다.')
    return ADAPTERS[provider]


def encoded(meta, data, prefix=''):
    return binary_reference(meta,prefix) if isinstance(data,StoredFile) else prefix+base64.b64encode(data).decode()


def llm_request(prompt, files, config):
    config = validate_llm_config(config)
    provider, api, model = config['provider'], config['api'], config['model']
    if not isinstance(prompt, str) or not (prompt.strip() or files or config.get('history')):
        raise RunError('invalid_tool_arguments', '프롬프트 또는 파일이 필요합니다.')
    parts = []
    for meta, data, remote in files:
        kind, mime = meta['kind'], meta['mime']
        support = model_input_support(meta['format'])
        if provider == 'openai' and kind == 'video' and support['gemini/interactions']:
            raise RunError('unsupported_provider_modality',
                           '현재 OpenAI 경로는 원본 동영상을 지원하지 않습니다. 동영상 분석에는 config.provider를 gemini로 명시해 주세요.',
                           details={'mime': mime, 'provider': provider, 'suggested_provider': 'gemini'})
        if not support[f'{provider}/{api}']:
            raise RunError('unsupported_provider_modality', '현재 어댑터가 지원하지 않는 파일 형식입니다. 변환·다른 API 선택은 호출자가 결정합니다.',
                           details={'mime':mime, 'extension':meta['extension'], 'provider':provider, 'api':api, 'modelInputs':support})
        if provider == 'gemini':
            part = {'type':kind,'mime_type':mime}
            part.update({'uri':remote} if remote else {'data':encoded(meta,data)})
            # Static video input has no provider-side agent actions or tool loop.
            if kind=='video': part['processing']='static'
        elif api == 'chat/completions':
            if kind == 'audio' and meta['extension'] in {'mp3','wav'}:
                part={'type':'input_audio','input_audio':{'data':encoded(meta,data),'format':meta['extension']}}
            elif kind=='image':
                part={'type':'image_url','image_url':{'url':encoded(meta,data,f'data:{mime};base64,')}}
            else:
                raise RunError('unsupported_provider_modality', '지정한 Chat Completions 입력 규격이 지원하지 않는 파일입니다.', details={'mime':mime})
        elif kind=='image':
            part={'type':'input_image','image_url':encoded(meta,data,f'data:{mime};base64,')}
        elif kind=='document' and support['openai/responses']:
            part={'type':'input_file','filename':meta['filename'],'file_data':encoded(meta,data,f'data:{mime};base64,')}
        else:
            raise RunError('unsupported_provider_modality', '지정한 Responses 입력 규격이 지원하지 않는 파일입니다. 변환·다른 API 선택은 호출자가 결정합니다.', details={'mime':mime})
        parts.append(part)
    text_type = 'input_text' if api=='responses' else 'text'
    content = ([{'type':text_type,'text':prompt}] if prompt else []) + parts
    thinking, summary = config['thinking'], config.get('thinking_summary')
    if provider=='gemini':
        generation = {'max_output_tokens': config['max_output_tokens']}
        if thinking is not None: generation['thinking_level'] = thinking
        if summary is not None: generation['thinking_summaries'] = summary
        for name in ('seed', 'stop_sequences'):
            if config.get(name) is not None: generation[name] = config[name]
        body={'model':model,'input':content,'generation_config':generation,'store':False}
        if config.get('system_instruction') is not None: body['system_instruction'] = config['system_instruction']
        if config.get('response_schema') is not None:
            body['response_format'] = {'type':'text','mime_type':'application/json','schema':config['response_schema']['schema']}
        url='https://generativelanguage.googleapis.com/v1beta/interactions'
    elif api=='chat/completions':
        messages = ([{'role':'system','content':config['system_instruction']}] if config.get('system_instruction') is not None else []) + [{'role':'user','content':content}]
        body={'model':model,'modalities':['text'],'messages':messages,
              'max_completion_tokens':config['max_output_tokens'],'store':False,'stream':False}
        if config.get('seed') is not None: body['seed'] = config['seed']
        if config.get('stop_sequences') is not None: body['stop'] = config['stop_sequences']
        if config.get('response_schema') is not None:
            body['response_format'] = {'type':'json_schema','json_schema':config['response_schema']}
        url='https://api.openai.com/v1/chat/completions'
    else:
        body={'model':model,'input':[{'role':'user','content':content}],
              'max_output_tokens':config['max_output_tokens'],'store':False,'stream':False,'truncation':'disabled'}
        if not content: body['input'] = []
        body['input'].extend(copy.deepcopy(config.get('history', [])))
        for key in ('tools', 'tool_choice', 'parallel_tool_calls'):
            if key in config: body[key] = copy.deepcopy(config[key])
        if config.get('tools') or config.get('history'):
            body['include'] = ['reasoning.encrypted_content']
        reasoning = {}
        if thinking is not None: reasoning['effort'] = thinking
        if summary not in (None, 'none'): reasoning['summary'] = summary
        if reasoning: body['reasoning'] = reasoning
        if config.get('system_instruction') is not None: body['instructions'] = config['system_instruction']
        if config.get('response_schema') is not None:
            body['text'] = {'format': {'type':'json_schema', **config['response_schema']}}
        url='https://api.openai.com/v1/responses'
    if provider == 'openai':
        for name in ('temperature', 'top_p'):
            if config.get(name) is not None: body[name] = config[name]
    return {'provider':provider,'api':api,'role':'llm','method':'POST','url':url,'body':body}


def token_request(request):
    """Official counting endpoint, separate from the single generation request."""
    if request['provider']=='openai':
        body = {k: request['body'][k] for k in ('model', 'input', 'instructions', 'text', 'tools', 'tool_choice', 'parallel_tool_calls', 'reasoning') if k in request['body']}
        return {**request,'role':'token_count','url':'https://api.openai.com/v1/responses/input_tokens', 'body':body}
    from urllib.parse import quote
    parts=[]
    for part in request['body']['input']:
        if part['type']=='text': parts.append({'text':part['text']})
        elif 'uri' in part: parts.append({'fileData':{'mimeType':part['mime_type'],'fileUri':part['uri']}})
        else: parts.append({'inlineData':{'mimeType':part['mime_type'],'data':part['data']}})
    contents = [{'role':'user','parts':parts}]
    body = {'contents':contents}
    if 'system_instruction' in request['body']:
        body = {'generateContentRequest': {'model':'models/'+request['body']['model'], 'contents':contents,
                 'systemInstruction':{'parts':[{'text':request['body']['system_instruction']}]}}}
    return {**request,'role':'token_count','url':'https://generativelanguage.googleapis.com/v1beta/models/'+quote(request['body']['model'],safe='')+':countTokens','body':body}


def response_text(payload, provider, api, *, allow_tool_calls=False):
    if not isinstance(payload,dict): raise RunError('invalid_model_response','모델 응답은 JSON 객체여야 합니다.')
    if provider=='openai' and api=='chat/completions':
        choices=payload.get('choices',[])
        if not choices or not isinstance(choices[0], dict) or not isinstance(choices[0].get('finish_reason'), str):
            raise RunError('invalid_model_response','모델 완료 상태를 읽을 수 없습니다.',details={'response':payload})
        if choices[0]['finish_reason']!='stop':
            raise RunError('model_incomplete','모델 응답이 정상 종료되지 않았습니다.',details={'response':payload})
        text = choices[0]['message']['content']
        if not isinstance(text, str) or not text.strip():
            raise RunError('invalid_model_response', '응답 텍스트가 없습니다.', details={'response':payload})
        return text
    if payload.get('status') not in {'completed', 'incomplete', 'failed', 'cancelled', 'queued', 'in_progress', 'requires_action'}:
        raise RunError('invalid_model_response','모델 완료 상태를 읽을 수 없습니다.',details={'response':payload})
    if payload['status']!='completed':
        raise RunError('model_incomplete','모델 응답이 정상 완료되지 않았습니다.',details={'response':payload})
    if provider == 'openai' and any(part.get('type') == 'refusal' for item in payload.get('output', [])
                                  if isinstance(item, dict) and item.get('type') == 'message'
                                  for part in item.get('content', []) if isinstance(part, dict)):
        raise RunError('model_refused', '모델이 요청 처리를 거부했습니다.', details={'response': payload})
    if provider=='gemini':
        text=payload.get('output_text') or '\n'.join(p['text'] for step in payload.get('steps',[]) if step.get('type')=='model_output' for p in step.get('content',[]) if p.get('type')=='text')
    else:
        text='\n'.join(p['text'] for item in payload.get('output',[]) if item.get('type')=='message' for p in item.get('content',[]) if p.get('type')=='output_text')
    if not isinstance(text,str) or not text.strip() and not (allow_tool_calls and any(item.get('type') == 'function_call' for item in payload.get('output', []))):
        raise RunError('invalid_model_response','응답 텍스트가 없습니다.',details={'response':payload})
    return text


def llm_response(payload, provider, api, config):
    """A function-only response is complete; no functions are executed here."""
    from .function_calling import function_calls
    text = response_text(payload, provider, api, allow_tool_calls=(provider, api) == ('openai', 'responses'))
    calls = function_calls(payload, config) if (provider, api) == ('openai', 'responses') else []
    if text and config.get('response_schema') is not None:
        from .tools import validate_schema
        def reject(value): raise ValueError(value)
        try:
            parsed = json.loads(text, parse_constant=reject)
            validate_schema(parsed, config['response_schema']['schema'], 'invalid_model_response')
        except (ValueError, TypeError) as exc:
            raise RunError('invalid_model_response', '구조화 응답이 지정한 JSON 스키마와 일치하지 않습니다.') from exc
    return text, calls

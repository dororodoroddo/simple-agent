"""Explicit local test responses. These never represent actual file analysis."""
from .errors import RunError
import json


class FixtureTransport:
    def __init__(self, scenario='normal', replay=None, *, outputs=None, env=None):
        self.scenario, self.replay = scenario, replay
        if outputs is not None and (not isinstance(outputs, list) or any(not isinstance(item, str) for item in outputs)):
            raise RunError('invalid_fixture_outputs', '가상 모델 출력은 문자열 목록이어야 합니다.')
        self.outputs = tuple(outputs) if outputs is not None else None
        self.output_index = 0
        if env is not None:
            self.env = env

    async def model(self, request, run):
        if self.replay:
            raise RunError('replay_external_call', '재생은 저장된 실행 결과만 사용해야 합니다.')
        if request.get('role') == 'embedding':
            raise RunError('fixture_embedding_unavailable', '가상 임베딩은 생성하지 않습니다. 검색 시험에는 직접 준비한 벡터를 사용하세요.')
        if self.scenario=='api_error':
            raise RunError('model_http_error','가상 API HTTP 503',details={'synthetic':True,'httpStatus':503})
        text='가상 LLM 응답입니다. 실제 파일을 분석하지 않았습니다.'
        body = request['body']
        schema = (body.get('text', {}).get('format', {}).get('schema') if request.get('api')=='responses'
                  else body.get('response_format', {}).get('json_schema', {}).get('schema') if request.get('api')=='chat/completions'
                  else body.get('response_format', {}).get('schema'))
        if self.outputs is not None:
            if self.output_index >= len(self.outputs):
                raise RunError('fixture_response_missing', '이번 모델 호출에 사용할 가상 출력이 없습니다.')
            text = self.outputs[self.output_index]
            self.output_index += 1
        elif schema is not None:
            raise RunError('fixture_response_required', '구조화 출력 시험에는 fixtureOutputs로 가상 JSON 응답을 지정해야 합니다.')
        if body.get('tools'):
            if self.outputs is None:
                raise RunError('fixture_response_required', '함수 호출 시험에는 fixtureOutputs로 가상 Responses 응답을 지정해야 합니다.')
            from .function_calling import json_object
            from .providers import llm_response
            try:
                native = json_object(text)
                if not isinstance(native.get('output'), list):
                    raise ValueError('output must be a list')
                native = {**native, 'status': native.get('status', 'completed'), 'synthetic': True,
                          'model': body['model'], 'usage': {'input_tokens': 1, 'output_tokens': 1}}
                options = {key: body[key] for key in ('tools', 'tool_choice', 'parallel_tool_calls') if key in body}
                if schema is not None: options['response_schema'] = {'schema': schema}
                options['history'] = body['input'][1:] if body['input'] and body['input'][0].get('role') == 'user' else body['input']
                if native['status'] == 'completed':
                    try: llm_response(native, 'openai', 'responses', options)
                    except RunError as exc:
                        if exc.code != 'model_refused': raise
                return native
            except (ValueError, TypeError, KeyError, RunError) as exc:
                raise RunError('invalid_fixture_response', '가상 Responses 출력이 도구 호출 계약과 일치하지 않습니다.') from exc
        if schema is not None:
            from .tools import validate_schema
            try:
                value = json.loads(text, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
                validate_schema(value, schema, 'invalid_fixture_response')
            except (ValueError, TypeError) as exc:
                raise RunError('invalid_fixture_response', '가상 출력이 JSON 또는 지정한 스키마와 일치하지 않습니다.') from exc
        base={'model':request['body']['model'],'synthetic':True}
        if request['provider']=='gemini':
            return {**base,'status':'completed','steps':[{'type':'model_output','content':[{'type':'text','text':text}]}],
                    'usage':{'total_input_tokens':1,'total_output_tokens':1,'total_thought_tokens':0,'total_cached_tokens':0}}
        if request.get('api')=='chat/completions':
            return {**base,'choices':[{'finish_reason':'stop','message':{'content':text}}],
                    'usage':{'prompt_tokens':1,'completion_tokens':1,'prompt_tokens_details':{'audio_tokens':1},'completion_tokens_details':{'audio_tokens':0}}}
        return {**base,'status':'completed','output':[{'type':'message','role':'assistant','content':[{'type':'output_text','text':text}]}],
                'usage':{'input_tokens':1,'output_tokens':1}}

    async def download_to_store(self,*args):
        raise RunError('fixture_network_blocked','가상 시험에서는 외부 파일을 다운로드하지 않습니다.')

    async def upload_file(self,data,artifact,run):
        return 'fixture://media/'+artifact['id'],{'source':'synthetic','state':'ACTIVE'}

    async def count_tokens(self, request, run):
        if self.replay: raise RunError('replay_external_call','재생 중 계수 API 호출 금지')
        return {'totalTokens':1,'input_tokens':1,'synthetic':True}

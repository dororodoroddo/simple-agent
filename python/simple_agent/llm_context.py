"""Stage-local Responses conversations; no model calls or persistence of their own."""
import asyncio
import copy
from .errors import RunError
from .files import BinaryReference
from .function_calling import validate_history
from .protocol import json_text


def restore_items(value):
    if isinstance(value, list):
        return [restore_items(item) for item in value]
    if isinstance(value, dict):
        restored = {key: restore_items(item) for key, item in value.items()}
        return BinaryReference(restored) if set(value) == {'$binary', 'sha256', 'bytes', 'encoding', 'prefix'} else restored
    return value


class LLMContext:
    def __init__(self, snapshot=None):
        value = copy.deepcopy(snapshot or {})
        self.system = value.get('system')
        self.model = value.get('model')
        self.items = restore_items(value.get('items', []))
        self.routes = {}
        self.last_start = len(self.items)
        self.lock = asyncio.Lock()

    def snapshot(self):
        return copy.deepcopy({'system': self.system, 'model': self.model, 'items': self.items})

    def prepare(self, options):
        if (options['provider'], options['api']) != ('openai', 'responses'):
            raise RunError('unsupported_context_api', '단계 문맥 연결은 OpenAI Responses에서 지원합니다.')
        if options.get('history'):
            raise RunError('managed_llm_history', '단계 문맥은 자동 연결됩니다. 함수 결과는 tools.add_llm_tool_result(call_id, output)으로 추가하세요.')
        if self.model is not None and self.system != options.get('system_instruction'):
            raise RunError('llm_context_system_changed', '같은 단계 문맥의 시스템 프롬프트가 바뀌었습니다.')
        # Validate returned function calls/results while leaving file inputs to file adapters.
        validate_history([item for item in self.items if item.get('role') != 'user'])

    def complete(self, options, current, output):
        self.system = options.get('system_instruction')
        self.model = options['model']
        self.last_start = len(self.items)
        self.items.extend(copy.deepcopy([*current, *output]))

    def add_tool_result(self, call_id, output):
        calls = {item['call_id'] for item in self.items if item.get('type') == 'function_call'}
        completed = {item['call_id'] for item in self.items if item.get('type') == 'function_call_output'}
        if call_id not in calls or call_id in completed:
            raise RunError('invalid_llm_history', '결과를 추가할 미완료 함수 호출이 없습니다.', details={'call_id': call_id})
        self.items.append({'type': 'function_call_output', 'call_id': call_id,
                           'output': output if isinstance(output, str) else json_text(output)})

    def accept_route(self, node, result):
        call = self.routes.pop(node, None)
        if call is None:
            return
        if call['name'] != result['next_node']:
            raise RunError('invalid_node_result', '모델이 선택한 노드와 실제 전이 결과가 다릅니다.')
        self.add_tool_result(call['call_id'], {'status': 'handoff_accepted',
            'next_node': result['next_node'], 'payload': result['payload']})

    def start_new_task(self):
        """Keep only the current turn after the planner identifies an independent task."""
        self.items = self.items[self.last_start:]
        self.last_start = 0

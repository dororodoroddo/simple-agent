"""Typed, code-owned node connections. Nodes are registered tools, never generated code."""
import copy
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Any, NotRequired, TypedDict
from .errors import RunError, TurnInterrupted
from .tools import ToolSpec, object_schema
from .node_execution import NodePolicy, NodeDecision, execute_node


class NodeKind(StrEnum):
    PLANNING = 'planning'
    RETRIEVAL = 'retrieval'
    GENERATE = 'generate'
    RESPONSE = 'response'


class NodeResult(TypedDict):
    next_node: str | None
    payload: Any
    plan: NotRequired[Any]


class NodeState(TypedDict):
    original_input: dict
    plan: Any
    history: list[dict]
    available_data: tuple[str, ...]


@dataclass(frozen=True, kw_only=True)
class NodeSpec(ToolSpec):
    kind: NodeKind
    next_nodes: tuple[str, ...] = ()
    allowed_tools: tuple[str, ...] = ()
    interruptible: bool = False
    prompt_template: str = ''
    system_prompt: str | None = None
    data_refs: tuple[str, ...] = ()
    child_nodes: tuple[str, ...] = ()
    child_only: bool = False
    on_accepted: object = None
    policy: NodePolicy | None = None


TRANSITIONS = MappingProxyType({
    NodeKind.PLANNING: frozenset((NodeKind.PLANNING, NodeKind.RETRIEVAL, NodeKind.RESPONSE)),
    NodeKind.RETRIEVAL: frozenset((NodeKind.RETRIEVAL, NodeKind.GENERATE)),
    NodeKind.GENERATE: frozenset((NodeKind.GENERATE, NodeKind.RESPONSE)),
    NodeKind.RESPONSE: frozenset(),
})


class Workflow:
    def __init__(self, registry, *, entry_node, stage_tools, stage_prompts=None):
        self.nodes = MappingProxyType({name: spec for name, spec in registry.specs.items() if isinstance(spec, NodeSpec)})
        self.entry_node = entry_node
        self.stage_tools = MappingProxyType({NodeKind(kind): frozenset(names) for kind, names in stage_tools.items()})
        self.stage_prompts = MappingProxyType({NodeKind(kind): prompt for kind, prompt in (stage_prompts or {}).items()})
        if any(kind not in {NodeKind.PLANNING, NodeKind.GENERATE} or not isinstance(prompt, str)
               for kind, prompt in self.stage_prompts.items()):
            raise RunError('invalid_workflow', 'stage_prompts에는 planning·generate의 시스템 프롬프트 문자열을 지정하세요.')
        if entry_node not in self.nodes or self.nodes[entry_node].kind != NodeKind.PLANNING:
            raise RunError('invalid_workflow', '진입점은 등록된 플래닝 노드여야 합니다.')
        for node in self.nodes.values():
            if node.policy is not None:
                if not isinstance(node.policy, NodePolicy):
                    raise RunError("invalid_workflow", "policy는 NodePolicy여야 합니다.")
                node.policy.validate()
            if not isinstance(node.interruptible,bool):
                raise RunError('invalid_workflow','interruptible은 bool이어야 합니다.',details={'node':node.name})
            if not isinstance(node.prompt_template, str) or node.system_prompt is not None and not isinstance(node.system_prompt, str):
                raise RunError('invalid_workflow', '노드 프롬프트는 문자열이어야 합니다.', details={'node': node.name})
            if not isinstance(node.kind, NodeKind) or not node.description.strip() or node.kind not in self.stage_tools:
                raise RunError('invalid_workflow', '노드 유형·설명·단계별 도구 선언이 필요합니다.')
            if len(set(node.next_nodes)) != len(node.next_nodes):
                raise RunError('invalid_workflow', '다음 노드가 중복되었습니다.')
            if node.child_only:
                if node.kind not in {NodeKind.RETRIEVAL, NodeKind.GENERATE} or node.next_nodes:
                    raise RunError('invalid_workflow', '하위 작업은 retrieval/generate 유형과 반환값만 선언합니다.')
            elif ((node.kind == NodeKind.RESPONSE and node.next_nodes) or
                  (node.kind != NodeKind.RESPONSE and not node.next_nodes)):
                raise RunError('invalid_workflow', '최종 응답만 후속 노드 없이 종료할 수 있습니다.')
            for target in node.next_nodes:
                if target not in self.nodes or self.nodes[target].child_only or self.nodes[target].kind not in TRANSITIONS[node.kind]:
                    raise RunError('invalid_workflow', '미등록 노드 또는 금지된 단계 연결입니다.', details={'node': node.name, 'target': target})
            for target in node.child_nodes:
                if target not in self.nodes or not self.nodes[target].child_only:
                    raise RunError('invalid_workflow', '하위 작업 목록에는 등록된 child_only 노드를 지정하세요.')
            for name in (*node.allowed_tools, *self.stage_tools[node.kind]):
                if isinstance(registry.get(name), NodeSpec):
                    raise RunError('invalid_workflow', '노드 연결은 next_nodes에 선언하세요.')

    def allowed_tools(self, node):
        return frozenset(node.allowed_tools) & self.stage_tools[node.kind]

    async def run(self, input_data, tools, *, payload):
        if tools._workflow.get() is not None:
            raise RunError('node_execution_forbidden', '실행 중인 노드에서 새 흐름을 시작할 수 없습니다.')
        state: NodeState = {'original_input': copy.deepcopy(input_data), 'plan': None, 'history': [], 'available_data': ()}
        # The registration snapshot cannot be replaced while the flow is running.
        for name, node in self.nodes.items():
            if tools.ctx.registry.get(name) is not node:
                raise RunError('invalid_workflow', '등록 이후 노드 정의가 변경되었습니다.')
        workflow_token = tools._workflow.set((self, state))
        context_token = tools._llm_contexts.set({})
        current, previous = self.entry_node, None
        try:
            while True:
                node = self.nodes[current]
                state['available_data'] = node.data_refs
                transition = {'from': previous, 'to': current, 'kind': node.kind.value,
                              'next_nodes': list(node.next_nodes), 'allowed_tools': sorted(self.allowed_tools(node))}
                await tools.ctx.effect('node_transition', transition, lambda: None)
                await tools.ctx.stage('node_transition', transition, checkpoint=False)
                result, accepted_input = await execute_node(node, payload, tools)
                if not isinstance(result, dict) or not {'next_node', 'payload'}.issubset(result) or set(result) - {'next_node', 'payload', 'plan'}:
                    raise RunError('invalid_node_result', '노드 결과에는 next_node와 payload가 필요하며 plan을 추가할 수 있습니다.')
                target = result['next_node']
                if node.kind == NodeKind.RESPONSE:
                    if target is not None or not isinstance(result['payload'], str) or not result['payload'].strip():
                        raise RunError('invalid_node_result', '최종 응답 노드는 비어 있지 않은 문자열로 종료해야 합니다.')
                elif not isinstance(target, str) or target not in node.next_nodes:
                    raise RunError('node_transition_denied', '현재 노드에 허용되지 않은 다음 노드입니다.', details={'node': current, 'target': target})
                state['history'].append({'node': current, 'kind': node.kind.value,
                                         'input': copy.deepcopy(accepted_input), 'result': copy.deepcopy(result)})
                if 'plan' in result: state['plan'] = copy.deepcopy(result['plan'])
                if node.interruptible and tools.ctx.config.interrupts_enabled and (
                    tools.ctx.run.get('conversationId') or tools.ctx.replay and tools.ctx.replay.get('conversationId')):
                    changing = any(op.get('changing') for op in tools.ctx.run['operations'])
                    decision = await tools.ctx.effect('turn_boundary', {'node': current, 'changingStarted': changing},
                        lambda: tools.ctx.store.turn_boundary(tools.ctx.run['id'], changing))
                    await tools.ctx.stage('turn_boundary', decision, checkpoint=False)
                    if decision['interrupt']:
                        tools.ctx.run['interruption'] = {'node':current, **decision}
                        raise TurnInterrupted()
                context = tools._llm_contexts.get().get(node.kind)
                if context is not None:
                    context.accept_route(node.name, result)
                if node.on_accepted:
                    # Persist operational intent only after the boundary accepted this result.
                    await node.on_accepted(result, tools.ctx)
                if node.kind == NodeKind.RESPONSE: return result['payload']
                previous, current, payload = current, target, result['payload']
        finally:
            tools._llm_contexts.reset(context_token)
            tools._workflow.reset(workflow_token)

    async def run_child(self, name, payload, tools):
        parent = tools._node.get()
        scope = tools._workflow.get()
        if scope is None or parent is None or name not in parent.child_nodes:
            raise RunError('child_node_not_allowed', '현재 노드가 선언한 하위 작업만 호출할 수 있습니다.')
        node = self.nodes[name]
        state = {'original_input': copy.deepcopy(scope[1]['original_input']),
                 'plan': copy.deepcopy(scope[1]['plan']), 'history': [], 'available_data': node.data_refs}
        wt = tools._workflow.set((self, state))
        # Child calls have their own conversation; parents pass needed data explicitly.
        ct = tools._llm_contexts.set({})
        try:
            result, _ = await execute_node(node, payload, tools)
            return result
        finally:
            tools._workflow.reset(wt)
            tools._llm_contexts.reset(ct)


def result_schema(payload_schema):
    return {'type': 'object', 'properties': {'next_node': {'type': ['string', 'null']},
            'payload': payload_schema, 'plan': {}}, 'required': ['next_node', 'payload'], 'additionalProperties': False}

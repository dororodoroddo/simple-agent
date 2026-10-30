"""Code-owned completion checks and bounded retries for declared nodes."""
import asyncio
import copy
import json
import math
from dataclasses import dataclass
from .errors import RunError, TurnInterrupted
from .llm_context import LLMContext


@dataclass(frozen=True, kw_only=True)
class NodeDecision:
    action: str  # accept, stop, retry
    reason: str
    payload: object = None  # New node input, required for a result retry.


@dataclass(frozen=True, kw_only=True)
class NodePolicy:
    max_attempts: int  # Includes the first attempt.
    retry_delay_seconds: float
    evaluate: object = None  # Pure synchronous function (payload, result) -> NodeDecision.

    def validate(self):
        if (type(self.max_attempts) is not int or self.max_attempts < 1 or
            type(self.retry_delay_seconds) not in (int, float) or
            not math.isfinite(self.retry_delay_seconds) or self.retry_delay_seconds < 0 or
            self.evaluate is not None and not callable(self.evaluate)):
            raise RunError('invalid_workflow', '노드 정책에는 양의 시도 횟수·유한한 대기 시간·완료 판정 함수를 지정하세요.')


def transient_api_error(error):
    if not isinstance(error, RunError):
        return False
    if error.code == 'network_error':
        return True  # Uncertain model/write operations have already become indeterminate.
    if error.code not in {'model_http_error', 'api_http_error'}:
        return False
    response = error.details.get('response', {})
    detail = response.get('error', {}) if isinstance(response, dict) else {}
    if isinstance(detail, dict) and (detail.get('type') == 'insufficient_quota' or detail.get('code') in {
        'insufficient_quota', 'billing_hard_limit_reached', 'credit_balance_exhausted',
        'organization_spend_limit_exceeded', 'project_spend_limit_exceeded', 'organization_usage_limit_exceeded'}):
        return False
    return error.details.get('httpStatus') in {408, 429, 500, 502, 503, 504}


def fingerprint(payload):
    return json.dumps(payload, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(',', ':'))


def restore_contexts(contexts, snapshots):
    contexts.clear()
    for kind, (snapshot, routes, last_start) in snapshots.items():
        context = LLMContext(snapshot)
        context.routes, context.last_start = copy.deepcopy(routes), last_start
        contexts[kind] = context


async def execute_node(node, payload, tools):
    """Retry only this node; accepted transitions and on_accepted run outside here."""
    policy = node.policy
    attempts = policy.max_attempts if policy else 1
    contexts = tools._llm_contexts.get()
    snapshots = {kind: (ctx.snapshot(), copy.deepcopy(ctx.routes), ctx.last_start)
                 for kind, ctx in contexts.items()}
    current = copy.deepcopy(payload)
    seen = {fingerprint(current)}
    for attempt in range(1, attempts + 1):
        nt = tools._node.set(node)
        pt = tools._node_permit.set(node.name)
        parent_lease = tools._node_lease.get()
        lease = {'active': True, 'tasks': set(), 'changing': False, 'indeterminate': False, 'parent': parent_lease}
        lt = tools._node_lease.set(lease)
        error, result = None, None
        try:
            try:
                result = await tools.invoke(node.name, copy.deepcopy(current))
            except TurnInterrupted:
                raise
            except Exception as exc:
                error = exc
        finally:
            lease['active'] = False
            pending = [t for t in lease['tasks'] if t is not asyncio.current_task() and not t.done()]
            for task in pending: task.cancel()
            if pending: await asyncio.gather(*pending, return_exceptions=True)
            tools._node_lease.reset(lt)
            tools._node_permit.reset(pt)
            tools._node.reset(nt)
        if pending:
            raise RunError('unawaited_node_tools', '노드가 기다리지 않은 도구를 종료했습니다.')
        if policy is None:
            if error is not None: raise error
            return result, current
        transient = transient_api_error(error)
        if error is not None:
            decision = NodeDecision(action='retry' if transient else 'stop',
                                    reason=tools.ctx.error(error)['message'], payload=current)
        elif policy.evaluate is not None:
            decision = policy.evaluate(copy.deepcopy(current), copy.deepcopy(result))
        else:
            decision = NodeDecision(action='accept', reason='노드 반환 계약 충족')
        if (not isinstance(decision, NodeDecision) or decision.action not in {'accept', 'stop', 'retry'} or
            not isinstance(decision.reason, str) or not decision.reason.strip()):
            raise RunError('invalid_node_policy', '판정은 accept·stop·retry와 사유를 반환해야 합니다.')
        action, reason = decision.action, decision.reason
        stop_code = 'node_stopped'
        delay = 0
        if action == 'retry':
            next_key = fingerprint(decision.payload)
            if not node.read_only or lease['changing'] or lease['indeterminate']:
                action, stop_code = 'stop', 'node_retry_unsafe'
                reason = '시작한 변경 작업이나 결과가 불명확한 작업을 포함한 노드를 반복하지 않습니다.'
            elif attempt == attempts:
                action, stop_code = 'stop', 'node_retry_exhausted'
                reason = '선언한 노드 시도 횟수를 소진했습니다.'
            elif not transient and next_key in seen:
                action, stop_code = 'stop', 'node_retry_no_progress'
                reason = '결과 재시도 입력이 이전 시도와 같습니다.'
            else:
                delay = policy.retry_delay_seconds
                header_delay = error.details.get('retryAfterSeconds') if isinstance(error, RunError) else None
                if type(header_delay) in (int, float) and math.isfinite(header_delay) and header_delay >= 0:
                    delay = max(delay, header_delay)
        event = {'node': node.name, 'attempt': attempt, 'maxAttempts': attempts,
                 'action': action, 'reason': reason, 'input': current,
                 'error': tools.ctx.error(error) if error is not None else None,
                 'retryInput': decision.payload if action == 'retry' else None, 'delaySeconds': delay}
        await tools.ctx.effect('node_decision', event, lambda: None, parent_id=tools.parent_id)
        if action == 'accept':
            return result, current
        if action == 'stop':
            if decision.action == 'stop' and error is not None: raise error
            raise RunError(stop_code, reason, details={'node': node.name, 'attempts': attempt,
                'decisionReason': decision.reason, 'cause': event['error']}) from error
        seen.add(next_key)
        await tools.ctx.effect('node_retry_wait', {'node': node.name, 'attempt': attempt, 'seconds': delay},
                               lambda: asyncio.sleep(delay), parent_id=tools.parent_id)
        restore_contexts(contexts, snapshots)
        current = copy.deepcopy(decision.payload)

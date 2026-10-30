# 최초 플래너와 코드로 선언하는 노드

`handler.py`의 `handle(input_data, tools)`는 `planner.py`의 `run_default`를 호출합니다. 최초 플래너는 이번 턴·보류 작업·파일 메타데이터를 받아 **등록된 후보 하나**와 그 입력을 생성합니다.

```text
initial_planner → direct_response / clarify_request / 등록한 업무 진입 노드
업무 진입 노드 → 개발자가 선언한 후속 노드와 하위 작업 → response
```

기본 등록부에는 `initial_planner`, `direct_response`, `clarify_request`만 있습니다. 업무 노드는 아직 등록되어 있지 않습니다. 즉시 응답과 질문은 최초 플래너가 만든 문자열을 그대로 반환하고 모델을 추가 호출하지 않습니다. 수행할 업무가 명확해도 노드가 없으면 미구현 상태를 안내합니다.

최초 플래너 설정은 호출 코드에 다음과 같이 명시되어 있습니다. 다른 `callLLM`의 기본값이나 환경 설정이 아닙니다.

```python
PLANNER_CONFIG = {
    'provider': 'openai', 'api': 'responses', 'model': 'gpt-6-luna',
    'max_input_tokens': 50000, 'max_output_tokens': 1000, 'thinking': 'high',
}
```

제공된 플래닝의 `max_input_tokens`는 **50000**이며, 제너레이션은 **None**을 시작 설정으로 사용합니다. 각 호출 config에서 명시적으로 조정할 수 있습니다. 50000은 시스템 지시·도구·이전 대화·이번 입력을 포함한 전체 입력 기준입니다. None은 자체 입력 한도와 사전 계수 검사를 생략하고 제공자 한도를 적용합니다. 아직 업무 제너레이션 노드는 없으며 아래 설정을 출발점으로 업무에 맞게 조정합니다. 공통 실행기가 호출자의 값을 덮어쓰지는 않습니다.

공식 입력 토큰 계수와 생성을 별도로 기록합니다. 생성은 한 번만 시도하며 모델의 실패·잘못된 선택을 자동 재호출하지 않습니다. 실제 실행에는 모델 API ON과 인증키가 필요합니다. `preview`는 이 흐름을 실행하지 않습니다.

## 단계별 프롬프트와 문맥 계약

OpenAI Responses를 사용하는 planning·generate 노드의 `callLLM`은 단계별 대화를 자동 누적합니다. 시스템 지시는 고정하고 이전 입력·전체 모델 출력(암호화된 reasoning·phase 포함)·함수 결과 뒤에 현재 입력을 추가합니다. planning과 generate의 문맥은 분리되며, retrieval 호출은 매번 독립 입력입니다. 워크플로 밖의 직접 `callLLM`은 기존처럼 호출자가 `history`를 관리합니다. Gemini·Chat Completions는 현재 단계 문맥 연결 대상이 아니며, 해당 API를 planning·generate에서 사용하면 명시적인 지원 범위 오류를 반환합니다.

같은 실행 안에서는 문맥을 메모리에 보관합니다. Clarification으로 대기를 확정할 때 `ctx.store`의 `pending_work.llm_context`에 시스템 지시와 메시지 이력을 함께 보관하고 다음 턴에 복원합니다. 기본 `MemoryStore`의 보류 작업은 프로세스 메모리에 있으므로 프로세스가 종료되면 사라집니다. 실제 API 요청·응답은 실행 기록에 남습니다.

| 단계 | 시스템 프롬프트 | 노드별 사용자 프롬프트 | 문맥 유지 범위 |
|---|---|---|---|
| planning | 플래닝 단계에서 공유하는 충분한 작업 규칙·후보 설명·판단 기준 | 현재 노드에서 결정할 사항, 새 사용자 발언·파일 메타데이터·추가 확인 정보 | 어떤 업무를 수행하며 어떤 리트리벌을 선택할지 확정할 때까지 같은 작업의 대화를 유지 |
| retrieval | 해당 노드의 검색·자료 확인 작업과 결과 규격을 개발자가 작성 | 확정된 목표, 검색 조건, 이번 호출에서 볼 문서·자료 | 매 LLM 호출마다 독립 입력. 이전 LLM 대화 이력을 넣지 않음 |
| generate | 제너레이션 단계에서 공유하는 답변 작성 규칙·근거 사용 기준 | 현재 노드의 작성·검토·종합 지시와 필요한 리트리벌 결과 | 같은 답변 작업의 제너레이션 호출 사이에서 대화를 유지 |

### 문맥을 이어가는 planning·generate

각 단계의 시스템 프롬프트를 고정하고, 노드마다 새 작업 지시를 **사용자 메시지로 뒤에 추가**합니다. 앞선 사용자 메시지, 모델 응답, 실제 도구 실행 결과는 순서대로 유지합니다. 노드가 바뀔 때마다 시스템 프롬프트를 해당 노드의 지시로 교체하지 않습니다. 시스템 프롬프트의 길이는 필요한 규칙으로 정하며 캐시를 위해 의미 없는 내용을 채우지 않습니다.

```text
planning 문맥
  고정 시스템 프롬프트
  → 사용자: 원본 요청 + 첫 플래닝 노드의 지시
  → 모델: 판단 / 추가 질문 / 도구 선택
  → 필요한 도구 결과 또는 사용자의 추가 답변 + 다음 노드 지시
  → 업무 목표와 리트리벌 선택 확정

retrieval 호출 A: A의 시스템 프롬프트 + 목표·이번 문서 → 근거 A
retrieval 호출 B: B의 시스템 프롬프트 + 목표·이번 문서 → 근거 B

generate 문맥 (planning 문맥과 별도)
  고정 시스템 프롬프트
  → 사용자: 답변 목표 + 근거 A/B + 첫 제너레이션 노드의 지시
  → 모델: 결과
  → 사용자: 다음 제너레이션 노드의 지시 + 추가 자료
  → 모델: 결과
```

플래닝 문맥은 같은 대화·사용자의 같은 업무를 범위로 합니다. Clarification 답변은 이 문맥에 추가합니다. 독립된 새 업무로 판단하면 이전 이력을 분리하며, 새 업무의 입력과 해당 턴에서 확정한 선택·질문으로 문맥을 시작합니다. 새 업무를 분류할 때 참조했던 이전 업무·reasoning은 실행 로그에 보존하고 새 업무 문맥에는 넣지 않습니다. 취소·완료 시 보류 내용을 지웁니다. 제너레이션은 확정된 목표·필요한 사용자 조건·리트리벌 결과로 별도 문맥을 시작하며 플래닝 대화 전체를 자동 복사하지 않습니다. 같은 답변 작업 안에서 이어가며 실행 종료 후 다른 업무로 자동 연결하지 않습니다. 인터럽트로 폐기된 플랜·응답은 확정된 문맥에 추가하지 않습니다.

여기서 문맥을 잇는 턴은 LLM 호출 사이의 메시지 턴입니다. Clarification에서는 다음 입력의 추가 답변까지 같은 작업에 연결합니다. 문맥 유지 자체가 추가 호출이나 반복문을 실행하지는 않습니다. 호출·전이·종료는 개발자가 작성한 업무 코드가 결정합니다.

### 독립 입력을 사용하는 retrieval

리트리벌에서 LLM을 사용하는 각 노드는 **시스템 프롬프트와 사용자 프롬프트를 모두 설계**합니다. 여러 LLM 호출이 있으면 각각 해당 목표와 자료를 갖춘 새 입력을 구성합니다. 이전 플래닝·리트리벌·제너레이션의 LLM 대화나 모델 응답 전체를 `history`로 물려주지 않습니다.

플래너가 확정한 목표·조건이나 앞선 검색 결과가 필요하면 필요한 값을 이번 사용자 프롬프트의 명시적 입력으로 전달합니다. 이는 이전 대화를 그대로 이어가는 것과 구분합니다. 찾아낸 내용·출처·근거와 미확인 사항을 결과로 반환하고, 제너레이션이 이를 사용합니다. 이 규칙은 수신 시 자동 다운로드·분석하거나 LLM 호출을 강제하는 지시가 아닙니다.

하위 호출은 직렬·병렬 모두 부모·다른 하위 호출과 독립된 문맥으로 시작합니다. 한 하위 generate 호출 안에서 여러 번 LLM을 사용하면 그 호출의 문맥을 이어갑니다. 부모는 필요한 데이터를 하위 입력으로 전달하고 반환된 결과를 자기 문맥에 명시적으로 반영합니다. 같은 단계 문맥에 동시 LLM 호출이 들어오면 요청과 이력 추가를 순서대로 처리하여 이력이 뒤섞이지 않게 합니다.

### 문맥 유지와 프롬프트 캐싱

문맥은 실제 요청 메시지로 보존하고, 캐싱은 반복되는 입력 앞부분의 계산을 재사용하는 최적화로 사용합니다. planning·generate에서 고정 시스템 프롬프트와 기존 메시지의 내용·순서를 유지하고 새 메시지만 뒤에 붙여 재사용 가능한 앞부분을 유지합니다. 캐시가 없어도 동일한 문맥으로 요청해야 합니다. retrieval은 대화 이력이 없어도 고정된 프롬프트 부분의 캐싱을 사용할 수 있습니다.

같은 앞부분이 있다는 것만으로 캐시 적중을 보장하지 않습니다. 모델·도구 명세·출력 규격 등 요청 설정과 캐시 경계도 영향을 줍니다. 명시적 캐시 경계·보존 시간 옵션은 아직 연결하지 않았으며 제공자의 기본 캐싱 동작을 사용합니다. 실제 적중은 응답의 `cached_tokens`로 확인해야 합니다. 캐시를 맞추기 위해 노드의 도구 권한을 넓히거나 실제로 필요한 지시를 생략하지 않습니다. [OpenAI 프롬프트 캐싱](https://developers.openai.com/api/docs/guides/prompt-caching), [문맥 관리](https://developers.openai.com/api/docs/guides/conversation-state)

### 선언과 호출

`Workflow(..., stage_prompts={...})` 또는 `run_default(..., stage_prompts={...})`에 planning·generate의 공통 시스템 지시를 지정합니다. 기본 플래너의 시스템 지시는 `PLANNER_PROMPT`입니다. `NodeSpec.prompt_template`은 현재 노드의 사용자 지시로, 명시적 `callLLM` 호출의 `prompt` 앞에 붙습니다. retrieval의 시스템 지시는 `NodeSpec.system_prompt`에 지정합니다. 선언만으로 모델을 호출하지는 않습니다.

```python
# 노드 등록: 리트리벌에는 시스템 지시와 사용자 지시를 각각 작성합니다.
NodeSpec(..., kind=NodeKind.RETRIEVAL,
         system_prompt=RETRIEVAL_SYSTEM,
         prompt_template=RETRIEVAL_NODE_INSTRUCTION)

# 같은 generate 단계의 노드들은 공통 시스템 지시와 누적 문맥을 사용합니다.
await run_default(input_data, tools, task_nodes=TASK_NODES, stage_tools=STAGE_TOOLS,
    stage_prompts={NodeKind.GENERATE: GENERATE_SYSTEM})

# generate 노드 안: 나머지 생성 설정은 업무 구현 시 개발자가 명시합니다.
result = await ctx.services.callLLM(current_node_data, files, config={
    "model": GENERATE_MODEL,
    "max_input_tokens": None,
    "max_output_tokens": GENERATE_OUTPUT_TOKENS,
    "thinking": GENERATE_THINKING,
})
```

위 대문자 값은 개발자가 정의하는 변수입니다. 선언된 시스템 지시를 호출 config로 덮어쓰거나 같은 단계에서 시스템 지시를 바꾸면 원인을 포함한 오류를 반환합니다. 선언하지 않았다면 명시한 `config.system_instruction`을 사용하며, 임의 시스템 프롬프트나 모델 설정을 채우지 않습니다. 모델·도구·출력 설정은 호출마다 명시한 값을 사용하고 제공자 호환성은 해당 API 규격을 따릅니다.

문맥을 관리하는 노드에서는 수동 `config.history`를 함께 넣지 않습니다. 실제 함수 도구 결과는 아래처럼 추가합니다. 함수 실행이나 재호출을 자동으로 수행하지 않습니다.

```python
first = await tools.callLLM(prompt, files, config=FIRST_CONFIG)
call = first['tool_calls'][0]
actual_result = await tools.invoke(call['name'], call['arguments'])
await tools.add_llm_tool_result(call['call_id'], actual_result)
second = await tools.callLLM(next_instruction, [], config=NEXT_CONFIG)
```

`select_next_node`와 최초 플래너의 함수 선택은 인터럽트·전이 검증을 통과한 뒤 `handoff_accepted` 결과를 문맥에 추가합니다. 이는 선택한 노드로 입력을 넘겼다는 기록이며 해당 업무 완료를 의미하지 않습니다. 실제 함수 호출 결과가 누락되면 다음 모델 호출 전에 오류를 반환합니다. 실패한 모델 호출은 성공 응답으로 문맥에 추가하지 않습니다.

## 노드 선언과 등록

`workflow.py`의 `NodeSpec`은 공통 `ToolSpec`을 상속합니다. `name`은 노드 ID, `kind`는 유형입니다. `tools.py`의 `default_registry()`가 공통 노드·도구를 등록합니다. 업무 노드는 실행 컨텍스트 생성 전에 같은 등록부에 추가하세요. 최초 플래너에 노출할 업무 진입 ID만 `run_default(..., task_nodes=(...))`로 지정합니다. 노드 설명·입력 스키마만 후보로 전달하며 하위 작업 목록이나 실행 조합을 모델이 만들지 않습니다.

| 속성 | 의미 |
|---|---|
| `name`, `description` | 고유 ID와 선택 후보의 용도 설명 |
| `kind` | `NodeKind.PLANNING`, `RETRIEVAL`, `GENERATE`, `RESPONSE` |
| `prompt_template` | 노드 사용자 지시. `callLLM`의 현재 prompt 앞에 추가하며 `ctx.services.node_prompt`로도 읽을 수 있음 |
| `system_prompt` | 노드 시스템 지시. retrieval에서 호출별로 선언. planning·generate는 `stage_prompts` 공통 지시를 우선 사용 |
| `input_schema`, `output_schema` | payload·반환값의 JSON Schema |
| `handler` | 일반 함수 또는 `async def handler(payload, ctx)` |
| `permission` | 등록부가 검사하는 실행 권한 |
| `next_nodes` | 상위 흐름에서 선택 가능한 후속 노드 ID |
| `allowed_tools` | 노드 안에서 허용할 일반 도구 ID |
| `data_refs` | `node_state.available_data`에 전달할 데이터 참조. 자동 조회하지 않음 |
| `child_nodes` | 이 노드가 호출할 수 있는 등록된 하위 작업 ID |
| `child_only` | 하위 작업 전용. retrieval/generate 유형, 후속 전이 없이 자기 결과를 반환 |
| `interruptible` | 기본 False. 전역 ON일 때 반환 후 추가 입력 확인 |
| `policy` | `NodePolicy`: 완료 판정·최대 시도 횟수·재시도 대기. 상위·하위 노드 공통 |
| `on_accepted` | 상위 노드의 반환 검증·인터럽트 확인을 통과한 뒤 실행할 처리 |
| `external`, `read_only` | 외부 I/O·변경 작업 기록 계약. 조합 노드는 `external=False` |

`prompt_template`·`system_prompt`·`data_refs`를 선언해도 모델 호출이나 자료 조회를 자동 수행하지 않습니다. 업무 함수가 `callLLM`을 명시적으로 호출할 때 프롬프트와 단계 문맥을 구성합니다. `prompt_template`을 prompt 인자에도 중복해서 넣지 마세요.

상위 단계의 전이와 단계·노드 도구 허용 목록·권한의 교집합은 기존 계약을 유지합니다.

| 현재 유형 | 가능한 다음 유형 |
|---|---|
| planning | planning, retrieval, response |
| retrieval | retrieval, generate |
| generate | generate, response |
| response | 없음 |

이 표와 현재 노드의 `next_nodes`를 모두 만족해야 합니다. 하위 작업은 상위 단계 전이와 별개로 `child_nodes`에 선언해야 하며, 실행이 끝나면 부모로 결과를 반환합니다. 상위 후속 노드에는 `child_only`를 넣을 수 없습니다.

## 실행 입력과 반환값

노드 입력은 선언한 스키마로 검증합니다. `ctx.services.node_state`는 다음 값의 복사본입니다.

```python
class NodeState(TypedDict):
    original_input: dict      # 이번 턴의 원래 입력·발언·파일 참조
    plan: Any                 # 플래너 결정과 이어지는 업무 코드의 계획
    history: list[dict]       # 이 흐름의 앞선 노드 입력·결과. 별도로 관리하는 LLM 대화 이력과 구분
    available_data: tuple[str, ...]  # 현재 노드에 선언된 data_refs
```

기본 플래너의 `plan`에는 선택한 노드·요청 관계·대화 ID·이번 요청의 누적 `messages`·`files`·`fileMetadata`가 있습니다. 보류 작업의 추가 답변이면 여기의 발언·파일 목록에 원래 요청을 합칩니다. `original_input` 자체를 이전 요청으로 덮어쓰지 않습니다. 각 메시지의 원문·ID·파일 참조를 유지합니다.

상위 노드는 다음 형태로 반환합니다. `plan`을 생략하면 기존 계획을 유지합니다. 마지막 response 노드는 비어 있지 않은 최종 문자열로 종료합니다.

```python
return {'next_node': '다음_ID', 'payload': 다음_입력, 'plan': 계획}
# 최종 response 노드
return {'next_node': None, 'payload': '최종 답장'}
```

`result_schema(payload_schema)`로 상위 반환 형태를 선언합니다. 하위 작업의 `output_schema`는 자기 결과 스키마이며 `NodeResult`로 감싸지 않습니다. 실행이 끝나지 않은 도구를 남기고 노드를 종료하면 해당 호출을 취소하고 후속 전이를 막습니다. 변경 결과가 불명확하면 `indeterminate`로 기록합니다.

## 노드 완료·종료·재시도

경로와 도구 호출은 개발자가 정합니다. 실행기는 같은 노드의 제한된 재시도만 관리하며, 모델이 도구나 노드를 새로 만들지 않습니다. `workflow`에서 `NodePolicy`, `NodeDecision`을 가져와 `NodeSpec.policy`에 연결합니다.

- `NodePolicy(max_attempts=..., retry_delay_seconds=..., evaluate=...)`: 최초 실행을 포함한 최대 횟수와 초 단위 대기를 반드시 명시합니다. `evaluate`는 선택 사항인 순수 동기 함수로 `(현재 payload, 반환 결과)`의 복사본을 받아 `NodeDecision`을 반환합니다. 여기서 I/O나 모델 호출을 하지 않습니다.
- `NodeDecision(action='accept', reason=...)`: 노드의 완료 기준을 충족했습니다. 원래 반환값을 검증하고 선언된 후속 경로로 진행합니다.
- `NodeDecision(action='stop', reason=...)`: 더 진행할 수 없습니다. `node_stopped`로 현재 실행을 종료합니다. 부분 실패를 다음 노드에서 처리할 수 있다면 실패 상태를 반환값에 담고 `accept`로 인계합니다.
- `NodeDecision(action='retry', reason=..., payload=...)`: 아직 완료되지 않았으며 보완한 입력으로 같은 노드를 실행합니다. payload에는 오류 설명만 덧붙이기보다 결과를 바꿀 수 있는 자료·조회 조건·판단 조건을 담습니다. 동일 입력이나 이전 시도 입력으로 되돌아가면 `node_retry_no_progress`, 횟수를 소진하면 `node_retry_exhausted`로 종료합니다. 입력 비교는 JSON 값 기준이며 의미상 진전 여부는 업무의 완료 판정이 책임집니다.

`evaluate`를 생략하면 정상 반환을 완료로 판단합니다. `policy`를 생략한 노드는 한 번만 실행하고 오류를 종료 조건으로 처리합니다. 업무 노드를 작성할 때 완료 기준과 복구 가능한 실패를 확인하여 정책을 연결하세요. 기본 `initial_planner`는 최초 포함 **3회**, 최소 **1초** 대기를 명시하며, 정상 반환은 완료입니다. 이 값은 다른 노드나 모델 호출에 적용되는 기본값이 아닙니다.

API 오류는 결과 판정 함수에 전달하지 않고 코드에서 분류합니다. `model_http_error`·`api_http_error`의 HTTP 408·429·500·502·503·504와 확정된 읽기 작업의 `network_error`는 재시도 후보입니다. quota·billing 오류, 그 밖의 API 거부·권한·스키마·코드 오류는 종료합니다. API 어댑터는 HTTP 상태를 `details.httpStatus`, 응답을 `details.response`, Retry-After를 초로 변환해 `details.retryAfterSeconds`에 보존해야 합니다. 제공자 어댑터는 JSON이 아닌 오류 응답도 HTTP 상태를 보존합니다. 재시도는 노드에 선언한 대기와 Retry-After 중 긴 시간을 기다립니다. [OpenAI 오류 계약](https://developers.openai.com/api/docs/guides/error-codes)

일시적인 API 오류만 같은 입력으로 재시도할 수 있습니다. 모델 생성의 연결 단절처럼 완료 여부를 알 수 없는 `indeterminate`는 자동 반복하지 않습니다. 변경 노드나 해당 시도에서 변경 작업을 시작한 노드도 반복하지 않습니다(`node_retry_unsafe`). 복합 노드는 실패한 읽기 작업을 별도 하위 노드로 분리해야 앞서 완료한 변경을 반복하지 않고 복구할 수 있습니다. 형제 하위 노드의 변경은 다른 읽기 하위 노드의 재시도를 막지 않지만, 부모 전체 재시도는 막습니다. 기존 전송 계층의 복구는 그 API 계약대로 동작하며 노드 시도 횟수는 HTTP 전송 횟수와 다릅니다.

모든 시도의 도구·모델·오류 기록과 `node_decision`, `node_retry_wait`를 보존합니다. 재생은 같은 판정·입력·순서를 검사하며 API 호출과 실제 대기는 하지 않습니다. 재시도 전 단계 문맥을 노드 진입 시점으로 복원해 거부한 결과·미완료 함수 호출이 다음 시도를 오염시키지 않게 합니다. 부족분은 새 payload로 명시적으로 전달합니다. 취소·인터럽트는 재시도하지 않으며 `on_accepted`는 최종 수락 후 한 번 실행하고 재시도 대상에 포함하지 않습니다.

## 직렬·병렬 하위 실행

모든 하위 노드는 등록부와 부모 `child_nodes`에 선언합니다. 하위 노드도 자신의 스키마·권한·단계별/노드별 도구 허용 목록으로 실행합니다. 부모의 도구 목록을 그대로 상속하지 않습니다.

```python
async def combine(payload, ctx):
    tools = ctx.services
    # 직렬 의존성: 앞선 결과로 다음 입력을 개발자가 구성합니다.
    first = await tools.run_child('lookup', {'key': payload['key']})
    second = await tools.run_child('summarize', {'data': first})

    # 독립 작업: 결과는 입력 순서대로 모으며 개별 오류를 포함합니다.
    results = await tools.run_children([
        {'node': 'check_a', 'payload': {'data': second}},
        {'node': 'check_b', 'payload': {'data': second}},
    ], mode='parallel')
    # results: [{node, status, output, error}, ...]
    # 결과 종합·실패 시 정책은 업무 코드에서 작성합니다.
    return {'next_node': 'task_response', 'payload': build_reply(results)}
```

`run_child`는 검증한 결과를 바로 반환하며 오류는 호출자에게 전달합니다. `run_children`은 `parallel` 또는 `serial`로 실행하고 개별 결과·오류를 목록으로 반환합니다. serial 그룹도 각 payload를 그대로 사용합니다. 앞선 결과에 의존하면 위처럼 순차 `await run_child`로 다음 인자를 구성하세요. 자동 재호출은 없습니다.

하위 문맥에는 원본 요청과 현재 계획의 복사본, 자기 데이터 참조, 별도 빈 history를 제공합니다. 부모 history를 변경하지 않습니다. 도구 기록에 노드 유형·프롬프트·데이터 참조·부모 호출 ID를 남기고 그룹 기록에는 직렬/병렬 방식과 각 입력을 남깁니다. 그룹이 아닌 직접 직렬 호출은 실행 순서·부모 ID로 연결됩니다.

아래처럼 노드를 등록한 뒤 `handler.py`에서 업무 진입 ID를 지정합니다. handler 안에 임의의 별도 실행기나 모델 반복문을 만들 필요는 없습니다.

```python
# tools.py의 default_registry()에서 return registry 이전에 등록
registry.register(NodeSpec(
    name='task_retrieve', kind=NodeKind.RETRIEVAL,
    description='문제 공개 후 확정할 업무 설명', prompt_template='확정한 작업 지시',
    input_schema=task_input_schema, output_schema=result_schema(next_input_schema),
    permission='workflow', handler=task_retrieve,
    next_nodes=('task_generate',), child_nodes=('lookup',),
    data_refs=('업무 데이터 참조',), allowed_tools=(),
))
# task_generate·task_response와 child_only=True인 lookup도 같은 등록부에 등록

# handler.py
async def handle(input_data, tools):
    return await run_default(input_data, tools, task_nodes=('task_retrieve',),
        stage_tools={NodeKind.RETRIEVAL: ('sql_query',),
                     NodeKind.GENERATE: ('callLLM',)})
```

노드의 도구 호출에는 **단계별 목록 ∩ 노드별 allowed_tools ∩ 실행 권한**을 적용합니다. 하위 lookup에서 SQL을 쓰려면 lookup 자신의 `allowed_tools=('sql_query',)`도 필요합니다. 파일을 LLM에 전달할 때는 `file` 허용과 `file_read` 권한도 필요합니다.

## Clarification과 파일 메타데이터

첫 플래너는 이번 턴과 보류 작업의 문맥을 받습니다. 저장된 LLM 문맥이 있으면 그 이력을 복원하고 새 발언·새 파일 메타데이터와 보류 작업 존재 표시만 뒤에 붙입니다. 이전 발언·파일 목록 전체를 새 메시지에 다시 복사하거나 이전 파일의 HEAD를 반복하지 않습니다. 선택한 함수의 입력에는 예약 필드 `_request_relation`을 붙입니다. 값은 `new`·`continue`·`cancel`이며 실제 업무 노드 payload에서는 제거합니다. 개발자는 이 필드를 자기 후보 입력에 선언하지 않습니다.

- `clarify_request`: 원래 발언·파일 참조·메타데이터·확인된 내용·질문과 플래닝 LLM 문맥을 공통 보류 작업 저장소의 `pending_work`에 저장한 뒤 질문을 반환합니다. 반복 질문은 갱신합니다.
- `continue`: 기존 질문에 대한 추가 답변으로 원래 요청과 현재 턴을 합칩니다. 업무 성공 시 보류 작업을 정리하며 실패 시 유지합니다.
- `new`: 독립된 새 요청이면 이전 보류 작업을 대체합니다. 새 질문이면 새 내용으로 저장하고 그 외에는 지웁니다.
- `cancel`: 즉시 응답으로 취소를 알리고 보류 작업을 지웁니다.

키는 기존 턴 수집의 대화·사용자 식별자를 사용합니다. 전체 과거 대화를 매번 모델에 넣지 않습니다. 보류 작업 변경은 최초 플래너의 인터럽트 검사가 끝난 뒤에만 확정합니다. 폐기된 플랜은 보류 내용을 바꾸지 않습니다. 단계별 진행 상태를 저장소에 쓰지 않습니다.

파일 메타데이터와 본문 접근은 등록된 도구와 권한 경계를 따릅니다. 메타데이터 누락이나 조회 실패는 미확인으로 구분하며, 파일 본문은 명시적인 도구 호출 없이 분석하지 않습니다. 보류 작업의 첨부 참조는 다음 턴에도 이어집니다.

## callLLM의 네이티브 함수 호출

OpenAI Responses config에 `tools`, `tool_choice`, `parallel_tool_calls`, `history`를 전달할 수 있습니다. 다른 제공자·API에 이 옵션을 전달하면 명시적으로 거부합니다. `tools`는 `type='function'`, `name`, `description`, `parameters`, `strict`를 갖는 목록입니다. `spec.model_schema(strict=True)` 또는 `strict=False`로 생성하세요.

`strict=True`는 모든 object의 `additionalProperties=False`와 전체 속성의 `required` 선언을 요구합니다. 자유형 JSON 같은 기존 도구 계약을 임의로 좁히지 않습니다. 부분집합의 명백한 비지원 구문과 object 규칙은 로컬 검사하며, 서비스·모델의 추가 제약은 제공자 오류로 확인합니다. `response_schema.strict=True`에도 같은 검사를 적용합니다.

`tool_choice`는 `auto`, `none`, `required`, 특정 함수 지정, `allowed_tools` 부분집합을 지원합니다. `parallel_tool_calls=False`이면 복수 함수 응답을 거부합니다. 반환값에는 `text`, 정규화한 `tool_calls`(`name`, `arguments`, `call_id`), 원문 `response`가 들어 있습니다. 함수 호출만 반환하면 `text`는 빈 문자열이며 정상 결과입니다. 호출 함수의 이름·인자·ID와 명시된 호출 제약도 로컬에서 검사합니다.

워크플로 밖에서 연속 도구 실행이 필요한 업무 함수는 명시적으로 다음과 같이 구성할 수 있습니다. planning·generate 노드 안에서는 앞 절의 자동 문맥과 `add_llm_tool_result`를 사용합니다.

```python
first = await tools.callLLM(prompt, files, config=config)
call = first['tool_calls'][0]
result = await tools.invoke(call['name'], call['arguments'])
history = [*first['response']['output'], {
    'type': 'function_call_output', 'call_id': call['call_id'],
    'output': json.dumps(result, ensure_ascii=False),
}]
second = await tools.callLLM(prompt, files, config={**config, 'history': history})
```

도구 실행에도 현재 노드의 도구 제한이 적용됩니다. `config`의 `tool_choice`는 다음 호출의 목적에 맞게 지정하세요. `history`는 원래 사용자 프롬프트·파일 뒤에 추가하는 모델 출력·reasoning·함수 호출·함수 결과와 추가 사용자 텍스트입니다. 여러 번 이어갈 때는 이전 history도 누적하세요. 존재하지 않거나 중복된 호출 ID, 결과가 없는 호출, 새 미디어를 우회 전달하는 이력은 거부합니다. 반환된 reasoning 필드를 보존하고, stateless 연속 호출을 위해 encrypted reasoning을 요청합니다.

공식 입력 토큰 계수에는 도구·호출 제약·이력을 포함합니다. `callLLM`은 호출당 생성 한 번만 수행하고 함수를 자동 실행하거나 재호출하지 않습니다. 구조화 출력과 함수 호출은 독립적으로 처리합니다. [OpenAI 함수 호출](https://developers.openai.com/api/docs/guides/function-calling)

## 연속 입력과 노드 인터럽트

`initial_planner`만 기본 `interruptible=True`입니다. 전역 OFF이거나 노드가 비허용이면 추가 입력을 조회하지 않습니다. 노드 반환 후 허용 경계에서 새 입력을 발견하면 후속 전이·답장을 중단하고 발언을 합쳐 새 실행에서 시작합니다. 이전 계획·payload를 재사용하지 않고 기존 사용량·기록은 유지합니다.

변경 작업을 이미 시작했다면 현재 흐름을 마치고 새 입력은 다음 턴으로 넘깁니다. 진행 중인 API를 강제로 취소하지 않습니다. 재생은 저장된 경계 판단을 사용합니다.

## 로컬 실행과 E2E fixture

실행 기록에는 입력, 모델 요청·응답, 노드 선택·보류 변경, 도구 입출력, 부모·자식 관계와 그룹 결과가 포함됩니다. `node_transition`, `toolCalls`, `child_group`, `planner_decision`, `pending_load`·`pending_save`는 같은 실행·호출 ID로 연결됩니다. 재생은 기록된 외부 결과와 보류 상태를 사용하며 모델을 호출하지 않습니다.

가상 시험은 모델 호출 순서대로 `fixtureOutputs` 문자열 목록을 지정합니다. 기본 플래너의 즉시 응답 예입니다.

```json
{
  "fixtureOutputs": [
    "{\"output\":[{\"type\":\"function_call\",\"name\":\"direct_response\",\"call_id\":\"fixture-plan\",\"arguments\":\"{\\\"message\\\":\\\"안녕하세요.\\\",\\\"_request_relation\\\":\\\"new\\\"}\"}]}"
  ]
}
```

추가 질문의 함수명은 `clarify_request`, arguments는 `question`, `known_details`, `_request_relation`입니다. 이어지는 답이면 `continue`를 사용합니다. 가상 응답을 자동으로 채우거나 실제 모델 품질로 평가하지 않습니다. 이 저장소의 격리된 E2E 검사는 `npm run test:e2e`로 실행합니다. fixture 응답을 사용하며 실제 모델 API나 외부 서비스를 호출하지 않습니다.

## 내부 구현·회귀 검사

| 변경 대상 | 소스에서 확인할 흐름 | 관련 검사 |
|---|---|---|
| 업무 진입·후보 선택 | [handler.py](../python/simple_agent/handler.py) → [planner.py](../python/simple_agent/planner.py)의 initial_plan·accept_plan·run_default | E2E 실행 경로는 `npm run test:e2e` |
| 노드·하위 작업·전이 | [workflow.py](../python/simple_agent/workflow.py)의 NodeSpec·Workflow → [node_execution.py](../python/simple_agent/node_execution.py)의 완료·재시도 → [tool_runtime.py](../python/simple_agent/tool_runtime.py)의 ToolRuntime | E2E 실행 경로는 `npm run test:e2e` |
| 단계별 모델 문맥 | [llm_context.py](../python/simple_agent/llm_context.py)의 LLMContext → [function_calling.py](../python/simple_agent/function_calling.py) → `planner.py`의 보류 작업 처리 | E2E 실행 경로는 `npm run test:e2e` |

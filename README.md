# Simple Agent

인프라나 웹 프레임워크 없이 Python 프로세스 안에서 실행되는 에이전트 코어입니다. 업무 로직은 `python/simple_agent/handler.py`의 `handle(input_data, tools)` 함수에 작성합니다. 모델·파일·데이터 도구는 등록된 기능과 권한 범위 안에서만 호출됩니다.

이 저장소에는 에이전트 코어와 격리된 종단 간(E2E) 검사만 포함됩니다. 테스트는 실제 모델 API나 외부 서비스에 연결하지 않고 가상 모델 응답과 프로세스 내 메모리 저장소를 사용합니다. 별도 웹 서버, 클라우드 저장소, 작업 큐, 배포 설정은 포함하지 않습니다.

## 실행 및 검증

```sh
npm run setup:python
npm run test:e2e
```

`npm test`도 동일한 E2E 검사를 실행합니다. E2E는 별도 Python 프로세스에서 실제 `handler.handle` 진입점과 에이전트 실행 흐름을 검증합니다. 실제 외부 모델을 호출하거나 서비스를 배포하는 명령은 없습니다.

실행 설정은 호출자가 `Config`와 `Transport`를 만들어 전달합니다. 기본 모델 호출은 꺼져 있으며, 실제 제공자 API를 사용할 때만 `SIMPLE_AGENT_ENABLE_LIVE_API=1`과 제공자 키를 프로세스에 명시적으로 설정합니다. E2E에서는 이 설정과 관계없이 가상 응답만 사용합니다.

## 에이전트 개발 문서

- [에이전트 구성과 실행 흐름](docs/agent.md)
- [노드 구조와 워크플로 작성법](docs/workflow.md)
- [업무 프롬프트 작성 안내](prompts/agent.md)

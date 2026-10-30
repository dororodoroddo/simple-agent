# 스트리밍 JSON 의존성

`ijson-3.6.0-py3-none-any.whl`은 공식 ijson 3.6.0 소스의 순수 Python 빌드입니다. 모든 로컬 실행환경에서 동일한 Python 구현을 사용하며 라이브러리 동작은 수정하지 않았습니다.

원본: https://files.pythonhosted.org/packages/75/61/4066af787ed25bfca02c3edd2d7fd489b1b5ca27b54b400b187e5f2865e7/ijson-3.6.0.tar.gz

빌드: `IJSON_BUILD_YAJL2C=0 uv build --wheel <공식 소스 디렉터리> --out-dir vendor`

라이선스는 wheel 내부 `ijson-3.6.0.dist-info/licenses/LICENSE.txt`에 포함되어 있습니다. `pyproject.toml`과 `uv.lock`에 이 wheel을 고정합니다. 프로젝트 기록 형식 버전과는 관계없습니다.

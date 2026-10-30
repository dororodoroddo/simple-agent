"""Launch a separate process and exercise the real agent handler without network I/O."""
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
response = json.dumps(
    {"output": [{"type": "function_call", "name": "direct_response", "call_id": "e2e-call",
                 "arguments": json.dumps({"_request_relation": "new", "message": "안녕하세요. 에이전트가 응답했습니다."}, ensure_ascii=False)}]},
    ensure_ascii=False,
)
environment = {**os.environ, "PYTHONPATH": str(ROOT / "python"), "SIMPLE_AGENT_TEST_OFFLINE": "1"}
process = subprocess.run(
    [sys.executable, "-m", "simple_agent"], cwd=ROOT, env=environment,
    input=json.dumps({"message": "안녕하세요", "fixtureOutputs": [response]}, ensure_ascii=False),
    text=True, capture_output=True, check=False,
)
if process.returncode:
    raise SystemExit(process.stderr or process.stdout or f"agent exited {process.returncode}")
result = json.loads(process.stdout)
assert result["status"] == "completed", result
assert result["output"] == "안녕하세요. 에이전트가 응답했습니다.", result
assert result["reply"] == result["output"], result
assert [call["name"] for call in result["toolCalls"]] == ["initial_planner", "callLLM", "direct_response"], result
assert all(call["authorized"] for call in result["toolCalls"]), result
print("E2E passed: separate process → handler → fixture model → authorized response")

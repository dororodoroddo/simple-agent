"""One shared execution shell: input → explicit handler/tool → result.

No file ingestion, model selection or agent loop runs on receipt.
"""
import asyncio
import copy
from dataclasses import fields, replace
from pathlib import PurePosixPath
from urllib.parse import unquote, urlsplit
from .config import Config
from .errors import RunError, TurnInterrupted
from .protocol import normalize_event, media_url, reply_text, json_text, redact, handler_event_raw, input_reply
from .runtime import RunContext, Dependencies
from .tools import default_registry
from .usage import aggregate_usage
from .tool_runtime import ToolRuntime
from . import handler
from .file_formats import VIDEO

VIDEO_WARNING = "동영상으로 보이는 파일입니다. 현재 OpenAI 경로는 원본 동영상을 지원하지 않으므로 Gemini 선택과 해당 파일 형식의 지원 여부를 확인해 주세요."


def static_checks(run):
    interrupted = run.get('status') == 'interrupted'
    return [
        {"id": "input_preserved", "pass": isinstance(run.get("input", {}).get("eventRaw"), str), "evidence": "input.eventRaw"},
        {"id": "execution_completed", "pass": None if interrupted else run.get("status") in {"completed", "preview"}, "evidence": "status"},
        {"id": "reply_present", "pass": None if interrupted else bool(run.get("reply")), "evidence": "status" if interrupted else "reply"},
        {"id": "tool_permissions", "pass": all(t.get("authorized") is True for t in run.get("toolCalls", [])), "evidence": "toolCalls.authorized"},
        {"id": "business_rubric", "pass": None, "status": "not_assessed", "reason": "업무 평가 기준 미제공"},
    ]


def input_view(raw, *, fixture=False):
    if raw.get('messages'):
        messages = [input_view({k:v for k,v in item.items() if k != 'messages'}, fixture=fixture)
                    | {k:item[k] for k in ('receivedAtEpoch','sequence','inputId') if k in item}
                    for item in raw['messages']]
        refs = list(dict.fromkeys(ref for m in messages for ref in m['files']))
        refs = [ref for ref in refs if not (fixture and urlsplit(ref).hostname == 'fixtures.simple-agent.invalid'
                 and urlsplit(ref).path.strip('/') in refs)]
        return {**messages[-1], 'message':'\n'.join(m['message'] for m in messages),
                'messages':messages, 'files':refs,
                'warnings':[w for m in messages for w in m['warnings']], 'task':raw.get('task',''),
                'turnId':raw.get('turnId'), 'conversationId':raw.get('conversationId')}
    event = normalize_event(raw["eventRaw"])
    if event is None:
        raise RunError("ignored_event", "메시지 생성 이벤트가 아닙니다.")
    refs = list(dict.fromkeys(raw.get("artifactIds", [])))
    url = media_url(event["message"], fixture=fixture)
    fixture_id = urlsplit(url).path.strip("/") if url and urlsplit(url).hostname == "fixtures.simple-agent.invalid" else None
    if url and url not in refs and fixture_id not in refs:
        refs.append(url)
    extension = PurePosixPath(unquote(urlsplit(url).path)).suffix.lower().lstrip(".") if url else ""
    warnings = [{"code": "video_requires_gemini", "message": VIDEO_WARNING, "basis": "url_extension"}] if extension in VIDEO else []
    single = {"eventRaw": handler_event_raw(raw["eventRaw"]), "messageId": event["messageId"],
              "message": event["message"], "task": raw.get("task", ""), "files": refs, "warnings": warnings}
    return single


async def execute_run(run, store, transport, config=None, *, deps=None, processor=None):
    source = getattr(transport, "replay", None)
    config = config or Config()
    if source:
        recorded = source.get("config", {})
        config = replace(config, **{f.name: recorded[f.name] for f in fields(Config) if f.name in recorded})
    deps = deps or Dependencies(registry=default_registry())
    ctx = RunContext(run, store, transport, config, deps)
    transport.store = store
    transport.run = run
    run.update(status="running", startedAt=ctx.now(),
               config=config.public(), stages=[], operations=[], analyses=[], modelCalls=[],
               toolCalls=[], usage=[], artifactIds=list(run["input"].get("artifactIds", [])))
    services = ToolRuntime(ctx)
    ctx.services = services
    error = None
    try:
        async def process():
            if source and source.get("evidenceMissing"):
                raise RunError("evidence_incomplete", "재생에 필요한 실행 증거가 완전하지 않습니다.",
                               details={"missing": source["evidenceMissing"]})
            data = input_view(run["input"], fixture=run["source"] != "external_input")
            if 'messages' not in data:
                data = {**data, 'messages':[dict(data)], 'turnId':run.get('turnId',run['id']),
                        'conversationId':run.get('conversationId')}
            run["messageId"] = data["messageId"]
            await ctx.stage("input", data, checkpoint=False)
            explicit = run["input"].get("toolCall")
            if explicit is not None:
                if not isinstance(explicit, dict) or set(explicit) != {"name", "arguments"}:
                    raise RunError("invalid_tool_call", "toolCall에는 name과 arguments가 필요합니다.")
                if run["mode"] == "preview":
                    spec = ctx.registry.get(explicit["name"])
                    ctx.registry.authorize(spec, ctx)
                    ctx.registry.validate_arguments(spec, explicit["arguments"], ctx)
            if run["mode"] == "preview":
                run["preview"] = {"input": data, "config": config.public(), "tools": ctx.registry.schemas(ctx.permissions),
                                  "toolCall": explicit, "executed": False, "handlerStatus": "not_executed"}
                run["output"] = input_reply(data['message'], data['files'], [])
                run["status"] = "preview"
            else:
                if explicit:
                    result = await services.invoke(explicit["name"], explicit["arguments"])
                    run["output"] = (result.get("text") or json_text(result)) if isinstance(result, dict) else str(result)
                else:
                    run["output"] = await (processor or handler.handle)(data, services)
                    if not isinstance(run["output"], str) or not run["output"].strip():
                        raise RunError("invalid_handler_output", "처리 함수는 비어 있지 않은 답장 문자열을 반환해야 합니다.")
                ctx.verify_replay_end()
                if source and run["output"] != source.get("output"):
                    raise RunError("replay_mismatch","재생의 최종 결과가 원본과 다릅니다.")
                run["status"] = "completed"
            run["reply"] = reply_text(run["output"])
        await process()
    except TurnInterrupted:
        run.update(status='interrupted', output=None, reply=None)
        try: ctx.verify_replay_end()
        except Exception as exc: error = ctx.error(exc)
    except (Exception, asyncio.CancelledError) as exc:
        error = ctx.error(exc)
    orphaned = await services.close()
    if orphaned and error is None:
        error = RunError("unawaited_tool_calls", "처리 함수가 기다리지 않은 도구 호출을 종료했습니다.").record()
    unfinished = [name for name in ("toolCalls", "operations", "modelCalls")
                  if any(item.get("status") == "started" for item in run[name])]
    if unfinished:
        run["evidenceMissing"] = list(dict.fromkeys([*run.get("evidenceMissing", []), *unfinished]))
        error = error or RunError("evidence_incomplete", "종료 결과가 없는 작업 기록이 있습니다.").record()
    if error is None and any(op.get("changing") and op.get("status") == "indeterminate" for op in run["operations"]):
        error = RunError("indeterminate", "변경 작업의 완료 여부가 불명확합니다.").record()
    if error is not None:
        if error["code"] != "indeterminate" and any(op["status"] == "indeterminate" for op in run["operations"]):
            error = {"code": "indeterminate", "message": "종료된 외부 작업의 결과가 불명확합니다.", "details": {"cause": error}}
        run.update(status="indeterminate" if error["code"] == "indeterminate" else "failed",
                   error=error, output=None, reply=reply_text(f"처리를 완료하지 못했습니다. [{error['code']}] {error['message']}"))
        await ctx.stage("error", error, run["status"], checkpoint=False)
    elif run['status'] == 'interrupted':
        await ctx.stage('turn_interrupted', run['interruption'], 'interrupted', checkpoint=False)
    else:
        await ctx.stage("final_output", {"output": run["output"], "reply": run["reply"]}, checkpoint=False)
    # Archived replay must never silently consume values changed by redaction.
    missing = []
    for name in ("input", "output"):
        value = run.get(name)
        if name in run and redact(value) != value:
            missing.append("redacted_" + name)
    if any(redact({key: op[key] for key in ("request", "result", "error") if key in op}) !=
           {key: op[key] for key in ("request", "result", "error") if key in op} for op in run["operations"]):
        missing.append("redacted_operations")
    if missing:
        run["evidenceMissing"] = list(dict.fromkeys([*run.get("evidenceMissing", []), *missing]))
    run["completedAt"] = ctx.now()
    run["staticChecks"] = static_checks(run)
    run["usageSummary"] = aggregate_usage(run["usage"], "replay" if source else run["mode"])
    await ctx.save()
    return run

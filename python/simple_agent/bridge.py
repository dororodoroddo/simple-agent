"""Artifact references accepted by the in-process agent runtime."""
from urllib.parse import urlsplit

from .protocol import media_url, normalize_event


def input_artifact_ids(run):
    input_ids = list(run["input"].get("artifactIds", []))
    fixture = run.get("source") == "synthetic_fixture"
    event = normalize_event(run["input"]["eventRaw"])
    url = media_url(event["message"], fixture=fixture) if event else None
    refs = [url] if url else []
    explicit = run["input"].get("toolCall")
    if isinstance(explicit, dict) and isinstance(explicit.get("arguments"), dict):
        arguments = explicit["arguments"]
        if isinstance(arguments.get("files"), list):
            refs.extend(arguments["files"])
        if isinstance(arguments.get("ref"), str):
            refs.append(arguments["ref"])
    for ref in refs:
        if not isinstance(ref, str):
            continue
        if "://" not in ref:
            input_ids.append(ref)
        elif fixture and urlsplit(ref).hostname == "fixtures.simple-agent.invalid":
            input_ids.append(urlsplit(ref).path.strip("/"))
    return list(dict.fromkeys(input_ids))


def artifact_allowed(meta, run_id, input_ids):
    return meta.get("runId") == run_id or (meta.get("runId") is None and meta["id"] in input_ids)

"""Official rate-based estimates; accounting never controls execution."""
from collections import defaultdict
from datetime import date

CHECKED_AT = "2026-10-27"
OPENAI_SOURCE = "https://developers.openai.com/api/docs/pricing"
GEMINI_SOURCE = "https://ai.google.dev/gemini-api/docs/pricing#gemini-3.6-flash"
VERIFIED_PRICES = {
    "text-embedding-3-small": {"input": .02, "source": "https://developers.openai.com/api/docs/models/text-embedding-3-small"},
    "gpt-6-luna": {"input": .10, "cached": .01, "write30": .125, "write12": .25, "output": .50, "source": OPENAI_SOURCE},
    "gpt-6.1-sol": {"input": 2., "cached": .10, "write30": 2.50, "write12": 5., "output": 10., "source": OPENAI_SOURCE},
    "gpt-6.1-terra": {"input": 1., "cached": .05, "write30": 1.25, "write12": 2.50, "output": 5., "source": "https://developers.openai.com/api/docs/models/gpt-6.1-terra", "checked_at": "2026-10-27"},
    "gpt-audio-1.5": {"input": 2.50, "output": 10., "audio_input": 32., "audio_output": 64., "source": "https://developers.openai.com/api/docs/models/gpt-audio-1.5"},
    "gemini-3.6-flash": {"input": .75, "cached": .075, "output": 3.75, "source": GEMINI_SOURCE, "intro_end": "2026-12-31"},
}


def count(value):
    return value if type(value) is int and value >= 0 else None


def usage_record(payload, provider, model=None, *, origin="live", at=None):
    raw_usage = payload.get("usage", payload.get("usageMetadata"))
    usage = raw_usage if isinstance(raw_usage, dict) else {}
    model = payload.get("model") or model or "unknown"
    # Only exact, dated snapshot suffixes of a configured family inherit its rates.
    import re
    family = next((m for m in VERIFIED_PRICES if model == m or re.fullmatch(re.escape(m)+r"-\d{4}-\d{2}-\d{2}", model)), model)
    price = VERIFIED_PRICES.get(family)
    inp = count(usage.get("total_input_tokens", usage.get("input_tokens", usage.get("prompt_tokens", usage.get("promptTokenCount")))))
    out = count(usage.get("total_output_tokens", usage.get("output_tokens", usage.get("completion_tokens"))))
    lines, assumptions, cost, reason = [], [], None, None
    tier = payload.get("service_tier") or "standard"
    def line(name, tokens, rate):
        if count(tokens) is None:
            raise ValueError("invalid token breakdown")
        lines.append({"category": name, "tokens": tokens, "usd_per_million": rate, "estimated_usd": tokens * rate / 1_000_000})
    try:
        if not price or inp is None:
            raise ValueError("model rate or usage absent")
        if provider == "openai" and family == "text-embedding-3-small":
            if tier not in {"standard", "default", "auto"}:
                raise ValueError("unsupported embedding service tier")
            line("input", inp, price["input"])
            # Embeddings have input usage only; no output tokens are billed.
            out = 0
        elif out is None:
            raise ValueError("output usage absent")
        elif provider == "gemini" and family == "gemini-3.6-flash":
            if tier not in {"standard", "default", "auto"}:
                raise ValueError("unsupported Gemini service tier")
            factor = 2 if (at or date.today().isoformat())[:10] > price["intro_end"] else 1
            cached = usage.get("total_cached_tokens", 0)
            thought = usage.get("total_thought_tokens", 0)
            if count(cached) is None or cached > inp:
                raise ValueError("invalid cached usage")
            line("input", inp-cached, price["input"]*factor)
            line("cached_input", cached, price["cached"]*factor)
            line("output", out, price["output"]*factor)
            line("thinking_output", thought, price["output"]*factor)
            assumptions.append("Gemini Developer API Standard paid-tier estimate; free-tier credits and account discounts excluded")
        elif provider == "openai" and family == "gpt-audio-1.5":
            if tier not in {"standard", "default", "auto"}:
                raise ValueError("unsupported service tier")
            idetail, odetail = usage.get("prompt_tokens_details", {}), usage.get("completion_tokens_details", {})
            ai, ao = idetail.get("audio_tokens"), odetail.get("audio_tokens", 0)
            if count(ai) is None or count(ao) is None or ai > inp or ao > out or idetail.get("cached_tokens", 0):
                raise ValueError("audio/cache breakdown missing or unsupported")
            line("text_input", inp-ai, price["input"]); line("audio_input", ai, price["audio_input"])
            line("text_output", out-ao, price["output"]); line("audio_output", ao, price["audio_output"])
        elif provider == "openai" and family in {"gpt-6-luna", "gpt-6.1-sol", "gpt-6.1-terra"}:
            factor = {"standard": 1, "default": 1, "auto": 1, "flex": .5, "batch": .5, "fast": 2, "priority": 2, "ultrafast": 6}.get(tier)
            if factor is None or (tier == "ultrafast" and family != "gpt-6.1-sol"):
                raise ValueError("unsupported service tier")
            details = usage.get("input_tokens_details", {})
            cached, writes = details.get("cached_tokens", 0), details.get("cache_write_tokens", 0)
            split = details.get("cache_write_tokens_details", {})
            w12 = split.get("ttl_12h_tokens", 0)
            w30 = split.get("ttl_30m_tokens", writes-w12)
            if any(count(v) is None for v in (cached, writes, w30, w12)) or w30+w12 != writes or cached+writes > inp:
                raise ValueError("invalid cache breakdown")
            if not details:
                assumptions.append("Cache usage omitted by response; uncached input estimate")
            if writes and not split:
                assumptions.append("Cache TTL split omitted; request default 30m applied")
            input_factor, output_factor = (2, 1.5) if inp > 272000 else (1, 1)
            for key, n in [("input", inp-cached-writes), ("cached", cached), ("write30", w30), ("write12", w12)]:
                line(key, n, price[key]*input_factor*factor)
            line("output_including_reasoning", out, price["output"]*output_factor*factor)
        else:
            raise ValueError("provider/model mismatch")
        cost = round(sum(x["estimated_usd"] for x in lines), 12)
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        reason = str(exc)
        lines = []
    thought = count(usage.get("total_thought_tokens", 0)) if provider == "gemini" else 0
    output_total = out + thought if out is not None and thought is not None else None
    return {"provider": provider, "model": model, "origin": origin, "raw": raw_usage, "input_tokens": inp, "output_tokens": output_total,
            "reasoning_tokens": thought if provider == "gemini" else None,
            "estimated_cost_usd": cost, "cost_status": "estimated" if cost is not None else "unknown",
            "actual_billed_usd": None, "price_source": price["source"] if price else None, "price_date": price.get("checked_at", CHECKED_AT) if price else CHECKED_AT,
            "service_tier": tier, "breakdown": lines, "assumptions": assumptions, "unknown_reason": reason,
            "cost_note": "공식 요율 기반 추정치. 실제 청구액·계정 할인·크레딧과 다를 수 있습니다."}

def aggregate_usage(records, mode):
    groups = defaultdict(list)
    for record in records:
        groups[(record["provider"], record["model"])].append(record)
    def total(items):
        missing = sum(x.get("estimated_cost_usd") is None for x in items)
        return {"calls": len(items), "input_tokens": sum(x.get("input_tokens") or 0 for x in items),
                "output_tokens": sum(x.get("output_tokens") or 0 for x in items),
                "unknown_usage_calls": sum(x.get("input_tokens") is None or x.get("output_tokens") is None for x in items),
                "unknown_cost_calls": missing,
                "estimated_cost_usd": None if missing else sum(x["estimated_cost_usd"] for x in items),
                "known_estimated_subtotal_usd": sum(x.get("estimated_cost_usd") or 0 for x in items), "actual_billed_usd": None}
    actual = [r for r in records if r.get("origin") == "live"]
    return {"costLimit": None, "mode": mode, "newUsage": total(actual), "referenceUsage": total([r for r in records if r.get("origin") != "live"]),
            "byModel": [{"provider": p, "model": m, "origin": sorted(set(r["origin"] for r in rows)), **total(rows)} for (p, m), rows in groups.items()]}

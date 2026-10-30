"""Transport recovery only; never wrap generation or token-count calls.

RFC 9110 §9.2.2 permits replaying idempotent requests after connection failure
and advises against automatically retrying a failed automatic retry. We use
that single-recovery policy, not an application workload/cost limit.
HTTP responses are retried only with an explicit server Retry-After and a
retryable endpoint contract. No invented sleep/backoff defaults.
"""
import asyncio
import time
from email.utils import parsedate_to_datetime
from .errors import RunError

SOURCE = 'https://www.rfc-editor.org/rfc/rfc9110.html#section-9.2.2'


def retry_after(headers, clock=time.time):
    value = next((str(v) for k,v in headers.items() if k.lower()=='retry-after'), None)
    if value is None: return None
    try:
        if value.isdigit(): return float(value)
        return max(0, parsedate_to_datetime(value).timestamp()-clock())
    except (ValueError, TypeError, OverflowError): return None


async def transfer(send, *, safe, events, operation, retryable=None, sleep=asyncio.sleep):
    """send returns (status, headers, result). Streams are reopened by send."""
    for attempt in (1,2):
        event={'operation':operation,'attempt':attempt,'source':SOURCE}
        events.append(event)
        try:
            result=await send()
        except Exception as exc:
            code=getattr(exc,'code',type(exc).__name__)
            event.update(status='failed',error=code)
            network=code in {'network_error','ConnectError','ReadError','WriteError','RemoteProtocolError','ConnectTimeout','ReadTimeout'}
            if not (safe and network and attempt==1): raise
            event['retryReason']='idempotent_connection_recovery'
            continue
        status,headers,_=result
        event.update(status='received',httpStatus=status)
        delay=retry_after(headers)
        eligible=retryable(result) if retryable else safe and status in {408,429,500,502,503,504}
        if attempt==1 and eligible and delay is not None:
            event.update(retryReason='server_retry_after',waitSeconds=delay)
            await sleep(delay)
            continue
        return result

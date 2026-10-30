"""User-owned entrypoint. Add registered business nodes when the task is known."""
from .planner import run_default


async def handle(input_data, tools):
    """Return reply text. tools.invoke(name, arguments) is an explicit operation.

    input_data contains the original event, message, task and file references.
    Business logic and tool selection belong here when the task is known.
    """
    return await run_default(input_data, tools)

# Project Instructions

This repository contains a provider-neutral Python agent core. The user-owned handler is `python/simple_agent/handler.py`.

## Working rules

- Keep provider credentials and local environment files out of commits. Use `.env.example` for names and safe defaults only.
- Keep provider I/O behind the existing transport and tool boundaries. Preserve permission checks and explicit model configuration.
- Keep this repository limited to the agent core and isolated E2E checks.
- For node and workflow changes, follow `docs/workflow.md`. For prompt work, follow `docs/prompt-writing.md`.
- Run `npm run test:e2e` for changes. Do not skip checks or weaken assertions.
- E2E checks use fixture responses and must not call real model APIs or external services.
- Do not publish releases or deploy.
- Keep project-local Git credentials under `.git`; never copy them into tracked files or change global Git identity settings.
